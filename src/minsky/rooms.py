"""Room implementations for the Society of Mind architecture.

Each room is implemented as a set of functions that process incoming messages
and generate outgoing messages. Rooms do not directly communicate - all
message passing goes through the orchestrator.

Data flow per room:
1. llm_fn (RWKV base model) generates text via next-token prediction
2. edit_fn (T5) refines the raw text into structured output
3. Only T5's output enters the room (RWKV output is never used directly)
4. If T5 is unavailable (--no-t5), RWKV output is used as fallback

RWKV is a BASE MODEL (not chat-tuned). Prompts must be document-continuation
style with few-shot examples so the model knows what to predict next.
"""

import re
from minsky.types import (
    Message,
    RoomType,
    RoomState,
    MessageType,
    Hypothesis,
    Plan,
    MESSAGE_MAX_LENGTH,
    truncate_message,
)
from minsky.tools import execute_tool, get_tools_description, ToolResult
from minsky.prompts.rooms import (
    SENSORY_PROMPT_TEMPLATE,
    PLANNING_PROMPT_TEMPLATE,
    MOTOR_PROMPT_TEMPLATE,
)


def apply_edit(raw_llm_output: str, edit_fn, context: str = "", task_prefix: str = "edit") -> str:
    """Pass RWKV output through T5 edit model.

    RWKV output should NEVER reach rooms directly. T5 is the gate.
    If edit_fn is None (--no-t5 mode), raw output passes through.
    If edit_fn raises, logs error and returns raw output as last resort.

    Args:
        raw_llm_output: Raw text from RWKV.
        edit_fn: T5 edit function (text, context, task_prefix) -> str.
        context: Short description of input context.
        task_prefix: Room-specific prefix (edit_sensory, edit_planning, edit_motor).
    """
    if edit_fn is None:
        return raw_llm_output
    try:
        edited = edit_fn(raw_llm_output, context, task_prefix)
        if edited and edited.strip():
            return edited
        print(f"WARNING: T5 returned empty output for context='{context}', using raw RWKV")
        return raw_llm_output
    except Exception as e:
        print(f"ERROR: T5 edit failed for context='{context}': {e}")
        return raw_llm_output


# -----------------------------------------------------------------------------
# Sensory Room
# -----------------------------------------------------------------------------

def sensory_process(
    state: RoomState,
    incoming: list[Message],
    llm_fn=None,
    edit_fn=None,
) -> tuple[RoomState, list[Message]]:
    """
    Process incoming messages in the Sensory room.

    Outputs exactly 2 messages per global step:
    - sensory_to_planning: Summary for strategic decisions
    - sensory_to_motor: Context for action execution
    """
    perceptions: list[str] = []
    tool_outputs: list[str] = []
    motor_actions: list[str] = []
    attention_request: str = ""

    for msg in incoming:
        state.add_message(msg)
        if msg.message_type == MessageType.PERCEPTION:
            perceptions.append(msg.content)
            state.current_context = msg.content
            state.metadata["last_perception"] = msg.content
        elif msg.message_type == MessageType.ATTENTION_REQUEST:
            attention_request = msg.content
        elif msg.message_type == MessageType.ACTION:
            motor_actions.append(msg.content)
            state.metadata["last_motor_action"] = msg.content
        elif msg.message_type == MessageType.TOOL_OUTPUT:
            tool_outputs.append(msg.content)
            state.metadata["last_tool_output"] = msg.content
            state.current_context += f"\n[Tool]: {msg.content[:100]}"

    # Build input data string
    parts = []
    if perceptions:
        parts.append(f"Perceptions: {'; '.join(p[:80] for p in perceptions)}")
    if tool_outputs:
        parts.append(f"Tool results: {'; '.join(t[:80] for t in tool_outputs)}")
    if motor_actions:
        parts.append(f"Actions taken: {'; '.join(a[:80] for a in motor_actions)}")
    if attention_request:
        parts.append(f"Planning asks you to focus on: {attention_request[:80]}")
    combined = " | ".join(parts) if parts else "No new data this cycle."

    if llm_fn:
        prompt = SENSORY_PROMPT_TEMPLATE.format(input_data=combined)
        raw = llm_fn(prompt)
        response = "TO_PLANNING:" + raw  # Prepend since prompt ends at TO_PLANNING:
        response = apply_edit(response, edit_fn, context=combined[:80], task_prefix="edit_sensory")

        planning_match = re.search(r'TO_PLANNING:\s*(.+?)(?=TO_MOTOR:|$)', response, re.IGNORECASE | re.DOTALL)
        motor_match = re.search(r'TO_MOTOR:\s*(.+?)$', response, re.IGNORECASE | re.DOTALL)

        to_planning = planning_match.group(1).strip() if planning_match else combined
        to_motor = motor_match.group(1).strip() if motor_match else state.metadata.get("last_tool_output", "")
    else:
        to_planning = combined
        to_motor = state.metadata.get("last_tool_output", "No context")

    cycle = incoming[0].cycle if incoming else 0

    outgoing = [
        Message(
            content=to_planning,
            source=RoomType.SENSORY,
            target=RoomType.PLANNING,
            message_type=MessageType.PERCEPTION,
            cycle=cycle,
        ),
        Message(
            content=to_motor,
            source=RoomType.SENSORY,
            target=RoomType.MOTOR,
            message_type=MessageType.PERCEPTION,
            cycle=cycle,
        ),
    ]

    return state, outgoing


# -----------------------------------------------------------------------------
# Planning Room
# -----------------------------------------------------------------------------

def planning_process(
    state: RoomState,
    incoming: list[Message],
    llm_fn=None,
    edit_fn=None,
) -> tuple[RoomState, list[Message]]:
    """
    Process incoming messages in the Planning room.

    Outputs exactly 2 messages per global step:
    - planning_to_sensory: Attention focus request
    - planning_to_motor: High-level command (what, not how)
    """
    sensory_data: str = ""
    action_result: str = ""

    for msg in incoming:
        state.add_message(msg)
        if msg.message_type == MessageType.PERCEPTION:
            sensory_data = msg.content
            state.current_context = msg.content
        elif msg.message_type == MessageType.ACTION_RESULT:
            action_result = msg.content
            state.metadata["last_action_result"] = msg.content

    input_parts = [sensory_data]
    if action_result:
        input_parts.append(f"Previous action result: {action_result}")
    input_data = " | ".join(input_parts)

    if llm_fn:
        prompt = PLANNING_PROMPT_TEMPLATE.format(input_data=input_data)
        raw = llm_fn(prompt)
        response = "HYPOTHESES:" + raw  # Prepend since prompt ends at HYPOTHESES:
        response = apply_edit(response, edit_fn, context=sensory_data[:80], task_prefix="edit_planning")
        state.metadata["last_planning_response"] = response

        sensory_match = re.search(r'TO_SENSORY:\s*(.+?)(?=TO_MOTOR:|$)', response, re.IGNORECASE | re.DOTALL)
        motor_match = re.search(r'TO_MOTOR:\s*(.+?)$', response, re.IGNORECASE | re.DOTALL)

        to_sensory = sensory_match.group(1).strip() if sensory_match else f"Observe: {sensory_data[:80]}"
        to_motor = motor_match.group(1).strip() if motor_match else f"Investigate: {sensory_data[:80]}"
    else:
        to_sensory = f"Focus on: {sensory_data[:100]}"
        to_motor = f"Act on: {sensory_data[:100]}"

    cycle = incoming[0].cycle if incoming else 0

    outgoing = [
        Message(
            content=to_sensory,
            source=RoomType.PLANNING,
            target=RoomType.SENSORY,
            message_type=MessageType.ATTENTION_REQUEST,
            cycle=cycle,
        ),
        Message(
            content=to_motor,
            source=RoomType.PLANNING,
            target=RoomType.MOTOR,
            message_type=MessageType.MOTOR_COMMAND,
            cycle=cycle,
        ),
    ]

    return state, outgoing


# -----------------------------------------------------------------------------
# Motor Room
# -----------------------------------------------------------------------------

def parse_motor_output(output: str) -> tuple[str, dict | None, str | None]:
    """Parse Motor's LLM output into action type, args, and response."""
    import json
    output = output.strip()

    # Check for tool call
    tool_match = re.search(r'TOOL:\s*(\w+)', output, re.IGNORECASE)
    if tool_match:
        tool_name = tool_match.group(1).lower()

        # Find ARGS: and parse JSON with balanced brace matching
        args = {}
        args_start = re.search(r'ARGS:\s*\{', output, re.IGNORECASE)
        if args_start:
            start_idx = args_start.end() - 1
            brace_count = 0
            end_idx = start_idx
            for i, char in enumerate(output[start_idx:], start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break

            json_str = output[start_idx:end_idx]
            try:
                args = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"Failed to parse tool args: {e}")
                args = {}

        return ("tool", {"tool": tool_name, **args}, None)

    # Check for direct response
    response_match = re.search(r'TO_EXTERNAL:\s*(.+?)(?=\nTO_|$)', output, re.IGNORECASE | re.DOTALL)
    if response_match:
        return ("response", None, response_match.group(1).strip())

    return ("response", None, output)


def motor_process(
    state: RoomState,
    incoming: list[Message],
    llm_fn=None,
    edit_fn=None,
    action_fn=None,
) -> tuple[RoomState, list[Message]]:
    """
    Process incoming messages in the Motor room.

    Outputs exactly 3 messages per global step:
    - motor_to_sensory: Tool output (Motor doesn't see this)
    - motor_to_planning: Action result summary
    - motor_to_external: User-facing response (if any)
    """
    command: str = ""
    sensory_context: str = ""

    for msg in incoming:
        state.add_message(msg)
        if msg.message_type == MessageType.MOTOR_COMMAND:
            command = msg.content
        elif msg.message_type == MessageType.PERCEPTION:
            sensory_context = msg.content
            state.metadata["sensory_context"] = msg.content

    to_sensory = ""
    to_planning = ""
    to_external = ""

    if command:
        if llm_fn:
            prompt = MOTOR_PROMPT_TEMPLATE.format(
                command=command[:200],
                context=sensory_context[:200],
            )
            raw = llm_fn(prompt)
            output = "ACTION:" + raw  # Prepend since prompt ends at ACTION:
            output = apply_edit(output, edit_fn, context=command[:80], task_prefix="edit_motor")

            action_type, tool_args, response = parse_motor_output(output)

            if action_type == "tool" and tool_args:
                tool_name = tool_args.pop("tool")
                result = execute_tool(tool_name, **tool_args)

                state.metadata.setdefault("tool_history", []).append({
                    "tool": tool_name,
                    "args": tool_args,
                    "success": result.success,
                })

                to_sensory = f"[{tool_name}] {'OK' if result.success else 'FAIL'}: {result.output if result.success else result.error}"
                to_planning = f"Called {tool_name}({tool_args})"
                to_external = ""
            else:
                # Parse structured fields from T5-edited response
                sensory_match = re.search(r'TO_SENSORY:\s*(.+?)(?=\nTO_PLANNING:|\nTO_EXTERNAL:|$)', output, re.IGNORECASE | re.DOTALL)
                planning_match = re.search(r'TO_PLANNING:\s*(.+?)(?=\nTO_SENSORY:|\nTO_EXTERNAL:|$)', output, re.IGNORECASE | re.DOTALL)
                external_match = re.search(r'TO_EXTERNAL:\s*(.+?)(?=\nTO_SENSORY:|\nTO_PLANNING:|$)', output, re.IGNORECASE | re.DOTALL)

                to_sensory = sensory_match.group(1).strip() if sensory_match else f"Executed: {command[:80]}"
                to_planning = planning_match.group(1).strip() if planning_match else f"Done: {command[:80]}"
                to_external = external_match.group(1).strip() if external_match else (response or "")
        else:
            to_sensory = f"[Stub] Executing: {command[:100]}"
            to_planning = f"Action taken: {command[:100]}"
            to_external = f"[Processing: {command[:100]}]"

        if action_fn:
            action_fn(to_planning)
    else:
        to_sensory = "No command received"
        to_planning = "Idle"
        to_external = ""

    cycle = incoming[0].cycle if incoming else 0

    outgoing = [
        Message(
            content=to_sensory,
            source=RoomType.MOTOR,
            target=RoomType.SENSORY,
            message_type=MessageType.TOOL_OUTPUT,
            cycle=cycle,
        ),
        Message(
            content=to_planning,
            source=RoomType.MOTOR,
            target=RoomType.PLANNING,
            message_type=MessageType.ACTION_RESULT,
            cycle=cycle,
        ),
        Message(
            content=to_external,
            source=RoomType.MOTOR,
            target=RoomType.EXTERNAL,
            message_type=MessageType.ACTION,
            cycle=cycle,
        ),
    ]

    return state, outgoing


# -----------------------------------------------------------------------------
# Room Registry
# -----------------------------------------------------------------------------

ROOM_PROCESSORS = {
    RoomType.SENSORY: sensory_process,
    RoomType.PLANNING: planning_process,
    RoomType.MOTOR: motor_process,
}


def create_room_state(room_type: RoomType) -> RoomState:
    """Create initial state for a room."""
    return RoomState(room_type=room_type)
