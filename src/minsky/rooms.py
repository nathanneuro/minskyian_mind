"""Room implementations for the Society of Mind architecture.

Each room is implemented as a set of functions that process incoming messages
and generate outgoing messages. Rooms do not directly communicate - all
message passing goes through the orchestrator.

Each room can optionally have:
- llm_fn: Frozen LLM (RWKV) for generating responses
- edit_fn: T5 edit model for modifying LLM outputs

Motor room has access to tools:
- web_search: Search the web
- scratchpad_write/read: Persistent storage
- python_exec: Sandboxed Python execution
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


def apply_edit(text: str, edit_fn, context: str = "") -> str:
    """Apply edit function to text if available."""
    if edit_fn is None:
        return text
    try:
        return edit_fn(text, context)
    except Exception as e:
        print(f"Edit failed: {e}")
        return text


# -----------------------------------------------------------------------------
# Sensory Room
# -----------------------------------------------------------------------------
# Gets flooded with ALL incoming data (perceptions, tool outputs, actions)
# Must summarize and decide what to pass to Planning and Motor
# Responds to attention requests from Planning

SENSORY_SYSTEM_PROMPT = f"""You are the Sensory module. You receive ALL incoming data.
Summarize what matters in {MESSAGE_MAX_LENGTH} chars or less.
Output TWO lines:
TO_PLANNING: <summary for strategic decisions>
TO_MOTOR: <context for action execution>"""


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

    Each message is limited to MESSAGE_MAX_LENGTH chars.
    """
    # Collect all incoming data this cycle
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

    # Build combined data summary
    all_data = []
    if perceptions:
        all_data.append(f"Perceptions: {'; '.join(p[:50] for p in perceptions)}")
    if tool_outputs:
        all_data.append(f"Tools: {'; '.join(t[:50] for t in tool_outputs)}")
    if motor_actions:
        all_data.append(f"Actions: {'; '.join(a[:50] for a in motor_actions)}")
    if attention_request:
        all_data.append(f"Attention focus: {attention_request[:50]}")

    combined = " | ".join(all_data) if all_data else "No new data"

    # Generate summaries for Planning and Motor
    if llm_fn:
        response = llm_fn(
            f"{SENSORY_SYSTEM_PROMPT}\n\n"
            f"---\n\n"
            f"Incoming data: {combined}\n"
            f"Current context: {state.current_context[:200]}\n\n"
            f"Generate your TO_PLANNING and TO_MOTOR summaries:"
        )
        response = apply_edit(response, edit_fn, context="sensory_summarize")

        # Parse response
        import re
        planning_match = re.search(r'TO_PLANNING:\s*(.+?)(?=TO_MOTOR:|$)', response, re.IGNORECASE | re.DOTALL)
        motor_match = re.search(r'TO_MOTOR:\s*(.+?)$', response, re.IGNORECASE | re.DOTALL)

        to_planning = planning_match.group(1).strip() if planning_match else combined
        to_motor = motor_match.group(1).strip() if motor_match else state.metadata.get("last_tool_output", "")
    else:
        to_planning = combined
        to_motor = state.metadata.get("last_tool_output", "No context")

    cycle = incoming[0].cycle if incoming else 0

    # Output exactly 2 messages (auto-truncated by Message class)
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
# No direct contact with the external world
# Has two "tools":
#   1. Attention requests -> Sensory (query what to focus on)
#   2. High-level commands -> Motor (what to do, not how)
# Motor decides how to implement commands (which tools to use)

PLANNING_SYSTEM_PROMPT = f"""You are the Planning module. No direct world contact.
Generate hypotheses, pick highest EV action.
Output TWO lines ({MESSAGE_MAX_LENGTH} chars each):
TO_SENSORY: <attention focus request>
TO_MOTOR: <high-level command - what, not how>"""


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

    Each message is limited to MESSAGE_MAX_LENGTH chars.
    Motor decides HOW to implement commands.
    """
    # Collect incoming data
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

    # Build context
    context = sensory_data
    if action_result:
        context += f" | Last action result: {action_result[:50]}"

    # Generate outputs for Sensory and Motor
    if llm_fn:
        response = llm_fn(
            f"{PLANNING_SYSTEM_PROMPT}\n\n"
            f"---\n\n"
            f"Sensory input: {sensory_data[:200]}\n"
            f"Previous context: {state.metadata.get('last_action_result', 'none')[:100]}\n\n"
            f"1. Form hypotheses about the situation.\n"
            f"2. Estimate expected value of actions.\n"
            f"3. Generate TO_SENSORY (what to focus on) and TO_MOTOR (what to do):"
        )
        response = apply_edit(response, edit_fn, context=f"planning: {sensory_data[:50]}")
        state.metadata["last_planning_response"] = response

        # Parse response
        import re
        sensory_match = re.search(r'TO_SENSORY:\s*(.+?)(?=TO_MOTOR:|$)', response, re.IGNORECASE | re.DOTALL)
        motor_match = re.search(r'TO_MOTOR:\s*(.+?)$', response, re.IGNORECASE | re.DOTALL)

        to_sensory = sensory_match.group(1).strip() if sensory_match else "Focus on current situation"
        to_motor = motor_match.group(1).strip() if motor_match else "Respond appropriately"
    else:
        to_sensory = f"Focus on: {sensory_data[:100]}"
        to_motor = f"Act on: {sensory_data[:100]}"

    cycle = incoming[0].cycle if incoming else 0

    # Output exactly 2 messages (auto-truncated by Message class)
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
# Executes actions in the world (messages, tool calls, etc.)
# Receives instructions from Planning
# Gets context from Sensory
# Has access to tools: web_search, scratchpad, python_exec

MOTOR_SYSTEM_PROMPT = f"""You are the Motor module. Execute actions based on Planning's commands.
You do NOT see tool results - those go to Sensory.
{get_tools_description()}

Output THREE lines ({MESSAGE_MAX_LENGTH} chars each):
TO_SENSORY: <tool output or action description>
TO_PLANNING: <action result summary>
TO_EXTERNAL: <response to user, or empty if tool call>

For tool calls, format: TOOL:name ARGS:{{"key":"val"}}"""


def parse_motor_output(output: str) -> tuple[str, dict | None, str | None]:
    """Parse Motor's LLM output into action type, args, and response."""
    output = output.strip()

    # Check for tool call
    tool_match = re.search(r'TOOL:\s*(\w+)', output, re.IGNORECASE)
    if tool_match:
        tool_name = tool_match.group(1).lower()
        args_match = re.search(r'ARGS:\s*(\{.*?\})', output, re.IGNORECASE | re.DOTALL)
        try:
            import json
            args = json.loads(args_match.group(1)) if args_match else {}
        except:
            args = {}
        return ("tool", {"tool": tool_name, **args}, None)

    # Check for direct response
    response_match = re.search(r'TO_EXTERNAL:\s*(.+?)(?=TO_|$)', output, re.IGNORECASE | re.DOTALL)
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
    - motor_to_sensory: Tool output (Motor doesn't see this) or action description
    - motor_to_planning: Action result summary
    - motor_to_external: User-facing response (if any)

    Each message is limited to MESSAGE_MAX_LENGTH chars.
    Motor decides HOW to implement Planning's commands.
    """
    # Collect incoming data
    command: str = ""
    sensory_context: str = ""

    for msg in incoming:
        state.add_message(msg)

        if msg.message_type == MessageType.MOTOR_COMMAND:
            command = msg.content

        elif msg.message_type == MessageType.PERCEPTION:
            sensory_context = msg.content
            state.metadata["sensory_context"] = msg.content

    # Generate action based on command and context
    to_sensory = ""
    to_planning = ""
    to_external = ""

    if command:
        prompt = (
            f"{MOTOR_SYSTEM_PROMPT}\n\n"
            f"---\n\n"
            f"Command from Planning: {command}\n"
            f"Context from Sensory: {sensory_context[:150]}\n\n"
            f"Decide how to implement this command:"
        )

        if llm_fn:
            llm_output = llm_fn(prompt)
            llm_output = apply_edit(llm_output, edit_fn, context=f"command: {command[:50]}")

            action_type, tool_args, response = parse_motor_output(llm_output)

            if action_type == "tool" and tool_args:
                tool_name = tool_args.pop("tool")
                result = execute_tool(tool_name, **tool_args)

                state.metadata.setdefault("tool_history", []).append({
                    "tool": tool_name,
                    "args": tool_args,
                    "success": result.success,
                })

                # Motor doesn't see result - it goes to Sensory
                to_sensory = f"[{tool_name}] {'OK' if result.success else 'FAIL'}: {result.output if result.success else result.error}"
                to_planning = f"Called {tool_name}({tool_args})"
                to_external = ""  # No external response for tool calls
            else:
                to_sensory = f"Response generated: {response[:100] if response else llm_output[:100]}"
                to_planning = f"Responded to user"
                to_external = response or llm_output
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

    # Output exactly 3 messages (auto-truncated by Message class)
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
