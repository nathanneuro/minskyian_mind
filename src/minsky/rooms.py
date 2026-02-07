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

SENSORY_SYSTEM_PROMPT = """You are the Sensory module of a cognitive system.
You receive ALL incoming data: perceptions, tool outputs, action observations.
Your job is to SUMMARIZE what's relevant and pass it along.

You do NOT filter what you receive - everything comes in.
You DO decide what summaries to pass to:
- Planning: Needs strategic info for decision-making
- Motor: Needs context relevant to action execution

Be concise. Extract signal from noise. Highlight what matters."""


def sensory_process(
    state: RoomState,
    incoming: list[Message],
    llm_fn=None,
    edit_fn=None,
) -> tuple[RoomState, list[Message]]:
    """
    Process incoming messages in the Sensory room.

    Sensory's role:
    - Gets flooded with ALL data (no input filtering)
    - Summarizes data for Planning and Motor
    - Responds to attention requests from Planning
    - Provides context to Motor for action execution
    """
    outgoing: list[Message] = []

    # Collect all incoming data this cycle
    perceptions: list[str] = []
    tool_outputs: list[str] = []
    motor_actions: list[str] = []

    for msg in incoming:
        state.add_message(msg)

        if msg.message_type == MessageType.PERCEPTION:
            # External perception - accumulate
            perceptions.append(msg.content)
            state.current_context = msg.content
            state.metadata["last_perception"] = msg.content

        elif msg.message_type == MessageType.ATTENTION_REQUEST:
            # Planning wants us to focus on something specific
            attention_target = msg.content

            if llm_fn:
                response_content = llm_fn(
                    f"{SENSORY_SYSTEM_PROMPT}\n\n"
                    f"---\n\n"
                    f"Planning requests attention on: {attention_target}\n"
                    f"Current context: {state.current_context}\n"
                    f"Recent tool outputs: {state.metadata.get('last_tool_output', 'none')}\n\n"
                    f"Summarize what you observe about this focus area:"
                )
                response_content = apply_edit(
                    response_content,
                    edit_fn,
                    context=f"attention_request: {attention_target}",
                )
            else:
                response_content = f"[Sensory observed: {attention_target} in context of: {state.current_context[:100]}]"

            outgoing.append(
                Message(
                    content=response_content,
                    source=RoomType.SENSORY,
                    target=RoomType.PLANNING,
                    message_type=MessageType.ATTENTION_RESPONSE,
                    cycle=msg.cycle,
                )
            )

        elif msg.message_type == MessageType.ACTION:
            # Motor is taking an action - observe it
            motor_actions.append(msg.content)
            state.metadata["last_motor_action"] = msg.content

        elif msg.message_type == MessageType.TOOL_OUTPUT:
            # Tool output from Motor - Sensory perceives the results
            tool_output = msg.content
            tool_outputs.append(tool_output)
            state.metadata["last_tool_output"] = tool_output
            state.current_context += f"\n[Tool Result]: {tool_output}"

    # After processing all messages, summarize and forward to Planning
    if tool_outputs or perceptions:
        # Summarize accumulated data for Planning
        all_data = []
        if perceptions:
            all_data.append(f"Perceptions: {'; '.join(perceptions)}")
        if tool_outputs:
            all_data.append(f"Tool results: {'; '.join(tool_outputs)}")
        if motor_actions:
            all_data.append(f"Motor actions: {'; '.join(motor_actions)}")

        combined = "\n".join(all_data)

        if llm_fn:
            summary = llm_fn(
                f"{SENSORY_SYSTEM_PROMPT}\n\n"
                f"---\n\n"
                f"Incoming data this cycle:\n{combined}\n\n"
                f"Summarize the key information Planning needs to know:"
            )
            summary = apply_edit(summary, edit_fn, context="summarize_for_planning")
        else:
            summary = combined[:500]

        # Forward summary to Planning
        outgoing.append(
            Message(
                content=summary,
                source=RoomType.SENSORY,
                target=RoomType.PLANNING,
                message_type=MessageType.PERCEPTION,
                cycle=incoming[0].cycle if incoming else 0,
            )
        )

        # Also provide relevant context to Motor
        motor_context = state.metadata.get("last_tool_output", "")
        if motor_context:
            outgoing.append(
                Message(
                    content=f"Context for action: {motor_context[:300]}",
                    source=RoomType.SENSORY,
                    target=RoomType.MOTOR,
                    message_type=MessageType.PERCEPTION,
                    cycle=incoming[0].cycle if incoming else 0,
                )
            )

    return state, outgoing


# -----------------------------------------------------------------------------
# Planning Room
# -----------------------------------------------------------------------------
# No direct contact with the external world
# Has two "tools":
#   1. Attention requests -> Sensory (query what to focus on)
#   2. High-level commands -> Motor (what to do, not how)
# Motor decides how to implement commands (which tools to use)

PLANNING_SYSTEM_PROMPT = """You are the Planning module of a cognitive system.
You have NO direct contact with the external world.

Your tools:
1. ATTENTION: Ask Sensory to focus on something (gather info)
2. COMMAND: Tell Motor what to accomplish (high-level goal)

Motor decides HOW to implement your commands (which tools to use).
You just say WHAT you want done.

Generate hypotheses, estimate expected values, pick best action.
Avoid vacillation - commit to a plan once decided."""


def planning_process(
    state: RoomState,
    incoming: list[Message],
    llm_fn=None,
    edit_fn=None,
) -> tuple[RoomState, list[Message]]:
    """
    Process incoming messages in the Planning room.

    Planning's role:
    - Generate hypotheses to explain sensory data
    - Estimate expected value of possible actions
    - Send attention requests to Sensory (queries)
    - Send high-level commands to Motor (goals, not methods)

    Motor decides HOW to implement the commands.
    """
    outgoing: list[Message] = []

    for msg in incoming:
        state.add_message(msg)

        if msg.message_type == MessageType.PERCEPTION:
            # Sensory summary received - time to hypothesize and act
            perception = msg.content

            if llm_fn:
                # Use LLM to generate hypotheses and high-level command
                planning_response = llm_fn(
                    f"{PLANNING_SYSTEM_PROMPT}\n\n"
                    f"---\n\n"
                    f"Sensory reports: {perception}\n\n"
                    f"1. Generate 2+ hypotheses explaining this situation.\n"
                    f"2. For each, estimate expected value (0-1) of acting on it.\n"
                    f"3. Choose the highest EV action.\n\n"
                    f"Output ONE of:\n"
                    f"ATTENTION: <what to focus on> (if you need more info)\n"
                    f"COMMAND: <high-level goal for Motor> (what, not how)"
                )
                planning_response = apply_edit(
                    planning_response,
                    edit_fn,
                    context=f"perception: {perception[:100]}",
                )
                state.metadata["last_planning_response"] = planning_response

                # Parse Planning's output
                if "ATTENTION:" in planning_response.upper():
                    # Planning wants more information
                    import re
                    match = re.search(r'ATTENTION:\s*(.+)', planning_response, re.IGNORECASE | re.DOTALL)
                    attention_content = match.group(1).strip() if match else perception[:50]

                    outgoing.append(
                        Message(
                            content=attention_content,
                            source=RoomType.PLANNING,
                            target=RoomType.SENSORY,
                            message_type=MessageType.ATTENTION_REQUEST,
                            cycle=msg.cycle,
                        )
                    )
                else:
                    # Planning wants Motor to act
                    import re
                    match = re.search(r'COMMAND:\s*(.+)', planning_response, re.IGNORECASE | re.DOTALL)
                    command_content = match.group(1).strip() if match else planning_response

                    outgoing.append(
                        Message(
                            content=command_content,
                            source=RoomType.PLANNING,
                            target=RoomType.MOTOR,
                            message_type=MessageType.MOTOR_COMMAND,
                            cycle=msg.cycle,
                        )
                    )
            else:
                # Stub: generate a simple command
                plan_content = f"[Command based on perception: {perception[:100]}]"
                outgoing.append(
                    Message(
                        content=plan_content,
                        source=RoomType.PLANNING,
                        target=RoomType.MOTOR,
                        message_type=MessageType.MOTOR_COMMAND,
                        cycle=msg.cycle,
                    )
                )

        elif msg.message_type == MessageType.ATTENTION_RESPONSE:
            # Sensory responded to attention request - now decide action
            sensory_info = msg.content

            if llm_fn:
                planning_response = llm_fn(
                    f"{PLANNING_SYSTEM_PROMPT}\n\n"
                    f"---\n\n"
                    f"Attention response from Sensory: {sensory_info}\n"
                    f"Previous context: {state.current_context}\n\n"
                    f"Based on this information:\n"
                    f"COMMAND: <high-level goal for Motor>"
                )
                planning_response = apply_edit(
                    planning_response,
                    edit_fn,
                    context=f"sensory_info: {sensory_info[:100]}",
                )

                # Extract command
                import re
                match = re.search(r'COMMAND:\s*(.+)', planning_response, re.IGNORECASE | re.DOTALL)
                command_content = match.group(1).strip() if match else planning_response
            else:
                command_content = f"[Command based on sensory info: {sensory_info[:50]}]"

            # Send high-level command to Motor
            outgoing.append(
                Message(
                    content=command_content,
                    source=RoomType.PLANNING,
                    target=RoomType.MOTOR,
                    message_type=MessageType.MOTOR_COMMAND,
                    cycle=msg.cycle,
                )
            )

        elif msg.message_type == MessageType.ACTION_RESULT:
            # Motor reported back - evaluate success, possibly replan
            result = msg.content
            state.metadata["last_action_result"] = result

    return state, outgoing


# -----------------------------------------------------------------------------
# Motor Room
# -----------------------------------------------------------------------------
# Executes actions in the world (messages, tool calls, etc.)
# Receives instructions from Planning
# Gets context from Sensory
# Has access to tools: web_search, scratchpad, python_exec

MOTOR_SYSTEM_PROMPT = f"""You are the Motor module of a cognitive system.
Your job is to execute actions in the world based on instructions from Planning.

IMPORTANT: You do NOT see the results of tool calls directly.
Tool outputs go to the Sensory module, which will perceive and report them.
You only know that you issued a command - not what happened.

{get_tools_description()}

To use a tool, output in this format:
TOOL: tool_name
ARGS: {{"arg1": "value1", "arg2": "value2"}}

To respond directly to the user without a tool:
RESPONSE: Your response here

Choose the action that best follows Planning's instruction."""


def parse_motor_output(output: str) -> tuple[str, dict | None, str | None]:
    """Parse Motor's LLM output into action type, args, and response.

    Returns:
        Tuple of (action_type, tool_args, response_text)
        action_type: "tool" or "response"
    """
    output = output.strip()

    # Check for tool call
    tool_match = re.search(r'TOOL:\s*(\w+)', output, re.IGNORECASE)
    if tool_match:
        tool_name = tool_match.group(1).lower()

        # Parse args
        args_match = re.search(r'ARGS:\s*(\{.*?\})', output, re.IGNORECASE | re.DOTALL)
        try:
            import json
            args = json.loads(args_match.group(1)) if args_match else {}
        except:
            args = {}

        return ("tool", {"tool": tool_name, **args}, None)

    # Check for direct response
    response_match = re.search(r'RESPONSE:\s*(.*)', output, re.IGNORECASE | re.DOTALL)
    if response_match:
        return ("response", None, response_match.group(1).strip())

    # Default: treat entire output as response
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

    Motor's goals:
    - Successfully follow instructions from Planning
    - Execute actions in the world using tools
    - Report actions taken (but NOT tool results - those go to Sensory)

    IMPORTANT: Motor does not see tool outputs directly.
    Tool outputs are sent to Sensory, which perceives them.
    This enforces the perception-action separation.
    """
    outgoing: list[Message] = []

    for msg in incoming:
        state.add_message(msg)

        if msg.message_type == MessageType.MOTOR_COMMAND:
            # Planning sent us an instruction - execute it
            instruction = msg.content
            sensory_context = state.metadata.get("sensory_context", "")

            # Build prompt with tools
            prompt = (
                f"{MOTOR_SYSTEM_PROMPT}\n\n"
                f"---\n\n"
                f"Instruction from Planning: {instruction}\n"
                f"Sensory context: {sensory_context}\n\n"
                f"What action should you take?"
            )

            if llm_fn:
                llm_output = llm_fn(prompt)

                # Apply edit model
                llm_output = apply_edit(
                    llm_output,
                    edit_fn,
                    context=f"instruction: {instruction[:100]}",
                )

                action_type, tool_args, response = parse_motor_output(llm_output)

                if action_type == "tool" and tool_args:
                    tool_name = tool_args.pop("tool")

                    # Execute the tool
                    result = execute_tool(tool_name, **tool_args)

                    # Log tool call in Motor's state (but Motor doesn't see output)
                    state.metadata.setdefault("tool_history", []).append({
                        "tool": tool_name,
                        "args": tool_args,
                        "success": result.success,
                    })

                    # Motor only knows it called a tool
                    action_content = f"Called tool: {tool_name} with args: {tool_args}"

                    # TOOL OUTPUT GOES TO SENSORY (not Motor!)
                    tool_output_msg = (
                        f"[Tool Output: {tool_name}]\n"
                        f"{'SUCCESS' if result.success else 'FAILED'}\n"
                        f"{result.output if result.success else result.error}"
                    )
                    outgoing.append(
                        Message(
                            content=tool_output_msg,
                            source=RoomType.MOTOR,
                            target=RoomType.SENSORY,
                            message_type=MessageType.TOOL_OUTPUT,
                            cycle=msg.cycle,
                            metadata={"tool": tool_name, "success": result.success},
                        )
                    )

                else:
                    # Direct response (no tool call)
                    action_content = response or llm_output

                    # Direct response goes to external world
                    outgoing.append(
                        Message(
                            content=action_content,
                            source=RoomType.MOTOR,
                            target=RoomType.EXTERNAL,
                            message_type=MessageType.ACTION,
                            cycle=msg.cycle,
                        )
                    )
            else:
                action_content = f"[Motor executing: {instruction[:100]}]"

            # Execute custom action function if provided
            if action_fn:
                action_fn(action_content)

            # Notify Sensory of Motor's action (what Motor did, not results)
            outgoing.append(
                Message(
                    content=action_content,
                    source=RoomType.MOTOR,
                    target=RoomType.SENSORY,
                    message_type=MessageType.ACTION,
                    cycle=msg.cycle,
                )
            )

            # Report to Planning what action was taken (not results)
            outgoing.append(
                Message(
                    content=f"Action taken: {action_content[:100]}",
                    source=RoomType.MOTOR,
                    target=RoomType.PLANNING,
                    message_type=MessageType.ACTION_RESULT,
                    cycle=msg.cycle,
                )
            )

        elif msg.message_type == MessageType.PERCEPTION:
            # Context from Sensory
            state.metadata["sensory_context"] = msg.content

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
