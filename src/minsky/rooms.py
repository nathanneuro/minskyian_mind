"""Room implementations for the Society of Mind architecture.

Each room is implemented as a set of functions that process incoming messages
and generate outgoing messages. Rooms do not directly communicate - all
message passing goes through the orchestrator.

Each room can optionally have:
- llm_fn: Frozen LLM (RWKV) for generating responses
- edit_fn: T5 edit model for modifying LLM outputs
"""

from minsky.types import (
    Message,
    RoomType,
    RoomState,
    MessageType,
    Hypothesis,
    Plan,
)


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
# Receives perceptions from the external world
# Responds to attention requests from Planning
# Observes Motor's actions as context


def sensory_process(
    state: RoomState,
    incoming: list[Message],
    llm_fn=None,
    edit_fn=None,
) -> tuple[RoomState, list[Message]]:
    """
    Process incoming messages in the Sensory room.

    Sensory's goals:
    - Predict what comes next (world modeling)
    - Direct attention according to Planning's instructions
    - Observe Motor's actions as contextual clues
    """
    outgoing: list[Message] = []

    for msg in incoming:
        state.add_message(msg)

        if msg.message_type == MessageType.PERCEPTION:
            # New perception from external world - update context
            state.current_context = msg.content
            state.metadata["last_perception"] = msg.content

        elif msg.message_type == MessageType.ATTENTION_REQUEST:
            # Planning wants us to focus on something specific
            attention_target = msg.content

            if llm_fn:
                response_content = llm_fn(
                    f"Focus attention on: {attention_target}\n"
                    f"Current context: {state.current_context}\n"
                    f"What do you observe?"
                )
                # Apply edit model
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
            # Motor is taking an action - observe it as context
            state.metadata["last_motor_action"] = msg.content

    return state, outgoing


# -----------------------------------------------------------------------------
# Planning Room
# -----------------------------------------------------------------------------
# No direct contact with the external world
# Sends attention requests to Sensory
# Sends high-level commands to Motor
# Generates hypotheses and plans with expected values


def planning_process(
    state: RoomState,
    incoming: list[Message],
    llm_fn=None,
    edit_fn=None,
) -> tuple[RoomState, list[Message]]:
    """
    Process incoming messages in the Planning room.

    Planning's goals:
    - Generate at least two hypotheses to explain the data
    - Generate at least one plan per hypothesis
    - Predict expected value of each plan
    - Send highest EV plan to Motor
    - Avoid action vacillation/churn
    """
    outgoing: list[Message] = []

    for msg in incoming:
        state.add_message(msg)

        if msg.message_type == MessageType.PERCEPTION:
            # New perception relayed from Sensory - time to hypothesize
            perception = msg.content

            if llm_fn:
                # Use LLM to generate hypotheses and plan
                planning_response = llm_fn(
                    f"Given this perception: {perception}\n\n"
                    f"1. Generate at least 2 hypotheses explaining this situation.\n"
                    f"2. For each hypothesis, generate a plan.\n"
                    f"3. Estimate the expected value (0-1) of each plan.\n"
                    f"4. Select the highest EV plan.\n\n"
                    f"What is your highest EV plan, stated as a clear instruction?"
                )
                # Apply edit model
                planning_response = apply_edit(
                    planning_response,
                    edit_fn,
                    context=f"perception: {perception[:100]}",
                )
                # Store hypotheses in metadata for later judge evaluation
                state.metadata["last_planning_response"] = planning_response
                plan_content = planning_response
            else:
                # Stub: generate a simple plan
                plan_content = f"[Plan based on perception: {perception[:100]}]"

            # Request more info from Sensory if needed
            if "need more information" in plan_content.lower() or not state.metadata.get("attention_requested"):
                outgoing.append(
                    Message(
                        content=f"What are the key details about: {perception[:50]}?",
                        source=RoomType.PLANNING,
                        target=RoomType.SENSORY,
                        message_type=MessageType.ATTENTION_REQUEST,
                        cycle=msg.cycle,
                    )
                )
                state.metadata["attention_requested"] = True
            else:
                # Send plan to Motor
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
            # Sensory responded to our attention request - incorporate and plan
            sensory_info = msg.content

            if llm_fn:
                plan_content = llm_fn(
                    f"Additional sensory information: {sensory_info}\n"
                    f"Previous context: {state.current_context}\n\n"
                    f"Generate the highest EV plan as a clear instruction for Motor."
                )
                # Apply edit model
                plan_content = apply_edit(
                    plan_content,
                    edit_fn,
                    context=f"sensory_info: {sensory_info[:100]}",
                )
            else:
                plan_content = f"[Refined plan with sensory info: {sensory_info[:50]}]"

            # Now send to Motor
            outgoing.append(
                Message(
                    content=plan_content,
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
    - Execute actions in the world
    - Report results back
    """
    outgoing: list[Message] = []

    for msg in incoming:
        state.add_message(msg)

        if msg.message_type == MessageType.MOTOR_COMMAND:
            # Planning sent us an instruction - execute it
            instruction = msg.content
            sensory_context = state.metadata.get("sensory_context", "")

            if llm_fn:
                # Translate high-level instruction into concrete action
                action_content = llm_fn(
                    f"Instruction from Planning: {instruction}\n"
                    f"Sensory context: {sensory_context}\n\n"
                    f"Translate this into a concrete action or response."
                )
                # Apply edit model
                action_content = apply_edit(
                    action_content,
                    edit_fn,
                    context=f"instruction: {instruction[:100]}",
                )
            else:
                action_content = f"[Motor executing: {instruction[:100]}]"

            # Execute the action
            if action_fn:
                result = action_fn(action_content)
            else:
                result = f"[Action completed: {action_content[:50]}]"

            # Notify other rooms of the action
            outgoing.append(
                Message(
                    content=action_content,
                    source=RoomType.MOTOR,
                    target=RoomType.SENSORY,
                    message_type=MessageType.ACTION,
                    cycle=msg.cycle,
                )
            )

            # Report result back to Planning
            outgoing.append(
                Message(
                    content=result,
                    source=RoomType.MOTOR,
                    target=RoomType.PLANNING,
                    message_type=MessageType.ACTION_RESULT,
                    cycle=msg.cycle,
                )
            )

            # This is the actual output to the external world
            outgoing.append(
                Message(
                    content=action_content,
                    source=RoomType.MOTOR,
                    target=RoomType.EXTERNAL,
                    message_type=MessageType.ACTION,
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
