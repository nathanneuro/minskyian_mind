"""End-to-end test of the Society of Mind cycle.

Tests the full message flow through all three rooms without LLM.
Each global step:
1. External input → Sensory
2. Sensory → (Planning, Motor)
3. Planning → (Sensory, Motor)
4. Motor → (Sensory, Planning, External)
"""

from minsky.types import Message, RoomType, MessageType, MESSAGE_MAX_LENGTH
from minsky.rooms import (
    sensory_process,
    planning_process,
    motor_process,
    create_room_state,
)


def run_global_step(
    sensory_state,
    planning_state,
    motor_state,
    external_input: str | None = None,
    cycle: int = 0,
):
    """Run one global step through all rooms.

    Returns:
        Tuple of (updated_states, messages_by_target, external_output)
    """
    # Collect all messages for this step
    sensory_inbox = []
    planning_inbox = []
    motor_inbox = []

    # Phase 1: External input goes to Sensory
    if external_input:
        sensory_inbox.append(Message(
            content=external_input,
            source=RoomType.EXTERNAL,
            target=RoomType.SENSORY,
            message_type=MessageType.PERCEPTION,
            cycle=cycle,
        ))

    # Process Sensory first (it needs external input)
    sensory_state, sensory_out = sensory_process(sensory_state, sensory_inbox)

    # Route Sensory outputs
    for msg in sensory_out:
        if msg.target == RoomType.PLANNING:
            planning_inbox.append(msg)
        elif msg.target == RoomType.MOTOR:
            motor_inbox.append(msg)

    # Process Planning (needs Sensory output)
    planning_state, planning_out = planning_process(planning_state, planning_inbox)

    # Route Planning outputs
    for msg in planning_out:
        if msg.target == RoomType.SENSORY:
            # This would go to next cycle's Sensory inbox
            pass  # For now, just track it
        elif msg.target == RoomType.MOTOR:
            motor_inbox.append(msg)

    # Process Motor (needs Sensory context and Planning command)
    motor_state, motor_out = motor_process(motor_state, motor_inbox)

    # Collect external output
    external_output = None
    for msg in motor_out:
        if msg.target == RoomType.EXTERNAL and msg.content:
            external_output = msg.content

    return (
        (sensory_state, planning_state, motor_state),
        {
            "sensory": sensory_out,
            "planning": planning_out,
            "motor": motor_out,
        },
        external_output,
    )


def print_messages(label: str, messages: list[Message]):
    """Print messages in a formatted way."""
    print(f"\n{label}:")
    for msg in messages:
        print(f"  {msg.source.value:8} → {msg.target.value:8} [{msg.message_type.value:20}]")
        print(f"    \"{msg.content[:60]}{'...' if len(msg.content) > 60 else ''}\"")


def main():
    print("=" * 70)
    print("MINSKY SOCIETY OF MIND - END-TO-END CYCLE TEST")
    print("=" * 70)
    print(f"\nMessage max length: {MESSAGE_MAX_LENGTH} chars")

    # Initialize room states
    sensory_state = create_room_state(RoomType.SENSORY)
    planning_state = create_room_state(RoomType.PLANNING)
    motor_state = create_room_state(RoomType.MOTOR)

    # Test input
    user_input = "Hello! Can you help me search for information about RWKV language models?"

    print(f"\n{'─' * 70}")
    print("GLOBAL STEP 1")
    print(f"{'─' * 70}")
    print(f"\nExternal Input: \"{user_input}\"")

    # Run first global step
    states, messages, external_output = run_global_step(
        sensory_state, planning_state, motor_state,
        external_input=user_input,
        cycle=1,
    )
    sensory_state, planning_state, motor_state = states

    # Print all messages
    print_messages("Sensory outputs", messages["sensory"])
    print_messages("Planning outputs", messages["planning"])
    print_messages("Motor outputs", messages["motor"])

    print(f"\n{'─' * 70}")
    print("EXTERNAL OUTPUT")
    print(f"{'─' * 70}")
    print(f"\n\"{external_output}\"")

    # Run second global step (to show continuation)
    print(f"\n{'─' * 70}")
    print("GLOBAL STEP 2 (continuation)")
    print(f"{'─' * 70}")

    # Feed back previous outputs as next inputs
    # Sensory gets Motor's tool output from previous step
    motor_to_sensory = next(
        (m for m in messages["motor"] if m.target == RoomType.SENSORY),
        None
    )

    states, messages2, external_output2 = run_global_step(
        sensory_state, planning_state, motor_state,
        external_input=motor_to_sensory.content if motor_to_sensory else None,
        cycle=2,
    )

    print_messages("Sensory outputs", messages2["sensory"])
    print_messages("Planning outputs", messages2["planning"])
    print_messages("Motor outputs", messages2["motor"])

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"""
Room Communication Pattern (per global step):
  Sensory  → Planning (summary for decisions)
  Sensory  → Motor    (context for action)
  Planning → Sensory  (attention focus)
  Planning → Motor    (high-level command)
  Motor    → Sensory  (tool output - Motor doesn't see this!)
  Motor    → Planning (action result)
  Motor    → External (user response)

Total messages per step: 7 (2 + 2 + 3)
Each message capped at {MESSAGE_MAX_LENGTH} chars

Perception-Action Separation:
  - Motor calls tools but doesn't see results
  - Sensory perceives tool outputs
  - Planning directs attention and commands
""")


def test_with_tool():
    """Test a cycle where Motor uses a memory tool."""
    from minsky.tools import execute_tool

    print("\n" + "=" * 70)
    print("TOOL INTEGRATION TEST")
    print("=" * 70)

    # Simulate Motor deciding to use memory_store
    print("\n1. Motor stores a memory:")
    result = execute_tool("memory_store",
        content="User prefers RWKV models over transformers",
        tags="preferences,models",
        source="conversation"
    )
    print(f"   Tool result: {result.output}")

    # Simulate Motor querying memory
    print("\n2. Motor queries memory:")
    result = execute_tool("memory_query",
        query="model preferences",
        limit=3
    )
    print(f"   Tool result:\n{result.output}")

    # Key point: Motor doesn't see the result - it goes to Sensory
    print("\n3. Perception-action separation:")
    print("   Motor called memory_query but the result goes to Sensory.")
    print("   Sensory would receive this as TOOL_OUTPUT message type.")
    print("   Sensory then summarizes and forwards to Planning.")

    # Clean up test data
    import os
    from pathlib import Path
    db_path = Path(__file__).parent / "data" / "memory.db"
    if db_path.exists():
        os.remove(db_path)
    print("\n   (Test memory cleaned up)")


if __name__ == "__main__":
    main()
    test_with_tool()
