"""Unit tests for minsky.rooms module."""

import pytest
from unittest.mock import MagicMock

from minsky.types import Message, InternalMessage, RoomType, MessageType, MESSAGE_MAX_LENGTH
from minsky.rooms import (
    sensory_process,
    planning_process,
    motor_process,
    parse_motor_output,
    create_room_state,
    run_dual_agents,
    ROOM_PROCESSORS,
)


class TestCreateRoomState:
    """Tests for create_room_state function."""

    def test_creates_sensory_state(self):
        """Should create a valid Sensory room state."""
        state = create_room_state(RoomType.SENSORY)
        assert state.room_type == RoomType.SENSORY
        assert state.message_history == []

    def test_creates_planning_state(self):
        """Should create a valid Planning room state."""
        state = create_room_state(RoomType.PLANNING)
        assert state.room_type == RoomType.PLANNING

    def test_creates_motor_state(self):
        """Should create a valid Motor room state."""
        state = create_room_state(RoomType.MOTOR)
        assert state.room_type == RoomType.MOTOR


class TestRoomProcessors:
    """Tests for ROOM_PROCESSORS registry."""

    def test_all_rooms_registered(self):
        """All room types should have processors registered."""
        assert RoomType.SENSORY in ROOM_PROCESSORS
        assert RoomType.PLANNING in ROOM_PROCESSORS
        assert RoomType.MOTOR in ROOM_PROCESSORS

    def test_processors_are_callable(self):
        """All processors should be callable."""
        for room_type, processor in ROOM_PROCESSORS.items():
            assert callable(processor)


class TestSensoryProcess:
    """Tests for sensory_process function."""

    def test_outputs_two_messages(self):
        """Sensory should output exactly 2 messages (to Planning and Motor)."""
        state = create_room_state(RoomType.SENSORY)
        incoming = [
            Message(
                content="User says hello",
                source=RoomType.EXTERNAL,
                target=RoomType.SENSORY,
                message_type=MessageType.PERCEPTION,
                cycle=1,
            )
        ]

        new_state, outgoing, raw_output = sensory_process(state, incoming)

        assert len(outgoing) == 2
        targets = {msg.target for msg in outgoing}
        assert RoomType.PLANNING in targets
        assert RoomType.MOTOR in targets

    def test_messages_truncated(self):
        """Output messages should be truncated to MESSAGE_MAX_LENGTH."""
        state = create_room_state(RoomType.SENSORY)
        incoming = [
            Message(
                content="x" * 500,
                source=RoomType.EXTERNAL,
                target=RoomType.SENSORY,
                message_type=MessageType.PERCEPTION,
                cycle=1,
            )
        ]

        _, outgoing, _ = sensory_process(state, incoming)

        for msg in outgoing:
            assert len(msg.content) <= MESSAGE_MAX_LENGTH

    def test_updates_state_context(self):
        """Sensory should update current_context from perception."""
        state = create_room_state(RoomType.SENSORY)
        incoming = [
            Message(
                content="Important observation",
                source=RoomType.EXTERNAL,
                target=RoomType.SENSORY,
                message_type=MessageType.PERCEPTION,
                cycle=1,
            )
        ]

        new_state, _, _ = sensory_process(state, incoming)

        assert "Important observation" in new_state.current_context

    def test_handles_empty_incoming(self):
        """Should handle empty incoming messages gracefully."""
        state = create_room_state(RoomType.SENSORY)

        new_state, outgoing, raw_output = sensory_process(state, [])

        # Should still output 2 messages (possibly empty/default)
        assert len(outgoing) == 2
        assert raw_output == ""


class TestPlanningProcess:
    """Tests for planning_process function."""

    def test_outputs_two_messages(self):
        """Planning should output exactly 2 messages (to Sensory and Motor)."""
        state = create_room_state(RoomType.PLANNING)
        incoming = [
            Message(
                content="Sensory summary",
                source=RoomType.SENSORY,
                target=RoomType.PLANNING,
                message_type=MessageType.PERCEPTION,
                cycle=1,
            )
        ]

        new_state, outgoing, raw_output = planning_process(state, incoming)

        assert len(outgoing) == 2
        targets = {msg.target for msg in outgoing}
        assert RoomType.SENSORY in targets
        assert RoomType.MOTOR in targets

    def test_sends_attention_request_to_sensory(self):
        """Planning should send ATTENTION_REQUEST to Sensory."""
        state = create_room_state(RoomType.PLANNING)
        incoming = [
            Message(
                content="Something happened",
                source=RoomType.SENSORY,
                target=RoomType.PLANNING,
                message_type=MessageType.PERCEPTION,
                cycle=1,
            )
        ]

        _, outgoing, _ = planning_process(state, incoming)

        sensory_msg = next(m for m in outgoing if m.target == RoomType.SENSORY)
        assert sensory_msg.message_type == MessageType.ATTENTION_REQUEST

    def test_sends_motor_command_to_motor(self):
        """Planning should send MOTOR_COMMAND to Motor."""
        state = create_room_state(RoomType.PLANNING)
        incoming = [
            Message(
                content="Something happened",
                source=RoomType.SENSORY,
                target=RoomType.PLANNING,
                message_type=MessageType.PERCEPTION,
                cycle=1,
            )
        ]

        _, outgoing, _ = planning_process(state, incoming)

        motor_msg = next(m for m in outgoing if m.target == RoomType.MOTOR)
        assert motor_msg.message_type == MessageType.MOTOR_COMMAND


class TestMotorProcess:
    """Tests for motor_process function."""

    def test_outputs_three_messages(self):
        """Motor should output exactly 3 messages (Sensory, Planning, External)."""
        state = create_room_state(RoomType.MOTOR)
        incoming = [
            Message(
                content="Do something",
                source=RoomType.PLANNING,
                target=RoomType.MOTOR,
                message_type=MessageType.MOTOR_COMMAND,
                cycle=1,
            )
        ]

        new_state, outgoing, raw_output = motor_process(state, incoming)

        assert len(outgoing) == 3
        targets = {msg.target for msg in outgoing}
        assert RoomType.SENSORY in targets
        assert RoomType.PLANNING in targets
        assert RoomType.EXTERNAL in targets

    def test_tool_output_goes_to_sensory(self):
        """Motor's tool output should go to Sensory (perception-action separation)."""
        state = create_room_state(RoomType.MOTOR)
        incoming = [
            Message(
                content="Search for something",
                source=RoomType.PLANNING,
                target=RoomType.MOTOR,
                message_type=MessageType.MOTOR_COMMAND,
                cycle=1,
            )
        ]

        _, outgoing, _ = motor_process(state, incoming)

        sensory_msg = next(m for m in outgoing if m.target == RoomType.SENSORY)
        assert sensory_msg.message_type == MessageType.TOOL_OUTPUT


class TestParseMotorOutput:
    """Tests for parse_motor_output function."""

    def test_parses_simple_tool_call(self):
        """Should parse a simple tool call."""
        output = 'TOOL: web_search ARGS: {"query": "test"}'
        action_type, args, response = parse_motor_output(output)

        assert action_type == "tool"
        assert args["tool"] == "web_search"
        assert args["query"] == "test"
        assert response is None

    def test_parses_nested_json_args(self):
        """Should correctly parse nested JSON in tool args."""
        output = 'TOOL: memory_store ARGS: {"content": "test {with} braces", "tags": "a,b"}'
        action_type, args, response = parse_motor_output(output)

        assert action_type == "tool"
        assert args["tool"] == "memory_store"
        assert args["content"] == "test {with} braces"
        assert args["tags"] == "a,b"

    def test_parses_tool_without_args(self):
        """Should handle tool call without ARGS."""
        output = "TOOL: memory_stats"
        action_type, args, response = parse_motor_output(output)

        assert action_type == "tool"
        assert args["tool"] == "memory_stats"

    def test_parses_external_response(self):
        """Should parse TO_EXTERNAL response."""
        output = "TO_EXTERNAL: Hello, how can I help?"
        action_type, args, response = parse_motor_output(output)

        assert action_type == "response"
        assert args is None
        assert response == "Hello, how can I help?"

    def test_parses_plain_response(self):
        """Should treat plain text as response."""
        output = "Just a plain response"
        action_type, args, response = parse_motor_output(output)

        assert action_type == "response"
        assert response == "Just a plain response"

    def test_handles_malformed_json(self):
        """Should handle malformed JSON gracefully."""
        output = 'TOOL: web_search ARGS: {"query": broken}'
        action_type, args, response = parse_motor_output(output)

        assert action_type == "tool"
        assert args["tool"] == "web_search"
        # Args should be empty dict due to parse failure
        assert "query" not in args or args.get("query") is None


class TestRoomIntegration:
    """Integration tests for room communication."""

    def test_full_cycle_without_llm(self):
        """Test a full cycle through all rooms without LLM."""
        # Initialize states
        sensory_state = create_room_state(RoomType.SENSORY)
        planning_state = create_room_state(RoomType.PLANNING)
        motor_state = create_room_state(RoomType.MOTOR)

        # External input
        external_input = Message(
            content="Hello world",
            source=RoomType.EXTERNAL,
            target=RoomType.SENSORY,
            message_type=MessageType.PERCEPTION,
            cycle=1,
        )

        # Process through Sensory
        sensory_state, sensory_out, sensory_raw = sensory_process(sensory_state, [external_input])
        assert len(sensory_out) == 2
        assert sensory_raw == ""  # No LLM, so raw is empty

        # Route to Planning
        planning_input = [m for m in sensory_out if m.target == RoomType.PLANNING]
        planning_state, planning_out, planning_raw = planning_process(planning_state, planning_input)
        assert len(planning_out) == 2

        # Route to Motor (from both Sensory and Planning)
        motor_input = [m for m in sensory_out if m.target == RoomType.MOTOR]
        motor_input += [m for m in planning_out if m.target == RoomType.MOTOR]
        motor_state, motor_out, motor_raw = motor_process(motor_state, motor_input)
        assert len(motor_out) == 3

        # Check external output exists
        external_out = [m for m in motor_out if m.target == RoomType.EXTERNAL]
        assert len(external_out) == 1


class TestDualAgents:
    """Tests for dual-agent (left/right) conversation in rooms."""

    def _make_chat_llm_fn(self, responses):
        """Create a mock llm_fn that returns responses in sequence."""
        call_count = [0]
        def llm_fn(prompt):
            idx = call_count[0]
            call_count[0] += 1
            return responses[idx % len(responses)]
        llm_fn.use_chat_template = True
        return llm_fn

    def test_dual_agents_sensory(self):
        """Dual-agent sensory should produce 2 internal history entries."""
        state = create_room_state(RoomType.SENSORY)
        incoming = [
            Message(
                content="User says hello",
                source=RoomType.EXTERNAL,
                target=RoomType.SENSORY,
                message_type=MessageType.PERCEPTION,
                cycle=1,
            )
        ]
        llm_fn = self._make_chat_llm_fn([
            "The user is greeting us. Simple social interaction.",
            "TO_PLANNING: User greeted us, likely wants conversation.\nTO_MOTOR: Prepare a friendly response.",
        ])

        new_state, outgoing, raw_output = sensory_process(state, incoming, llm_fn=llm_fn)

        assert len(new_state.internal_history) == 2
        assert len(outgoing) == 2
        agents = {m.agent for m in new_state.internal_history}
        assert "left" in agents
        assert "right" in agents

    def test_dual_agents_planning(self):
        """Dual-agent planning should produce 2 internal history entries."""
        state = create_room_state(RoomType.PLANNING)
        incoming = [
            Message(
                content="User wants weather info",
                source=RoomType.SENSORY,
                target=RoomType.PLANNING,
                message_type=MessageType.PERCEPTION,
                cycle=1,
            )
        ]
        llm_fn = self._make_chat_llm_fn([
            "User wants weather. Could search or use memory.",
            "HYPOTHESES: 1) Wants current weather. 2) Wants forecast.\nBEST_ACTION: Search.\nTO_SENSORY: Focus on location.\nTO_MOTOR: Search for weather data.",
        ])

        new_state, outgoing, raw_output = planning_process(state, incoming, llm_fn=llm_fn)

        assert len(new_state.internal_history) == 2
        assert len(outgoing) == 2

    def test_internal_messages_not_truncated(self):
        """InternalMessage content should NOT be truncated (unlike between-room Message)."""
        long_content = "x" * 500
        msg = InternalMessage(
            content=long_content,
            agent="left",
            room_type=RoomType.SENSORY,
            cycle=1,
        )
        assert len(msg.content) == 500
        assert msg.content == long_content

    def test_internal_history_sliding_window(self):
        """RoomState.add_internal should keep at most 20 entries."""
        state = create_room_state(RoomType.SENSORY)
        for i in range(25):
            state.add_internal(InternalMessage(
                content=f"msg {i}",
                agent="left" if i % 2 == 0 else "right",
                room_type=RoomType.SENSORY,
                cycle=i,
            ))
        assert len(state.internal_history) == 20
        assert state.internal_history[0].content == "msg 5"

    def test_motor_programmatic_between_room(self):
        """Motor's between-room messages should be programmatic, not LLM-parsed."""
        state = create_room_state(RoomType.MOTOR)
        incoming = [
            Message(
                content="Respond to the user with a greeting",
                source=RoomType.PLANNING,
                target=RoomType.MOTOR,
                message_type=MessageType.MOTOR_COMMAND,
                cycle=1,
            )
        ]
        llm_fn = self._make_chat_llm_fn([
            "We should use TO_EXTERNAL to greet the user.",
            "ACTION: TO_EXTERNAL: Hello there! How can I help?",
        ])

        new_state, outgoing, raw_output = motor_process(state, incoming, llm_fn=llm_fn)

        sensory_msg = next(m for m in outgoing if m.target == RoomType.SENSORY)
        planning_msg = next(m for m in outgoing if m.target == RoomType.PLANNING)
        external_msg = next(m for m in outgoing if m.target == RoomType.EXTERNAL)

        # Between-room messages are programmatic
        assert sensory_msg.content.startswith("Responded to user:")
        assert planning_msg.content == "Sent response to user"
        # External message is LLM-generated
        assert "Hello there" in external_msg.content

    def test_agent_order_randomized(self):
        """Both 'left' and 'right' should appear as first agent over many runs."""
        first_agents = set()
        for _ in range(50):
            state = create_room_state(RoomType.SENSORY)
            incoming = [
                Message(
                    content="test",
                    source=RoomType.EXTERNAL,
                    target=RoomType.SENSORY,
                    message_type=MessageType.PERCEPTION,
                    cycle=1,
                )
            ]
            llm_fn = self._make_chat_llm_fn([
                "Analysis output.",
                "TO_PLANNING: Summary.\nTO_MOTOR: Context.",
            ])
            new_state, _, _ = sensory_process(state, incoming, llm_fn=llm_fn)
            # First entry in internal_history is the first agent
            first_agents.add(new_state.internal_history[0].agent)
            if len(first_agents) == 2:
                break

        assert "left" in first_agents
        assert "right" in first_agents
