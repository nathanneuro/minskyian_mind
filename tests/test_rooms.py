"""Unit tests for minsky.rooms module."""

import pytest

from minsky.types import Message, RoomType, MessageType, MESSAGE_MAX_LENGTH
from minsky.rooms import (
    sensory_process,
    planning_process,
    motor_process,
    parse_motor_output,
    create_room_state,
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

        new_state, outgoing = sensory_process(state, incoming)

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

        _, outgoing = sensory_process(state, incoming)

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

        new_state, _ = sensory_process(state, incoming)

        assert "Important observation" in new_state.current_context

    def test_handles_empty_incoming(self):
        """Should handle empty incoming messages gracefully."""
        state = create_room_state(RoomType.SENSORY)

        new_state, outgoing = sensory_process(state, [])

        # Should still output 2 messages (possibly empty/default)
        assert len(outgoing) == 2


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

        new_state, outgoing = planning_process(state, incoming)

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

        _, outgoing = planning_process(state, incoming)

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

        _, outgoing = planning_process(state, incoming)

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

        new_state, outgoing = motor_process(state, incoming)

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

        _, outgoing = motor_process(state, incoming)

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
        sensory_state, sensory_out = sensory_process(sensory_state, [external_input])
        assert len(sensory_out) == 2

        # Route to Planning
        planning_input = [m for m in sensory_out if m.target == RoomType.PLANNING]
        planning_state, planning_out = planning_process(planning_state, planning_input)
        assert len(planning_out) == 2

        # Route to Motor (from both Sensory and Planning)
        motor_input = [m for m in sensory_out if m.target == RoomType.MOTOR]
        motor_input += [m for m in planning_out if m.target == RoomType.MOTOR]
        motor_state, motor_out = motor_process(motor_state, motor_input)
        assert len(motor_out) == 3

        # Check external output exists
        external_out = [m for m in motor_out if m.target == RoomType.EXTERNAL]
        assert len(external_out) == 1
