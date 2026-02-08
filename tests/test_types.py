"""Unit tests for minsky.types module."""

import pytest
from datetime import datetime

from minsky.types import (
    Message,
    RoomType,
    MessageType,
    RoomState,
    MESSAGE_MAX_LENGTH,
    truncate_message,
)


class TestTruncateMessage:
    """Tests for truncate_message function."""

    def test_short_message_unchanged(self):
        """Messages shorter than limit should not be modified."""
        msg = "Hello, world!"
        assert truncate_message(msg) == msg

    def test_exact_length_unchanged(self):
        """Messages exactly at limit should not be modified."""
        msg = "x" * MESSAGE_MAX_LENGTH
        assert truncate_message(msg) == msg
        assert len(truncate_message(msg)) == MESSAGE_MAX_LENGTH

    def test_long_message_truncated(self):
        """Messages longer than limit should be truncated with ellipsis."""
        msg = "x" * (MESSAGE_MAX_LENGTH + 100)
        result = truncate_message(msg)
        assert len(result) == MESSAGE_MAX_LENGTH
        assert result.endswith("...")

    def test_custom_max_length(self):
        """Custom max_length should be respected."""
        msg = "Hello, this is a test message"
        result = truncate_message(msg, max_length=10)
        assert len(result) == 10
        assert result == "Hello, ..."

    def test_empty_message(self):
        """Empty messages should remain empty."""
        assert truncate_message("") == ""


class TestMessage:
    """Tests for Message dataclass."""

    def test_message_creation(self):
        """Basic message creation should work."""
        msg = Message(
            content="Test content",
            source=RoomType.SENSORY,
            target=RoomType.PLANNING,
            message_type=MessageType.PERCEPTION,
        )
        assert msg.content == "Test content"
        assert msg.source == RoomType.SENSORY
        assert msg.target == RoomType.PLANNING
        assert msg.message_type == MessageType.PERCEPTION
        assert msg.cycle == 0

    def test_message_auto_truncates(self):
        """Long message content should be auto-truncated."""
        long_content = "x" * 500
        msg = Message(
            content=long_content,
            source=RoomType.SENSORY,
            target=RoomType.PLANNING,
            message_type=MessageType.PERCEPTION,
        )
        assert len(msg.content) == MESSAGE_MAX_LENGTH
        assert msg.content.endswith("...")

    def test_message_timestamp(self):
        """Message should have a timestamp."""
        before = datetime.now()
        msg = Message(
            content="Test",
            source=RoomType.SENSORY,
            target=RoomType.PLANNING,
            message_type=MessageType.PERCEPTION,
        )
        after = datetime.now()
        assert before <= msg.timestamp <= after

    def test_message_str(self):
        """Message string representation should be readable."""
        msg = Message(
            content="Test content here",
            source=RoomType.SENSORY,
            target=RoomType.PLANNING,
            message_type=MessageType.PERCEPTION,
        )
        str_repr = str(msg)
        assert "sensory" in str_repr
        assert "planning" in str_repr
        assert "perception" in str_repr


class TestRoomState:
    """Tests for RoomState dataclass."""

    def test_room_state_creation(self):
        """Basic room state creation should work."""
        state = RoomState(room_type=RoomType.SENSORY)
        assert state.room_type == RoomType.SENSORY
        assert state.message_history == []
        assert state.current_context == ""
        assert state.metadata == {}

    def test_add_message(self):
        """Adding messages should update history."""
        state = RoomState(room_type=RoomType.SENSORY)
        msg = Message(
            content="Test",
            source=RoomType.EXTERNAL,
            target=RoomType.SENSORY,
            message_type=MessageType.PERCEPTION,
        )
        state.add_message(msg)
        assert len(state.message_history) == 1
        assert state.message_history[0] == msg

    def test_get_recent_messages(self):
        """Should return the N most recent messages."""
        state = RoomState(room_type=RoomType.SENSORY)

        # Add 15 messages
        for i in range(15):
            msg = Message(
                content=f"Message {i}",
                source=RoomType.EXTERNAL,
                target=RoomType.SENSORY,
                message_type=MessageType.PERCEPTION,
            )
            state.add_message(msg)

        # Get last 10
        recent = state.get_recent_messages(10)
        assert len(recent) == 10
        assert recent[0].content == "Message 5"
        assert recent[-1].content == "Message 14"

    def test_get_recent_messages_fewer_than_requested(self):
        """Should return all messages if fewer than N exist."""
        state = RoomState(room_type=RoomType.SENSORY)

        # Add only 3 messages
        for i in range(3):
            msg = Message(
                content=f"Message {i}",
                source=RoomType.EXTERNAL,
                target=RoomType.SENSORY,
                message_type=MessageType.PERCEPTION,
            )
            state.add_message(msg)

        recent = state.get_recent_messages(10)
        assert len(recent) == 3


class TestRoomType:
    """Tests for RoomType enum."""

    def test_room_types_exist(self):
        """All expected room types should exist."""
        assert RoomType.SENSORY.value == "sensory"
        assert RoomType.PLANNING.value == "planning"
        assert RoomType.MOTOR.value == "motor"
        assert RoomType.EXTERNAL.value == "external"


class TestMessageType:
    """Tests for MessageType enum."""

    def test_message_types_exist(self):
        """Core message types should exist."""
        assert MessageType.PERCEPTION.value == "perception"
        assert MessageType.ATTENTION_REQUEST.value == "attention_request"
        assert MessageType.MOTOR_COMMAND.value == "motor_command"
        assert MessageType.ACTION.value == "action"
        assert MessageType.TOOL_OUTPUT.value == "tool_output"
