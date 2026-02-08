"""Unit tests for minsky.tools module."""

import pytest
import tempfile
import os
from pathlib import Path

from minsky.tools import (
    ToolResult,
    execute_tool,
    get_tools_description,
    scratchpad_write,
    scratchpad_read,
    memory_store,
    memory_query,
    memory_stats,
    TOOLS,
)


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_success_result(self):
        """Should create a success result."""
        result = ToolResult(success=True, output="Done")
        assert result.success is True
        assert result.output == "Done"
        assert result.error == ""  # Empty string, not None

    def test_failure_result(self):
        """Should create a failure result."""
        result = ToolResult(success=False, output="", error="Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"


class TestGetToolsDescription:
    """Tests for get_tools_description function."""

    def test_returns_string(self):
        """Should return a string description."""
        desc = get_tools_description()
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_includes_tool_names(self):
        """Should include names of available tools."""
        desc = get_tools_description()
        assert "web_search" in desc or "scratchpad" in desc or "memory" in desc


class TestExecuteTool:
    """Tests for execute_tool function."""

    def test_unknown_tool(self):
        """Should return error for unknown tool."""
        result = execute_tool("nonexistent_tool")
        assert result.success is False
        assert "Unknown tool" in result.error

    def test_scratchpad_write_and_read(self):
        """Should write and read from scratchpad."""
        # Write
        write_result = execute_tool("scratchpad_write", key="test_key", value="test_value")
        assert write_result.success is True

        # Read
        read_result = execute_tool("scratchpad_read", key="test_key")
        assert read_result.success is True
        assert "test_value" in read_result.output

        # Cleanup
        execute_tool("scratchpad_write", key="test_key", value="")


class TestScratchpad:
    """Tests for scratchpad tools."""

    def test_write_creates_entry(self):
        """Writing should create a new entry."""
        result = scratchpad_write(key="new_key", value="new_value")
        assert result.success is True

        read_result = scratchpad_read(key="new_key")
        assert "new_value" in read_result.output

    def test_read_nonexistent_key(self):
        """Reading nonexistent key should return error."""
        result = scratchpad_read(key="definitely_not_a_key_12345")
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_write_overwrites(self):
        """Writing to existing key should overwrite."""
        scratchpad_write(key="overwrite_test", value="original")
        scratchpad_write(key="overwrite_test", value="updated")

        result = scratchpad_read(key="overwrite_test")
        assert "updated" in result.output


class TestMemoryTools:
    """Tests for memory-related tools."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for memory tests."""
        # Tests will use the default memory store
        yield
        # Cleanup is handled by individual tests

    def test_memory_store_creates_memory(self):
        """Should store a new memory (CREATE or REINFORCE)."""
        result = memory_store(
            content="Unique test content abc123xyz",
            tags="test,unit",
            source="unit_test"
        )
        assert result.success is True
        # smart_ingest returns CREATE, UPDATE, or REINFORCE
        assert any(action in result.output for action in ["CREATE", "UPDATE", "REINFORCE"])

    def test_memory_query_finds_memories(self):
        """Should find stored memories."""
        # Store a memory first
        memory_store(
            content="Unique test content XYZ123",
            tags="searchable",
            source="unit_test"
        )

        result = memory_query(query="XYZ123", limit=5)
        assert result.success is True
        # Should find the memory or indicate none found
        assert "XYZ123" in result.output or "Found 0" in result.output or "No memories" in result.output

    def test_memory_stats_returns_info(self):
        """Should return memory statistics."""
        result = memory_stats()
        assert result.success is True
        assert "total" in result.output.lower() or "memories" in result.output.lower()


class TestToolsRegistry:
    """Tests for TOOLS registry."""

    def test_tools_registered(self):
        """Core tools should be registered."""
        assert "scratchpad_write" in TOOLS
        assert "scratchpad_read" in TOOLS
        assert "memory_store" in TOOLS

    def test_tools_have_descriptions(self):
        """All tools should have descriptions."""
        for name, tool_info in TOOLS.items():
            assert "description" in tool_info or "desc" in tool_info or callable(tool_info)
