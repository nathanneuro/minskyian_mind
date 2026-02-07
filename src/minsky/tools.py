"""Tools available to the Motor room.

Tools:
- web_search: Search the web using Exa API
- scratchpad: Persistent key-value storage for notes
- python_exec: Sandboxed Python code execution
- memory_*: Long-term memory with FSRS-6 decay and semantic search

IMPORTANT: Motor executes tools but does NOT see outputs.
Tool outputs go to Sensory, which perceives them.
"""

import os
import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Data directory for scratchpad
DATA_DIR = Path(__file__).parent.parent.parent / "data"
SCRATCHPAD_FILE = DATA_DIR / "scratchpad.json"


# =============================================================================
# Tool Results
# =============================================================================

@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    output: str
    error: str = ""
    metadata: dict = field(default_factory=dict)


# =============================================================================
# Web Search (Exa API)
# =============================================================================

EXA_API_KEY = os.getenv("EXA_API_KEY", "")
EXA_API_URL = "https://api.exa.ai/search"


def web_search(
    query: str,
    num_results: int = 5,
    use_autoprompt: bool = True,
) -> ToolResult:
    """Search the web using Exa API.

    Args:
        query: Search query.
        num_results: Number of results to return.
        use_autoprompt: Let Exa optimize the query.

    Returns:
        ToolResult with search results.
    """
    if not EXA_API_KEY:
        return ToolResult(
            success=False,
            output="",
            error="EXA_API_KEY not set in environment",
        )

    try:
        response = httpx.post(
            EXA_API_URL,
            headers={
                "x-api-key": EXA_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "numResults": num_results,
                "useAutoprompt": use_autoprompt,
                "type": "neural",
            },
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()

        # Format results
        results = []
        for i, result in enumerate(data.get("results", []), 1):
            results.append(
                f"{i}. {result.get('title', 'No title')}\n"
                f"   URL: {result.get('url', '')}\n"
                f"   {result.get('text', '')[:200]}..."
            )

        output = f"Search results for: {query}\n\n" + "\n\n".join(results)

        return ToolResult(
            success=True,
            output=output,
            metadata={"num_results": len(results), "query": query},
        )

    except httpx.HTTPStatusError as e:
        return ToolResult(
            success=False,
            output="",
            error=f"HTTP error: {e.response.status_code} - {e.response.text}",
        )
    except Exception as e:
        return ToolResult(
            success=False,
            output="",
            error=f"Search failed: {str(e)}",
        )


# =============================================================================
# Scratchpad (Persistent Key-Value Store)
# =============================================================================

class Scratchpad:
    """Persistent key-value store for notes and intermediate results."""

    def __init__(self, filepath: Path = SCRATCHPAD_FILE):
        self.filepath = filepath
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load scratchpad from disk."""
        if self.filepath.exists():
            try:
                with open(self.filepath) as f:
                    self._data = json.load(f)
            except:
                self._data = {}

    def _save(self) -> None:
        """Save scratchpad to disk."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, "w") as f:
            json.dump(self._data, f, indent=2, default=str)

    def write(self, key: str, value: Any) -> ToolResult:
        """Write a value to the scratchpad."""
        self._data[key] = value
        self._save()
        return ToolResult(
            success=True,
            output=f"Wrote to '{key}': {str(value)[:100]}...",
            metadata={"key": key},
        )

    def read(self, key: str) -> ToolResult:
        """Read a value from the scratchpad."""
        if key not in self._data:
            return ToolResult(
                success=False,
                output="",
                error=f"Key '{key}' not found in scratchpad",
            )
        value = self._data[key]
        return ToolResult(
            success=True,
            output=f"{key}: {value}",
            metadata={"key": key, "value": value},
        )

    def list_keys(self) -> ToolResult:
        """List all keys in the scratchpad."""
        keys = list(self._data.keys())
        return ToolResult(
            success=True,
            output=f"Scratchpad keys: {', '.join(keys) if keys else '(empty)'}",
            metadata={"keys": keys},
        )

    def delete(self, key: str) -> ToolResult:
        """Delete a key from the scratchpad."""
        if key in self._data:
            del self._data[key]
            self._save()
            return ToolResult(
                success=True,
                output=f"Deleted '{key}' from scratchpad",
            )
        return ToolResult(
            success=False,
            output="",
            error=f"Key '{key}' not found",
        )

    def clear(self) -> ToolResult:
        """Clear the entire scratchpad."""
        self._data = {}
        self._save()
        return ToolResult(
            success=True,
            output="Scratchpad cleared",
        )


# Global scratchpad instance
_scratchpad: Scratchpad | None = None


def get_scratchpad() -> Scratchpad:
    """Get the global scratchpad instance."""
    global _scratchpad
    if _scratchpad is None:
        _scratchpad = Scratchpad()
    return _scratchpad


def scratchpad_write(key: str, value: Any) -> ToolResult:
    """Write to the scratchpad."""
    return get_scratchpad().write(key, value)


def scratchpad_read(key: str) -> ToolResult:
    """Read from the scratchpad."""
    return get_scratchpad().read(key)


def scratchpad_list() -> ToolResult:
    """List scratchpad keys."""
    return get_scratchpad().list_keys()


# =============================================================================
# Sandboxed Python Interpreter
# =============================================================================

PYTHON_TIMEOUT = 10  # seconds
PYTHON_MAX_OUTPUT = 10000  # characters


def python_exec(code: str, timeout: int = PYTHON_TIMEOUT) -> ToolResult:
    """Execute Python code in a sandboxed subprocess.

    The code runs in a separate process with:
    - Limited execution time
    - No network access (via restricted imports)
    - Limited output size

    Args:
        code: Python code to execute.
        timeout: Maximum execution time in seconds.

    Returns:
        ToolResult with stdout/stderr output.
    """
    # Create a wrapper script that restricts dangerous operations
    wrapper = f'''
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

# Restrict dangerous imports
BLOCKED_MODULES = {{'os', 'subprocess', 'shutil', 'socket', 'requests', 'urllib', 'httpx'}}

class RestrictedImporter:
    def find_module(self, name, path=None):
        if name.split('.')[0] in BLOCKED_MODULES:
            raise ImportError(f"Import of '{{name}}' is not allowed in sandbox")
        return None

sys.meta_path.insert(0, RestrictedImporter())

# Capture output
stdout_capture = io.StringIO()
stderr_capture = io.StringIO()

try:
    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
        exec("""
{code.replace(chr(34)*3, chr(39)*3)}
""")
    print(stdout_capture.getvalue(), end='')
    if stderr_capture.getvalue():
        print("STDERR:", stderr_capture.getvalue(), file=sys.stderr)
except Exception as e:
    print(f"Error: {{type(e).__name__}}: {{e}}", file=sys.stderr)
'''

    try:
        # Write wrapper to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(wrapper)
            temp_path = f.name

        try:
            # Run in subprocess
            result = subprocess.run(
                ['python', temp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
            )

            stdout = result.stdout[:PYTHON_MAX_OUTPUT]
            stderr = result.stderr[:PYTHON_MAX_OUTPUT]

            if result.returncode == 0:
                return ToolResult(
                    success=True,
                    output=stdout if stdout else "(no output)",
                    error=stderr if stderr else "",
                    metadata={"returncode": result.returncode},
                )
            else:
                return ToolResult(
                    success=False,
                    output=stdout,
                    error=stderr if stderr else f"Process exited with code {result.returncode}",
                    metadata={"returncode": result.returncode},
                )

        finally:
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)

    except subprocess.TimeoutExpired:
        return ToolResult(
            success=False,
            output="",
            error=f"Execution timed out after {timeout} seconds",
        )
    except Exception as e:
        return ToolResult(
            success=False,
            output="",
            error=f"Execution failed: {str(e)}",
        )


# =============================================================================
# Memory System (Vestige-inspired)
# =============================================================================

from minsky.memory import get_memory_store


def memory_store(
    content: str,
    tags: str = "",
    source: str = "",
) -> ToolResult:
    """Store content in long-term memory with smart duplicate detection.

    Uses prediction error gating:
    - Similar content (>92%) just reinforces existing memory
    - Related content (>75%) updates existing memory
    - Novel content creates new memory

    Args:
        content: The information to remember.
        tags: Comma-separated tags for categorization.
        source: Source of the information.

    Returns:
        ToolResult with action taken (CREATE/UPDATE/REINFORCE) and memory ID.
    """
    try:
        store = get_memory_store()
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

        action, memory_id = store.smart_ingest(
            content=content,
            tags=tag_list,
            source=source,
        )

        return ToolResult(
            success=True,
            output=f"Memory {action}: {memory_id}\nContent: {content[:100]}...",
            metadata={"action": action, "memory_id": memory_id},
        )

    except Exception as e:
        return ToolResult(
            success=False,
            output="",
            error=f"Memory store failed: {str(e)}",
        )


def memory_query(
    query: str,
    limit: int = 5,
    min_accessibility: float = 0.1,
) -> ToolResult:
    """Search long-term memory using hybrid keyword + semantic search.

    Memories decay over time (FSRS-6 power law). Retrieving a memory
    strengthens it (testing effect).

    Args:
        query: What to search for.
        limit: Maximum number of results.
        min_accessibility: Minimum accessibility threshold (0-1).

    Returns:
        ToolResult with matching memories and their accessibility scores.
    """
    try:
        store = get_memory_store()
        results = store.search(
            query=query,
            limit=limit,
            min_accessibility=min_accessibility,
        )

        if not results:
            return ToolResult(
                success=True,
                output=f"No memories found for: {query}",
                metadata={"count": 0, "query": query},
            )

        # Format results
        lines = [f"Found {len(results)} memories for: {query}\n"]
        for i, (memory, score) in enumerate(results, 1):
            state = memory.get_state().value
            lines.append(
                f"{i}. [{memory.id}] (state={state}, accessibility={memory.accessibility:.2f})\n"
                f"   {memory.content[:200]}..."
            )
            if memory.tags:
                lines.append(f"   Tags: {', '.join(memory.tags)}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={
                "count": len(results),
                "query": query,
                "memory_ids": [m.id for m, _ in results],
            },
        )

    except Exception as e:
        return ToolResult(
            success=False,
            output="",
            error=f"Memory query failed: {str(e)}",
        )


def memory_promote(memory_id: str) -> ToolResult:
    """Mark a memory as helpful, strengthening it.

    Increases storage strength and makes it easier to recall.

    Args:
        memory_id: The ID of the memory to promote.

    Returns:
        ToolResult indicating success or failure.
    """
    try:
        store = get_memory_store()
        success = store.promote(memory_id)

        if success:
            return ToolResult(
                success=True,
                output=f"Memory {memory_id} promoted (strengthened)",
                metadata={"memory_id": memory_id},
            )
        else:
            return ToolResult(
                success=False,
                output="",
                error=f"Memory {memory_id} not found",
            )

    except Exception as e:
        return ToolResult(
            success=False,
            output="",
            error=f"Memory promote failed: {str(e)}",
        )


def memory_demote(memory_id: str) -> ToolResult:
    """Mark a memory as wrong or unhelpful, weakening it.

    Reduces retrieval strength, making it less likely to surface.

    Args:
        memory_id: The ID of the memory to demote.

    Returns:
        ToolResult indicating success or failure.
    """
    try:
        store = get_memory_store()
        success = store.demote(memory_id)

        if success:
            return ToolResult(
                success=True,
                output=f"Memory {memory_id} demoted (weakened)",
                metadata={"memory_id": memory_id},
            )
        else:
            return ToolResult(
                success=False,
                output="",
                error=f"Memory {memory_id} not found",
            )

    except Exception as e:
        return ToolResult(
            success=False,
            output="",
            error=f"Memory demote failed: {str(e)}",
        )


def memory_stats() -> ToolResult:
    """Get statistics about the memory system.

    Returns counts by state (active/dormant/silent/unavailable)
    and average accessibility.
    """
    try:
        store = get_memory_store()
        stats = store.stats()

        lines = [
            f"Memory Statistics:",
            f"  Total memories: {stats['total']}",
        ]

        if stats['total'] > 0:
            lines.append(f"  Average accessibility: {stats['avg_accessibility']:.2f}")
            lines.append(f"  By state:")
            for state, count in stats['by_state'].items():
                lines.append(f"    {state}: {count}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata=stats,
        )

    except Exception as e:
        return ToolResult(
            success=False,
            output="",
            error=f"Memory stats failed: {str(e)}",
        )


# =============================================================================
# Tool Registry
# =============================================================================

TOOLS = {
    "web_search": {
        "function": web_search,
        "description": "Search the web for information. Args: query (str), num_results (int, optional)",
    },
    "scratchpad_write": {
        "function": scratchpad_write,
        "description": "Write a value to persistent storage. Args: key (str), value (any)",
    },
    "scratchpad_read": {
        "function": scratchpad_read,
        "description": "Read a value from persistent storage. Args: key (str)",
    },
    "scratchpad_list": {
        "function": scratchpad_list,
        "description": "List all keys in persistent storage. No args.",
    },
    "python_exec": {
        "function": python_exec,
        "description": "Execute Python code in a sandbox. Args: code (str). Limited imports, 10s timeout.",
    },
    # Memory tools
    "memory_store": {
        "function": memory_store,
        "description": "Store info in long-term memory. Args: content (str), tags (str, optional), source (str, optional). Auto-detects duplicates.",
    },
    "memory_query": {
        "function": memory_query,
        "description": "Search memory using hybrid keyword+semantic. Args: query (str), limit (int, optional). Retrieval strengthens memories.",
    },
    "memory_promote": {
        "function": memory_promote,
        "description": "Mark memory as helpful (strengthens it). Args: memory_id (str).",
    },
    "memory_demote": {
        "function": memory_demote,
        "description": "Mark memory as wrong (weakens it). Args: memory_id (str).",
    },
    "memory_stats": {
        "function": memory_stats,
        "description": "Get memory system statistics. No args.",
    },
}


def get_tools_description() -> str:
    """Get a formatted description of all available tools."""
    lines = ["Available tools:"]
    for name, info in TOOLS.items():
        lines.append(f"  - {name}: {info['description']}")
    return "\n".join(lines)


def execute_tool(tool_name: str, **kwargs) -> ToolResult:
    """Execute a tool by name.

    Args:
        tool_name: Name of the tool to execute.
        **kwargs: Arguments to pass to the tool.

    Returns:
        ToolResult from the tool execution.
    """
    if tool_name not in TOOLS:
        return ToolResult(
            success=False,
            output="",
            error=f"Unknown tool: {tool_name}. Available: {list(TOOLS.keys())}",
        )

    try:
        return TOOLS[tool_name]["function"](**kwargs)
    except Exception as e:
        return ToolResult(
            success=False,
            output="",
            error=f"Tool execution failed: {str(e)}",
        )
