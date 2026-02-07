"""Minsky - A Society of Mind architecture with frozen LLMs and learnable edit models.

Architecture:
- Three rooms: Sensory (summarizes data), Planning (hypotheses), Motor (actions)
- Frozen LLM (RWKV 7B on GPU 0) for cognition
- Learnable T5 edit model (T5Gemma 270M on GPU 1) for adaptation
- Summarizer agents (RWKV only) every N global steps
- Memory system (FSRS-6 decay, dual-strength, hybrid search)

Motor has tools: web_search, scratchpad, python_exec, memory_*
Motor executes tools but does NOT see outputs - those go to Sensory.
Sensory summarizes all data and forwards to Planning and Motor.

Each global step:
1. Batch all room prompts through RWKV (GPU 0)
2. Batch all outputs through T5 edit model (GPU 1)
3. Route edited outputs to target rooms
"""

from minsky.types import Message, RoomType, MessageType, RoomState
from minsky.orchestrator import Orchestrator, BatchedLLM, BatchedEditModel
from minsky.config import RWKVConfig, EditModelConfig, SummarizerConfig, OrchestratorConfig
from minsky.judges import JudgeInput, JudgeOutput, JUDGE_PROMPTS
from minsky.edit_model import EditModel, TrainingPair, EditModelTrainer
from minsky.tools import (
    web_search,
    scratchpad_write,
    scratchpad_read,
    python_exec,
    memory_store,
    memory_query,
    memory_promote,
    memory_demote,
    memory_stats,
    execute_tool,
    get_tools_description,
    ToolResult,
)
from minsky.memory import Memory, MemoryStore, MemoryState, get_memory_store

__all__ = [
    # Types
    "Message",
    "RoomType",
    "MessageType",
    "RoomState",
    # Orchestrator
    "Orchestrator",
    "BatchedLLM",
    "BatchedEditModel",
    # Config
    "RWKVConfig",
    "EditModelConfig",
    "SummarizerConfig",
    "OrchestratorConfig",
    # Judges
    "JudgeInput",
    "JudgeOutput",
    "JUDGE_PROMPTS",
    # Edit Model
    "EditModel",
    "TrainingPair",
    "EditModelTrainer",
    # Memory
    "Memory",
    "MemoryStore",
    "MemoryState",
    "get_memory_store",
    "memory_store",
    "memory_query",
    "memory_promote",
    "memory_demote",
    "memory_stats",
]
