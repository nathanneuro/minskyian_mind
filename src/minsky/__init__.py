"""Minsky - A Society of Mind architecture with frozen LLMs and learnable edit models.

Architecture:
- Three rooms: Sensory (summarizes data), Planning (hypotheses), Motor (actions)
- Frozen LLM (configurable, default Qwen3-8B on GPU 0) for cognition
- Learnable T5 edit model (T5Gemma 270M on GPU 1) for adaptation
- Summarizer agents every N global steps
- Memory system (FSRS-6 decay, dual-strength, hybrid search)

Motor has tools: web_search, scratchpad, memory_*
Motor executes tools but does NOT see outputs - those go to Sensory.
Sensory summarizes all data and forwards to Planning and Motor.

Each global step:
1. Batch all room prompts through LLM (GPU 0)
2. Batch all outputs through T5 edit model (GPU 1)
3. Route edited outputs to target rooms
"""

from minsky.types import Message, RoomType, MessageType, RoomState, InternalMessage
from minsky.orchestrator import Orchestrator, LLMWrapper, RWKVWrapper, T5EditWrapper
from minsky.config import load_config
from minsky.judges import JudgeInput, JudgeOutput
from minsky.edit_model import EditModel, TrainingPair, EditModelTrainer
from minsky.tools import (
    web_search,
    scratchpad_write,
    scratchpad_read,
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
from minsky.rooms import ROOM_PROCESSORS, sensory_process, planning_process, motor_process, run_dual_agents
from minsky.prompts import (
    SENSORY_PROMPT_TEMPLATE,
    PLANNING_PROMPT_TEMPLATE,
    MOTOR_PROMPT_TEMPLATE,
    ARCHITECTURE_PREAMBLE,
    JUDGE_PROMPTS,
    FORECAST_PROMPT_TEMPLATE,
    SUMMARIZER_PROMPT_TEMPLATE,
    format_t5_prompt,
)

__all__ = [
    # Types
    "Message",
    "InternalMessage",
    "RoomType",
    "MessageType",
    "RoomState",
    # Orchestrator
    "Orchestrator",
    "LLMWrapper",
    "RWKVWrapper",
    "T5EditWrapper",
    # Room processors
    "ROOM_PROCESSORS",
    "sensory_process",
    "planning_process",
    "motor_process",
    "run_dual_agents",
    # Config
    "load_config",
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
    # Prompts
    "SENSORY_PROMPT_TEMPLATE",
    "PLANNING_PROMPT_TEMPLATE",
    "MOTOR_PROMPT_TEMPLATE",
    "ARCHITECTURE_PREAMBLE",
    "FORECAST_PROMPT_TEMPLATE",
    "SUMMARIZER_PROMPT_TEMPLATE",
    "format_t5_prompt",
]
