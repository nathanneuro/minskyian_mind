"""Consolidated prompt templates for the Minsky Society of Mind architecture.

All prompt templates are defined here and imported by the modules that use them.
"""

from minsky.prompts.rooms import (
    SENSORY_PROMPT_TEMPLATE,
    PLANNING_PROMPT_TEMPLATE,
    MOTOR_PROMPT_TEMPLATE,
)
from minsky.prompts.judges import (
    ARCHITECTURE_PREAMBLE,
    SENSORY_JUDGE_PROMPT,
    PLANNING_JUDGE_PROMPT,
    MOTOR_JUDGE_PROMPT,
    JUDGE_PROMPTS,
)
from minsky.prompts.forecast import FORECAST_PROMPT_TEMPLATE
from minsky.prompts.summarizer import SUMMARIZER_PROMPT_TEMPLATE
from minsky.prompts.t5 import format_t5_prompt

__all__ = [
    # Room prompts
    "SENSORY_PROMPT_TEMPLATE",
    "PLANNING_PROMPT_TEMPLATE",
    "MOTOR_PROMPT_TEMPLATE",
    # Judge prompts
    "ARCHITECTURE_PREAMBLE",
    "SENSORY_JUDGE_PROMPT",
    "PLANNING_JUDGE_PROMPT",
    "MOTOR_JUDGE_PROMPT",
    "JUDGE_PROMPTS",
    # Forecast
    "FORECAST_PROMPT_TEMPLATE",
    # Summarizer
    "SUMMARIZER_PROMPT_TEMPLATE",
    # T5
    "format_t5_prompt",
]
