"""T5 edit prompt formatting with per-room instructions.

T5 is the gate between RWKV's raw output and the rooms. It receives
the raw continuation from RWKV (a base model) and must clean it into
structured room messages. The instructions tell T5 what each room's
output format looks like and what "good" output means.
"""

# Per-room instructions prepended to the T5 input.
# These teach T5 what the target format is for each room.
T5_INSTRUCTIONS = {
    "edit": (
        "Rewrite the text as a clear, structured room message. "
        "Fix formatting, remove artifacts, and keep only relevant content."
    ),
    "edit_sensory": (
        "Rewrite as Sensory room output. "
        "Sensory observes the world and sends messages to two rooms: "
        "TO_PLANNING: a concise summary of perceptions for strategic decisions. "
        "TO_MOTOR: relevant context for action execution. "
        "Output must contain both TO_PLANNING: and TO_MOTOR: sections. "
        "Remove RWKV artifacts and repetition."
    ),
    "edit_planning": (
        "Rewrite as Planning room output. "
        "Planning analyzes situations and sends messages to two rooms: "
        "TO_SENSORY: an attention request telling Sensory what to focus on or investigate. "
        "TO_MOTOR: a high-level command saying what to do (not how). "
        "Motor can: search the web, store/recall memories, use scratchpad, or speak to the user. "
        "Output must contain HYPOTHESES:, TO_SENSORY:, and TO_MOTOR: sections. "
        "Ensure hypotheses are distinct and the motor command is actionable. Remove artifacts."
    ),
    "edit_motor": (
        "Rewrite as Motor room output. "
        "Motor executes commands using tools or by speaking to the user. "
        "Available tools: "
        "web_search (search the web and return page contents), "
        "memory_store (save info to long-term memory with tags), "
        "memory_query (recall info from long-term memory by search), "
        "scratchpad_write (save a value to short-term storage), "
        "scratchpad_read (read a value from short-term storage). "
        "Or use TO_EXTERNAL: to speak directly to the user. "
        "Output must contain ACTION: with either TOOL: name ARGS: {{...}} or TO_EXTERNAL: response. "
        "Include TO_SENSORY: and TO_PLANNING: feedback sections. Remove artifacts."
    ),
    "forecast_sensory": (
        "Rewrite as a prediction of what Sensory will observe next. "
        "Be specific and concise."
    ),
}


def format_t5_prompt(text: str, context: str = "", task_prefix: str = "edit") -> str:
    """Format input for T5 edit model with room-specific instructions.

    Args:
        text: The text to edit (RWKV raw output).
        context: Additional context (e.g. room input summary).
        task_prefix: Task prefix (edit, edit_sensory, edit_planning, edit_motor, forecast_sensory).

    Returns:
        Formatted prompt string for T5.
    """
    instruction = T5_INSTRUCTIONS.get(task_prefix, T5_INSTRUCTIONS["edit"])

    parts = [f"{task_prefix}: {instruction}"]
    if context:
        parts.append(f"context: {context}")
    parts.append(f"text: {text}")

    return " ".join(parts)
