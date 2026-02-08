"""T5 edit prompt formatting.

T5 edits RWKV base model output into structured room messages.
The task prefix tells T5 which room's output format to target.
"""


def format_t5_prompt(text: str, context: str = "", task_prefix: str = "edit") -> str:
    """Format input for T5 edit model.

    T5 edits RWKV base model output into structured room messages.
    The task prefix tells T5 which room's output format to target.

    Args:
        text: The text to edit (RWKV raw output).
        context: Additional context for the edit.
        task_prefix: Task prefix (edit, edit_sensory, edit_planning, edit_motor, forecast_sensory).

    Returns:
        Formatted prompt string for T5.
    """
    if context:
        return f"{task_prefix}: context: {context} text: {text}"
    return f"{task_prefix}: {text}"
