"""Summarizer prompt template for compressing room message histories.

Summarizers run every N global steps using RWKV only (no T5 edit).
"""

SUMMARIZER_PROMPT_TEMPLATE = (
    "Summarize the following {room_type} room activity in 2-3 sentences. "
    "Focus on key observations, decisions, and actions.\n\n"
    "Activity:\n{history}\n\n"
    "Summary:"
)
