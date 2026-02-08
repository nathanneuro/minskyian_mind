"""Forecast prompt template for self-supervised sensory prediction.

The forecast system generates predictions about what will happen next,
then compares against reality to create training data without judges.
"""

FORECAST_PROMPT_TEMPLATE = """Given recent events, predict what the Sensory module will observe next.

---
RECENT_EVENTS: Motor searched web for "quantum computing". Planning requested focus on error correction.
<forecast>Sensory will receive search results about quantum error correction advances and summarize key findings for Planning.</forecast>
---
RECENT_EVENTS: User asked about weather. Motor called weather API. Sensory reported temperature data.
<forecast>Sensory will receive the full weather API response and focus on temperature trends as directed by Planning.</forecast>
---
RECENT_EVENTS: {recent_events}
<forecast>"""
