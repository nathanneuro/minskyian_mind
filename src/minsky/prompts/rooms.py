"""Room prompt templates for the Society of Mind architecture.

RWKV is a BASE MODEL (not chat-tuned). Prompts must be document-continuation
style with few-shot examples so the model knows what to predict next.
"""

from minsky.tools import get_tools_description

_TOOLS_DESC = get_tools_description()

SENSORY_PROMPT_TEMPLATE = """The Sensory module receives data and writes concise summaries.

---
INPUT: Perceptions: A cat sat on the mat | Tool results: search returned 3 results about cats
TO_PLANNING: A cat is present on the mat. Web search found 3 relevant results about cats.
TO_MOTOR: Cat on mat. Search results available for review.
---
INPUT: Perceptions: User asks about weather | Planning asks you to focus on: temperature data
TO_PLANNING: User wants weather info. Focus on temperature as requested by Planning.
TO_MOTOR: Retrieve temperature data for the user's query.
---
INPUT: {input_data}
TO_PLANNING:"""

PLANNING_PROMPT_TEMPLATE = """The Planning module analyzes situations and issues commands.

---
INPUT: User wants to know about climate change impacts on agriculture.
HYPOTHESES: 1) User needs a broad overview. 2) User wants specific crop data.
BEST_ACTION: Search for recent studies on climate-agriculture intersection.
TO_SENSORY: Focus on agricultural impact data and recent climate studies.
TO_MOTOR: Search the web for "climate change impact on agriculture 2025 studies"
---
INPUT: Search returned 5 papers on crop yield decline due to rising temperatures.
HYPOTHESES: 1) User wants a summary of findings. 2) User wants actionable recommendations.
BEST_ACTION: Summarize key findings and present to user.
TO_SENSORY: Focus on the key statistics and conclusions from the papers.
TO_MOTOR: Compose a response summarizing the top findings about crop yield decline.
---
INPUT: {input_data}
HYPOTHESES:"""

MOTOR_PROMPT_TEMPLATE = """The Motor module executes commands. It can call tools or respond to users.
Available tools: {tools}

---
COMMAND: Search the web for "quantum computing breakthroughs 2025"
CONTEXT: User asked about quantum computing progress.
ACTION: TOOL: web_search ARGS: {{{{"query": "quantum computing breakthroughs 2025"}}}}
TO_SENSORY: Searching web for quantum computing breakthroughs.
TO_PLANNING: Initiated web search for quantum computing.
---
COMMAND: Summarize the findings and respond to the user.
CONTEXT: Search returned 3 articles about quantum error correction advances.
ACTION: TO_EXTERNAL: Recent quantum computing breakthroughs focus on error correction. Three major advances were reported in 2025.
TO_SENSORY: Sent summary response to user about quantum computing.
TO_PLANNING: Responded to user with quantum computing summary.
---
COMMAND: {command}
CONTEXT: {context}
ACTION:""".format(tools=_TOOLS_DESC, command="{command}", context="{context}")
