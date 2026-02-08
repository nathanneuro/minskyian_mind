"""Judges for evaluating room outputs and generating training signals.

Each room has a dedicated judge with a specific evaluation rubric.
Judges use DeepSeek-V3.2 via OpenAI-compatible API and output:
1. A score (0-1)
2. A counterfactual: "what should have been done instead"

The counterfactual becomes the training target for the T5 edit model.
"""

import asyncio
import os
from dataclasses import dataclass

from openai import AsyncOpenAI

from minsky.types import RoomType
from minsky.prompts.judges import JUDGE_PROMPTS

# Judge API configuration
JUDGE_MODEL = "QuantTrio/DeepSeek-V3.2-AWQ"
JUDGE_BASE_URL = "https://api.infinity.inc/v1"
JUDGE_MAX_TOKENS = 1000


# =============================================================================
# Message History Formatting
# =============================================================================

def format_message_history(
    history: list[tuple[str, str, str]],
    max_chars: int = 1500,
) -> str:
    """Format message history tuples into a compact string for judge context.

    Args:
        history: List of (source, message_type, content) tuples.
        max_chars: Maximum total characters for the formatted output.

    Returns:
        Formatted history string, truncated to max_chars.
    """
    if not history:
        return ""

    lines = []
    total = 0
    for source, msg_type, content in history:
        line = f"[{source}->{msg_type}] {content}"
        if total + len(line) + 1 > max_chars:
            remaining = max_chars - total - 4  # room for "..."
            if remaining > 20:
                lines.append(line[:remaining] + "...")
            break
        lines.append(line)
        total += len(line) + 1  # +1 for newline

    return "\n".join(lines)


# =============================================================================
# Judge Data Structures
# =============================================================================

@dataclass
class JudgeInput:
    """Input to a judge for evaluation."""
    room_type: RoomType
    room_output: str  # What the room actually produced
    context: str  # The input/request the room received
    ground_truth: str = ""  # Optional: what actually happened / correct answer
    message_history: list[tuple[str, str, str]] | None = None  # (source, msg_type, content)
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class JudgeOutput:
    """Output from a judge evaluation."""
    room_type: RoomType
    score: float  # 0.0 to 1.0
    reasoning: str
    counterfactual: str  # What should have been said instead
    original: str  # The original room output


def build_judge_prompt(judge_input: JudgeInput) -> str:
    """Build the full prompt for a judge evaluation.

    Args:
        judge_input: The input to evaluate.

    Returns:
        Full prompt string to send to RWKV.
    """
    system_prompt = JUDGE_PROMPTS[judge_input.room_type]

    user_content = f"""Context/Request:
{judge_input.context}

{judge_input.room_type.value.title()} Module's Response:
{judge_input.room_output}
"""

    if judge_input.message_history:
        history_str = format_message_history(judge_input.message_history)
        user_content += f"""
Recent Message History:
{history_str}
"""

    if judge_input.ground_truth:
        user_content += f"""
Ground Truth/Outcome:
{judge_input.ground_truth}
"""

    return f"""{system_prompt}

---

{user_content}

---

Evaluate the response:"""


def parse_judge_output(raw_output: str, judge_input: JudgeInput) -> JudgeOutput:
    """Parse the raw RWKV output into a structured JudgeOutput.

    Args:
        raw_output: Raw text from RWKV.
        judge_input: The original input (for context).

    Returns:
        Parsed JudgeOutput.
    """
    # Default values
    score = 0.5
    reasoning = ""
    counterfactual = judge_input.room_output  # Default to original if parsing fails

    lines = raw_output.strip().split("\n")

    for i, line in enumerate(lines):
        line_upper = line.upper().strip()

        if line_upper.startswith("SCORE:"):
            try:
                score_str = line.split(":", 1)[1].strip()
                # Handle various formats: "0.8", "0.8/1.0", "8/10", etc.
                if "/" in score_str:
                    parts = score_str.split("/")
                    score = float(parts[0]) / float(parts[1])
                else:
                    score = float(score_str)
                score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
            except:
                pass

        elif line_upper.startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()

        elif line_upper.startswith("COUNTERFACTUAL:"):
            # Everything after COUNTERFACTUAL: is the improved response
            counterfactual_parts = [line.split(":", 1)[1].strip()]
            # Also include subsequent lines until end or next section
            for j in range(i + 1, len(lines)):
                next_line = lines[j].strip()
                if next_line.upper().startswith(("SCORE:", "REASONING:")):
                    break
                counterfactual_parts.append(next_line)
            counterfactual = "\n".join(counterfactual_parts).strip()

    return JudgeOutput(
        room_type=judge_input.room_type,
        score=score,
        reasoning=reasoning,
        counterfactual=counterfactual,
        original=judge_input.room_output,
    )


# =============================================================================
# Batch Judge Processing (API-based, concurrent)
# =============================================================================

def _build_chat_messages(judge_input: JudgeInput) -> list[dict]:
    """Build chat messages for the judge API call."""
    system_prompt = JUDGE_PROMPTS[judge_input.room_type]

    user_content = f"""Context/Request:
{judge_input.context}

{judge_input.room_type.value.title()} Module's Response:
{judge_input.room_output}
"""

    if judge_input.message_history:
        history_str = format_message_history(judge_input.message_history)
        user_content += f"""
Recent Message History:
{history_str}
"""

    if judge_input.ground_truth:
        user_content += f"""
Ground Truth/Outcome:
{judge_input.ground_truth}
"""
    user_content += "\nEvaluate the response:"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


async def _judge_one(
    client: AsyncOpenAI,
    judge_input: JudgeInput,
) -> JudgeOutput:
    """Evaluate a single input via the judge API."""
    messages = _build_chat_messages(judge_input)
    try:
        resp = await client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=messages,
            max_tokens=JUDGE_MAX_TOKENS,
        )
        raw = resp.choices[0].message.content
    except Exception as e:
        print(f"Judge API error: {e}")
        raw = f"SCORE: 0.5\nREASONING: API error: {e}\nCOUNTERFACTUAL: {judge_input.room_output}"

    return parse_judge_output(raw, judge_input)


async def _judge_batch_async(inputs: list[JudgeInput]) -> list[JudgeOutput]:
    """Run all judge evaluations concurrently."""
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.environ.get("INF_API_KEY", "")
    if not api_key:
        print("ERROR: INF_API_KEY not set in .env")
        return [
            JudgeOutput(
                room_type=inp.room_type,
                score=0.5,
                reasoning="INF_API_KEY not configured",
                counterfactual=inp.room_output,
                original=inp.room_output,
            )
            for inp in inputs
        ]

    client = AsyncOpenAI(base_url=JUDGE_BASE_URL, api_key=api_key)
    return await asyncio.gather(*[_judge_one(client, inp) for inp in inputs])


def judge_batch(inputs: list[JudgeInput]) -> list[JudgeOutput]:
    """Evaluate a batch of judge inputs via DeepSeek API (concurrent).

    Args:
        inputs: List of JudgeInput to evaluate.

    Returns:
        List of JudgeOutput (same order as inputs).
    """
    if not inputs:
        return []
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already in an async context — run in a new thread to avoid deadlock
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, _judge_batch_async(inputs)).result()
    else:
        return asyncio.run(_judge_batch_async(inputs))


# =============================================================================
# Summarizer via Judge Model (concurrent)
# =============================================================================

async def _summarize_one(client: AsyncOpenAI, prompt: str) -> str:
    """Generate a single summary via the judge model."""
    try:
        resp = await client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Summarizer API error: {e}")
        return f"(summary unavailable: {e})"


async def _summarize_batch_async(prompts: list[str]) -> list[str]:
    """Run all summarizations concurrently."""
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.environ.get("INF_API_KEY", "")
    if not api_key:
        print("ERROR: INF_API_KEY not set in .env")
        return ["(INF_API_KEY not configured)"] * len(prompts)

    client = AsyncOpenAI(base_url=JUDGE_BASE_URL, api_key=api_key)
    return await asyncio.gather(*[_summarize_one(client, p) for p in prompts])


def summarize_batch(prompts: list[str]) -> list[str]:
    """Summarize a batch of room histories via DeepSeek API (concurrent).

    Args:
        prompts: List of summarizer prompts.

    Returns:
        List of summary strings (same order as prompts).
    """
    if not prompts:
        return []
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, _summarize_batch_async(prompts)).result()
    else:
        return asyncio.run(_summarize_batch_async(prompts))


# Keep old functions for backward compat with tests
def build_judge_batch(inputs: list[JudgeInput]) -> list[str]:
    """Build prompts for a batch of judge evaluations."""
    return [build_judge_prompt(inp) for inp in inputs]


def parse_judge_batch(
    raw_outputs: list[str],
    inputs: list[JudgeInput],
) -> list[JudgeOutput]:
    """Parse a batch of raw judge outputs."""
    return [
        parse_judge_output(raw, inp)
        for raw, inp in zip(raw_outputs, inputs)
    ]
