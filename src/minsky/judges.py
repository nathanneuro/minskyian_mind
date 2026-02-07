"""Judges for evaluating room outputs and generating training signals.

Each room has a dedicated judge with a specific evaluation rubric.
Judges run through RWKV (same as rooms) and output:
1. A score (0-1)
2. A counterfactual: "what should have been done instead"

The counterfactual becomes the training target for the T5 edit model.
"""

from dataclasses import dataclass
from enum import Enum

from minsky.types import RoomType


# =============================================================================
# Judge System Prompts
# =============================================================================

SENSORY_JUDGE_PROMPT = """You are evaluating the Sensory module of a cognitive system.

The Sensory module's goals are:
1. Accurately perceive and describe relevant details from the world
2. Successfully direct attention according to Planning's requests
3. Provide useful context without unnecessary noise
4. Predict what information will be needed next

Evaluation criteria:
- Relevance: Did it focus on what was asked?
- Accuracy: Are the observations correct and specific?
- Completeness: Did it capture the key details?
- Conciseness: Did it avoid irrelevant information?

You will be given:
- The attention request from Planning
- The Sensory module's response
- Any available ground truth or context

Output format (you MUST follow this exactly):
SCORE: [0.0-1.0]
REASONING: [1-2 sentences explaining the score]
COUNTERFACTUAL: [What Sensory should have said instead. Write the improved response directly.]
"""

PLANNING_JUDGE_PROMPT = """You are evaluating the Planning module of a cognitive system.

The Planning module's goals are:
1. Generate at least 2 hypotheses to explain the situation
2. Create actionable plans for each hypothesis
3. Accurately predict expected value (EV) of each plan
4. Select and communicate the highest EV plan clearly
5. Avoid action vacillation/churn (don't flip-flop on decisions)

Evaluation criteria:
- Hypothesis quality: Are the hypotheses distinct and plausible?
- Plan specificity: Are plans concrete and actionable?
- EV calibration: Are probability estimates reasonable?
- Decision quality: Was the best plan selected?
- Clarity: Can Motor easily execute the instruction?

You will be given:
- The perception/context Planning received
- The Planning module's response (hypotheses + plan)
- The eventual outcome (if available)

Output format (you MUST follow this exactly):
SCORE: [0.0-1.0]
REASONING: [1-2 sentences explaining the score]
COUNTERFACTUAL: [What Planning should have said instead. Write the improved response directly.]
"""

MOTOR_JUDGE_PROMPT = """You are evaluating the Motor module of a cognitive system.

The Motor module's goals are:
1. Successfully follow instructions from Planning
2. Translate high-level plans into concrete actions
3. Execute actions accurately in the world
4. Report results honestly back to Planning

Evaluation criteria:
- Fidelity: Did it follow the instruction correctly?
- Translation: Was the high-level plan properly concretized?
- Execution: Was the action performed successfully?
- Communication: Is the output clear and appropriate?

You will be given:
- The instruction from Planning
- Any sensory context available
- The Motor module's action/output
- The actual result (if available)

Output format (you MUST follow this exactly):
SCORE: [0.0-1.0]
REASONING: [1-2 sentences explaining the score]
COUNTERFACTUAL: [What Motor should have said instead. Write the improved response directly.]
"""

# Map room types to their judge prompts
JUDGE_PROMPTS = {
    RoomType.SENSORY: SENSORY_JUDGE_PROMPT,
    RoomType.PLANNING: PLANNING_JUDGE_PROMPT,
    RoomType.MOTOR: MOTOR_JUDGE_PROMPT,
}


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
# Batch Judge Processing
# =============================================================================

def build_judge_batch(inputs: list[JudgeInput]) -> list[str]:
    """Build prompts for a batch of judge evaluations.

    Args:
        inputs: List of JudgeInput to evaluate.

    Returns:
        List of prompt strings (same order as inputs).
    """
    return [build_judge_prompt(inp) for inp in inputs]


def parse_judge_batch(
    raw_outputs: list[str],
    inputs: list[JudgeInput],
) -> list[JudgeOutput]:
    """Parse a batch of raw judge outputs.

    Args:
        raw_outputs: Raw RWKV outputs.
        inputs: Original inputs (for context).

    Returns:
        List of JudgeOutput (same order as inputs).
    """
    return [
        parse_judge_output(raw, inp)
        for raw, inp in zip(raw_outputs, inputs)
    ]
