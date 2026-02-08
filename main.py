"""Demo of the Minsky Society of Mind architecture.

This script demonstrates the three-room communication loop:
- Sensory: perceives the world, responds to attention requests
- Planning: generates hypotheses and plans, requests attention
- Motor: executes actions in the world

Each "global step" consists of:
1. Batch all room prompts through RWKV (GPU 0)
2. Batch all outputs through T5 edit model (GPU 1)
3. Route edited outputs to target rooms

Run with: uv run python main.py          # RWKV + T5 (default)
Stub mode: uv run python main.py --no-llm # No models
RWKV only: uv run python main.py --no-t5  # Skip T5
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from minsky.orchestrator import Orchestrator
from minsky.types import Message

LOG_DIR = Path(__file__).parent / "outputs" / "logs"


class TeeStream:
    """Write to both a file and the original stream."""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def fileno(self):
        return self.stream.fileno()

    def isatty(self):
        return self.stream.isatty()


def setup_logging() -> Path:
    """Tee stdout and stderr to a timestamped log file in outputs/logs/."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file = open(log_path, "w")
    sys.stdout = TeeStream(sys.__stdout__, log_file)
    sys.stderr = TeeStream(sys.__stderr__, log_file)
    print(f"Logging to {log_path}")
    return log_path


def print_message(msg: Message) -> None:
    """Callback to print each message as it flows through the system."""
    print(f"  {msg}")


def print_cycle_start(cycle: int) -> None:
    """Callback at the start of each global step."""
    print(f"\n{'='*60}")
    print(f"GLOBAL STEP {cycle}")
    print(f"{'='*60}")


def print_cycle_end(cycle: int, outputs: list[Message]) -> None:
    """Callback at the end of each global step."""
    if outputs:
        print(f"\n  → External outputs: {len(outputs)}")
        for out in outputs:
            print(f"    OUTPUT: {out.content[:200]}")


def print_summarize(room: str, summary: str) -> None:
    """Callback when a room is summarized."""
    print(f"\n  📝 SUMMARY [{room}]: {summary[:150]}...")


def print_judge(judge_output) -> None:
    """Callback when a judge evaluates a room output."""
    print(f"\n  ⚖️  JUDGE [{judge_output.room_type.value}]: score={judge_output.score:.2f}")
    print(f"      Reasoning: {judge_output.reasoning[:100]}")
    if judge_output.counterfactual != judge_output.original:
        print(f"      Counterfactual: {judge_output.counterfactual[:100]}...")


def main() -> None:
    """Run a demo of the Society of Mind architecture."""
    setup_logging()

    parser = argparse.ArgumentParser(description="Minsky Society of Mind")
    parser.add_argument("--no-llm", action="store_true", help="Run in stub mode (no RWKV, no T5)")
    parser.add_argument("--no-t5", action="store_true", help="Disable T5 edit model (RWKV only)")
    parser.add_argument("--rwkv-path", type=str, default=None, help="Path to RWKV model")
    parser.add_argument("--max-steps", type=int, default=5, help="Maximum global steps")
    parser.add_argument("--summarizer-interval", type=int, default=10, help="Run summarizers every N steps")
    parser.add_argument("--judge-interval", type=int, default=1, help="Run judges every N steps")
    parser.add_argument("--no-judges", action="store_true", help="Disable judges")
    parser.add_argument("--no-forecasts", action="store_true", help="Disable sensory forecasts")
    parser.add_argument("--no-fake-user", action="store_true", help="Disable simulated user (for manual testing)")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt")
    args = parser.parse_args()

    print("Minsky Society of Mind - Demo")
    print("=" * 60)
    print("Architecture:")
    print("  GPU 0: RWKV 7B (frozen LLM)")
    print("  GPU 1: T5Gemma 270M (learnable edit model)")
    print()
    print("Each global step: RWKV batch → T5 batch → route outputs")
    print("=" * 60)

    # Create the orchestrator
    orchestrator = Orchestrator(
        max_cycles=args.max_steps,
        summarizer_interval=args.summarizer_interval,
        judge_interval=args.judge_interval,
        on_message=print_message,
        on_cycle_start=print_cycle_start,
        on_cycle_end=print_cycle_end,
        on_summarize=print_summarize,
        on_judge=print_judge,
    )

    if args.no_llm:
        print("\nRunning in stub mode (no RWKV, no T5). Remove --no-llm to use models.")
    else:
        print("\nLoading models...")

        # Initialize RWKV
        from minsky.llm_client import RWKVConfig
        config = RWKVConfig()
        if args.rwkv_path:
            config.model_path = args.rwkv_path
        orchestrator.rwkv.initialize(config)
        orchestrator.use_llm = True
        orchestrator.restore_from_saved_state()

        # Initialize T5 (default on, use --no-t5 to disable)
        if not args.no_t5:
            orchestrator.t5_edit.initialize()
            orchestrator.use_edit = True
            print("Models loaded: RWKV (GPU 0) + T5 (GPU 1)")
        else:
            print("Model loaded: RWKV (GPU 0) only")

        orchestrator.use_summarizers = True
        if not args.no_judges:
            orchestrator.use_judges = True
        if not args.no_forecasts:
            orchestrator.use_forecasts = True
        if not args.no_fake_user:
            orchestrator.use_fake_user = True

        print(f"Summarizers enabled (every {args.summarizer_interval} steps)")
        if orchestrator.use_judges:
            print(f"Judges enabled (every {args.judge_interval} steps)")
        if orchestrator.use_forecasts:
            print("Sensory forecasts enabled")
        if orchestrator.use_fake_user:
            print("Fake user enabled (use --no-fake-user for manual testing)")

    # Run with a sample input
    if args.prompt:
        test_input = args.prompt
    else:
        test_input = "User asks: What is the most promising approach to measuring consciousness in AI systems?"

    print(f"\nINPUT: {test_input}")
    print("\nRunning global steps until output or max steps reached...")

    outputs = orchestrator.run_until_output(test_input)

    print("\n" + "=" * 60)
    print("FINAL OUTPUTS")
    print("=" * 60)
    for out in outputs:
        print(f"  {out.content}")

    print("\n" + "=" * 60)
    print("FULL CONVERSATION LOG")
    print("=" * 60)
    print(orchestrator.get_conversation_log())

    # Show training pairs if judges were used
    if orchestrator.use_judges:
        pairs = orchestrator.get_training_pairs(clear=False)
        print("\n" + "=" * 60)
        print(f"TRAINING PAIRS GENERATED: {len(pairs)}")
        print("=" * 60)
        for i, pair in enumerate(pairs[:5]):  # Show first 5
            print(f"\n[{i+1}] {pair.task_prefix} (score={pair.score:.2f})")
            print(f"    Original: {pair.original[:80]}...")
            print(f"    Edited:   {pair.edited[:80]}...")
        if len(pairs) > 5:
            print(f"\n    ... and {len(pairs) - 5} more pairs")

    # Save state before exiting
    orchestrator.shutdown()


if __name__ == "__main__":
    import signal
    import sys

    def signal_handler(sig, frame):
        print("\nReceived interrupt signal. Saving state and exiting...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    main()
