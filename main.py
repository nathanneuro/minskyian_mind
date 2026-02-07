"""Demo of the Minsky Society of Mind architecture.

This script demonstrates the three-room communication loop:
- Sensory: perceives the world, responds to attention requests
- Planning: generates hypotheses and plans, requests attention
- Motor: executes actions in the world

Each "global step" consists of:
1. Batch all room prompts through RWKV (GPU 0)
2. Batch all outputs through T5 edit model (GPU 1)
3. Route edited outputs to target rooms

Run with: uv run python main.py
Run with RWKV+T5: uv run python main.py --use-models
"""

import argparse

from minsky.orchestrator import Orchestrator
from minsky.types import Message


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
    parser = argparse.ArgumentParser(description="Minsky Society of Mind Demo")
    parser.add_argument("--use-models", action="store_true", help="Use RWKV + T5 models")
    parser.add_argument("--use-rwkv", action="store_true", help="Use RWKV only (no T5 edit)")
    parser.add_argument("--rwkv-path", type=str, default=None, help="Path to RWKV model")
    parser.add_argument("--max-steps", type=int, default=5, help="Maximum global steps")
    parser.add_argument("--summarizer-interval", type=int, default=10, help="Run summarizers every N steps")
    parser.add_argument("--judge-interval", type=int, default=1, help="Run judges every N steps")
    parser.add_argument("--no-judges", action="store_true", help="Disable judges")
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

    if args.use_models or args.use_rwkv:
        print("\nLoading models...")

        # Initialize RWKV
        if args.rwkv_path:
            from minsky.llm_client import RWKVConfig
            config = RWKVConfig(model_path=args.rwkv_path)
            orchestrator.batched_llm.initialize(config)
        else:
            orchestrator.batched_llm.initialize()

        orchestrator.use_llm = True

        # Initialize T5 if using full model stack
        if args.use_models:
            orchestrator.batched_edit.initialize()
            orchestrator.use_edit = True
            orchestrator.use_summarizers = True
            if not args.no_judges:
                orchestrator.use_judges = True
            print("Models loaded: RWKV (GPU 0) + T5 (GPU 1)")
            print(f"Summarizers enabled (every {args.summarizer_interval} steps)")
            if orchestrator.use_judges:
                print(f"Judges enabled (every {args.judge_interval} steps)")
        else:
            orchestrator.use_summarizers = True
            if not args.no_judges:
                orchestrator.use_judges = True
            print("Models loaded: RWKV only (GPU 0)")
            print(f"Summarizers enabled (every {args.summarizer_interval} steps)")
            if orchestrator.use_judges:
                print(f"Judges enabled (every {args.judge_interval} steps)")
    else:
        print("\nRunning in stub mode (no models). Use --use-models or --use-rwkv to enable.")

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


if __name__ == "__main__":
    main()
