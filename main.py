"""Demo of the Minsky Society of Mind architecture.

This script demonstrates the three-room communication loop:
- Sensory: perceives the world, responds to attention requests
- Planning: generates hypotheses and plans, requests attention
- Motor: executes actions in the world

Each "global step" consists of:
1. Batch all room prompts through RWKV (GPU 0)
2. Batch all outputs through T5 edit model (GPU 1)
3. Route edited outputs to target rooms

Run with: uv run python main.py                    # loads config.toml
Custom:   uv run python main.py --config other.toml
"""

import argparse
import re
import sys
import tomllib
from datetime import datetime
from pathlib import Path

from minsky.orchestrator import Orchestrator
from minsky.types import Message

LOG_DIR = Path(__file__).parent / "outputs" / "logs"


class TeeStream:
    """Write to both a file and the original stream.

    Progress bar output (carriage returns, tqdm-style lines) is shown on
    the terminal but filtered out of the log file to keep logs clean.
    """

    # Matches tqdm-style progress: "  3%|██         | 1/30 [00:01<...]"
    _PROGRESS_RE = re.compile(r"\d+%\|")

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def _is_progress(self, data: str) -> bool:
        """Return True if data looks like a progress bar update."""
        if "\r" in data and "\n" not in data:
            return True
        if self._PROGRESS_RE.search(data):
            return True
        return False

    def write(self, data):
        self.stream.write(data)
        if not self._is_progress(data):
            self.log_file.write(data)
            self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def fileno(self):
        return self.stream.fileno()

    def isatty(self):
        return self.stream.isatty()


def setup_logging() -> tuple[Path, Path]:
    """Tee stdout/stderr to a full log file; also create an external-view log.

    Returns (full_log_path, external_log_path).
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    log_path = LOG_DIR / f"{timestamp}.log"
    log_file = open(log_path, "w")
    sys.stdout = TeeStream(sys.__stdout__, log_file)
    sys.stderr = TeeStream(sys.__stderr__, log_file)

    ext_log_path = LOG_DIR / f"{timestamp}_external.log"
    print(f"Logging to {log_path}")
    print(f"External view log: {ext_log_path}")
    return log_path, ext_log_path


def load_config(config_path: str) -> dict:
    """Load configuration from a TOML file."""
    path = Path(config_path)
    if not path.exists():
        print(f"ERROR: Config file not found: {path.resolve()}")
        sys.exit(1)
    with open(path, "rb") as f:
        return tomllib.load(f)


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
            print(f"    OUTPUT: {out.content}")


def print_summarize(room: str, summary: str) -> None:
    """Callback when a room is summarized."""
    print(f"\n  📝 SUMMARY [{room}]: {summary}")


def print_judge(judge_output) -> None:
    """Callback when a judge evaluates a room output."""
    print(f"\n  ⚖️  JUDGE [{judge_output.room_type.value}]: score={judge_output.score:.2f}")
    print(f"      Reasoning: {judge_output.reasoning}")
    if judge_output.counterfactual != judge_output.original:
        print(f"      Counterfactual: {judge_output.counterfactual}")


def main() -> None:
    """Run a demo of the Society of Mind architecture."""
    _, ext_log_path = setup_logging()
    ext_log = open(ext_log_path, "w")

    def write_external(text: str) -> None:
        """Write a line to the external view log."""
        ext_log.write(text + "\n")
        ext_log.flush()

    parser = argparse.ArgumentParser(description="Minsky Society of Mind")
    parser.add_argument("--config", type=str, default="config.toml", help="Path to TOML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run = cfg.get("run", {})
    features = run.get("features", {})
    intervals = run.get("intervals", {})
    llm_cfg = cfg.get("llm", {})
    t5_cfg = cfg.get("t5", {})

    model_name = llm_cfg.get("model_name", "Qwen/Qwen3-8B")
    llm_backend = llm_cfg.get("backend", "hf")
    llm_device = llm_cfg.get("device", "cuda:0")

    print("Minsky Society of Mind - Demo")
    print("=" * 60)
    print(f"Config: {args.config}")
    print("Architecture:")
    if llm_backend == "api":
        base = llm_cfg.get("api_base_url") or "OpenAI"
        print(f"  API: {model_name} via {base} (frozen LLM)")
    else:
        print(f"  {llm_device}: {model_name} (frozen LLM)")
    print("  GPU 1: T5Gemma 270M (learnable edit model)")
    print()
    print("Each global step: LLM batch → T5 batch → route outputs")
    print("=" * 60)

    max_steps = run.get("max_steps", 100)
    summarizer_interval = intervals.get("summarizer", 10)
    judge_interval = intervals.get("judge", 1)

    def on_cycle_end_with_ext(cycle: int, outputs: list[Message]) -> None:
        """Callback at end of step: print + write external outputs to ext log."""
        print_cycle_end(cycle, outputs)
        for out in outputs:
            if out.content:
                write_external(f"[step {cycle}] Assistant: {out.content}")

    def on_fake_user(reply: str) -> None:
        """Write fake user reply to external log."""
        write_external(f"User: {reply}")

    # Create the orchestrator
    orchestrator = Orchestrator(
        max_cycles=max_steps,
        summarizer_interval=summarizer_interval,
        judge_interval=judge_interval,
        on_message=print_message,
        on_cycle_start=print_cycle_start,
        on_cycle_end=on_cycle_end_with_ext,
        on_summarize=print_summarize,
        on_judge=print_judge,
        on_fake_user=on_fake_user,
    )

    use_llm = features.get("llm", True)
    use_t5 = features.get("t5", True)
    use_summarizers = features.get("summarizers", True)
    use_judges = features.get("judges", True)
    use_forecasts = features.get("forecasts", True)
    use_fake_user = features.get("fake_user", True)

    if not use_llm:
        print("\nRunning in stub mode (no LLM, no T5).")
    else:
        print("\nLoading models...")

        # Initialize LLM
        from minsky.llm_client import LLMConfig
        config = LLMConfig(
            backend=llm_cfg.get("backend", "hf"),
            model_name=llm_cfg.get("model_name", "Qwen/Qwen3-8B"),
            device=llm_cfg.get("device", "cuda:0"),
            dtype=llm_cfg.get("dtype", "float16"),
            max_tokens=llm_cfg.get("max_tokens", 256),
            temperature=llm_cfg.get("temperature", 0.7),
            top_p=llm_cfg.get("top_p", 0.9),
            state_file=llm_cfg.get("state_file", "llm_state.json"),
            use_chat_template=llm_cfg.get("use_chat_template", True),
            strategy=llm_cfg.get("strategy", "cuda fp16"),
            state_save_interval=llm_cfg.get("state_save_interval", 100),
            use_cudagraph=llm_cfg.get("use_cudagraph", False),
            api_base_url=llm_cfg.get("api_base_url", ""),
            api_key_env=llm_cfg.get("api_key_env", ""),
        )
        orchestrator.rwkv.initialize(config)
        orchestrator.use_llm = True
        orchestrator.restore_from_saved_state()

        # Initialize T5
        if use_t5:
            orchestrator.t5_edit.initialize()
            orchestrator.use_edit = True
            if llm_backend == "api":
                print(f"Models loaded: {model_name} (API) + T5 (GPU 1)")
            else:
                print(f"Models loaded: {model_name} ({llm_device}) + T5 (GPU 1)")
        else:
            if llm_backend == "api":
                print(f"Model loaded: {model_name} (API) only")
            else:
                print(f"Model loaded: {model_name} ({llm_device}) only")

        if use_summarizers:
            orchestrator.use_summarizers = True
        if use_judges:
            orchestrator.use_judges = True
        if use_forecasts:
            orchestrator.use_forecasts = True
        if use_fake_user:
            orchestrator.use_fake_user = True

        print(f"Summarizers enabled (every {summarizer_interval} steps)")
        if orchestrator.use_judges:
            print(f"Judges enabled (every {judge_interval} steps)")
        if orchestrator.use_forecasts:
            print("Sensory forecasts enabled")
        if orchestrator.use_fake_user:
            print("Fake user enabled")

    # Run with configured prompt
    prompt = run.get("prompt", "What is the most promising approach to measuring consciousness in AI systems?")
    test_input = f"User asks: {prompt}"

    print(f"\nINPUT: {test_input}")
    write_external(f"User: {prompt}")
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
            print(f"    Raw:      {pair.raw}")
            print(f"    T5:       {pair.t5_edited}")
            print(f"    Improved: {pair.improved}")
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
