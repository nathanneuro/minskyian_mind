"""Configuration for the Minsky Society of Mind architecture."""

from dataclasses import dataclass, field
import torch


@dataclass
class RWKVConfig:
    """Configuration for RWKV model."""

    model_path: str = "/mnt/e/RWKV-Runner/models/rwkv7-g0a-7.2b-20250829-ctx4096"
    vocab_size: int = 65536
    head_size: int = 64
    device: str = "cuda:0"  # GPU 0 for RWKV inference
    dtype: torch.dtype = torch.float16
    max_tokens: int = 256
    temperature: float = 0.7
    noise: float = 0.5


@dataclass
class EditModelConfig:
    """Configuration for the T5Gemma edit model."""

    model_name: str = "google/t5gemma-2-270m-270m"
    device: str = "cuda:1"  # GPU 1 for T5 inference + training
    dtype: torch.dtype = torch.float32  # T5 works better with float32
    max_input_length: int = 512
    max_output_length: int = 512


@dataclass
class SummarizerConfig:
    """Configuration for summarizer agents."""

    # How often to run summarizers (in global steps)
    run_every_n_steps: int = 10

    # Which room pairs to summarize between
    # Each tuple is (source_room, target_room)
    summarize_pairs: list[tuple[str, str]] = field(default_factory=lambda: [
        ("sensory", "planning"),   # Summarize sensory context for planning
        ("planning", "motor"),     # Summarize plans for motor
        ("motor", "sensory"),      # Summarize actions for sensory context
    ])

    # Max tokens for summarization
    max_summary_tokens: int = 128


@dataclass
class OrchestratorConfig:
    """Main configuration for the orchestrator."""

    # Model configs
    rwkv: RWKVConfig = field(default_factory=RWKVConfig)
    edit_model: EditModelConfig = field(default_factory=EditModelConfig)
    summarizer: SummarizerConfig = field(default_factory=SummarizerConfig)

    # Orchestrator settings
    max_global_steps: int = 100
    use_llm: bool = True
    use_edit: bool = True
    use_summarizers: bool = True

    # Callbacks
    verbose: bool = True
