"""RWKV client for the Society of Mind architecture.

Uses the official rwkv pip package (v0.8.31+) with CUDAGraph acceleration.

IMPORTANT: RWKV is stateful - the state represents the model's "memory" of
everything it has processed. This state must be:
1. Maintained across generations (not reset each time)
2. Saved to disk periodically (every N steps)
3. Saved on shutdown
4. Loaded when resuming
"""

import os
import atexit
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

import torch
import numpy as np

# Set RWKV environment before importing
os.environ["RWKV_V7_ON"] = "1"
os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "1"

from rwkv.model import RWKV
from rwkv.utils import PIPELINE

# Default paths
MODELS_DIR = Path(__file__).parent.parent.parent / "models"
STATE_DIR = Path(__file__).parent.parent.parent / "data" / "state"


@dataclass
class RWKVConfig:
    """Configuration for RWKV model."""

    model_path: str = ""  # Set by download_model.py
    strategy: str = "cuda fp16"  # or "cuda fp16 *20+" for offloading
    device: str = "cuda:0"
    state_save_interval: int = 100  # Save state every N generations
    state_file: str = "rwkv_state.pt"  # State filename


@dataclass
class RWKVClient:
    """Client for RWKV inference with persistent state.

    RWKV state represents the model's accumulated context/memory.
    State is:
    - Maintained across all generations (not reset)
    - Saved to disk every N steps and on shutdown
    - Loaded when resuming
    """

    config: RWKVConfig = field(default_factory=RWKVConfig)
    model: Any = None
    pipeline: Any = None
    _initialized: bool = False

    # Persistent state
    state: list = field(default_factory=list)
    generation_count: int = 0

    # CUDAGraph components for fast inference
    _use_cudagraph: bool = True
    _cudagraph: Any = None
    _static_input: Any = None
    _static_state_in: list = field(default_factory=list)
    _static_state_out: list = field(default_factory=list)
    _static_output: Any = None

    def initialize(self) -> None:
        """Load model, set up CUDAGraph, and restore state if available."""
        if self._initialized:
            return

        if not self.config.model_path:
            # Try to find a model in models dir
            if MODELS_DIR.exists():
                pth_files = list(MODELS_DIR.glob("*.pth"))
                if pth_files:
                    self.config.model_path = str(pth_files[0])
                    print(f"Found model: {self.config.model_path}")

        if not self.config.model_path:
            raise ValueError(
                "No model path specified. Run: uv run python scripts/download_model.py"
            )

        print(f"Loading RWKV model: {self.config.model_path}")
        print(f"Strategy: {self.config.strategy}")

        self.model = RWKV(
            model=self.config.model_path,
            strategy=self.config.strategy,
        )
        self.pipeline = PIPELINE(self.model, "rwkv_vocab_v20230424")

        # Load or initialize state
        self._load_state()

        # Set up CUDAGraph for fast inference
        if self._use_cudagraph:
            self._setup_cudagraph()

        # Register shutdown handler to save state
        atexit.register(self.save_state)

        self._initialized = True
        print("RWKV model loaded.")

    def _get_state_path(self) -> Path:
        """Get path to state file."""
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        return STATE_DIR / self.config.state_file

    def _load_state(self) -> None:
        """Load state from disk or initialize fresh."""
        state_path = self._get_state_path()

        if state_path.exists():
            try:
                checkpoint = torch.load(state_path, weights_only=False)
                self.state = checkpoint["state"]
                self.generation_count = checkpoint.get("generation_count", 0)
                print(f"Loaded RWKV state from {state_path}")
                print(f"  Resuming from generation {self.generation_count}")
            except Exception as e:
                print(f"Failed to load state: {e}")
                print("Starting with fresh state.")
                self.state = self.model.generate_zero_state()
                self.generation_count = 0
        else:
            print("No saved state found. Starting fresh.")
            self.state = self.model.generate_zero_state()
            self.generation_count = 0

    def save_state(self) -> None:
        """Save state to disk."""
        if not self._initialized or not self.state:
            return

        state_path = self._get_state_path()
        try:
            # Move state to CPU for saving
            state_cpu = [s.cpu() if hasattr(s, 'cpu') else s for s in self.state]
            torch.save({
                "state": state_cpu,
                "generation_count": self.generation_count,
            }, state_path)
            print(f"Saved RWKV state to {state_path} (gen {self.generation_count})")
        except Exception as e:
            print(f"Failed to save state: {e}")

    def reset_state(self) -> None:
        """Reset state to fresh (loses all context)."""
        self.state = self.model.generate_zero_state()
        self.generation_count = 0
        print("RWKV state reset.")

    def _setup_cudagraph(self) -> None:
        """Set up CUDAGraph for accelerated inference."""
        state = self.model.generate_zero_state()

        self._static_input = torch.empty(
            (self.model.n_embd,), device="cuda", dtype=torch.half
        )
        self._static_state_in = [
            torch.empty_like(x, device="cuda") for x in state
        ]
        self._static_state_out = [
            torch.empty_like(x, device="cuda") for x in state
        ]
        self._static_output = torch.empty(
            (self.model.args.vocab_size,), device="cuda", dtype=torch.half
        )

        self._cudagraph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._cudagraph):
            self._static_output, self._static_state_out = self.model.forward_one_alt(
                self._static_input, self._static_state_in
            )

        print("CUDAGraph initialized for fast inference.")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.0,
        stop_tokens: list[str] | None = None,
    ) -> str:
        """Generate text using persistent state.

        The state accumulates context across all generations.
        State is saved every N generations (config.state_save_interval).

        Args:
            prompt: Input text to continue from.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (1.0 = neutral).
            top_p: Nucleus sampling threshold (0.0 = greedy).
            stop_tokens: Strings that stop generation.

        Returns:
            Generated text (excluding the prompt).
        """
        if not self._initialized:
            self.initialize()

        stop_tokens = stop_tokens or ["\n\n", "User:", "Human:", "---"]

        # Forward pass with prompt using PERSISTENT state
        out, self.state = self.model.forward(self.pipeline.encode(prompt), self.state)

        if self._use_cudagraph and self._cudagraph:
            result = self._generate_with_cudagraph(
                out, max_tokens, temperature, top_p, stop_tokens
            )
        else:
            result = self._generate_slow(
                out, max_tokens, temperature, top_p, stop_tokens
            )

        # Update generation count and save periodically
        self.generation_count += 1
        if self.generation_count % self.config.state_save_interval == 0:
            self.save_state()

        return result

    def _generate_with_cudagraph(
        self,
        out: torch.Tensor,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_tokens: list[str],
    ) -> str:
        """Fast generation using CUDAGraph with persistent state."""
        # Copy current state to static buffers
        for i in range(len(self.state)):
            self._static_state_in[i].copy_(self.state[i])
        self._static_output.copy_(out)

        all_tokens = []
        out_last = 0

        for i in range(max_tokens):
            # Sample token
            token = self.pipeline.sample_logits(
                self._static_output, temperature=temperature, top_p=top_p
            )
            all_tokens.append(token)

            # Check for stop conditions
            try:
                text = self.pipeline.decode(all_tokens[out_last:])
                if "\ufffd" not in text:
                    out_last = i + 1
                    for stop in stop_tokens:
                        if stop in self.pipeline.decode(all_tokens):
                            full_text = self.pipeline.decode(all_tokens)
                            # Update persistent state from static buffers
                            for n in range(len(self.state)):
                                self.state[n].copy_(self._static_state_in[n])
                            return full_text.split(stop)[0].strip()
            except:
                pass

            # Fast forward using CUDAGraph
            self._static_input.copy_(self.model.z["emb.weight"][token])
            self._cudagraph.replay()
            for n in range(len(self.state)):
                self._static_state_in[n].copy_(self._static_state_out[n])

        # Update persistent state from static buffers
        for n in range(len(self.state)):
            self.state[n].copy_(self._static_state_in[n])

        try:
            return self.pipeline.decode(all_tokens).strip()
        except:
            return "[decode error]"

    def _generate_slow(
        self,
        out: torch.Tensor,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_tokens: list[str],
    ) -> str:
        """Slower generation without CUDAGraph (fallback), updates persistent state."""
        all_tokens = []

        for _ in range(max_tokens):
            token = self.pipeline.sample_logits(
                out, temperature=temperature, top_p=top_p
            )
            all_tokens.append(token)

            try:
                text = self.pipeline.decode(all_tokens)
                for stop in stop_tokens:
                    if stop in text:
                        return text.split(stop)[0].strip()
            except:
                pass

            # Update persistent state
            out, self.state = self.model.forward(token, self.state)

        try:
            return self.pipeline.decode(all_tokens).strip()
        except:
            return "[decode error]"

    def chat(
        self,
        message: str,
        system_prompt: str = "",
        use_thinking: bool = False,
    ) -> str:
        """Chat interface with RWKV-7 prompt format.

        Args:
            message: User message (will strip trailing whitespace).
            system_prompt: Optional system prompt.
            use_thinking: If True, uses <think> prefix for reasoning.

        Returns:
            Assistant response.
        """
        # Strip trailing whitespace (important for RWKV tokenizer)
        message = message.rstrip()

        # Replace \n\n with \n in user message (RWKV uses \n\n as round separator)
        message = message.replace("\n\n", "\n")

        prompt = ""
        if system_prompt:
            prompt = f"System: {system_prompt.rstrip()}\n\n"

        prompt += f"User: {message}\n\nA:"

        if use_thinking:
            prompt += " <think"  # Model will reason first
        else:
            prompt += " <think></think"  # Fake think (recommended)

        return self.generate(
            prompt,
            temperature=1.0,
            top_p=0.5,  # Recommended for chat
            stop_tokens=["User:", "\n\nUser"],
        )


# Singleton for shared model
_shared_client: RWKVClient | None = None


def get_shared_client(config: RWKVConfig | None = None) -> RWKVClient:
    """Get or create the shared RWKV client."""
    global _shared_client

    if _shared_client is None:
        _shared_client = RWKVClient(config=config or RWKVConfig())
        _shared_client.initialize()

    return _shared_client


def create_room_llm(
    room_name: str,
    system_prompt: str,
    config: RWKVConfig | None = None,
    use_thinking: bool = False,
) -> callable:
    """Create an LLM function for a specific room.

    Returns a function that takes a prompt and returns a response.
    Uses RWKV-7 chat format with recommended decoding params.
    """
    client = get_shared_client(config)

    # Clean system prompt
    system_prompt = system_prompt.rstrip().replace("\n\n", "\n")

    def llm_fn(prompt: str) -> str:
        # Strip and clean prompt
        prompt = prompt.rstrip().replace("\n\n", "\n")

        # Build RWKV-7 format
        full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nA:"
        if use_thinking:
            full_prompt += " <think"
        else:
            full_prompt += " <think></think"

        return client.generate(
            full_prompt,
            max_tokens=256,
            temperature=1.0,
            top_p=0.5,
        )

    return llm_fn
