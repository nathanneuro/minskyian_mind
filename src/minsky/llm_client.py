"""LLM client for the Society of Mind architecture.

Supports two backends (selected via config.toml [llm] backend):

- "hf": HuggingFace transformers (e.g. Qwen3-8B). Uses apply_chat_template()
  for chat-tuned models, or raw prompts for base models.
- "rwkv": Official rwkv pip package (v0.8.31+) with CUDAGraph acceleration
  and persistent KV state.

Both backends expose the same interface: initialize(), generate(),
save_state(), saved_metadata.
"""

import json
import atexit
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

import torch

DATA_DIR = Path(__file__).parent.parent.parent / "data"
MODELS_DIR = DATA_DIR / "models"
STATE_DIR = DATA_DIR / "state"


# ---------------------------------------------------------------------------
# Unified config
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """Configuration for all LLM backends."""

    backend: str = "hf"                    # "hf" or "rwkv"
    model_name: str = "Qwen/Qwen3-8B"     # HF model id, or RWKV .pth path
    device: str = "cuda:0"
    dtype: str = "float16"
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    state_file: str = "llm_state.json"     # metadata persistence file
    use_chat_template: bool = True         # True for chat models, False for RWKV/base

    # RWKV-specific
    strategy: str = "cuda fp16"
    state_save_interval: int = 100
    use_cudagraph: bool = False


# Backward-compat alias
RWKVConfig = LLMConfig


# ---------------------------------------------------------------------------
# HuggingFace transformers backend
# ---------------------------------------------------------------------------

@dataclass
class HFClient:
    """HuggingFace transformers client.

    - use_chat_template=True: wraps prompt in chat messages, calls apply_chat_template()
    - use_chat_template=False: sends raw prompt (for base/document-continuation models)

    State persistence is metadata-only (generation_count, global_step).
    """

    config: LLMConfig = field(default_factory=LLMConfig)
    model: Any = None
    tokenizer: Any = None
    _initialized: bool = False

    generation_count: int = 0
    saved_metadata: dict = field(default_factory=dict)

    def initialize(self) -> None:
        """Load model and tokenizer, restore metadata."""
        if self._initialized:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.dtype, torch.float16)

        print(f"Loading HF model: {self.config.model_name}")
        print(f"Device: {self.config.device}, dtype: {self.config.dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch_dtype,
            device_map=self.config.device,
        )
        self.model.eval()

        self._load_metadata()
        atexit.register(self.save_state)

        self._initialized = True
        print(f"HF model loaded: {self.config.model_name}")

    def _get_state_path(self) -> Path:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        return STATE_DIR / self.config.state_file

    def _load_metadata(self) -> None:
        state_path = self._get_state_path()
        if state_path.exists():
            try:
                data = json.loads(state_path.read_text())
                self.generation_count = data.get("generation_count", 0)
                self.saved_metadata = data.get("metadata", {})
                print(f"Loaded metadata from {state_path}")
                print(f"  Resuming from generation {self.generation_count}")
                if self.saved_metadata:
                    print(f"  Metadata: {self.saved_metadata}")
            except Exception as e:
                print(f"Failed to load metadata: {e}")
                self.generation_count = 0
                self.saved_metadata = {}
        else:
            print("No saved metadata found. Starting fresh.")
            self.generation_count = 0
            self.saved_metadata = {}

    def save_state(self, metadata: dict | None = None) -> None:
        if not self._initialized:
            return
        if metadata:
            self.saved_metadata.update(metadata)
        state_path = self._get_state_path()
        try:
            data = {
                "generation_count": self.generation_count,
                "metadata": self.saved_metadata,
            }
            state_path.write_text(json.dumps(data, indent=2))
            print(f"Saved metadata to {state_path} (gen {self.generation_count})")
        except Exception as e:
            print(f"Failed to save metadata: {e}")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_tokens: list[str] | None = None,
    ) -> str:
        """Generate text from a prompt.

        If use_chat_template=True, wraps prompt as a user message and applies
        the tokenizer's chat template. Otherwise sends the raw prompt.
        """
        if not self._initialized:
            self.initialize()

        stop_tokens = stop_tokens or ["\n\n", "User:", "Human:", "---"]

        if self.config.use_chat_template:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        prompt_len = input_ids.shape[-1]

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=20,
                do_sample=True,
            )

        new_tokens = output_ids[0][prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        for stop in stop_tokens:
            if stop in text:
                text = text.split(stop)[0].strip()

        self.generation_count += 1
        return text


# ---------------------------------------------------------------------------
# RWKV backend (lazy imports — rwkv package only needed when backend="rwkv")
# ---------------------------------------------------------------------------

_rwkv_env_ready = False


def _setup_rwkv_env() -> None:
    """Set CUDA_HOME and RWKV environment variables. Called once before RWKV import."""
    global _rwkv_env_ready
    if _rwkv_env_ready:
        return

    import os
    import subprocess
    import shutil

    if "CUDA_HOME" not in os.environ:
        nvcc_path = shutil.which("nvcc")
        if nvcc_path:
            cuda_home = str(Path(nvcc_path).parent.parent)
            os.environ["CUDA_HOME"] = cuda_home
            print(f"Set CUDA_HOME to: {cuda_home} (from nvcc)")
        else:
            for cuda_path in ["/usr/local/cuda", "/usr/lib/cuda", "/opt/cuda"]:
                if os.path.isdir(cuda_path) and os.path.exists(f"{cuda_path}/bin/nvcc"):
                    os.environ["CUDA_HOME"] = cuda_path
                    print(f"Set CUDA_HOME to: {cuda_path}")
                    break
            else:
                try:
                    result = subprocess.run(
                        ["find", "/usr", "-name", "nvcc", "-type", "f"],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.stdout.strip():
                        nvcc_found = result.stdout.strip().split('\n')[0]
                        cuda_home = str(Path(nvcc_found).parent.parent)
                        os.environ["CUDA_HOME"] = cuda_home
                        print(f"Set CUDA_HOME to: {cuda_home} (found via search)")
                except Exception:
                    pass

                if "CUDA_HOME" not in os.environ:
                    print("WARNING: CUDA toolkit (nvcc) not found.")
                    print("Install with: sudo apt install nvidia-cuda-toolkit")
                    print("Or set CUDA_HOME manually.")

    os.environ["RWKV_V7_ON"] = "1"
    os.environ["RWKV_JIT_ON"] = "1"

    cuda_on = os.environ.get("RWKV_CUDA_ON", "1")
    if cuda_on == "1":
        try:
            result = subprocess.run(
                ["nvcc", "--list-gpu-arch"],
                capture_output=True, text=True, timeout=5
            )
            if "compute_89" not in result.stdout and "sm_89" not in result.stdout:
                print("WARNING: CUDA toolkit doesn't support compute_89 (4090/Ada).")
                print("Disabling CUDA JIT. Install CUDA 12.x for GPU acceleration.")
                cuda_on = "0"
        except Exception:
            cuda_on = "0"
            print("WARNING: CUDA JIT disabled (nvcc check failed).")

    os.environ["RWKV_CUDA_ON"] = cuda_on
    _rwkv_env_ready = True


@dataclass
class RWKVClient:
    """Client for RWKV inference with persistent state.

    RWKV state represents the model's accumulated context/memory.
    State is:
    - Maintained across all generations (not reset)
    - Saved to disk every N steps and on shutdown
    - Loaded when resuming
    """

    config: LLMConfig = field(default_factory=LLMConfig)
    model: Any = None
    pipeline: Any = None
    _initialized: bool = False

    # Persistent state
    state: list = field(default_factory=list)
    generation_count: int = 0
    saved_metadata: dict = field(default_factory=dict)

    # CUDAGraph components
    _use_cudagraph: bool = False
    _cudagraph: Any = None
    _static_input: Any = None
    _static_state_in: list = field(default_factory=list)
    _static_state_out: list = field(default_factory=list)
    _static_output: Any = None

    def initialize(self) -> None:
        """Load model, set up CUDAGraph, and restore state if available."""
        if self._initialized:
            return

        _setup_rwkv_env()
        from rwkv.model import RWKV
        from rwkv.utils import PIPELINE

        model_path = self.config.model_name
        if not model_path:
            if MODELS_DIR.exists():
                pth_files = list(MODELS_DIR.glob("*.pth"))
                if pth_files:
                    model_path = str(pth_files[0])
                    print(f"Found model: {model_path}")

        if not model_path:
            raise ValueError(
                "No model path specified. Set model_name in [llm] config "
                "or run: uv run python scripts/download_model.py"
            )

        # RWKV library expects path WITHOUT .pth extension
        if model_path.endswith(".pth"):
            model_path = model_path[:-4]

        print(f"Loading RWKV model: {model_path}")
        print(f"Strategy: {self.config.strategy}")

        self.model = RWKV(
            model=model_path,
            strategy=self.config.strategy,
        )
        self.pipeline = PIPELINE(self.model, "rwkv_vocab_v20230424")

        self._load_state()

        self._use_cudagraph = self.config.use_cudagraph
        if self._use_cudagraph:
            try:
                self._setup_cudagraph()
                print("CUDAGraph initialized for fast inference.")
            except Exception as e:
                print(f"CUDAGraph setup failed: {e}")
                print("Falling back to standard inference (slower but reliable).")
                self._use_cudagraph = False
        else:
            print("Using standard inference (CUDAGraph disabled).")

        atexit.register(self.save_state)

        self._initialized = True
        print("RWKV model loaded.")

    def _get_state_path(self) -> Path:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        return STATE_DIR / self.config.state_file

    def _move_state_to_device(self, state: list) -> list:
        device = self.config.device
        return [s.to(device) if hasattr(s, 'to') else s for s in state]

    def _load_state(self) -> None:
        state_path = self._get_state_path()

        if state_path.exists():
            try:
                checkpoint = torch.load(state_path, weights_only=False)
                self.state = self._move_state_to_device(checkpoint["state"])
                self.generation_count = checkpoint.get("generation_count", 0)
                self.saved_metadata = checkpoint.get("metadata", {})
                print(f"Loaded RWKV state from {state_path}")
                print(f"  Resuming from generation {self.generation_count}")
                if self.saved_metadata:
                    print(f"  Metadata: {self.saved_metadata}")
            except Exception as e:
                print(f"Failed to load state: {e}")
                print("Starting with fresh state.")
                self.state = self._move_state_to_device(self.model.generate_zero_state())
                self.generation_count = 0
                self.saved_metadata = {}
        else:
            print("No saved state found. Starting fresh.")
            self.state = self._move_state_to_device(self.model.generate_zero_state())
            self.generation_count = 0
            self.saved_metadata = {}

    def save_state(self, metadata: dict | None = None) -> None:
        if not self._initialized or not self.state:
            return
        if metadata:
            self.saved_metadata.update(metadata)
        state_path = self._get_state_path()
        try:
            state_cpu = [s.cpu() if hasattr(s, 'cpu') else s for s in self.state]
            torch.save({
                "state": state_cpu,
                "generation_count": self.generation_count,
                "metadata": self.saved_metadata,
            }, state_path)
            print(f"Saved RWKV state to {state_path} (gen {self.generation_count}, metadata={self.saved_metadata})")
        except Exception as e:
            print(f"Failed to save state: {e}")

    def reset_state(self) -> None:
        self.state = self._move_state_to_device(self.model.generate_zero_state())
        self.generation_count = 0
        print("RWKV state reset.")

    def _setup_cudagraph(self) -> None:
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
        """Generate text using persistent state (document-continuation style)."""
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
        for i in range(len(self.state)):
            self._static_state_in[i].copy_(self.state[i])
        self._static_output.copy_(out)

        all_tokens = []
        out_last = 0

        for i in range(max_tokens):
            token = self.pipeline.sample_logits(
                self._static_output, temperature=temperature, top_p=top_p
            )
            all_tokens.append(token)

            try:
                text = self.pipeline.decode(all_tokens[out_last:])
                if "\ufffd" not in text:
                    out_last = i + 1
                    for stop in stop_tokens:
                        if stop in self.pipeline.decode(all_tokens):
                            full_text = self.pipeline.decode(all_tokens)
                            for n in range(len(self.state)):
                                self.state[n].copy_(self._static_state_in[n])
                            return full_text.split(stop)[0].strip()
            except (UnicodeDecodeError, ValueError, RuntimeError):
                pass

            self._static_input.copy_(self.model.z["emb.weight"][token])
            self._cudagraph.replay()
            for n in range(len(self.state)):
                self._static_state_in[n].copy_(self._static_state_out[n])

        for n in range(len(self.state)):
            self.state[n].copy_(self._static_state_in[n])

        try:
            return self.pipeline.decode(all_tokens).strip()
        except Exception:
            return "[decode error]"

    def _generate_slow(
        self,
        out: torch.Tensor,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_tokens: list[str],
    ) -> str:
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
            except Exception:
                pass

            out, self.state = self.model.forward(token, self.state)

        try:
            return self.pipeline.decode(all_tokens).strip()
        except Exception:
            return "[decode error]"

    def chat(
        self,
        message: str,
        system_prompt: str = "",
        use_thinking: bool = False,
    ) -> str:
        """Chat interface with RWKV-7 prompt format."""
        message = message.rstrip()
        message = message.replace("\n\n", "\n")

        prompt = ""
        if system_prompt:
            prompt = f"System: {system_prompt.rstrip()}\n\n"

        prompt += f"User: {message}\n\nA:"

        if use_thinking:
            prompt += " <think"
        else:
            prompt += " <think></think"

        return self.generate(
            prompt,
            temperature=1.0,
            top_p=0.5,
            stop_tokens=["User:", "\n\nUser"],
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_llm_client(config: LLMConfig) -> HFClient | RWKVClient:
    """Create the appropriate LLM client based on config.backend."""
    if config.backend == "rwkv":
        return RWKVClient(config=config)
    elif config.backend == "hf":
        return HFClient(config=config)
    else:
        raise ValueError(f"Unknown LLM backend: {config.backend!r}. Use 'hf' or 'rwkv'.")


# Backward-compat alias
RWKVClient_legacy = RWKVClient
