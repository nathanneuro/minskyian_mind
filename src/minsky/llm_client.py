"""RWKV client for the Society of Mind architecture.

Wraps the Albatross RWKV inference engine for use in rooms.
Each room can have its own RWKV state, allowing for parallel "thinking".
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import torch

# Add Albatross to path
ALBATROSS_PATH = Path(__file__).parent.parent.parent / "subrepos" / "Albatross"
sys.path.insert(0, str(ALBATROSS_PATH))

from reference.rwkv7 import RWKV_x070
from reference.utils import TRIE_TOKENIZER, sampler_simple_batch


@dataclass
class RWKVConfig:
    """Configuration for RWKV model."""

    model_path: str = "/mnt/e/RWKV-Runner/models/rwkv7-g0a-7.2b-20250829-ctx4096"
    vocab_size: int = 65536
    head_size: int = 64
    device: str = "cuda:0"  # GPU 0 for RWKV inference
    dtype: torch.dtype = torch.float16


@dataclass
class RWKVClient:
    """Client for RWKV inference with state management.

    Each client maintains its own state, allowing multiple "agents"
    to have independent conversation contexts.
    """

    config: RWKVConfig = field(default_factory=RWKVConfig)
    model: Any = None
    tokenizer: Any = None
    state: Any = None
    _initialized: bool = False

    def initialize(self) -> None:
        """Load model and tokenizer."""
        if self._initialized:
            return

        import types
        args = types.SimpleNamespace()
        args.vocab_size = self.config.vocab_size
        args.head_size = self.config.head_size
        args.MODEL_NAME = self.config.model_path

        print(f"Loading RWKV model: {self.config.model_path}")
        self.model = RWKV_x070(args)
        self.tokenizer = TRIE_TOKENIZER(str(ALBATROSS_PATH / "reference" / "rwkv_vocab_v20230424.txt"))
        self.state = self.model.generate_zero_state(1)  # batch size 1
        self._initialized = True
        print("RWKV model loaded.")

    def reset_state(self) -> None:
        """Reset the conversation state."""
        if self.model:
            self.state = self.model.generate_zero_state(1)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        noise: float = 0.5,
        stop_tokens: list[str] | None = None,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input text to continue from.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            noise: Noise for sampling.
            stop_tokens: List of strings to stop generation at.

        Returns:
            Generated text (excluding the prompt).
        """
        if not self._initialized:
            self.initialize()

        stop_tokens = stop_tokens or ["\n\n", "User:", "Human:"]

        # Encode prompt and process through model
        tokens = self.tokenizer.encode(prompt)
        out = self.model.forward_batch([tokens], self.state)

        generated_tokens = []
        generated_text = ""

        for _ in range(max_tokens):
            # Sample next token
            token = sampler_simple_batch(out, noise=noise, temp=temperature)
            token_list = token.tolist()
            generated_tokens.extend(token_list[0])

            # Decode to check for stop tokens
            try:
                generated_text = self.tokenizer.decode(generated_tokens, utf8_errors="ignore")
            except:
                pass

            # Check stop conditions
            if any(stop in generated_text for stop in stop_tokens):
                # Trim to stop token
                for stop in stop_tokens:
                    if stop in generated_text:
                        generated_text = generated_text.split(stop)[0]
                        break
                break

            # Check for end of document token
            if 0 in token_list[0]:
                break

            # Forward pass for next token
            out = self.model.forward_batch(token_list, self.state)

        return generated_text.strip()

    def chat(self, message: str, system_prompt: str = "") -> str:
        """Simple chat interface.

        Args:
            message: User message.
            system_prompt: Optional system prompt to prepend.

        Returns:
            Assistant response.
        """
        prompt = ""
        if system_prompt:
            prompt = f"{system_prompt}\n\n"
        prompt += f"User: {message}\n\nAssistant:"

        response = self.generate(prompt, stop_tokens=["User:", "\n\nUser", "\n\nHuman"])
        return response


# Singleton for shared model (saves memory when multiple rooms use same model)
_shared_model: RWKV_x070 | None = None
_shared_tokenizer: Any = None


def get_shared_model(config: RWKVConfig | None = None) -> tuple[RWKV_x070, Any]:
    """Get or create the shared RWKV model.

    Using a shared model saves GPU memory when multiple rooms
    need RWKV. Each room can still have its own state.
    """
    global _shared_model, _shared_tokenizer

    if _shared_model is None:
        config = config or RWKVConfig()

        import types
        args = types.SimpleNamespace()
        args.vocab_size = config.vocab_size
        args.head_size = config.head_size
        args.MODEL_NAME = config.model_path

        print(f"Loading shared RWKV model: {config.model_path}")
        _shared_model = RWKV_x070(args)
        _shared_tokenizer = TRIE_TOKENIZER(str(ALBATROSS_PATH / "reference" / "rwkv_vocab_v20230424.txt"))
        print("Shared RWKV model loaded.")

    return _shared_model, _shared_tokenizer


def create_room_llm(
    room_name: str,
    system_prompt: str,
    config: RWKVConfig | None = None,
) -> callable:
    """Create an LLM function for a specific room.

    Returns a function that takes a prompt and returns a response,
    suitable for use as the llm_fn parameter in room processors.

    Args:
        room_name: Name of the room (for logging).
        system_prompt: System prompt specific to this room's role.
        config: Optional RWKV config.

    Returns:
        A callable that takes str and returns str.
    """
    model, tokenizer = get_shared_model(config)
    state = model.generate_zero_state(1)

    def llm_fn(prompt: str) -> str:
        """Generate response for the room."""
        full_prompt = f"{system_prompt}\n\n{prompt}\n\nResponse:"

        tokens = tokenizer.encode(full_prompt)
        out = model.forward_batch([tokens], state)

        generated_tokens = []
        max_tokens = 256
        stop_tokens = ["\n\n", "User:", "---"]

        for _ in range(max_tokens):
            token = sampler_simple_batch(out, noise=0.5, temp=0.7)
            token_list = token.tolist()
            generated_tokens.extend(token_list[0])

            try:
                text = tokenizer.decode(generated_tokens, utf8_errors="ignore")
                if any(stop in text for stop in stop_tokens):
                    for stop in stop_tokens:
                        if stop in text:
                            text = text.split(stop)[0]
                            break
                    return text.strip()
            except:
                pass

            if 0 in token_list[0]:
                break

            out = model.forward_batch(token_list, state)

        try:
            return tokenizer.decode(generated_tokens, utf8_errors="ignore").strip()
        except:
            return "[generation error]"

    return llm_fn
