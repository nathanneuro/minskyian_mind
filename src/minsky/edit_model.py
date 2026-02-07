"""T5Gemma edit model for learning to modify LLM outputs.

The edit model is a small seq2seq transformer that learns to make
targeted edits to the outputs of frozen LLMs. It receives:
- The original LLM output
- Context about the room/task
- (Optional) Knowledge graph facts

And produces an edited version of the output.

Training data flow:
1. Judges generate TrainingPairs (original → counterfactual)
2. Trainer accumulates pairs in memory
3. After training on a batch, pairs are saved to data/used_train_data/
"""

from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
from datetime import datetime
import json

import torch

# Data directories
DATA_DIR = Path(__file__).parent.parent.parent / "data"
TRAIN_DATA_DIR = DATA_DIR / "train_data"  # New training pairs go here
USED_TRAIN_DATA_DIR = DATA_DIR / "used_train_data"  # After training, moved here


@dataclass
class EditModelConfig:
    """Configuration for the T5Gemma edit model."""

    model_name: str = "google/t5gemma-2-270m-270m"
    device: str = "cuda:1"  # GPU 1 for T5 inference + training
    dtype: torch.dtype = torch.bfloat16
    max_input_length: int = 512
    max_output_length: int = 512


@dataclass
class EditModel:
    """T5Gemma-based edit model for modifying LLM outputs.

    Uses task prefixes to distinguish between different rooms:
    - edit_sensory: Edit Sensory room outputs
    - edit_planning: Edit Planning room outputs
    - edit_motor: Edit Motor room outputs
    """

    config: EditModelConfig = field(default_factory=EditModelConfig)
    model: Any = None
    processor: Any = None
    _initialized: bool = False

    def initialize(self) -> None:
        """Load the T5Gemma model and processor."""
        if self._initialized:
            return

        from transformers import AutoProcessor, AutoModelForSeq2SeqLM

        print(f"Loading edit model: {self.config.model_name} on {self.config.device}")
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_name,
            torch_dtype=self.config.dtype,
        ).to(self.config.device)
        self._initialized = True
        print("Edit model loaded.")

    def edit(
        self,
        text: str,
        task_prefix: str = "edit",
        context: str = "",
        max_new_tokens: int = 256,
    ) -> str:
        """Edit the given text.

        Args:
            text: The text to edit (LLM output).
            task_prefix: Task prefix (edit_sensory, edit_planning, edit_motor).
            context: Additional context for the edit.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Edited text.
        """
        if not self._initialized:
            self.initialize()

        # Format input with task prefix
        if context:
            prompt = f"{task_prefix}: context: {context} text: {text}"
        else:
            prompt = f"{task_prefix}: {text}"

        # Truncate if too long
        if len(prompt) > self.config.max_input_length * 4:  # rough char estimate
            prompt = prompt[: self.config.max_input_length * 4]

        # Process and generate
        inputs = self.processor(text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        edited = self.processor.decode(outputs[0], skip_special_tokens=True)
        return edited


@dataclass
class TrainingPair:
    """A before/after training pair for the edit model."""

    original: str  # Original LLM output
    edited: str  # What it should have been (from judge)
    task_prefix: str  # Which room this came from
    context: str = ""  # Additional context
    score: float = 1.0  # Weight/importance of this example
    timestamp: str = ""  # When this pair was created

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "original": self.original,
            "edited": self.edited,
            "task_prefix": self.task_prefix,
            "context": self.context,
            "score": self.score,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingPair":
        """Create from dict."""
        return cls(
            original=data["original"],
            edited=data["edited"],
            task_prefix=data["task_prefix"],
            context=data.get("context", ""),
            score=data.get("score", 1.0),
            timestamp=data.get("timestamp", ""),
        )


def save_training_pairs(pairs: list[TrainingPair], batch_name: str | None = None) -> Path:
    """Save training pairs to data/train_data/.

    Args:
        pairs: List of training pairs to save.
        batch_name: Optional name for the batch file.

    Returns:
        Path to the saved file.
    """
    TRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if batch_name is None:
        batch_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    filepath = TRAIN_DATA_DIR / f"batch_{batch_name}.jsonl"

    with open(filepath, "w") as f:
        # Write metadata
        metadata = {
            "type": "metadata",
            "num_pairs": len(pairs),
            "timestamp": datetime.now().isoformat(),
        }
        f.write(json.dumps(metadata) + "\n")

        # Write pairs
        for pair in pairs:
            f.write(json.dumps(pair.to_dict()) + "\n")

    return filepath


def load_training_pairs_from_file(filepath: Path) -> list[TrainingPair]:
    """Load training pairs from a single file."""
    pairs = []
    with open(filepath) as f:
        for line in f:
            data = json.loads(line)
            if data.get("type") == "metadata":
                continue
            pairs.append(TrainingPair.from_dict(data))
    return pairs


class EditModelTrainer:
    """Trainer for the edit model using judge-generated pairs.

    Training data flow:
    1. Training pairs are saved to data/train_data/ (by orchestrator or manually)
    2. Trainer loads files from train_data/
    3. After training on a file, it's moved to data/used_train_data/
    """

    def __init__(
        self,
        model: EditModel,
        learning_rate: float = 1e-4,
        batch_size: int = 4,
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.training_pairs: list[TrainingPair] = []
        self.optimizer = None
        self.total_pairs_trained = 0
        self.current_file: Path | None = None  # Track which file we're training from

        # Ensure data directories exist
        TRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
        USED_TRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def add_training_pair(self, pair: TrainingPair) -> None:
        """Add a training pair to the in-memory buffer."""
        self.training_pairs.append(pair)

        # Duplicate high-scoring examples (simple weighting)
        if pair.score > 0.8:
            self.training_pairs.append(pair)
        if pair.score > 0.9:
            self.training_pairs.append(pair)

    def add_training_pairs(self, pairs: list[TrainingPair]) -> None:
        """Add multiple training pairs to the in-memory buffer."""
        for pair in pairs:
            self.add_training_pair(pair)

    def load_from_train_data(self) -> int:
        """Load all pending training files from data/train_data/.

        Returns:
            Number of pairs loaded.
        """
        files = sorted(TRAIN_DATA_DIR.glob("batch_*.jsonl"))
        loaded = 0

        for filepath in files:
            pairs = load_training_pairs_from_file(filepath)
            self.add_training_pairs(pairs)
            loaded += len(pairs)

        return loaded

    def get_pending_files(self) -> list[Path]:
        """Get list of files waiting to be trained on."""
        return sorted(TRAIN_DATA_DIR.glob("batch_*.jsonl"))

    def train_step(self) -> float | None:
        """Run a single training step on accumulated pairs.

        After training, the used pairs are saved to data/used_train_data/.

        Returns:
            Average loss, or None if not enough data.
        """
        if len(self.training_pairs) < self.batch_size:
            return None

        if not self.model._initialized:
            self.model.initialize()

        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.model.parameters(),
                lr=self.learning_rate,
            )

        # Take a batch
        batch = self.training_pairs[: self.batch_size]
        self.training_pairs = self.training_pairs[self.batch_size :]

        # Format inputs
        inputs_text = []
        targets_text = []
        for pair in batch:
            if pair.context:
                inp = f"{pair.task_prefix}: context: {pair.context} text: {pair.original}"
            else:
                inp = f"{pair.task_prefix}: {pair.original}"
            inputs_text.append(inp)
            targets_text.append(pair.edited)

        # Tokenize
        inputs = self.model.processor(
            text=inputs_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model.config.max_input_length,
        )
        inputs = {k: v.to(self.model.model.device) for k, v in inputs.items()}

        targets = self.model.processor(
            text=targets_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model.config.max_output_length,
        )
        labels = targets["input_ids"].to(self.model.model.device)

        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model.model(**inputs, labels=labels)
        loss = outputs.loss

        # Backward pass
        loss.backward()
        self.optimizer.step()

        self.total_pairs_trained += len(batch)

        return loss.item()

    def train_on_file(self, filepath: Path) -> list[float]:
        """Train on all pairs in a single file, then move it to used_train_data.

        Args:
            filepath: Path to the training file.

        Returns:
            List of losses from each batch.
        """
        # Load pairs from file
        pairs = load_training_pairs_from_file(filepath)
        self.add_training_pairs(pairs)

        # Train until all pairs are used
        losses = []
        while len(self.training_pairs) >= self.batch_size:
            loss = self.train_step()
            if loss is not None:
                losses.append(loss)

        # Move file to used_train_data
        dest = USED_TRAIN_DATA_DIR / filepath.name
        filepath.rename(dest)

        return losses

    def train_all_pending(self) -> dict:
        """Train on all pending files in data/train_data/.

        Returns:
            Dictionary with training stats.
        """
        files = self.get_pending_files()
        total_losses = []
        files_processed = 0

        for filepath in files:
            losses = self.train_on_file(filepath)
            total_losses.extend(losses)
            files_processed += 1

        return {
            "files_processed": files_processed,
            "batches_trained": len(total_losses),
            "avg_loss": sum(total_losses) / len(total_losses) if total_losses else 0,
            "total_pairs_trained": self.total_pairs_trained,
            "remaining_pairs": len(self.training_pairs),
        }

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        if self.model._initialized:
            self.model.model.save_pretrained(path)
            self.model.processor.save_pretrained(path)
            print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        from transformers import AutoProcessor, AutoModelForSeq2SeqLM

        self.model.processor = AutoProcessor.from_pretrained(path)
        self.model.model = AutoModelForSeq2SeqLM.from_pretrained(
            path,
            device_map="auto",
            torch_dtype=self.model.config.dtype,
        )
        self.model._initialized = True
        print(f"Loaded checkpoint from {path}")

    def load_used_data(self, max_files: int | None = None) -> list[TrainingPair]:
        """Load previously used training data for replay.

        Args:
            max_files: Maximum number of batch files to load.

        Returns:
            List of TrainingPair from used data.
        """
        pairs = []
        files = sorted(USED_TRAIN_DATA_DIR.glob("batch_*.jsonl"))

        if max_files:
            files = files[-max_files:]

        for filepath in files:
            with open(filepath) as f:
                for line in f:
                    data = json.loads(line)
                    if data.get("type") == "metadata":
                        continue
                    pairs.append(TrainingPair.from_dict(data))

        return pairs

    def get_stats(self) -> dict:
        """Get training statistics."""
        used_files = list(USED_TRAIN_DATA_DIR.glob("batch_*.jsonl"))
        return {
            "pending_pairs": len(self.training_pairs),
            "total_pairs_trained": self.total_pairs_trained,
            "used_data_files": len(used_files),
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
        }


def create_edit_fn(
    edit_model: EditModel,
    task_prefix: str,
) -> callable:
    """Create an edit function for a specific room.

    This wraps the edit model to be used in the room pipeline.

    Args:
        edit_model: The edit model instance.
        task_prefix: Task prefix for this room.

    Returns:
        A function that takes (text, context) and returns edited text.
    """

    def edit_fn(text: str, context: str = "") -> str:
        return edit_model.edit(text, task_prefix=task_prefix, context=context)

    return edit_fn


# Shared edit model singleton
_shared_edit_model: EditModel | None = None


def get_shared_edit_model(config: EditModelConfig | None = None) -> EditModel:
    """Get or create the shared edit model."""
    global _shared_edit_model

    if _shared_edit_model is None:
        _shared_edit_model = EditModel(config or EditModelConfig())

    return _shared_edit_model
