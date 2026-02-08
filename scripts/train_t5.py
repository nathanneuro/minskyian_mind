"""Standalone T5 training script.

Trains the T5 edit model on accumulated judge-generated training data
from data/train_data/batch_*.jsonl files.

Usage:
    uv run python scripts/train_t5.py                  # train on pending data
    uv run python scripts/train_t5.py --replay 5        # also replay last 5 used files
    uv run python scripts/train_t5.py --epochs 3        # 3 passes over each file
    uv run python scripts/train_t5.py --rollback        # rollback to previous checkpoint
    uv run python scripts/train_t5.py --stats           # show stats only
    uv run python scripts/train_t5.py --config o.toml   # custom config
"""

import argparse
import sys
from pathlib import Path

import tomllib

from minsky.edit_model import (
    EditModel,
    EditModelConfig,
    EditModelTrainer,
    TRAIN_DATA_DIR,
    USED_TRAIN_DATA_DIR,
    CHECKPOINTS_DIR,
)


def load_config(config_path: str) -> dict:
    """Load and return parsed TOML config."""
    path = Path(config_path)
    if not path.exists():
        print(f"Error: config file not found: {path}")
        sys.exit(1)
    with open(path, "rb") as f:
        return tomllib.load(f)


def print_stats(trainer: EditModelTrainer) -> None:
    """Print training statistics."""
    stats = trainer.get_stats()
    pending_files = trainer.get_pending_files()

    print("\n=== T5 Training Stats ===")
    print(f"  Pending files:        {len(pending_files)}")
    for f in pending_files:
        print(f"    - {f.name}")
    print(f"  Used data files:      {stats['used_data_files']}")
    print(f"  Total pairs trained:  {stats['total_pairs_trained']}")
    print(f"  Batch size:           {stats['batch_size']}")
    print(f"  Learning rate:        {stats['learning_rate']}")

    # Checkpoint info
    latest = CHECKPOINTS_DIR / "t5_latest"
    previous = CHECKPOINTS_DIR / "t5_previous"
    print(f"  Latest checkpoint:    {'yes' if latest.exists() else 'no'}")
    print(f"  Previous checkpoint:  {'yes' if previous.exists() else 'no'}")
    print()


def print_summary(all_results: list[dict]) -> None:
    """Print training summary across all epochs."""
    total_files = sum(r["files_processed"] for r in all_results)
    total_batches = sum(r["batches_trained"] for r in all_results)
    total_pairs = all_results[-1]["total_pairs_trained"] if all_results else 0

    all_losses = []
    for r in all_results:
        if r["avg_loss"] > 0:
            all_losses.append(r["avg_loss"])

    avg_loss = sum(all_losses) / len(all_losses) if all_losses else 0

    print("\n=== Training Summary ===")
    print(f"  Epochs:           {len(all_results)}")
    print(f"  Files processed:  {total_files}")
    print(f"  Batches trained:  {total_batches}")
    print(f"  Avg loss:         {avg_loss:.4f}" if avg_loss > 0 else "  Avg loss:         N/A")
    print(f"  Total pairs:      {total_pairs}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Train T5 edit model on accumulated data")
    parser.add_argument("--config", default="config.toml", help="Path to TOML config file")
    parser.add_argument("--stats", action="store_true", help="Show training stats and exit")
    parser.add_argument("--rollback", action="store_true", help="Rollback to previous checkpoint")
    parser.add_argument("--replay", type=int, default=0, metavar="N",
                        help="Replay last N used data files after training pending")
    parser.add_argument("--epochs", type=int, default=None, metavar="N",
                        help="Number of training epochs (overrides config)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    t5_cfg = config.get("t5", {})
    train_cfg = config.get("training", {})

    epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", 1)

    # Build EditModel + EditModelTrainer
    edit_model = EditModel(
        config=EditModelConfig(
            device=t5_cfg.get("device", "cuda:1"),
            max_input_length=t5_cfg.get("max_input_length", 512),
            max_output_length=t5_cfg.get("max_output_length", 512),
        )
    )
    trainer = EditModelTrainer(
        model=edit_model,
        learning_rate=train_cfg.get("learning_rate", 1e-4),
        batch_size=train_cfg.get("batch_size", 4),
    )

    # Try loading latest checkpoint
    if (CHECKPOINTS_DIR / "t5_latest").exists():
        trainer.load_checkpoint("latest")

    # --stats: print and exit
    if args.stats:
        print_stats(trainer)
        return

    # --rollback: rollback and exit
    if args.rollback:
        success = trainer.rollback()
        if success:
            print("Rolled back to previous checkpoint.")
            trainer.save_checkpoint("latest")
            print("Saved rolled-back model as latest checkpoint.")
        else:
            print("No previous checkpoint to rollback to.")
        return

    # Training loop
    all_results = []
    for epoch in range(1, epochs + 1):
        if epochs > 1:
            print(f"\n--- Epoch {epoch}/{epochs} ---")

        # Train on pending data (only meaningful in epoch 1 since files get moved)
        result = trainer.train_all_pending()
        if result["files_processed"] > 0:
            print(f"  Trained on {result['files_processed']} pending file(s), "
                  f"{result['batches_trained']} batches, "
                  f"avg loss: {result['avg_loss']:.4f}")

        # Replay used data
        if args.replay > 0:
            replay_pairs = trainer.load_used_data(args.replay)
            if replay_pairs:
                trainer.add_training_pairs(replay_pairs)
                replay_losses = []
                while len(trainer.training_pairs) >= trainer.batch_size:
                    loss = trainer.train_step()
                    if loss is not None:
                        replay_losses.append(loss)
                if replay_losses:
                    avg_replay_loss = sum(replay_losses) / len(replay_losses)
                    print(f"  Replayed {len(replay_pairs)} pairs from {args.replay} file(s), "
                          f"{len(replay_losses)} batches, avg loss: {avg_replay_loss:.4f}")
                    result["batches_trained"] += len(replay_losses)
                    result["avg_loss"] = (
                        (result["avg_loss"] * (result["batches_trained"] - len(replay_losses))
                         + avg_replay_loss * len(replay_losses))
                        / result["batches_trained"]
                    ) if result["batches_trained"] > 0 else avg_replay_loss
            else:
                print(f"  No used data files found for replay.")

        all_results.append(result)

    # Save final checkpoint if any training happened
    total_batches = sum(r["batches_trained"] for r in all_results)
    if total_batches > 0:
        trainer.save_checkpoint("latest")
        print_summary(all_results)
    else:
        print("\nNo training data found. Add batch files to data/train_data/ first.")
        print_stats(trainer)


if __name__ == "__main__":
    main()
