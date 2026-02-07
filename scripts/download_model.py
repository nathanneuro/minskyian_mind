#!/usr/bin/env python3
"""Download models from HuggingFace.

Downloads:
- RWKV7-G1 model (for inference)
- T5Gemma model (for edit/training)

Run with: uv run python scripts/download_model.py
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download

# T5 model for edit
T5_MODEL = "google/t5gemma-2-270m-270m"

# TODO: Alternative RWKV source (SafeTensors format, may be faster):
# https://modelscope.cn/models/shoumenchougou/RWKV-7-World-ST/resolve/master/rwkv7-g1d-7.2b-20260131-ctx8192.st

# Model configuration - RWKV7-G1 series from BlinkDL/rwkv7-g1
MODELS = {
    # G1d series (latest, best quality) - Jan 2026
    "g1d-0.1b": {
        "repo": "BlinkDL/rwkv7-g1",
        "filename": "rwkv7-g1d-0.1b-20260129-ctx8192.pth",
        "size": "382 MB",
    },
    "g1d-2.9b": {
        "repo": "BlinkDL/rwkv7-g1",
        "filename": "rwkv7-g1d-2.9b-20260131-ctx8192.pth",
        "size": "5.9 GB",
    },
    "g1d-7.2b": {
        "repo": "BlinkDL/rwkv7-g1",
        "filename": "rwkv7-g1d-7.2b-20260131-ctx8192.pth",
        "size": "14.4 GB",
    },
    "g1d-13.3b": {
        "repo": "BlinkDL/rwkv7-g1",
        "filename": "rwkv7-g1d-13.3b-20260131-ctx8192.pth",
        "size": "26.5 GB",
    },
    # G1c series - Dec 2025
    "g1c-1.5b": {
        "repo": "BlinkDL/rwkv7-g1",
        "filename": "rwkv7-g1c-1.5b-20260110-ctx8192.pth",
        "size": "3.06 GB",
    },
    "g1c-2.9b": {
        "repo": "BlinkDL/rwkv7-g1",
        "filename": "rwkv7-g1c-2.9b-20251231-ctx8192.pth",
        "size": "5.9 GB",
    },
    "g1c-7.2b": {
        "repo": "BlinkDL/rwkv7-g1",
        "filename": "rwkv7-g1c-7.2b-20251231-ctx8192.pth",
        "size": "14.4 GB",
    },
    # G1a series (older)
    "g1a-0.1b": {
        "repo": "BlinkDL/rwkv7-g1",
        "filename": "rwkv7-g1a-0.1b-20250728-ctx4096.pth",
        "size": "382 MB",
    },
    "g1a-0.4b": {
        "repo": "BlinkDL/rwkv7-g1",
        "filename": "rwkv7-g1a-0.4b-20250905-ctx4096.pth",
        "size": "902 MB",
    },
}

DEFAULT_MODEL = "g1d-7.2b"  # Best quality that fits on 4090 (24GB VRAM)
MODELS_DIR = Path(__file__).parent.parent / "data" / "models"


def download_model(model_key: str = DEFAULT_MODEL) -> Path:
    """Download a model from HuggingFace.

    Args:
        model_key: Key from MODELS dict

    Returns:
        Path to downloaded model
    """
    if model_key not in MODELS:
        print(f"Unknown model: {model_key}")
        print(f"Available: {list(MODELS.keys())}")
        raise ValueError(f"Unknown model: {model_key}")

    model_info = MODELS[model_key]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {model_key}...")
    print(f"  Repo: {model_info['repo']}")
    print(f"  File: {model_info['filename']}")

    model_path = hf_hub_download(
        repo_id=model_info["repo"],
        filename=model_info["filename"],
        local_dir=MODELS_DIR,
        local_dir_use_symlinks=False,
    )

    print(f"Downloaded to: {model_path}")
    return Path(model_path)


def download_t5() -> Path:
    """Download T5Gemma edit model to data/models/t5gemma/."""
    t5_path = MODELS_DIR / "t5gemma"

    print(f"\nDownloading T5Gemma edit model: {T5_MODEL}")
    print(f"Saving to: {t5_path}")

    from transformers import AutoProcessor, AutoModelForSeq2SeqLM
    import torch

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Download and save to local path
    processor = AutoProcessor.from_pretrained(T5_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        T5_MODEL,
        torch_dtype=torch.bfloat16,
    )

    # Save locally
    processor.save_pretrained(t5_path)
    model.save_pretrained(t5_path)

    print(f"T5Gemma saved to {t5_path}")
    del model, processor  # Free memory
    return t5_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download models for Minsky")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=list(MODELS.keys()),
        help=f"RWKV model to download (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available RWKV models"
    )
    parser.add_argument(
        "--skip-rwkv",
        action="store_true",
        help="Skip RWKV download"
    )
    parser.add_argument(
        "--skip-t5",
        action="store_true",
        help="Skip T5 download"
    )
    args = parser.parse_args()

    if args.list:
        print("Available RWKV7-G1 models:")
        print()
        for key, info in MODELS.items():
            print(f"  {key:12} {info['size']:>10}  {info['filename']}")
        print()
        print(f"Default: {DEFAULT_MODEL}")
        print(f"\nT5 model: {T5_MODEL} (~540 MB)")
        return

    # Download RWKV
    if not args.skip_rwkv:
        os.environ["RWKV_V7_ON"] = "1"
        os.environ["RWKV_JIT_ON"] = "1"
        os.environ["RWKV_CUDA_ON"] = "1"
        rwkv_path = download_model(args.model)
    else:
        print("Skipping RWKV download.")

    # Download T5
    if not args.skip_t5:
        download_t5()
    else:
        print("Skipping T5 download.")

    print("\n" + "=" * 60)
    print("Downloads complete! Run experiment with:")
    print("  uv run python main.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
