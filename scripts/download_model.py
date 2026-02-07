#!/usr/bin/env python3
"""Download RWKV model from HuggingFace.

Run with: uv run python scripts/download_model.py
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download

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
MODELS_DIR = Path(__file__).parent.parent / "models"


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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download RWKV model")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=list(MODELS.keys()),
        help=f"Model to download (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models"
    )
    args = parser.parse_args()

    if args.list:
        print("Available RWKV7-G1 models:")
        print()
        for key, info in MODELS.items():
            print(f"  {key:12} {info['size']:>10}  {info['filename']}")
        print()
        print(f"Default: {DEFAULT_MODEL}")
        return

    # Set environment for RWKV
    os.environ["RWKV_V7_ON"] = "1"
    os.environ["RWKV_JIT_ON"] = "1"
    os.environ["RWKV_CUDA_ON"] = "1"

    path = download_model(args.model)

    print("\nTo use this model, run:")
    print(f"  uv run python main.py --use-rwkv --rwkv-path {path}")


if __name__ == "__main__":
    main()
