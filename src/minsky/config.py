"""Configuration loading for the Minsky Society of Mind architecture.

All configuration lives in config.toml at the project root.
Model-specific config dataclasses live in their respective modules
(llm_client.py, edit_model.py).
"""

import sys
import tomllib
from pathlib import Path


def load_config(path: str | Path = "config.toml") -> dict:
    """Load and return the TOML configuration as a dict.

    Args:
        path: Path to the TOML config file.

    Returns:
        Parsed config dictionary.
    """
    path = Path(path)
    if not path.exists():
        print(f"ERROR: Config file not found: {path.resolve()}")
        sys.exit(1)
    with open(path, "rb") as f:
        return tomllib.load(f)
