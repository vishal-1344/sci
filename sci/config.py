"""Placeholder config module for SCI.

This file can be extended to expose default configuration
objects or helper loaders for YAML config files in `configs/`.
"""

from pathlib import Path

DEFAULTS = {
    "feature_dim": 128,
    "num_markers": 8,
    "num_classes": 10,
}

def load_yaml(path: str):
    try:
        import yaml
    except Exception:
        raise RuntimeError("PyYAML is required to load config files")
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
