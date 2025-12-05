"""
SCI: Surgical Cognitive Interpreter
Metacognitive control for signal dynamics.

This package is structured to keep the top-level import lightweight:
- `import sci` does NOT import torch or heavy submodules immediately.
- Actual components are imported lazily when accessed.

Author: Vishal Joshua Meesala
"""

from importlib import import_module
from typing import Any

__all__ = [
    "SCIController",
    "compute_sp",
    "Interpreter",
    "Decomposition",
    "ReliabilityWeighting",
]


def __getattr__(name: str) -> Any:
    """
    Lazy attribute access so that:

        import sci
        sci.SCIController

    does not import torch until the attribute is actually used.
    """
    if name == "SCIController":
        return import_module("sci.controller").SCIController
    if name == "compute_sp":
        return import_module("sci.sp").compute_sp
    if name == "Interpreter":
        return import_module("sci.interpreter").Interpreter
    if name == "Decomposition":
        return import_module("sci.decomposition").Decomposition
    if name == "ReliabilityWeighting":
        return import_module("sci.reliability").ReliabilityWeighting

    raise AttributeError(f"module 'sci' has no attribute {name!r}")

