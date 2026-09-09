"""tda_toolkit: modular Topological Data Analysis utilities."""

from __future__ import annotations

from importlib import import_module

from .engine import analyze, analyze_persistence

__all__ = [
    "io",
    "visualize",
    "persistence",
    "merge_tree",
    "mapper",
    "distances",
    "cluster",
    "profiles",
    "utils",
    "models",
    "backends",
    "engine",
    "representations",
    "generators",
    "app",
    "analyze",
    "analyze_persistence",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    if name in {
        "io",
        "visualize",
        "persistence",
        "merge_tree",
        "mapper",
        "distances",
        "cluster",
        "profiles",
        "utils",
        "models",
        "backends",
        "engine",
        "representations",
        "generators",
        "app",
    }:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
