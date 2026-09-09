from __future__ import annotations

import numpy as np
from typing import Any
try:
    import networkx as nx
except Exception:
    nx = None  # type: ignore


def load_point_cloud(file_path: str) -> np.ndarray:
    """Load point cloud data from NPY, CSV, or TXT."""
    if file_path.endswith(".npy"):
        return np.load(file_path)
    delimiter = "," if file_path.endswith(".csv") else None
    return np.loadtxt(file_path, delimiter=delimiter)


def load_scalar_field(file_path: str) -> np.ndarray:
    """Load scalar field data from NPY or CSV."""
    if file_path.endswith('.npy'):
        return np.load(file_path)
    else:
        return np.loadtxt(file_path, delimiter=',')


def load_graph(file_path: str) -> Any:
    """Load graph data (assumes edge list in CSV or TXT)."""
    if nx is None:
        raise ImportError("networkx is required for graph loading (pip install networkx)")
    return nx.read_edgelist(file_path, delimiter=',')
