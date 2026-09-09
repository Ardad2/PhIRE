"Backends for computing persistent homology in TDA Toolkit."
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .models import GraphData, PointCloudData, ScalarFieldData

try:
    import gudhi as gd
except Exception:  # pragma: no cover - optional dependency
    gd = None  # type: ignore


class BaseTDABackend:
    name = "base"

    def compute_persistence(
        self,
        data: Any,
        max_dim: int = 2,
        min_persistence: Optional[float] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass
class GudhiBackend(BaseTDABackend):
    name: str = "gudhi"

    def _require_gudhi(self) -> None:
        if gd is None:
            raise ImportError("Gudhi is required. Install with: pip install tda-toolkit[persistence]")

    def compute_persistence(
        self,
        data: Any,
        max_dim: int = 2,
        min_persistence: Optional[float] = None,
    ) -> Dict[str, Any]:
        self._require_gudhi()

        if isinstance(data, PointCloudData):
            complex_obj = gd.RipsComplex(points=data.points).create_simplex_tree(max_dimension=max_dim)
            complex_obj.compute_persistence()
            return {
                "complex_obj": complex_obj,
                "diagram": complex_obj.persistence(),
                "metadata": {"max_dim": max_dim},
            }

        if isinstance(data, ScalarFieldData):
            if data.field.ndim == 1:
                dims = (len(data.field),)
                cells = data.field
            elif data.field.ndim == 2:
                dims = data.field.shape
                cells = data.field.flatten()
            else:
                # 3D cubical support is naturally handled by Gudhi with flattened cells.
                dims = data.field.shape
                cells = data.field.flatten()

            complex_obj = gd.CubicalComplex(dimensions=dims, top_dimensional_cells=np.asarray(cells))
            if min_persistence is None:
                complex_obj.compute_persistence()
            else:
                complex_obj.compute_persistence(min_persistence=min_persistence)
            return {
                "complex_obj": complex_obj,
                "diagram": complex_obj.persistence(),
                "metadata": {"dimensions": tuple(int(d) for d in dims)},
            }

        if isinstance(data, GraphData):
            st = gd.SimplexTree()
            for i in range(data.num_nodes):
                st.insert([int(i)], filtration=0.0)
            if data.edge_weights is None:
                weights = np.ones(len(data.edges), dtype=float)
            else:
                weights = np.asarray(data.edge_weights, dtype=float)
            for (u, v), w in zip(data.edges, weights):
                st.insert([int(u), int(v)], filtration=float(w))
            st.expansion(max_dim)
            st.compute_persistence()
            return {
                "complex_obj": st,
                "diagram": st.persistence(),
                "metadata": {
                    "max_dim": max_dim,
                    "num_nodes": data.num_nodes,
                    "node_labels": data.node_labels,
                },
            }

        raise ValueError(f"Unsupported normalized input type: {type(data).__name__}")


def get_backend(backend: str = "gudhi") -> BaseTDABackend:
    key = backend.strip().lower()
    if key == "gudhi":
        return GudhiBackend()
    raise ValueError(f"Unsupported backend='{backend}'. Available: ['gudhi'].")
