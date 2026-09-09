"Models for normalized input data and persistence results in TDA Toolkit."

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import networkx as nx
except Exception:  # pragma: no cover - optional dependency
    nx = None  # type: ignore


InputKind = str


@dataclass
class PointCloudData:
    points: np.ndarray
    kind: InputKind = "point_cloud"


@dataclass
class ScalarFieldData:
    field: np.ndarray
    kind: InputKind = "scalar_field"


@dataclass
class GraphData:
    edges: np.ndarray
    num_nodes: int
    edge_weights: Optional[np.ndarray] = None
    node_labels: Optional[List[Any]] = None
    kind: InputKind = "graph"


NormalizedInput = Union[PointCloudData, ScalarFieldData, GraphData]


@dataclass
class PersistenceResult:
    kind: InputKind
    backend: str
    complex_obj: Any
    diagram: List[Tuple[int, Tuple[float, float]]]
    input_data: Optional[NormalizedInput] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def diagram_in_dimension(self, dim: int) -> List[Tuple[float, float]]:
        return [pair for d, pair in self.diagram if d == dim]

    def summary(self) -> Dict[str, Any]:
        dims = sorted({d for d, _ in self.diagram})
        finite_pairs = sum(1 for _, (_, death) in self.diagram if np.isfinite(death))
        return {
            "kind": self.kind,
            "backend": self.backend,
            "dimensions": dims,
            "num_pairs": len(self.diagram),
            "num_finite_pairs": finite_pairs,
            "metadata": self.metadata,
        }

    def vectorize(self, method: str = "summary", **kwargs: Any) -> np.ndarray:
        from .representations import vectorize_diagram

        return vectorize_diagram(self.diagram, method=method, **kwargs)

    def plot_diagram(self, dimension: Optional[int] = None, title: str = "", **kwargs: Any) -> None:
        from .persistence import plot_persistence_diagram

        plot_persistence_diagram(self.complex_obj, dimension=dimension, title=title, **kwargs)

    def plot_betti_curve(self, dim: int = 1, num_bins: int = 128, **kwargs: Any) -> None:
        from .visualize import plot_betti_curve

        plot_betti_curve(self.diagram, dim=dim, num_bins=num_bins, **kwargs)

    def plot_persistence_image(
        self,
        dim: int = 1,
        resolution: Tuple[int, int] = (32, 32),
        sigma: float = 0.05,
        **kwargs: Any,
    ) -> None:
        from .visualize import plot_persistence_image

        plot_persistence_image(self.diagram, dim=dim, resolution=resolution, sigma=sigma, **kwargs)

    def generator_mappings(self, dim: int = 1, **kwargs: Any):
        if self.input_data is None:
            raise ValueError("Generator mappings are unavailable because original input data was not retained.")

        if self.kind == "point_cloud":
            from .generators import compute_point_cloud_generator_mappings

            _, mappings = compute_point_cloud_generator_mappings(
                self.input_data.points,
                dim=dim,
                max_dim=int(self.metadata.get("max_dim", max(dim + 1, 2))),
                backend=self.backend,
            )
            return mappings

        if self.kind == "graph":
            from .generators import compute_graph_generator_mappings

            graph_payload = {
                "edges": self.input_data.edges,
                "num_nodes": self.input_data.num_nodes,
                "edge_weights": self.input_data.edge_weights,
                "node_labels": self.input_data.node_labels,
            }
            _, mappings = compute_graph_generator_mappings(
                graph_payload,
                dim=dim,
                max_dim=int(self.metadata.get("max_dim", max(dim + 1, 2))),
                backend=self.backend,
            )
            return mappings

        if self.kind == "scalar_field":
            from .generators import compute_scalar_field_generator_mappings

            _, mappings = compute_scalar_field_generator_mappings(
                self.input_data.field,
                dim=dim,
                direction=int(kwargs.get("direction", 1)),
            )
            return mappings

        raise ValueError(f"Unsupported kind for generator mappings: {self.kind}")


def _as_float_array(data: Any) -> np.ndarray:
    arr = np.asarray(data)
    if arr.size == 0:
        raise ValueError("Input data is empty.")
    return arr.astype(float, copy=False)


def _normalize_graph_input(data: Any) -> GraphData:
    if nx is not None and isinstance(data, nx.Graph):
        nodes = list(data.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        edges: List[Tuple[int, int]] = []
        weights: List[float] = []
        has_custom_weights = False
        for u, v, attrs in data.edges(data=True):
            edges.append((node_to_idx[u], node_to_idx[v]))
            if "weight" in attrs:
                has_custom_weights = True
            weights.append(float(attrs.get("weight", 1.0)))
        edge_arr = np.asarray(edges, dtype=int)
        weight_arr = np.asarray(weights, dtype=float) if has_custom_weights else None
        return GraphData(
            edges=edge_arr,
            num_nodes=len(nodes),
            edge_weights=weight_arr,
            node_labels=nodes,
        )

    if isinstance(data, dict) and "edges" in data:
        edges = np.asarray(data["edges"], dtype=int)
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("Graph edges must have shape (m, 2).")
        if "num_nodes" in data:
            num_nodes = int(data["num_nodes"])
        else:
            num_nodes = int(edges.max()) + 1 if edges.size else 0
        weights = data.get("edge_weights")
        weight_arr = None if weights is None else np.asarray(weights, dtype=float).reshape(-1)
        if weight_arr is not None and len(weight_arr) != len(edges):
            raise ValueError("edge_weights must have one entry per edge.")
        return GraphData(
            edges=edges,
            num_nodes=num_nodes,
            edge_weights=weight_arr,
            node_labels=data.get("node_labels"),
        )

    arr = np.asarray(data)
    if arr.ndim == 2 and arr.shape[1] == 2:
        edges = arr.astype(int, copy=False)
        num_nodes = int(edges.max()) + 1 if edges.size else 0
        return GraphData(edges=edges, num_nodes=num_nodes)

    raise ValueError(
        "Unsupported graph input. Use a networkx graph, an edge array with shape (m, 2), "
        "or a dict with keys: edges, num_nodes (optional), edge_weights (optional)."
    )


def normalize_input(data: Any, kind: Optional[InputKind] = None) -> NormalizedInput:
    if kind is None:
        if nx is not None and isinstance(data, nx.Graph):
            kind = "graph"
        elif isinstance(data, dict) and "edges" in data:
            kind = "graph"
        elif isinstance(data, (tuple, list)) and len(data) == 2:
            arr0 = np.asarray(data[0])
            arr1 = np.asarray(data[1])
            if arr0.ndim == 2 and arr0.shape[1] == 2 and arr1.ndim == 1:
                kind = "graph"
        else:
            arr = np.asarray(data)
            if arr.ndim == 1:
                kind = "scalar_field"
            elif arr.ndim == 2:
                # Heuristic: square-ish arrays are often scalar fields/images.
                kind = "scalar_field" if arr.shape[0] == arr.shape[1] else "point_cloud"
            elif arr.ndim == 3:
                kind = "scalar_field"
            else:
                raise ValueError(
                    "Could not infer input kind. Please pass kind='point_cloud', "
                    "'scalar_field', or 'graph'."
                )

    if kind == "point_cloud":
        points = _as_float_array(data)
        if points.ndim != 2:
            raise ValueError("Point cloud input must have shape (n_samples, n_features).")
        return PointCloudData(points=points)

    if kind == "scalar_field":
        field = _as_float_array(data)
        if field.ndim not in (1, 2, 3):
            raise ValueError("Scalar field input must be 1D, 2D, or 3D.")
        if field.ndim == 3 and field.shape[-1] == 1:
            field = np.squeeze(field, axis=-1)
        return ScalarFieldData(field=field)

    if kind == "graph":
        return _normalize_graph_input(data)

    raise ValueError(f"Unsupported kind='{kind}'.")
