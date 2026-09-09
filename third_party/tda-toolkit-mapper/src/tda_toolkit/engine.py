"Engine module for TDA Toolkit, providing a unified interface for computing persistent homology across different data types and backends."

from __future__ import annotations

from typing import Any, Optional

from .backends import get_backend
from .models import PersistenceResult, normalize_input


def analyze_persistence(
    data: Any,
    kind: Optional[str] = None,
    backend: str = "gudhi",
    max_dim: int = 2,
    min_persistence: Optional[float] = None,
) -> PersistenceResult:
    """Unified persistence entry point for point clouds, scalar fields, and graphs.

    Parameters
    ----------
    data
        Input data object. Supported forms include arrays, networkx graphs (optional),
        edge arrays, and dictionaries with graph metadata.
    kind
        One of {"point_cloud", "scalar_field", "graph"}. If omitted, a best-effort
        inference is used.
    backend
        TDA backend identifier. Currently: "gudhi".
    max_dim
        Maximum homology dimension (used for point cloud and graph filtrations).
    min_persistence
        Optional persistence threshold passed to cubical persistence.
    """
    normalized = normalize_input(data, kind=kind)
    selected_backend = get_backend(backend)
    result = selected_backend.compute_persistence(
        normalized,
        max_dim=max_dim,
        min_persistence=min_persistence,
    )
    return PersistenceResult(
        kind=normalized.kind,
        backend=selected_backend.name,
        complex_obj=result["complex_obj"],
        diagram=result["diagram"],
        input_data=normalized,
        metadata=result.get("metadata", {}),
    )


def analyze(
    data: Any,
    kind: Optional[str] = None,
    backend: str = "gudhi",
    max_dim: int = 2,
    min_persistence: Optional[float] = None,
) -> PersistenceResult:
    """User-friendly alias for analyze_persistence."""
    return analyze_persistence(
        data=data,
        kind=kind,
        backend=backend,
        max_dim=max_dim,
        min_persistence=min_persistence,
    )
