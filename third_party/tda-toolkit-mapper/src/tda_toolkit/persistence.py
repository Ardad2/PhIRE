from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

from .engine import analyze_persistence
from .models import PersistenceResult

try:
    import gudhi as gd
except Exception:
    gd = None  # type: ignore

try:
    from gudhi.wasserstein import wasserstein_distance  # optional
except Exception:
    wasserstein_distance = None  # type: ignore


def _require_gudhi() -> None:
    if gd is None:
        raise ImportError("This feature requires Gudhi. Install with: pip install tda-toolkit[persistence]")


def compute_rips_persistence(points: np.ndarray, max_dim: int = 2):
    """Compute Rips complex and persistence."""
    _require_gudhi()
    rips_complex = gd.RipsComplex(points=points)
    st = rips_complex.create_simplex_tree(max_dimension=max_dim)
    st.compute_persistence()
    return st


def compute_cubical_persistence(scalar_field: np.ndarray):
    """Compute Cubical complex and persistence from a 2D scalar field."""
    _require_gudhi()
    # Remove singleton third dimension if present
    if scalar_field.ndim == 3 and scalar_field.shape[2] == 1:
        scalar_field = np.squeeze(scalar_field, axis=2)

    assert scalar_field.ndim == 2, "Input must be a 2D array for cubical complex"

    cubical_complex = gd.CubicalComplex(top_dimensional_cells=scalar_field)
    cubical_complex.compute_persistence()
    return cubical_complex


def pd_common_limits(st_list, dimension: int = 0, pad: float = 1.05):
    """Find common [0, eps_max] for birth/death across several persistence outputs."""
    _require_gudhi()
    max_death = 0.0
    for st in st_list:
        for dim, (b, d) in st.persistence():
            if dim == dimension and np.isfinite(d):
                if d > max_death:
                    max_death = d
    eps_max = (max_death if max_death > 0 else 1.0) * pad
    return (0.0, eps_max), (0.0, eps_max)


def plot_persistence_diagram(simplex_tree, dimension: Optional[int] = None, ax=None, title: str = "", xlim=None, ylim=None):
    _require_gudhi()
    diag = simplex_tree.persistence()
    if dimension is not None:
        diag = [pt for pt in diag if pt[0] == dimension]
    ax = ax or plt.gca()
    gd.plot_persistence_diagram(diag, axes=ax)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title)
    plt.show()


def plot_barcode(simplex_tree, ax=None) -> None:
    _require_gudhi()
    diag = simplex_tree.persistence()
    ax = ax or plt.gca()
    gd.plot_persistence_barcode(diag, axes=ax)
    plt.show()


def get_persistence_pairs(simplex_tree) -> List[Tuple[float, float]]:
    _require_gudhi()
    return [(b, d) for (dim, (b, d)) in simplex_tree.persistence()]


def bottleneck_distance(p1, p2) -> float:
    """Compute bottleneck distance between two (0-dim) persistence diagrams provided as 1D arrays of values.

    If you have full (birth, death) pairs, use `compute_bottleneck_distance`.
    """
    _require_gudhi()
    ppd1 = [(0, float(val)) for val in np.asarray(p1).ravel()]
    ppd2 = [(0, float(val)) for val in np.asarray(p2).ravel()]
    return gd.bottleneck_distance(ppd1, ppd2)

def compute_bottleneck_distance(diag1, diag2) -> float:
    """General bottleneck distance for arbitrary persistence diagrams (any homology dim)."""
    _require_gudhi()
    return float(gd.bottleneck_distance(diag1, diag2))

def compute_wasserstein_distance(diag1, diag2, order: int = 1) -> float:
    if wasserstein_distance is None:
        raise ImportError(
            "gudhi.wasserstein is unavailable in your Gudhi build. "
            "Install a build with wasserstein support or use another metric."
        )
    return float(wasserstein_distance(diag1, diag2, order=order))


def map_critical_points(simplex_tree, points: np.ndarray, dim: int = 1):
    """Return the cycles (birth/death generators) mapped back to original points (best-effort)."""
    _require_gudhi()
    reps = simplex_tree.persistence_generators_in_dimension(dim)
    cycles = []
    for gen in reps:
        cycle_pts = np.unique([points[v] for simplex in gen[1] for v in simplex], axis=0)
        cycles.append(cycle_pts)
    return cycles


def compute_graph_persistence(graph, max_dim: int = 2):
    """Compute persistence on a graph using the configured graph filtration."""
    result = analyze_persistence(graph, kind="graph", backend="gudhi", max_dim=max_dim)
    return result.complex_obj


def compute_persistence(
    data,
    kind: Optional[str] = None,
    backend: str = "gudhi",
    max_dim: int = 2,
    min_persistence: Optional[float] = None,
) -> PersistenceResult:
    """Unified persistence API for point clouds, scalar fields, and graphs."""
    return analyze_persistence(
        data=data,
        kind=kind,
        backend=backend,
        max_dim=max_dim,
        min_persistence=min_persistence,
    )
