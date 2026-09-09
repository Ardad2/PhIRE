from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


Diagram = Sequence[Tuple[int, Tuple[float, float]]]


def _finite_pairs(diagram: Diagram, dim: Optional[int] = None) -> np.ndarray:
    pairs: List[Tuple[float, float]] = []
    for d, (birth, death) in diagram:
        if dim is not None and d != dim:
            continue
        if np.isfinite(death):
            pairs.append((float(birth), float(death)))
    if not pairs:
        return np.empty((0, 2), dtype=float)
    return np.asarray(pairs, dtype=float)


def summary_features(diagram: Diagram, dims: Optional[Iterable[int]] = None) -> np.ndarray:
    """Compact hand-crafted representation per homology dimension."""
    if dims is None:
        dims = sorted({d for d, _ in diagram})

    feats: List[float] = []
    for dim in dims:
        pairs = _finite_pairs(diagram, dim=dim)
        if len(pairs) == 0:
            feats.extend([0.0, 0.0, 0.0, 0.0])
            continue
        lifetimes = pairs[:, 1] - pairs[:, 0]
        feats.extend(
            [
                float(len(lifetimes)),
                float(np.sum(lifetimes)),
                float(np.max(lifetimes)),
                float(np.mean(lifetimes)),
            ]
        )
    return np.asarray(feats, dtype=float)


def betti_curve(
    diagram: Diagram,
    dim: int = 1,
    num_bins: int = 128,
    value_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Betti curve sampled on a regular filtration grid."""
    pairs = _finite_pairs(diagram, dim=dim)
    if value_range is None:
        if len(pairs) == 0:
            value_range = (0.0, 1.0)
        else:
            value_range = (float(np.min(pairs[:, 0])), float(np.max(pairs[:, 1])))
    t = np.linspace(value_range[0], value_range[1], num_bins)
    curve = np.zeros_like(t)
    for birth, death in pairs:
        curve += ((t >= birth) & (t < death)).astype(float)
    return t, curve


def persistence_image(
    diagram: Diagram,
    dim: int = 1,
    resolution: Tuple[int, int] = (32, 32),
    sigma: float = 0.05,
    value_range: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
) -> np.ndarray:
    """Simple persistence image over (birth, persistence)."""
    pairs = _finite_pairs(diagram, dim=dim)
    if len(pairs) == 0:
        return np.zeros(resolution, dtype=float)

    births = pairs[:, 0]
    pers = pairs[:, 1] - pairs[:, 0]
    if value_range is None:
        bx = (float(np.min(births)), float(np.max(births)))
        py = (0.0, float(np.max(pers)))
    else:
        bx, py = value_range

    xs = np.linspace(bx[0], bx[1], resolution[1])
    ys = np.linspace(py[0], py[1], resolution[0])
    X, Y = np.meshgrid(xs, ys)
    image = np.zeros_like(X)

    two_sigma2 = 2.0 * sigma * sigma
    for b, p in zip(births, pers):
        image += p * np.exp(-((X - b) ** 2 + (Y - p) ** 2) / two_sigma2)
    return image


def vectorize_diagram(
    diagram: Diagram,
    method: str = "summary",
    dim: int = 1,
    num_bins: int = 128,
    resolution: Tuple[int, int] = (32, 32),
    sigma: float = 0.05,
) -> np.ndarray:
    """Vectorize a persistence diagram for downstream ML tasks."""
    key = method.strip().lower()
    if key == "summary":
        return summary_features(diagram)
    if key == "betti_curve":
        _, curve = betti_curve(diagram, dim=dim, num_bins=num_bins)
        return curve
    if key == "persistence_image":
        return persistence_image(diagram, dim=dim, resolution=resolution, sigma=sigma).ravel()
    raise ValueError("Unsupported method. Use: 'summary', 'betti_curve', or 'persistence_image'.")
