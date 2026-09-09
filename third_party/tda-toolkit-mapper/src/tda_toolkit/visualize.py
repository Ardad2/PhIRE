from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
try:
    import networkx as nx
except Exception:
    nx = None  # type: ignore

from .representations import betti_curve, persistence_image


def visualize_point_cloud(points: np.ndarray, ax=None) -> None:
    if points.ndim != 2:
        raise ValueError("points must be 2D array of shape (n, d)")
    if points.shape[1] == 2:
        ax = ax or plt.gca()
        ax.scatter(points[:, 0], points[:, 1], alpha=0.7)
    elif points.shape[1] == 3:
        ax = ax or plt.figure().add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], alpha=0.7)
    else:
        raise ValueError("Can only visualize 2D/3D point clouds.")
    plt.show()


def visualize_scalar_field(scalar_field: np.ndarray, ax=None) -> None:
    ax = ax or plt.gca()
    im = ax.imshow(scalar_field, cmap='viridis')
    plt.colorbar(im, ax=ax)
    plt.show()


def visualize_graph(graph, ax=None) -> None:
    if nx is None:
        raise ImportError("networkx is required for graph visualization (pip install networkx)")
    pos = nx.spring_layout(graph)
    ax = ax or plt.gca()
    nx.draw(graph, pos, ax=ax, with_labels=True)
    plt.show()


def plot_betti_curve(diagram, dim: int = 1, num_bins: int = 128, ax=None) -> None:
    t, curve = betti_curve(diagram, dim=dim, num_bins=num_bins)
    ax = ax or plt.gca()
    ax.plot(t, curve, linewidth=2)
    ax.set_title(f"Betti Curve (H{dim})")
    ax.set_xlabel("Filtration Value")
    ax.set_ylabel("Betti Number")
    plt.show()


def plot_persistence_image(diagram, dim: int = 1, resolution=(32, 32), sigma: float = 0.05, ax=None) -> None:
    image = persistence_image(diagram, dim=dim, resolution=resolution, sigma=sigma)
    ax = ax or plt.gca()
    im = ax.imshow(image, origin="lower", cmap="magma", aspect="auto")
    ax.set_title(f"Persistence Image (H{dim})")
    ax.set_xlabel("Birth Axis (discretized)")
    ax.set_ylabel("Persistence Axis (discretized)")
    plt.colorbar(im, ax=ax)
    plt.show()
