from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def cluster_distance_matrix(D: np.ndarray, n_clusters: int = 2) -> np.ndarray:
    """Cluster a precomputed distance matrix with average-linkage agglomerative clustering."""
    try:
        model = AgglomerativeClustering(metric="precomputed", linkage="average", n_clusters=n_clusters)
    except TypeError:
        # Backward compatibility for older scikit-learn versions.
        model = AgglomerativeClustering(affinity="precomputed", linkage="average", n_clusters=n_clusters)
    return model.fit_predict(D)


def cluster_feature_matrix(
    X: np.ndarray,
    n_clusters: int = 2,
    method: str = "kmeans",
    random_state: Optional[int] = 0,
) -> np.ndarray:
    """Cluster vectorized topological representations."""
    key = method.strip().lower()
    if key == "kmeans":
        return _kmeans_numpy(X, n_clusters=n_clusters, random_state=random_state)
    if key == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        return model.fit_predict(X)
    raise ValueError("Unsupported method. Use 'kmeans' or 'agglomerative'.")


def _kmeans_numpy(
    X: np.ndarray,
    n_clusters: int,
    random_state: Optional[int] = 0,
    max_iter: int = 100,
) -> np.ndarray:
    """Small NumPy implementation to avoid backend-specific sklearn runtime issues."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("Feature matrix must have shape (n_samples, n_features).")
    n_samples = X.shape[0]
    if n_clusters < 1 or n_clusters > n_samples:
        raise ValueError("n_clusters must be in [1, n_samples].")

    rng = np.random.default_rng(random_state)
    centers = X[rng.choice(n_samples, size=n_clusters, replace=False)].copy()
    labels = np.zeros(n_samples, dtype=int)

    for _ in range(max_iter):
        dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for k in range(n_clusters):
            mask = labels == k
            if np.any(mask):
                centers[k] = X[mask].mean(axis=0)
            else:
                centers[k] = X[rng.integers(0, n_samples)]
    return labels


def cluster_and_plot(D: np.ndarray, n_clusters: int = 2):
    """Backward-compatible helper: cluster a distance matrix and visualize it."""
    labels = cluster_distance_matrix(D, n_clusters=n_clusters)
    plt.matshow(D)
    plt.title("Topological Distance Matrix")
    plt.colorbar()
    plt.show()
    return labels
