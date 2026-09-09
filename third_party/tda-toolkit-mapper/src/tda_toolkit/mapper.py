from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler

try:
    import kmapper as km
except Exception:
    km = None  # type: ignore

try:
    import networkx as nx
except Exception:
    nx = None  # type: ignore


@dataclass
class MapperResult:
    graph: Dict[str, Any]
    mapper: Any
    lens: np.ndarray
    clustering_data: np.ndarray
    points: np.ndarray
    lens_name: str
    projection: str
    clusterer_name: str


def _require_mapper() -> None:
    if km is None:
        raise ImportError("Mapper features require KeplerMapper. Install with: pip install tda-toolkit[mapper]")


def _require_networkx() -> None:
    if nx is None:
        raise ImportError("Static graph plotting requires networkx. Install with: pip install networkx")


def _as_2d_points(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D point cloud array, got shape {array.shape}.")
    return array


def preprocess_points(points: np.ndarray, scale: Optional[str] = None) -> np.ndarray:
    X = _as_2d_points(points)
    if scale in (None, "none"):
        return X
    if scale == "standard":
        return StandardScaler().fit_transform(X)
    if scale == "minmax":
        return MinMaxScaler().fit_transform(X)
    raise ValueError(f"Unsupported scale='{scale}'. Expected one of: none, standard, minmax.")


def project_points(points: np.ndarray, projection: Optional[str] = None, n_components: int = 2) -> np.ndarray:
    X = _as_2d_points(points)
    if projection in (None, "none"):
        return X
    if projection == "pca":
        n_components = max(1, min(int(n_components), X.shape[0], X.shape[1]))
        return PCA(n_components=n_components).fit_transform(X)
    raise ValueError(f"Unsupported projection='{projection}'. Expected one of: none, pca.")


def compute_mapper_lens(
    points: np.ndarray,
    lens: str = "coordinate",
    lens_dim: int = 0,
    n_components: int = 1,
    n_neighbors: int = 15,
) -> np.ndarray:
    X = _as_2d_points(points)
    lens_name = lens.lower()

    if lens_name == "coordinate":
        if lens_dim < 0 or lens_dim >= X.shape[1]:
            raise IndexError(f"lens_dim={lens_dim} out of bounds for point cloud with {X.shape[1]} dimensions.")
        values = X[:, [lens_dim]]
    elif lens_name == "pca":
        n_components = max(1, min(int(n_components), X.shape[0], X.shape[1]))
        values = PCA(n_components=n_components).fit_transform(X)
    elif lens_name == "eccentricity":
        centroid = np.mean(X, axis=0, keepdims=True)
        values = np.linalg.norm(X - centroid, axis=1, keepdims=True)
    elif lens_name == "norm":
        values = np.linalg.norm(X, axis=1, keepdims=True)
    elif lens_name == "knn_distance":
        neighbor_count = max(2, min(int(n_neighbors), X.shape[0]))
        nn = NearestNeighbors(n_neighbors=neighbor_count)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        values = distances[:, -1:]
    elif lens_name == "density":
        neighbor_count = max(2, min(int(n_neighbors), X.shape[0]))
        nn = NearestNeighbors(n_neighbors=neighbor_count)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        mean_distance = np.maximum(np.mean(distances[:, 1:], axis=1, keepdims=True), 1e-12)
        values = 1.0 / mean_distance
    else:
        raise ValueError(
            "Unsupported lens='{}'. Expected one of: coordinate, pca, eccentricity, norm, "
            "knn_distance, density.".format(lens)
        )

    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return np.asarray(values, dtype=float)


def build_mapper_clusterer(
    clusterer: str = "dbscan",
    *,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 3,
    n_clusters: int = 8,
) -> Any:
    clusterer_name = clusterer.lower()
    if clusterer_name == "dbscan":
        return DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    if clusterer_name == "kmeans":
        return KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    if clusterer_name == "agglomerative":
        return AgglomerativeClustering(n_clusters=n_clusters)
    raise ValueError(f"Unsupported clusterer='{clusterer}'. Expected one of: dbscan, kmeans, agglomerative.")


def summarize_mapper_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    nodes = graph.get("nodes", {})
    links = graph.get("links", {})
    member_sizes = [len(members) for members in nodes.values()]
    return {
        "num_nodes": len(nodes),
        "num_edges": sum(len(neighbors) for neighbors in links.values()) // 2,
        "num_points_covered": len({idx for members in nodes.values() for idx in members}),
        "largest_node_size": max(member_sizes, default=0),
        "avg_node_size": float(np.mean(member_sizes)) if member_sizes else 0.0,
    }


def mapper_graph_has_nodes(graph: Dict[str, Any]) -> bool:
    return len(graph.get("nodes", {})) > 0


def mapper_graph_to_networkx(graph: Dict[str, Any]) -> Any:
    _require_networkx()
    G = nx.Graph()
    for node_id, members in graph.get("nodes", {}).items():
        G.add_node(node_id, size=len(members), members=list(members))
    for node_id, neighbors in graph.get("links", {}).items():
        for neighbor in neighbors:
            G.add_edge(node_id, neighbor)
    return G


def save_mapper_graph_json(result: MapperResult, path: str) -> str:
    payload = {
        "mapper": {
            "nodes": result.graph.get("nodes", {}),
            "links": result.graph.get("links", {}),
        },
        "summary": summarize_mapper_graph(result.graph),
        "meta": {
            "lens_name": result.lens_name,
            "projection": result.projection,
            "clusterer": result.clusterer_name,
            "num_points": int(result.points.shape[0]),
            "ambient_dimension": int(result.points.shape[1]),
            "lens_dimension": int(result.lens.shape[1]),
        },
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def plot_mapper_graph(
    result: MapperResult,
    path_png: Optional[str] = None,
    *,
    layout: str = "spring",
    color_by: str = "lens_mean",
    ax=None,
):
    _require_networkx()
    if not mapper_graph_has_nodes(result.graph):
        raise ValueError(
            "Mapper graph has 0 nodes. Try increasing DBSCAN eps, lowering min_samples, "
            "increasing overlap, or switching to kmeans/agglomerative."
        )
    G = mapper_graph_to_networkx(result.graph)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 7))

    if layout == "spring":
        pos = nx.spring_layout(G, seed=7)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        raise ValueError("Unsupported layout='{}'. Expected one of: spring, kamada_kawai.".format(layout))

    node_names = list(G.nodes())
    node_sizes = np.array([max(30, 25 * G.nodes[name]["size"]) for name in node_names], dtype=float)
    lens_map = {node_id: float(np.mean(result.lens[members])) for node_id, members in result.graph.get("nodes", {}).items()}
    if color_by == "lens_mean":
        node_colors = np.array([lens_map[name] for name in node_names], dtype=float)
    elif color_by == "size":
        node_colors = np.array([G.nodes[name]["size"] for name in node_names], dtype=float)
    else:
        raise ValueError("Unsupported color_by='{}'. Expected one of: lens_mean, size.".format(color_by))

    nx.draw_networkx_edges(G, pos=pos, ax=ax, width=0.8, alpha=0.35, edge_color="#5f6f7f")
    nodes = nx.draw_networkx_nodes(
        G,
        pos=pos,
        ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        cmap="viridis",
        linewidths=0.8,
        edgecolors="white",
    )
    ax.set_title(
        f"Mapper Graph ({result.lens_name}, {result.clusterer_name})\n"
        f"{result.points.shape[0]} points in R^{result.points.shape[1]}"
    )
    ax.set_axis_off()
    plt.colorbar(nodes, ax=ax, fraction=0.046, pad=0.04, label=color_by)
    if path_png:
        ax.figure.savefig(path_png, dpi=220, bbox_inches="tight")
        plt.close(ax.figure)
    return ax


def run_mapper_pipeline(
    points: np.ndarray,
    n_cubes: int = 10,
    overlap: float = 0.1,
    filter_func=None,
    *,
    lens: str = "coordinate",
    lens_dim: int = 0,
    lens_components: int = 1,
    lens_neighbors: int = 15,
    scale: Optional[str] = None,
    projection: Optional[str] = None,
    projection_components: int = 2,
    clusterer: str = "dbscan",
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 3,
    n_clusters: int = 8,
) -> MapperResult:
    _require_mapper()
    X = _as_2d_points(points)
    scaled_points = preprocess_points(X, scale=scale)
    if filter_func is not None:
        lens_values = np.asarray(filter_func(scaled_points), dtype=float)
        if lens_values.ndim == 1:
            lens_values = lens_values.reshape(-1, 1)
        lens_name = getattr(filter_func, "__name__", "custom")
    else:
        lens_values = compute_mapper_lens(
            scaled_points,
            lens=lens,
            lens_dim=lens_dim,
            n_components=lens_components,
            n_neighbors=lens_neighbors,
        )
        lens_name = lens

    clustering_data = project_points(
        scaled_points,
        projection=projection,
        n_components=projection_components,
    )
    mapper = km.KeplerMapper(verbose=0)
    cover = km.Cover(n_cubes=n_cubes, perc_overlap=overlap)
    clustering_model = build_mapper_clusterer(
        clusterer,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        n_clusters=n_clusters,
    )
    graph = mapper.map(lens_values, clustering_data, clusterer=clustering_model, cover=cover)
    return MapperResult(
        graph=graph,
        mapper=mapper,
        lens=lens_values,
        clustering_data=clustering_data,
        points=X,
        lens_name=lens_name,
        projection=projection or "none",
        clusterer_name=clusterer,
    )


def run_mapper(points: np.ndarray, n_cubes: int = 10, overlap: float = 0.1, filter_func=None, **kwargs):
    result = run_mapper_pipeline(
        points,
        n_cubes=n_cubes,
        overlap=overlap,
        filter_func=filter_func,
        **kwargs,
    )
    return result.graph, result.mapper


def visualize_mapper_graph(mapper, graph, path_html: str = "mapper_graph.html") -> str:
    _require_mapper()
    mapper.visualize(graph, path_html=path_html)
    return path_html
