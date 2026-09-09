from __future__ import annotations

from typing import List, Tuple, Optional, Any, Dict, Literal
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

try:
    import gudhi as gd
except Exception:
    gd = None

try:
    import networkx as nx
except Exception:
    nx = None

def _require_gudhi_and_nx() -> None:
    if gd is None or nx is None:
        raise ImportError(
            "This feature requires Gudhi and NetworkX. Install with: "
            "pip install gudhi networkx"
        )


def _require_nx() -> None:
    if nx is None:
        raise ImportError("This feature requires NetworkX. Install with: pip install networkx")


def _compute_knn_neighbors(
    points: np.ndarray,
    n_neighbors: int = 8,
    *,
    algorithm: str = "auto",
    metric: str = "euclidean",
    n_jobs: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(points, dtype=float)
    if X.ndim != 2:
        raise ValueError("kNN construction expects a 2D point cloud array.")
    k = max(2, min(int(n_neighbors), X.shape[0]))
    nn = NearestNeighbors(
        n_neighbors=k,
        algorithm=algorithm,
        metric=metric,
        n_jobs=n_jobs,
    )
    nn.fit(X)
    return nn.kneighbors(X)


def _compute_point_cloud_scalar_function(
    points: np.ndarray,
    function: str = "eccentricity",
    function_dim: int = 0,
    n_neighbors: int = 8,
    knn_distances: Optional[np.ndarray] = None,
    knn_algorithm: str = "auto",
    metric: str = "euclidean",
    n_jobs: Optional[int] = None,
) -> np.ndarray:
    X = np.asarray(points, dtype=float)
    if X.ndim != 2:
        raise ValueError("Point-cloud merge trees require a 2D array of shape (n_samples, n_features).")

    if function == "coordinate":
        if function_dim < 0 or function_dim >= X.shape[1]:
            raise IndexError(f"function_dim={function_dim} out of range for point cloud with {X.shape[1]} dimensions.")
        values = X[:, function_dim]
    elif function == "pca":
        basis = PCA(n_components=max(1, min(function_dim + 1, X.shape[0], X.shape[1]))).fit_transform(X)
        values = basis[:, function_dim]
    elif function == "eccentricity":
        centroid = np.mean(X, axis=0, keepdims=True)
        values = np.linalg.norm(X - centroid, axis=1)
    elif function == "norm":
        values = np.linalg.norm(X, axis=1)
    elif function == "knn_distance":
        if knn_distances is None:
            knn_distances, _ = _compute_knn_neighbors(
                X,
                n_neighbors=n_neighbors,
                algorithm=knn_algorithm,
                metric=metric,
                n_jobs=n_jobs,
            )
        values = knn_distances[:, -1]
    elif function == "density":
        if knn_distances is None:
            knn_distances, _ = _compute_knn_neighbors(
                X,
                n_neighbors=n_neighbors,
                algorithm=knn_algorithm,
                metric=metric,
                n_jobs=n_jobs,
            )
        mean_distance = np.maximum(np.mean(knn_distances[:, 1:], axis=1), 1e-12)
        values = 1.0 / mean_distance
    else:
        raise ValueError(
            "Unsupported point-cloud scalar function='{}'. Expected one of: "
            "coordinate, pca, eccentricity, norm, knn_distance, density.".format(function)
        )
    return np.asarray(values, dtype=float)


def _project_point_cloud_to_2d(points: np.ndarray) -> np.ndarray:
    X = np.asarray(points, dtype=float)
    if X.shape[1] == 1:
        return np.column_stack([np.arange(X.shape[0], dtype=float), X[:, 0]])

    centered = X - np.mean(X, axis=0, keepdims=True)
    if not np.any(np.abs(centered) > 1e-12):
        return np.column_stack([np.arange(X.shape[0], dtype=float), np.zeros(X.shape[0])])

    n_components = min(2, X.shape[0], X.shape[1])
    coords = PCA(n_components=n_components).fit_transform(X)
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(X.shape[0])])
    if not np.all(np.isfinite(coords)):
        return np.column_stack([np.arange(X.shape[0], dtype=float), np.zeros(X.shape[0])])
    return np.asarray(coords, dtype=float)


def _build_knn_graph(
    points: np.ndarray,
    n_neighbors: int = 8,
    mutual: bool = False,
    *,
    indices: Optional[np.ndarray] = None,
    algorithm: str = "auto",
    metric: str = "euclidean",
    n_jobs: Optional[int] = None,
) -> nx.Graph:
    _require_nx()
    X = np.asarray(points, dtype=float)
    if X.ndim != 2:
        raise ValueError("kNN graph construction expects a 2D point cloud array.")
    if indices is None:
        _, indices = _compute_knn_neighbors(
            X,
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            metric=metric,
            n_jobs=n_jobs,
        )

    G = nx.Graph()
    G.add_nodes_from(range(X.shape[0]))
    neighbor_sets: Dict[int, set[int]] = {}
    for i in range(X.shape[0]):
        neigh = {int(j) for j in indices[i, 1:] if int(j) != i}
        neighbor_sets[i] = neigh

    for i, neigh in neighbor_sets.items():
        for j in neigh:
            if mutual and i not in neighbor_sets.get(j, set()):
                continue
            distance = float(np.linalg.norm(X[i] - X[j]))
            if G.has_edge(i, j):
                continue
            G.add_edge(i, j, weight=distance)
    return G


def _build_knn_edges(
    points: np.ndarray,
    n_neighbors: int = 8,
    mutual: bool = False,
    *,
    indices: Optional[np.ndarray] = None,
    algorithm: str = "auto",
    metric: str = "euclidean",
    n_jobs: Optional[int] = None,
) -> List[Tuple[int, int]]:
    X = np.asarray(points, dtype=float)
    if X.ndim != 2:
        raise ValueError("kNN edge construction expects a 2D point cloud array.")
    if indices is None:
        _, indices = _compute_knn_neighbors(
            X,
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            metric=metric,
            n_jobs=n_jobs,
        )

    neighbor_sets: Dict[int, set[int]] = {}
    for i in range(X.shape[0]):
        neighbor_sets[i] = {int(j) for j in indices[i, 1:] if int(j) != i}

    edges = set()
    for i, neighbors in neighbor_sets.items():
        for j in neighbors:
            if mutual and i not in neighbor_sets.get(j, set()):
                continue
            edges.add(tuple(sorted((int(i), int(j)))))
    return sorted(edges)


def _build_join_tree_graph(values: np.ndarray, adjacency: Dict[int, List[int]]) -> nx.DiGraph:
    _require_nx()
    f = np.asarray(values, dtype=float).ravel()
    n = len(f)
    uf = _UF(n)
    active = np.zeros(n, dtype=bool)
    T = nx.DiGraph()
    comp_rep: Dict[int, int] = {}

    def ensure(i: int, kind: str):
        if not T.has_node(i):
            T.add_node(i, idx=i, value=float(f[i]), type=kind)
        return i

    order = sorted(range(n), key=lambda i: (f[i], i))
    for i in order:
        active[i] = True
        uf.p[i] = i

        roots: List[int] = []
        for j in adjacency.get(i, []):
            if not active[j] or f[j] > f[i]:
                continue
            rj = uf.find(j)
            if rj not in roots:
                roots.append(rj)

        if len(roots) == 0:
            comp_rep[uf.find(i)] = ensure(i, "min")
            continue

        if len(roots) == 1:
            uf.union(i, roots[0])
            comp_rep[uf.find(i)] = comp_rep.get(roots[0], comp_rep.get(uf.find(i), None))
            continue

        saddle = ensure(i, "max")
        for r in roots:
            if r in comp_rep and not T.has_edge(saddle, comp_rep[r]):
                T.add_edge(saddle, comp_rep[r])
        new_root = i
        for r in roots:
            uf.union(new_root, r)
        comp_rep[uf.find(new_root)] = saddle

    return T


def _build_gudhi_lower_star_tree(values: np.ndarray, edges: List[Tuple[int, int]]) -> nx.DiGraph:
    _require_gudhi_and_nx()
    f = np.asarray(values, dtype=float).ravel()
    st = gd.SimplexTree()
    for i, value in enumerate(f):
        st.insert([int(i)], filtration=float(value))
    for u, v in edges:
        st.insert([int(u), int(v)], filtration=float(max(f[u], f[v])))
    st.compute_persistence(min_persistence=-1)

    adjacency: Dict[int, List[int]] = {int(i): [] for i in range(len(f))}
    for u, v in edges:
        adjacency[int(u)].append(int(v))
        adjacency[int(v)].append(int(u))

    tree = _build_join_tree_graph(f, adjacency)
    tree.graph["gudhi_persistence"] = [
        (int(dim), float(birth), float(death))
        for dim, (birth, death) in st.persistence()
        if dim == 0 and np.isfinite(death)
    ]
    if tree.number_of_nodes() == 0 and len(f) > 0:
        root = int(np.argmin(f))
        tree.add_node(root, idx=root, value=float(f[root]), type="min")
    return tree


def get_merge_tree_graph_point_cloud(
    points: np.ndarray,
    function: str = "eccentricity",
    function_dim: int = 0,
    n_neighbors: int = 8,
    direction: Literal[1, -1] = 1,
    mutual: bool = False,
    backend: Literal["gudhi", "python"] = "gudhi",
    knn_algorithm: str = "auto",
    metric: str = "euclidean",
    n_jobs: Optional[int] = None,
) -> Tuple[nx.DiGraph, np.ndarray, np.ndarray]:
    """
    Compute a merge tree for a point cloud by combining:
    1. a scalar function on points, and
    2. a neighborhood graph on the point cloud.

    The result is a join tree of the scalar function over the kNN graph.
    """
    _require_gudhi_and_nx() if backend == "gudhi" else _require_nx()
    X = np.asarray(points, dtype=float)
    if X.ndim != 2:
        raise ValueError("Point-cloud merge trees require a 2D array of shape (n_samples, n_features).")
    if X.shape[0] < 2:
        raise ValueError("Point-cloud merge trees require at least two points.")

    knn_distances, knn_indices = _compute_knn_neighbors(
        X,
        n_neighbors=n_neighbors,
        algorithm=knn_algorithm,
        metric=metric,
        n_jobs=n_jobs,
    )

    scalar_values = _compute_point_cloud_scalar_function(
        X,
        function=function,
        function_dim=function_dim,
        n_neighbors=n_neighbors,
        knn_distances=knn_distances,
        knn_algorithm=knn_algorithm,
        metric=metric,
        n_jobs=n_jobs,
    )
    work_values = scalar_values if direction == 1 else -scalar_values
    if backend == "gudhi":
        edges = _build_knn_edges(X, n_neighbors=n_neighbors, mutual=mutual, indices=knn_indices)
        tree = _build_gudhi_lower_star_tree(work_values, edges)
    elif backend == "python":
        graph = _build_knn_graph(X, n_neighbors=n_neighbors, mutual=mutual, indices=knn_indices)
        adjacency = {int(i): [int(j) for j in graph.neighbors(i)] for i in graph.nodes}
        tree = _build_join_tree_graph(work_values, adjacency)
    else:
        raise ValueError("Unsupported backend='{}'. Expected one of: gudhi, python.".format(backend))

    for node in tree.nodes:
        idx = int(tree.nodes[node]["idx"])
        tree.nodes[node]["value"] = float(scalar_values[idx])
        if direction == -1:
            tree.nodes[node]["type"] = "max" if tree.nodes[node]["type"] == "min" else "split_saddle"
    tree.graph["direction"] = direction
    tree.graph["function"] = function
    tree.graph["n_neighbors"] = n_neighbors
    tree.graph["backend"] = backend
    tree.graph["knn_algorithm"] = knn_algorithm
    tree.graph["metric"] = metric

    return tree, _project_point_cloud_to_2d(X), scalar_values

# --- Merge Tree Computation and Graph Reconstruction (2D) ---

def get_merge_tree_graph_2d(
    scalar_field: np.ndarray, 
    direction: Literal[1, -1] = 1,
) -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
    """
    Computes the Merge Tree (H0 persistence) for a 2D scalar field
    and reconstructs the connection graph structure.

    Args:
        scalar_field: A 2D numpy array representing the scalar field/image.
        direction: 1 for Merge Tree (minima-based), -1 for Split Tree (maxima-based).

    Returns:
        A tuple: (NetworkX Graph, 2D coordinates array, 1D scalar field values)
    """
    _require_gudhi_and_nx()
    
    # 1. Prepare data and complex
    assert scalar_field.ndim == 2, "Input must be a 2D array for cubical complex"
    
    # Apply direction: -1 inverts the field for Split Tree (Merge Tree of -f)
    if direction == -1:
        field_data = -scalar_field.copy()
    else:
        field_data = scalar_field.copy()

    # Flatten the field data for the cubical complex
    field_1d = field_data.flatten()
    dimensions = field_data.shape
    
    # GUDHI Cubical Complex: Constructs the filtered complex
    cc = gd.CubicalComplex(dimensions=dimensions, top_dimensional_cells=field_1d)
    cc.compute_persistence(min_persistence=-1)
    
    # 2. Extract Persistence Pairs and Cofaces (Birth/Death Indices)
    cofaces = cc.cofaces_of_persistence_pairs()
    h0_pairs_indices = cofaces[0][0] # Regular H0 pairs: (birth cell index, death cell index)

    # 3. Create the Critical Point Coordinates Array
    y_indices, x_indices = np.indices(dimensions)
    coords_2d = np.vstack([x_indices.ravel(), y_indices.ravel()]).T
    
    # Collect all unique critical point indices (births and deaths)
    critical_indices = np.unique(h0_pairs_indices.flatten())
    
    # 4. Construct the Graph (Merge Tree)
    G = nx.Graph()
    
    for idx in critical_indices:
        # Position in (x, y) space on the grid
        pos = coords_2d[idx] 
        # Scalar value at that position (using the original field)
        val = scalar_field.flat[idx]
        
        G.add_node(
            idx, 
            pos=pos,
            value=val,
            is_birth=idx in h0_pairs_indices[:, 0],
            is_death=idx in h0_pairs_indices[:, 1]
        )

    # Add edges based on H0 persistence pairs
    for birth_idx, death_idx in h0_pairs_indices:
        G.add_edge(birth_idx, death_idx)
        
    return G, coords_2d, scalar_field.flatten()


def plot_merge_tree_graph_2d(
    G: nx.Graph, 
    coords_2d: np.ndarray, 
    scalar_field: np.ndarray, 
    overlay: bool = True,
    ax: Optional[plt.Axes] = None,
) -> None:
    """
    Plots the Merge Tree graph structure, optionally overlaid on the scalar field.
    (Implementation from previous response)
    """
    _require_gudhi_and_nx()
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    n_points = scalar_field.size
    n_grid = int(np.sqrt(n_points))
    field_reshaped = scalar_field.reshape(n_grid, n_grid)
    
    # 1. Overlay Plot (Contour)
    if overlay:
        Xg = coords_2d[:, 0].reshape(n_grid, n_grid)
        Yg = coords_2d[:, 1].reshape(n_grid, n_grid)
        
        c = ax.contourf(Xg, Yg, field_reshaped, levels=30, cmap='viridis', alpha=0.5)
        plt.colorbar(c, ax=ax, label="Scalar Field Value")

    # 2. Graph Drawing
    node_positions = {n: G.nodes[n]['pos'] for n in G.nodes}
    
    # Customize node colors/sizes based on their type
    node_colors = []
    node_sizes = []
    for _, data in G.nodes(data=True):
        if data['is_birth']: 
            node_colors.append('blue')
            node_sizes.append(100)
        elif data['is_death']: 
            node_colors.append('red')
            node_sizes.append(150)
        else:
            node_colors.append('gray')
            node_sizes.append(50)
            
    nx.draw_networkx_nodes(G, pos=node_positions, ax=ax, node_size=node_sizes, 
                           node_color=node_colors, edgecolors='black')
    #nx.draw_networkx_edges(G, pos=node_positions, ax=ax, edge_color='black', 
    #                       width=2, alpha=0.8)
    
    ax.set_title(f"Merge Tree Graph Overlay ({'Minima-based' if G.nodes[list(G.nodes)[0]]['value'] == np.min(scalar_field) else 'Maxima-based'})")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------
# --- Merge Tree Computation and Graph Reconstruction (1D) ---
# ----------------------------------------------------------------------

def get_merge_tree_graph_1d(
    scalar_field: np.ndarray, 
    direction: Literal[1, -1] = 1,
) -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
    """
    Computes the Merge Tree (H0 persistence) for a 1D scalar field (time series)
    and reconstructs the connection graph structure.

    Args:
        scalar_field: A 1D numpy array representing the scalar field/signal.
        direction: 1 for Merge Tree (minima-based), -1 for Split Tree (maxima-based).

    Returns:
        A tuple: (NetworkX Graph, 2D coordinates array for plotting, 1D scalar field values)
    """
    _require_gudhi_and_nx()
    
    # 1. Prepare data and complex
    assert scalar_field.ndim == 1, "Input must be a 1D array for cubical complex"
    
    # Apply direction
    if direction == -1:
        field_data = -scalar_field.copy()
    else:
        field_data = scalar_field.copy()

    dimensions = (len(scalar_field),)
    
    # GUDHI Cubical Complex: 1D Cubical Complex
    cc = gd.CubicalComplex(dimensions=dimensions, top_dimensional_cells=field_data)
    cc.compute_persistence(min_persistence=-1)
    
    # 2. Extract Persistence Pairs and Cofaces (Birth/Death Indices)
    cofaces = cc.cofaces_of_persistence_pairs()
    h0_pairs_indices = cofaces[0][0] # Regular H0 pairs: (birth cell index, death cell index)

    # 3. Create the Critical Point Coordinates Array
    # Coords are (spatial index, value) for the 2D plot of the 1D function
    indices = np.arange(len(scalar_field))
    
    # Collect all unique critical point indices
    critical_indices = np.unique(h0_pairs_indices.flatten())
    
    # 4. Construct the Graph (Merge Tree)
    G = nx.Graph()
    
    for idx in critical_indices:
        # pos is (spatial index, scalar value) for the 1D function plot
        pos = np.array([indices[idx], scalar_field[idx]])
        val = scalar_field[idx]
        
        G.add_node(
            idx, 
            pos=pos,
            value=val,
            is_birth=idx in h0_pairs_indices[:, 0],
            is_death=idx in h0_pairs_indices[:, 1]
        )

    # Add edges based on H0 persistence pairs
    for birth_idx, death_idx in h0_pairs_indices:
        G.add_edge(birth_idx, death_idx)
        
    # Return the reconstructed graph and the 2D plot coordinates
    return G, np.vstack([indices, scalar_field]).T, scalar_field


# ---------- 1) Build a true 1D join tree (used for layout mode) ----------
class _UF:
    def __init__(self, n: int):
        self.p = np.arange(n)
    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a
    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return ra
        self.p[rb] = ra
        return ra

def _build_join_tree_1d(f: np.ndarray) -> nx.DiGraph:
    """Join tree (sublevel connectivity) for 1D f. Nodes: minima ('min') and maxima ('max')."""
    f = np.asarray(f, dtype=float).ravel()
    n = len(f)
    uf = _UF(n)
    active = np.zeros(n, dtype=bool)

    def is_min(i):
        L = f[i-1] if i > 0   else np.inf
        R = f[i+1] if i < n-1 else np.inf
        return f[i] < L and f[i] < R
    def is_max(i):
        L = f[i-1] if i > 0   else -np.inf
        R = f[i+1] if i < n-1 else -np.inf
        return f[i] > L and f[i] > R

    T = nx.DiGraph()
    comp_rep: Dict[int, int] = {}

    def ensure(i: int, kind: str):
        if not T.has_node(i):
            T.add_node(i, idx=i, value=float(f[i]), type=kind)
        return i

    order = sorted(range(n), key=lambda i: (f[i], i))
    for i in order:
        active[i] = True
        uf.p[i] = i

        nbrs = []
        if i > 0   and active[i-1] and f[i-1] <= f[i]: nbrs.append(i-1)
        if i < n-1 and active[i+1] and f[i+1] <= f[i]: nbrs.append(i+1)
        roots = []
        for j in nbrs:
            rj = uf.find(j)
            if rj not in roots:
                roots.append(rj)

        if len(roots) == 0:
            comp_rep[uf.find(i)] = ensure(i, 'min')
        elif len(roots) == 1:
            uf.union(i, roots[0])
            comp_rep[uf.find(i)] = comp_rep.get(roots[0], comp_rep.get(uf.find(i), None))
        else:
            m = ensure(i, 'max') if is_max(i) else ensure(i, 'max')
            for r in roots:
                if r in comp_rep and not T.has_edge(m, comp_rep[r]):
                    T.add_edge(m, comp_rep[r])
            new_root = i
            for r in roots:
                uf.union(new_root, r)
            comp_rep[uf.find(new_root)] = m
    return T

# ---------- 2) Tidy layout: x by subtree order, y by value ----------
# === modern palette ===
PALETTE = {
    "min": "#2563eb",          # blue-600
    "max": "#f59e0b",          # amber-500
    "saddle": "#14e893",
    "split_saddle": "#a855f7", # purple-500
    "regular": "#94a3b8",      # slate-400
    "edge": "#334155",         # slate-700
}

def _layout_tree_y_value(
    T: nx.DiGraph,
    spread: float = 1.4,
    jitter: float = 0.0,
    tie_eps: float = 0.0,   # add a tiny epsilon when multiple nodes share the same f-value
) -> Dict[int, Tuple[float, float]]:
    """x by subtree order (leaves left→right), y = node['value'] (with optional tiny tie split)."""
    if T.number_of_nodes() == 0:
        return {}

    T = T.copy()
    T.remove_edges_from(nx.selfloop_edges(T))

    # choose roots: highest value first for join, lowest first for split (direction set earlier)
    direction = T.graph.get("direction", 1)
    roots = [n for n in T.nodes if T.in_degree(n) == 0]
    if not roots:
        root = (max if direction == 1 else min)(T.nodes, key=lambda n: T.nodes[n]["value"])
        roots = [root]
    roots.sort(key=lambda n: (float(T.nodes[n]["value"]), T.nodes[n].get("idx", n)), reverse=(direction == 1))

    # small tie-handling per y-level to reduce stacked horizontals
    # build per-value buckets
    val_to_nodes: Dict[float, list] = {}
    for n in T.nodes:
        v = float(T.nodes[n]['value'])
        val_to_nodes.setdefault(v, []).append(n)
    if tie_eps > 0:
        for v, nodes in val_to_nodes.items():
            if len(nodes) > 1:
                nodes.sort()  # stable order
                for k, n in enumerate(nodes):
                    T.nodes[n]['value'] = v + tie_eps * (k - (len(nodes)-1)/2.0)

    # leaves ordered by spatial idx if present, else id
    leaves = [n for n in T.nodes if T.out_degree(n) == 0]
    leaves.sort(key=lambda n: T.nodes[n].get('idx', n))

    x: Dict[int, float] = {}
    y: Dict[int, float] = {n: float(T.nodes[n]['value']) for n in T.nodes}
    cursor = 0.0
    visiting: set[int] = set()

    def assign(u: int) -> float:
        nonlocal cursor
        if u in x:
            return x[u]
        if u in visiting:
            x[u] = cursor
            cursor += spread
            return x[u]
        visiting.add(u)
        kids = [v for v in T.successors(u) if v != u]
        if not kids:
            x[u] = cursor
            cursor += spread
            visiting.discard(u)
            return x[u]
        xs = [assign(v) for v in kids]
        x[u] = sum(xs) / len(xs)
        visiting.discard(u)
        return x[u]

    for root in roots:
        assign(root)
    for node in sorted(T.nodes, key=lambda n: T.nodes[n].get("idx", n)):
        assign(node)

    if jitter:
        rng = np.random.default_rng(0)
        for n in T.nodes:
            y[n] += jitter * rng.standard_normal()

    return {n: (x[n], y[n]) for n in T.nodes}

def _draw_orthogonal_edges(
    G: nx.DiGraph,
    pos: Dict[int, Tuple[float, float]],
    ax: plt.Axes,
    linewidth: float = 2.2,
    alpha: float = 0.95,
    edge_sep: float = 0.0,  # small vertical separation to avoid coincident horizontals
):
    """Draw parent→child as vertical then horizontal, with optional tiny separation."""
    # precompute per-(parent_y) counters to separate coincident horizontals slightly
    if edge_sep > 0:
        ycount: Dict[Tuple[float, float], int] = {}
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        # parent = higher y (for join); if split was relabeled earlier, values still monotone
        if y1 > y0:
            x0, y0, x1, y1 = x1, y1, x0, y0
        yy = y1
        if edge_sep > 0:
            key = (round(y0, 12), round(y1, 12))
            k = ycount.get(key, 0)
            yy = y1 - edge_sep * k
            ycount[key] = k + 1
        ax.plot([x0, x0, x1], [y0, yy, yy],
                color=PALETTE["edge"], linewidth=linewidth, alpha=alpha,
                solid_capstyle="round", antialiased=True, clip_on=False)

# ---------- 3) Final plotting function with both modes ----------
def plot_merge_tree_graph_1d(
    G: nx.Graph,
    coords_2d_plot: np.ndarray,
    scalar_field: np.ndarray,
    overlay: bool = True,
    ax: Optional[plt.Axes] = None,
    direction: Literal[1, -1] = 1,
    spread: float = 1.6,     # wider subtree spacing
    jitter: float = 0.0,
    tie_eps: float = 0.0,    # e.g., 1e-9 to split identical values slightly
    edge_sep: float = 0.002,   # e.g., 0.002 * (max(f)-min(f)) to separate stacked horizontals
) -> None:
    """
    overlay=True  -> nodes in (index, f(index)); usually draw nodes only (edge-free) to avoid spaghetti.
    overlay=False -> tidy abstract tree; x from tree structure, y=f value; orthogonal edges; modern palette.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor("white")
    ax.grid(False)

    is_join = (direction == 1)

    if overlay:
        # Map to data coordinates; no edges by default
        ax.plot(coords_2d_plot[:, 0], coords_2d_plot[:, 1],
                color="#9ca3af", linewidth=1.2, alpha=0.55, label="f(x)")
        pos = {n: tuple(G.nodes[n]["pos"]) for n in G.nodes}
        ax.set_xlabel("Spatial Index (x)")
        ax.set_title(f"1D Merge Tree Overlay ({'Minima-based' if is_join else 'Maxima-based'})")
    else:
        # Build a true tree just for layout; then draw with orthogonal edges
        f_for_tree = scalar_field if is_join else -scalar_field
        T = _build_join_tree_1d(f_for_tree)
        # restore original values and label types for split view
        for n in T.nodes:
            T.nodes[n]["value"] = float(scalar_field[T.nodes[n]["idx"]])
            if not is_join:
                T.nodes[n]["type"] = "max" if T.nodes[n]["type"] == "min" else "split_saddle"
        T.graph["direction"] = direction
        pos = _layout_tree_y_value(T, spread=spread, jitter=jitter, tie_eps=tie_eps)
        # choose a small separation relative to vertical range if requested
        if edge_sep > 0 and edge_sep < 1:
            ys = [p[1] for p in pos.values()]
            edge_sep = edge_sep * (max(ys) - min(ys) + 1e-12)
        _draw_orthogonal_edges(T, pos, ax, linewidth=2.2, alpha=0.95, edge_sep=edge_sep)
        G = T
        ax.set_xlabel("Hierarchical Position")
        ax.set_title(f"1D Merge Tree Structure ({'Minima-based' if is_join else 'Maxima-based'})")

    ax.set_ylabel("Function Value $f(x)$")

    # nodes above edges
    colors, sizes = [], []
    for _, d in G.nodes(data=True):
        if d.get("is_birth") or d.get("type") == "min":
            colors.append(PALETTE["min"]);           sizes.append(80)
        elif d.get("is_death") or d.get("type") in ("max", "split_saddle", "saddle"):
            colors.append(PALETTE.get(d.get("type"), PALETTE["max"])); sizes.append(110)
        else:
            colors.append(PALETTE["regular"]);       sizes.append(60)

    nx.draw_networkx_nodes(
        G, pos=pos, ax=ax,
        node_color=colors, node_size=sizes,
        edgecolors="white", linewidths=1.6,
        alpha=0.98
    )

    # limits with padding
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
    pad = 0.08 * (max(ys) - min(ys) if max(ys) > min(ys) else 1.0)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)

    # compact legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=PALETTE["min"], edgecolor="white", label="Birth"),
        Patch(facecolor=PALETTE["max"], edgecolor="white", label="Death"),
    ], loc="upper right", frameon=True, framealpha=0.9)

    plt.tight_layout()


def plot_merge_tree_graph_point_cloud(
    T: nx.DiGraph,
    point_coords_2d: np.ndarray,
    scalar_values: np.ndarray,
    overlay: bool = True,
    ax: Optional[plt.Axes] = None,
    direction: Literal[1, -1] = 1,
    spread: float = 1.6,
    jitter: float = 0.0,
    tie_eps: float = 0.0,
    edge_sep: float = 0.002,
) -> None:
    """
    Plot a point-cloud merge tree either:
    - over a 2D point-cloud embedding, or
    - as an abstract tree with y = scalar function value.
    """
    _require_nx()
    import matplotlib.pyplot as plt

    coords = np.asarray(point_coords_2d, dtype=float)
    values = np.asarray(scalar_values, dtype=float).ravel()
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("point_coords_2d must have shape (n_samples, 2).")

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("white")
    ax.grid(False)

    if overlay:
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=values,
            cmap="viridis",
            s=28,
            alpha=0.75,
            edgecolors="none",
        )
        critical_pos = {n: tuple(coords[int(T.nodes[n]["idx"])]) for n in T.nodes}
        colors, sizes = [], []
        for _, d in T.nodes(data=True):
            if d.get("type") == "min":
                colors.append(PALETTE["min"])
                sizes.append(70)
            else:
                colors.append(PALETTE.get(d.get("type"), PALETTE["max"]))
                sizes.append(100)
        nx.draw_networkx_nodes(
            T,
            pos=critical_pos,
            ax=ax,
            node_color=colors,
            node_size=sizes,
            edgecolors="white",
            linewidths=1.3,
            alpha=0.98,
        )
        ax.set_title(f"Point-Cloud Merge Tree Overlay ({'Minima-based' if direction == 1 else 'Maxima-based'})")
        ax.set_xlabel("Embedding Coordinate 1")
        ax.set_ylabel("Embedding Coordinate 2")
        plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Scalar Function Value")
        plt.tight_layout()
        return

    tree = T.copy()
    tree.graph["direction"] = direction
    pos = _layout_tree_y_value(tree, spread=spread, jitter=jitter, tie_eps=tie_eps)
    if not pos:
        ax.text(
            0.5,
            0.5,
            "No nontrivial merge-tree events were found for this scalar lens.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=PALETTE["edge"],
        )
        ax.set_axis_off()
        plt.tight_layout()
        return
    if edge_sep > 0 and edge_sep < 1:
        ys = [p[1] for p in pos.values()]
        edge_sep = edge_sep * (max(ys) - min(ys) + 1e-12)
    _draw_orthogonal_edges(tree, pos, ax, linewidth=2.2, alpha=0.95, edge_sep=edge_sep)

    colors, sizes = [], []
    for _, d in tree.nodes(data=True):
        if d.get("type") == "min":
            colors.append(PALETTE["min"])
            sizes.append(80)
        else:
            colors.append(PALETTE.get(d.get("type"), PALETTE["max"]))
            sizes.append(110)
    nx.draw_networkx_nodes(
        tree,
        pos=pos,
        ax=ax,
        node_color=colors,
        node_size=sizes,
        edgecolors="white",
        linewidths=1.6,
        alpha=0.98,
    )
    ax.set_title(f"Point-Cloud Merge Tree Structure ({'Minima-based' if direction == 1 else 'Maxima-based'})")
    ax.set_xlabel("Hierarchical Position")
    ax.set_ylabel("Scalar Function Value")
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
    pad = 0.08 * (max(ys) - min(ys) if max(ys) > min(ys) else 1.0)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=PALETTE["min"], edgecolor="white", label="Birth"),
        Patch(facecolor=PALETTE["max"], edgecolor="white", label="Merge / Death"),
    ], loc="upper right", frameon=True, framealpha=0.9)
    plt.tight_layout()



# ----------------------------------------------------------------------
# --- Placeholder for H1 (Loops) - H1 logic is the same for 1D/2D
# ----------------------------------------------------------------------

def compute_h1_persistence_2d(scalar_field: np.ndarray) -> np.ndarray:
    """
    Computes H1 (Loops/Cavities) persistence for a 2D scalar field.
    (H1 is only relevant for 2D or higher spatial dimensions)
    """
    _require_gudhi_and_nx()
    
    if scalar_field.ndim == 3 and scalar_field.shape[2] == 1:
        scalar_field = np.squeeze(scalar_field, axis=2)

    assert scalar_field.ndim == 2, "Input must be a 2D array for H1 cubical complex"

    cc = gd.CubicalComplex(top_dimensional_cells=scalar_field.flatten(), dimensions=scalar_field.shape)
    cc.compute_persistence(min_persistence=0)
    
    # Extract H1 pairs
    persistence_intervals = cc.persistence()
    h1_pairs = np.array([
        pair for dim, pair in persistence_intervals if dim == 1
    ])
    return h1_pairs
