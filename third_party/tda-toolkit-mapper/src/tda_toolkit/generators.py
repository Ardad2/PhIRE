from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .engine import analyze_persistence

try:
    import gudhi as gd
except Exception:  # pragma: no cover - optional dependency
    gd = None  # type: ignore

try:
    import networkx as nx
except Exception:  # pragma: no cover - optional dependency
    nx = None  # type: ignore


@dataclass
class GeneratorMapping:
    pair_index: int
    dimension: int
    birth: float
    death: float
    persistence: float
    birth_simplex: Tuple[int, ...]
    death_simplex: Tuple[int, ...]
    vertices: Tuple[int, ...]
    edges: Tuple[Tuple[int, int], ...]
    pair_edges: Tuple[Tuple[int, int], ...] = tuple()
    representative_edges: Tuple[Tuple[int, int], ...] = tuple()
    pixel_indices: Tuple[int, ...] = tuple()
    boundary_points: Tuple[Tuple[int, int], ...] = tuple()


def _require_gudhi() -> None:
    if gd is None:
        raise ImportError("Generator mapping requires Gudhi. Install with: pip install tda-toolkit[persistence]")


def _simplex_edges(simplex: Sequence[int]) -> List[Tuple[int, int]]:
    if len(simplex) < 2:
        return []
    return [tuple(sorted((int(u), int(v)))) for u, v in combinations(simplex, 2)]


def _mapping_from_pair(
    simplex_tree: Any,
    pair_index: int,
    birth_simplex: Sequence[int],
    death_simplex: Sequence[int],
) -> GeneratorMapping:
    b = tuple(int(v) for v in birth_simplex)
    d = tuple(int(v) for v in death_simplex)
    dim = max(0, len(b) - 1)
    birth_time = float(simplex_tree.filtration(b))
    death_time = float(simplex_tree.filtration(d)) if len(d) > 0 else float("inf")
    persistence = float(death_time - birth_time) if np.isfinite(death_time) else float("inf")

    vertices = tuple(sorted(set(b).union(set(d))))
    edge_set = {e for e in _simplex_edges(b)}
    edge_set.update(_simplex_edges(d))
    pair_edges = tuple(sorted(edge_set))

    return GeneratorMapping(
        pair_index=pair_index,
        dimension=dim,
        birth=birth_time,
        death=death_time,
        persistence=persistence,
        birth_simplex=b,
        death_simplex=d,
        vertices=vertices,
        edges=pair_edges,
        pair_edges=pair_edges,
        representative_edges=pair_edges,
    )


def _build_graph_at_filtration(simplex_tree: Any, threshold: float) -> Any:
    if nx is None:
        return None
    G = nx.Graph()
    for simplex, filt in simplex_tree.get_skeleton(1):
        if len(simplex) == 1:
            G.add_node(int(simplex[0]))
        elif len(simplex) == 2 and float(filt) <= threshold:
            u, v = int(simplex[0]), int(simplex[1])
            G.add_edge(u, v)
    return G


def _h1_cycle_from_birth_edge(simplex_tree: Any, mapping: GeneratorMapping) -> Tuple[Tuple[int, ...], Tuple[Tuple[int, int], ...]]:
    """Best-effort H1 representative cycle: path at birth scale + birth edge."""
    if nx is None:
        return mapping.vertices, mapping.edges
    if mapping.dimension != 1 or len(mapping.birth_simplex) != 2:
        return mapping.vertices, mapping.edges
    u, v = mapping.birth_simplex
    eps = max(1e-9, 1e-9 * abs(mapping.birth))
    threshold = mapping.birth + eps
    G = _build_graph_at_filtration(simplex_tree, threshold=threshold)
    if G is None:
        return mapping.vertices, mapping.edges
    if G.has_edge(u, v):
        G.remove_edge(u, v)
    if not (G.has_node(u) and G.has_node(v)):
        return mapping.vertices, mapping.edges
    try:
        path = nx.shortest_path(G, source=u, target=v)
    except Exception:
        return mapping.vertices, mapping.edges

    rep_edges: List[Tuple[int, int]] = []
    for a, b in zip(path[:-1], path[1:]):
        rep_edges.append(tuple(sorted((int(a), int(b)))))
    rep_edges.append(tuple(sorted((int(u), int(v)))))
    rep_edges = sorted(set(rep_edges))
    rep_vertices = tuple(sorted({node for edge in rep_edges for node in edge}))
    if len(rep_edges) < 3:
        return mapping.vertices, mapping.edges
    return rep_vertices, tuple(rep_edges)


def extract_simplex_tree_generator_mappings(
    simplex_tree: Any,
    dim: Optional[int] = None,
    finite_only: bool = True,
    sort_descending: bool = True,
    use_representative_cycles: bool = True,
) -> List[GeneratorMapping]:
    """Extract persistence-pair mappings from a Gudhi SimplexTree."""
    _require_gudhi()
    pairs = simplex_tree.persistence_pairs()
    mappings: List[GeneratorMapping] = []
    for pair_index, (birth_simplex, death_simplex) in enumerate(pairs):
        mapping = _mapping_from_pair(simplex_tree, pair_index, birth_simplex, death_simplex)
        if dim is not None and mapping.dimension != dim:
            continue
        if finite_only and not np.isfinite(mapping.death):
            continue
        if use_representative_cycles and mapping.dimension == 1:
            rep_vertices, rep_edges = _h1_cycle_from_birth_edge(simplex_tree, mapping)
            mapping.vertices = rep_vertices
            mapping.edges = rep_edges
            mapping.representative_edges = rep_edges
        mappings.append(mapping)
    if sort_descending:
        mappings.sort(
            key=lambda m: (
                np.isfinite(m.persistence),
                m.persistence if np.isfinite(m.persistence) else -np.inf,
            ),
            reverse=True,
        )
    return mappings


def select_generator_mapping(mappings: Sequence[GeneratorMapping], pair_index: int = 0) -> GeneratorMapping:
    if len(mappings) == 0:
        raise ValueError("No generator mappings available.")
    if pair_index < 0 or pair_index >= len(mappings):
        raise IndexError(f"pair_index={pair_index} out of range [0, {len(mappings) - 1}].")
    return mappings[pair_index]


def compute_point_cloud_generator_mappings(
    points: np.ndarray,
    dim: int = 1,
    max_dim: int = 2,
    backend: str = "gudhi",
) -> Tuple[Any, List[GeneratorMapping]]:
    result = analyze_persistence(points, kind="point_cloud", backend=backend, max_dim=max_dim)
    mappings = extract_simplex_tree_generator_mappings(result.complex_obj, dim=dim, finite_only=True)
    return result, mappings


def compute_graph_generator_mappings(
    graph: Any,
    dim: int = 1,
    max_dim: int = 2,
    backend: str = "gudhi",
) -> Tuple[Any, List[GeneratorMapping]]:
    result = analyze_persistence(graph, kind="graph", backend=backend, max_dim=max_dim)
    mappings = extract_simplex_tree_generator_mappings(result.complex_obj, dim=dim, finite_only=True)
    return result, mappings


def compute_scalar_field_generator_mappings(
    scalar_field: np.ndarray,
    dim: Optional[int] = 0,
    direction: int = 1,
) -> Tuple[Any, List[GeneratorMapping]]:
    """Best-effort scalar-field generator mapping for 2D H0 and H1 pairs."""
    _require_gudhi()
    field = np.asarray(scalar_field, dtype=float)
    if field.ndim == 3 and field.shape[-1] == 1:
        field = np.squeeze(field, axis=-1)
    if field.ndim != 2:
        raise ValueError("Scalar-field generator mapping currently supports 2D fields.")

    work_field = field if direction == 1 else -field
    cc = gd.CubicalComplex(dimensions=work_field.shape, top_dimensional_cells=work_field.flatten())
    cc.compute_persistence(min_persistence=-1)

    cofaces = cc.cofaces_of_persistence_pairs()
    mappings: List[GeneratorMapping] = []

    if dim in (None, 0):
        h0_pairs = np.asarray(cofaces[0][0], dtype=int) if len(cofaces[0]) > 0 else np.empty((0, 2), dtype=int)
        for rank, (birth_idx, death_idx) in enumerate(h0_pairs):
            birth = float(field.flat[birth_idx])
            death = float(field.flat[death_idx])
            persistence = float(abs(death - birth))
            mappings.append(
                GeneratorMapping(
                    pair_index=rank,
                    dimension=0,
                    birth=min(birth, death),
                    death=max(birth, death),
                    persistence=persistence,
                    birth_simplex=(int(birth_idx),),
                    death_simplex=(int(death_idx),),
                    vertices=(int(birth_idx), int(death_idx)),
                    edges=tuple(),
                )
            )

    if dim in (None, 1):
        h1_pairs = np.asarray(cofaces[0][1], dtype=int) if len(cofaces[0]) > 1 else np.empty((0, 2), dtype=int)
        for rank, (birth_idx, death_idx) in enumerate(h1_pairs):
            mapping = _build_scalar_field_h1_mapping(
                field=field,
                work_field=work_field,
                birth_idx=int(birth_idx),
                death_idx=int(death_idx),
                pair_index=rank,
            )
            mappings.append(mapping)

    mappings.sort(key=lambda m: m.persistence, reverse=True)
    return cc, mappings


def _build_scalar_field_h1_mapping(
    field: np.ndarray,
    work_field: np.ndarray,
    birth_idx: int,
    death_idx: int,
    pair_index: int,
) -> GeneratorMapping:
    birth_work = float(work_field.flat[birth_idx])
    death_work = float(work_field.flat[death_idx])
    threshold = 0.5 * (birth_work + death_work)

    active_mask = work_field <= threshold
    death_rc = np.unravel_index(death_idx, field.shape)
    hole_mask = _component_mask(active_mask=active_mask, seed_rc=death_rc)
    boundary_points = _boundary_points_from_hole(active_mask=active_mask, hole_mask=hole_mask)
    pixel_indices = tuple(int(idx) for idx in np.flatnonzero(hole_mask))
    boundary_indices = tuple(int(np.ravel_multi_index((r, c), field.shape)) for r, c in boundary_points)
    original_birth = float(field.flat[birth_idx])
    original_death = float(field.flat[death_idx])

    return GeneratorMapping(
        pair_index=pair_index,
        dimension=1,
        birth=min(original_birth, original_death),
        death=max(original_birth, original_death),
        persistence=float(abs(original_death - original_birth)),
        birth_simplex=(birth_idx,),
        death_simplex=(death_idx,),
        vertices=boundary_indices,
        edges=tuple(),
        pixel_indices=pixel_indices,
        boundary_points=boundary_points,
    )


def _component_mask(active_mask: np.ndarray, seed_rc: Tuple[int, int]) -> np.ndarray:
    hole_mask = ~active_mask
    component = np.zeros_like(hole_mask, dtype=bool)
    if not hole_mask[seed_rc]:
        return component

    stack = [seed_rc]
    component[seed_rc] = True
    nrows, ncols = hole_mask.shape

    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < nrows and 0 <= cc < ncols and hole_mask[rr, cc] and not component[rr, cc]:
                component[rr, cc] = True
                stack.append((rr, cc))
    return component


def _boundary_points_from_hole(
    active_mask: np.ndarray,
    hole_mask: np.ndarray,
) -> Tuple[Tuple[int, int], ...]:
    boundary: List[Tuple[int, int]] = []
    nrows, ncols = active_mask.shape
    for r, c in np.argwhere(hole_mask):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < nrows and 0 <= cc < ncols and active_mask[rr, cc]:
                boundary.append((rr, cc))
    return tuple(sorted(set(boundary)))


def _plot_mapping_scatter(ax: plt.Axes, mappings: Sequence[GeneratorMapping], active_idx: Optional[int]) -> None:
    births = np.asarray([m.birth for m in mappings], dtype=float)
    deaths = np.asarray([m.death for m in mappings], dtype=float)
    ax.clear()
    ax.scatter(births, deaths, alpha=0.75, color="#2563eb")
    finite = deaths[np.isfinite(deaths)]
    maxv = float(np.max(finite)) if len(finite) else 1.0
    ax.plot([0.0, maxv], [0.0, maxv], linestyle="--", color="#94a3b8", linewidth=1)
    if active_idx is not None and 0 <= active_idx < len(mappings):
        ax.scatter([births[active_idx]], [deaths[active_idx]], color="#dc2626", s=90)
    ax.set_title("Persistence Diagram (click pair)")
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")


def explore_generator_mappings(
    mappings: Sequence[GeneratorMapping],
    draw_selected: Callable[[GeneratorMapping, plt.Axes], None],
    title: str = "Generator Mapping Explorer",
) -> None:
    """Interactive explorer: click a diagram point to highlight its generator in data space."""
    if len(mappings) == 0:
        raise ValueError("No generator mappings available for exploration.")

    fig, (ax_pd, ax_data) = plt.subplots(1, 2, figsize=(12, 5))
    state: Dict[str, int] = {"active_idx": 0}

    def redraw() -> None:
        _plot_mapping_scatter(ax_pd, mappings, state["active_idx"])
        ax_data.clear()
        draw_selected(mappings[state["active_idx"]], ax_data)
        fig.suptitle(title)
        fig.canvas.draw_idle()

    def on_click(event) -> None:
        if event.inaxes is not ax_pd or event.xdata is None or event.ydata is None:
            return
        xy = np.asarray([[m.birth, m.death] for m in mappings], dtype=float)
        q = np.asarray([event.xdata, event.ydata], dtype=float)
        idx = int(np.argmin(np.sum((xy - q) ** 2, axis=1)))
        state["active_idx"] = idx
        redraw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    redraw()
    plt.tight_layout()
    plt.show()


def browse_generator_mappings(
    mappings: Sequence[GeneratorMapping],
    draw_selected: Callable[[GeneratorMapping, plt.Axes], None],
    title: str = "Generator Mapping Browser",
) -> None:
    """Notebook-friendly browser using a slider instead of click events."""
    if len(mappings) == 0:
        raise ValueError("No generator mappings available for browsing.")
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except Exception as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "Widget browsing requires ipywidgets. Install in your notebook kernel with: %pip install ipywidgets"
        ) from e

    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(mappings) - 1,
        step=1,
        description="rank",
        continuous_update=False,
    )
    out = widgets.Output()

    def render(idx: int) -> None:
        with out:
            out.clear_output(wait=True)
            fig, (ax_pd, ax_data) = plt.subplots(1, 2, figsize=(12, 5))
            _plot_mapping_scatter(ax_pd, mappings, idx)
            draw_selected(mappings[idx], ax_data)
            fig.suptitle(title)
            plt.tight_layout()
            plt.show()

    def on_value_change(change: Dict[str, Any]) -> None:
        render(int(change["new"]))

    slider.observe(on_value_change, names="value")
    render(0)
    display(widgets.VBox([slider, out]))


def plot_point_cloud_generator_mapping(
    points: np.ndarray,
    mapping: GeneratorMapping,
    ax: Optional[plt.Axes] = None,
) -> None:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError("Point cloud plotting supports shape (n,2) or (n,3).")

    if points.shape[1] == 2:
        ax = ax or plt.gca()
        ax.scatter(points[:, 0], points[:, 1], color="#cbd5e1", s=25, alpha=0.7)
        idx = np.asarray(mapping.vertices, dtype=int)
        if len(idx) > 0:
            ax.scatter(points[idx, 0], points[idx, 1], color="#dc2626", s=60)
        for u, v in mapping.edges:
            ax.plot([points[u, 0], points[v, 0]], [points[u, 1], points[v, 1]], color="#dc2626", linewidth=2.2)
        ax.set_title(f"Generator #{mapping.pair_index} | H{mapping.dimension} | pers={mapping.persistence:.3f}")
    else:
        if ax is None:
            ax = plt.figure().add_subplot(111, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="#cbd5e1", s=25, alpha=0.7)
        idx = np.asarray(mapping.vertices, dtype=int)
        if len(idx) > 0:
            ax.scatter(points[idx, 0], points[idx, 1], points[idx, 2], color="#dc2626", s=60)
        for u, v in mapping.edges:
            ax.plot(
                [points[u, 0], points[v, 0]],
                [points[u, 1], points[v, 1]],
                [points[u, 2], points[v, 2]],
                color="#dc2626",
                linewidth=2.2,
            )
        ax.set_title(f"Generator #{mapping.pair_index} | H{mapping.dimension} | pers={mapping.persistence:.3f}")


def plot_scalar_field_generator_mapping(
    scalar_field: np.ndarray,
    mapping: GeneratorMapping,
    ax: Optional[plt.Axes] = None,
) -> None:
    field = np.asarray(scalar_field, dtype=float)
    if field.ndim != 2:
        raise ValueError("Scalar-field mapping plot supports 2D fields.")
    ax = ax or plt.gca()
    im = ax.imshow(field, cmap="viridis")
    b_idx = mapping.birth_simplex[0]
    d_idx = mapping.death_simplex[0] if len(mapping.death_simplex) else b_idx
    by, bx = np.unravel_index(b_idx, field.shape)
    dy, dx = np.unravel_index(d_idx, field.shape)

    if mapping.dimension == 1 and mapping.pixel_indices:
        hole_rc = np.array(np.unravel_index(np.asarray(mapping.pixel_indices, dtype=int), field.shape)).T
        if len(hole_rc) > 0:
            ax.scatter(hole_rc[:, 1], hole_rc[:, 0], color="#fca5a5", s=18, alpha=0.45, label="Hole region")
        if mapping.boundary_points:
            boundary = np.asarray(mapping.boundary_points, dtype=int)
            ax.scatter(boundary[:, 1], boundary[:, 0], color="#dc2626", s=22, alpha=0.9, label="Hole boundary")
        ax.scatter([dx], [dy], color="#2563eb", s=90, label="Death cell")
        ax.scatter([bx], [by], color="#f59e0b", s=90, label="Birth cell")
        ax.set_title(f"H1 Pair #{mapping.pair_index} | pers={mapping.persistence:.3f}")
    else:
        ax.scatter([bx], [by], color="#2563eb", s=90, label="Birth")
        ax.scatter([dx], [dy], color="#dc2626", s=90, label="Death")
        ax.set_title(f"H0 Pair #{mapping.pair_index} | pers={mapping.persistence:.3f}")

    ax.legend(loc="upper right")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _to_indexed_graph(graph: Any) -> Tuple[Any, List[Any]]:
    if nx is None:
        raise ImportError("Graph plotting requires networkx.")
    if isinstance(graph, nx.Graph):
        labels = list(graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(labels)}
        G = nx.Graph()
        G.add_nodes_from(range(len(labels)))
        for u, v in graph.edges():
            G.add_edge(node_to_idx[u], node_to_idx[v])
        return G, labels

    arr = np.asarray(graph, dtype=int)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Graph input must be a networkx graph or an edge array of shape (m,2).")
    G = nx.Graph()
    G.add_edges_from((int(u), int(v)) for u, v in arr)
    labels = sorted(G.nodes())
    return G, labels


def plot_graph_generator_mapping(
    graph: Any,
    mapping: GeneratorMapping,
    ax: Optional[plt.Axes] = None,
) -> None:
    if nx is None:
        raise ImportError("Graph mapping plot requires networkx.")
    G, node_labels = _to_indexed_graph(graph)
    pos = nx.spring_layout(G, seed=0)
    ax = ax or plt.gca()
    nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color="#cbd5e1", width=1.2)
    nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_color="#cbd5e1", node_size=320)

    highlight_nodes = set(mapping.vertices)
    highlight_edges = set(tuple(sorted(e)) for e in mapping.edges)
    if highlight_edges:
        nx.draw_networkx_edges(
            G,
            pos=pos,
            edgelist=[(u, v) for u, v in G.edges() if tuple(sorted((u, v))) in highlight_edges],
            edge_color="#dc2626",
            width=2.8,
            ax=ax,
        )
    if highlight_nodes:
        nx.draw_networkx_nodes(
            G,
            pos=pos,
            nodelist=sorted(highlight_nodes),
            node_color="#dc2626",
            node_size=360,
            ax=ax,
        )
    if set(G.nodes()) == set(range(len(node_labels))):
        label_map = {i: str(node_labels[i]) for i in G.nodes()}
    else:
        label_map = {i: str(i) for i in G.nodes()}
    nx.draw_networkx_labels(G, pos=pos, labels=label_map, font_size=8, ax=ax)
    ax.set_title(f"Generator #{mapping.pair_index} | H{mapping.dimension} | pers={mapping.persistence:.3f}")
    ax.set_axis_off()


def explore_point_cloud_generators(points: np.ndarray, mappings: Sequence[GeneratorMapping]) -> None:
    def draw(mapping: GeneratorMapping, ax: plt.Axes) -> None:
        plot_point_cloud_generator_mapping(points, mapping, ax=ax)

    explore_generator_mappings(mappings, draw_selected=draw, title="Point Cloud Generator Explorer")


def browse_point_cloud_generators(points: np.ndarray, mappings: Sequence[GeneratorMapping]) -> None:
    def draw(mapping: GeneratorMapping, ax: plt.Axes) -> None:
        plot_point_cloud_generator_mapping(points, mapping, ax=ax)

    browse_generator_mappings(mappings, draw_selected=draw, title="Point Cloud Generator Browser")


def explore_scalar_field_generators(scalar_field: np.ndarray, mappings: Sequence[GeneratorMapping]) -> None:
    def draw(mapping: GeneratorMapping, ax: plt.Axes) -> None:
        plot_scalar_field_generator_mapping(scalar_field, mapping, ax=ax)

    explore_generator_mappings(mappings, draw_selected=draw, title="Scalar Field Generator Explorer")


def browse_scalar_field_generators(scalar_field: np.ndarray, mappings: Sequence[GeneratorMapping]) -> None:
    def draw(mapping: GeneratorMapping, ax: plt.Axes) -> None:
        plot_scalar_field_generator_mapping(scalar_field, mapping, ax=ax)

    browse_generator_mappings(mappings, draw_selected=draw, title="Scalar Field Generator Browser")


def explore_graph_generators(graph: Any, mappings: Sequence[GeneratorMapping]) -> None:
    def draw(mapping: GeneratorMapping, ax: plt.Axes) -> None:
        plot_graph_generator_mapping(graph, mapping, ax=ax)

    explore_generator_mappings(mappings, draw_selected=draw, title="Graph Generator Explorer")


def browse_graph_generators(graph: Any, mappings: Sequence[GeneratorMapping]) -> None:
    def draw(mapping: GeneratorMapping, ax: plt.Axes) -> None:
        plot_graph_generator_mapping(graph, mapping, ax=ax)

    browse_generator_mappings(mappings, draw_selected=draw, title="Graph Generator Browser")
