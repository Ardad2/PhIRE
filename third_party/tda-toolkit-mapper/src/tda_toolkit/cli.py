from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional

from .io import load_point_cloud, load_scalar_field
from .persistence import compute_rips_persistence, compute_cubical_persistence, plot_persistence_diagram
from .mapper import (
    plot_mapper_graph,
    run_mapper_pipeline,
    save_mapper_graph_json,
    summarize_mapper_graph,
    visualize_mapper_graph,
)
from .generators import (
    compute_graph_generator_mappings,
    compute_point_cloud_generator_mappings,
    compute_scalar_field_generator_mappings,
    explore_graph_generators,
    explore_point_cloud_generators,
    explore_scalar_field_generators,
    plot_graph_generator_mapping,
    plot_point_cloud_generator_mapping,
    plot_scalar_field_generator_mapping,
    select_generator_mapping,
)


def cmd_rips(args):
    X = load_point_cloud(args.points)
    st = compute_rips_persistence(X, max_dim=args.max_dim)
    if args.out:
        import matplotlib.pyplot as plt
        plot_persistence_diagram(st, dimension=args.dim, title=f"Rips PD H{args.dim}")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out, dpi=200, bbox_inches='tight')
    return 0


def cmd_cubical(args):
    F = load_scalar_field(args.field)
    st = compute_cubical_persistence(F)
    if args.out:
        import matplotlib.pyplot as plt
        plot_persistence_diagram(st, title="Cubical PD")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out, dpi=200, bbox_inches='tight')
    return 0


def cmd_mapper(args):
    X = load_point_cloud(args.points)
    result = run_mapper_pipeline(
        X,
        n_cubes=args.n_cubes,
        overlap=args.overlap,
        lens=args.lens,
        lens_dim=args.lens_dim,
        lens_components=args.lens_components,
        lens_neighbors=args.lens_neighbors,
        scale=args.scale,
        projection=args.projection,
        projection_components=args.projection_components,
        clusterer=args.clusterer,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        n_clusters=args.n_clusters,
    )
    out = visualize_mapper_graph(result.mapper, result.graph, path_html=args.out)
    summary = summarize_mapper_graph(result.graph)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        save_mapper_graph_json(result, args.json_out)
        print(f"Mapper JSON saved to: {args.json_out}")
    if args.png_out:
        Path(args.png_out).parent.mkdir(parents=True, exist_ok=True)
        plot_mapper_graph(
            result,
            path_png=args.png_out,
            layout=args.layout,
            color_by=args.color_by,
        )
        print(f"Mapper PNG saved to: {args.png_out}")
    print(f"Mapper HTML saved to: {out}")
    print(
        "Mapper summary: "
        f"nodes={summary['num_nodes']}, "
        f"edges={summary['num_edges']}, "
        f"covered_points={summary['num_points_covered']}, "
        f"largest_node={summary['largest_node_size']}"
    )
    return 0


def _save_or_show(ax, out: Optional[str]):
    import matplotlib.pyplot as plt

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(out, dpi=220, bbox_inches="tight")
        print(f"Saved to: {out}")
        plt.close(ax.figure)
    else:
        plt.show()


def cmd_generator_points(args):
    X = load_point_cloud(args.points)
    _, mappings = compute_point_cloud_generator_mappings(X, dim=args.dim, max_dim=args.max_dim)
    if len(mappings) == 0:
        raise RuntimeError("No finite generator mappings found for this point cloud.")
    if args.interactive:
        explore_point_cloud_generators(X, mappings)
        return 0
    mapping = select_generator_mapping(mappings, pair_index=args.pair_index)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_point_cloud_generator_mapping(X, mapping, ax=ax)
    _save_or_show(ax, args.out)
    return 0


def cmd_generator_field(args):
    F = load_scalar_field(args.field)
    _, mappings = compute_scalar_field_generator_mappings(F, dim=args.dim, direction=args.direction)
    if len(mappings) == 0:
        raise RuntimeError(f"No H{args.dim} generator mappings found for this scalar field.")
    if args.interactive:
        explore_scalar_field_generators(F, mappings)
        return 0
    mapping = select_generator_mapping(mappings, pair_index=args.pair_index)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    plot_scalar_field_generator_mapping(F, mapping, ax=ax)
    _save_or_show(ax, args.out)
    return 0


def cmd_generator_graph(args):
    from .io import load_graph

    G = load_graph(args.edges)
    _, mappings = compute_graph_generator_mappings(G, dim=args.dim, max_dim=args.max_dim)
    if len(mappings) == 0:
        raise RuntimeError("No finite generator mappings found for this graph.")
    if args.interactive:
        explore_graph_generators(G, mappings)
        return 0
    mapping = select_generator_mapping(mappings, pair_index=args.pair_index)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    plot_graph_generator_mapping(G, mapping, ax=ax)
    _save_or_show(ax, args.out)
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="tda-toolkit", description="TDA Toolkit CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("rips", help="Compute Rips persistence from a point cloud CSV/TXT")
    pr.add_argument("--points", required=True, help="Path to CSV/TXT with points")
    pr.add_argument("--dim", type=int, default=1, help="Homology dimension to display")
    pr.add_argument("--max-dim", type=int, default=2, help="Max homology dimension to compute")
    pr.add_argument("--out", help="Optional PNG output path")
    pr.set_defaults(func=cmd_rips)

    pc = sub.add_parser("cubical", help="Compute Cubical persistence from a scalar field CSV/NPY")
    pc.add_argument("--field", required=True, help="Path to NPY or CSV scalar field")
    pc.add_argument("--out", help="Optional PNG output path")
    pc.set_defaults(func=cmd_cubical)

    pm = sub.add_parser("mapper", help="Run KeplerMapper and export HTML")
    pm.add_argument("--points", required=True, help="Path to CSV/TXT with points")
    pm.add_argument("--n-cubes", type=int, default=10)
    pm.add_argument("--overlap", type=float, default=0.1)
    pm.add_argument(
        "--lens",
        default="coordinate",
        choices=["coordinate", "pca", "eccentricity", "norm", "knn_distance", "density"],
        help="Lens used to build the Mapper cover",
    )
    pm.add_argument("--lens-dim", type=int, default=0, help="Coordinate index when --lens=coordinate")
    pm.add_argument("--lens-components", type=int, default=1, help="Number of PCA lens coordinates")
    pm.add_argument("--lens-neighbors", type=int, default=15, help="Neighborhood size for KNN-based lenses")
    pm.add_argument(
        "--scale",
        default="none",
        choices=["none", "standard", "minmax"],
        help="Optional preprocessing before lens/projection construction",
    )
    pm.add_argument(
        "--projection",
        default="none",
        choices=["none", "pca"],
        help="Optional projection used for clustering inside each cover set",
    )
    pm.add_argument("--projection-components", type=int, default=2, help="Projection dimension when enabled")
    pm.add_argument(
        "--clusterer",
        default="dbscan",
        choices=["dbscan", "kmeans", "agglomerative"],
        help="Clustering algorithm inside each cover element",
    )
    pm.add_argument("--dbscan-eps", type=float, default=0.5, help="DBSCAN epsilon")
    pm.add_argument("--dbscan-min-samples", type=int, default=3, help="DBSCAN min_samples")
    pm.add_argument("--n-clusters", type=int, default=8, help="Cluster count for kmeans/agglomerative")
    pm.add_argument("--out", default="mapper_graph.html", help="HTML output path")
    pm.add_argument("--json-out", help="Optional JSON export path for the Mapper graph")
    pm.add_argument("--png-out", help="Optional static PNG graph output path")
    pm.add_argument("--layout", default="spring", choices=["spring", "kamada_kawai"], help="Static graph layout")
    pm.add_argument("--color-by", default="lens_mean", choices=["lens_mean", "size"], help="Static graph coloring")
    pm.set_defaults(func=cmd_mapper)

    pg = sub.add_parser("generator-points", help="Map a persistence generator back to point-cloud data")
    pg.add_argument("--points", required=True, help="Path to CSV/TXT with points")
    pg.add_argument("--dim", type=int, default=1, help="Homology dimension to map")
    pg.add_argument("--max-dim", type=int, default=2, help="Max homology dimension to compute")
    pg.add_argument("--pair-index", type=int, default=0, help="Index in persistence-sorted generator list")
    pg.add_argument("--interactive", action="store_true", help="Open click-to-highlight generator explorer")
    pg.add_argument("--out", help="Optional PNG output path for static mode")
    pg.set_defaults(func=cmd_generator_points)

    pf = sub.add_parser("generator-field", help="Map scalar-field H0/H1 generators back to image/field")
    pf.add_argument("--field", required=True, help="Path to NPY or CSV scalar field")
    pf.add_argument("--dim", type=int, default=0, choices=[0, 1], help="Homology dimension to map")
    pf.add_argument("--direction", type=int, default=1, choices=[1, -1], help="1 merge-tree, -1 split-tree")
    pf.add_argument("--pair-index", type=int, default=0, help="Index in persistence-sorted generator list")
    pf.add_argument("--interactive", action="store_true", help="Open click-to-highlight generator explorer")
    pf.add_argument("--out", help="Optional PNG output path for static mode")
    pf.set_defaults(func=cmd_generator_field)

    pgr = sub.add_parser("generator-graph", help="Map persistence generator back to graph nodes/edges")
    pgr.add_argument("--edges", required=True, help="Path to graph edge-list file")
    pgr.add_argument("--dim", type=int, default=1, help="Homology dimension to map")
    pgr.add_argument("--max-dim", type=int, default=2, help="Max homology dimension to compute")
    pgr.add_argument("--pair-index", type=int, default=0, help="Index in persistence-sorted generator list")
    pgr.add_argument("--interactive", action="store_true", help="Open click-to-highlight generator explorer")
    pgr.add_argument("--out", help="Optional PNG output path for static mode")
    pgr.set_defaults(func=cmd_generator_graph)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
