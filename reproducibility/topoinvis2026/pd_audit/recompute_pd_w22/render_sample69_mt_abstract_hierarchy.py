#!/usr/bin/env python3
"""
Render validated abstract Join-Tree hierarchies for sample 69.

This renderer uses ONLY the validated logical hierarchy:
    NodeId
    downNodeId -> upNodeId
    unique root

It deliberately does NOT use:
    - physical x/y coordinates for layout;
    - raw TTK Scalar as the vertical axis;
    - original wind_speed as a node-height surrogate;
    - historical F-order qualitative artifacts.

Vertical placement:
    topological depth from the validated root (root = depth 0).

Horizontal placement:
    deterministic tidy-tree spacing for readability only.
    It has NO physical, scalar, or metric meaning.

Outputs:
    sample_069_mt_t3_abstract_hierarchy.png
    sample_069_mt_t3_abstract_hierarchy.pdf
    sample_069_mt_t3_abstract_hierarchy_node_ids.png
    sample69_mt_abstract_layout.csv
    sample69_mt_abstract_hierarchy_summary.txt
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict, deque
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


W22 = Path(os.environ["W22"])

BASE = (
    W22
    / "corrected_pd_mt"
    / "discordance_visuals"
    / "sample_069"
)

NODES_CSV = BASE / "sample69_mt_logical_nodes.csv"
ARCS_CSV = BASE / "sample69_mt_logical_arcs.csv"

OUTDIR = BASE / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = ["gt", "cnn", "uv", "f1"]

DISPLAY_NAMES = {
    "gt": "GT",
    "cnn": "CNN",
    "uv": r"Ablation ($L_{uv}$ only)",
    "f1": "Candidate F",
}


def read_csv(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_int(row, key):
    return int(row[key])


def validate_and_build(method, node_rows, arc_rows):
    nodes = {
        as_int(r, "node_id"): r
        for r in node_rows
        if r["method"] == method
    }

    arcs = [
        r
        for r in arc_rows
        if r["method"] == method
    ]

    if not nodes:
        raise RuntimeError(f"{method}: no nodes")

    n = len(nodes)

    if sorted(nodes) != list(range(n)):
        raise RuntimeError(
            f"{method}: NodeId is not exactly 0..N-1"
        )

    if len(arcs) != n - 1:
        raise RuntimeError(
            f"{method}: expected N-1 logical arcs; "
            f"nodes={n}, arcs={len(arcs)}"
        )

    # Refuse to render if any previously validated structural condition
    # is absent in the logical-arc table.
    structural_flags = [
        "constant_up_down_pair",
        "sampled_geometry_simple_path",
        "sampled_path_endpoint_matches_nodes",
    ]

    for flag in structural_flags:
        if flag not in arcs[0]:
            raise RuntimeError(
                f"{method}: required audit column missing: {flag}"
            )

        bad = [
            r
            for r in arcs
            if int(r[flag]) != 1
        ]

        if bad:
            raise RuntimeError(
                f"{method}: structural audit flag failed: {flag}"
            )

    parents = {}
    children = defaultdict(list)

    for r in arcs:
        child = as_int(r, "down_node_id")
        parent = as_int(r, "up_node_id")

        if child not in nodes or parent not in nodes:
            raise RuntimeError(
                f"{method}: unknown node in edge {child}->{parent}"
            )

        if child == parent:
            raise RuntimeError(
                f"{method}: self-loop at node {child}"
            )

        if child in parents:
            raise RuntimeError(
                f"{method}: node {child} has multiple parents"
            )

        parents[child] = parent
        children[parent].append(child)

    roots = sorted(
        set(nodes)
        - set(parents)
    )

    if len(roots) != 1:
        raise RuntimeError(
            f"{method}: expected exactly one root; got {roots}"
        )

    root = roots[0]

    # Deterministic child ordering. Node IDs are used only to make
    # the drawing reproducible; horizontal position has no semantics.
    for p in children:
        children[p].sort()

    # Root-to-leaf depths.
    depth = {root: 0}
    q = deque([root])

    while q:
        parent = q.popleft()

        for child in children.get(parent, []):
            if child in depth:
                raise RuntimeError(
                    f"{method}: cycle/repeated traversal at {child}"
                )

            depth[child] = depth[parent] + 1
            q.append(child)

    if set(depth) != set(nodes):
        missing = sorted(set(nodes) - set(depth))
        raise RuntimeError(
            f"{method}: disconnected nodes: {missing}"
        )

    # ---------------------------------------------------------------
    # Tidy-tree x layout
    #
    # Leaves receive consecutive x positions.
    # Each internal node is centered over the mean x position of its
    # immediate children.
    # ---------------------------------------------------------------

    x = {}
    next_leaf = 0

    def place(node):
        nonlocal next_leaf

        ch = children.get(node, [])

        if not ch:
            x[node] = float(next_leaf)
            next_leaf += 1
            return x[node]

        child_x = [
            place(c)
            for c in ch
        ]

        x[node] = sum(child_x) / len(child_x)
        return x[node]

    place(root)

    leaves = sorted(
        node
        for node in nodes
        if len(children.get(node, [])) == 0
    )

    internals = sorted(
        node
        for node in nodes
        if node != root
        and len(children.get(node, [])) > 0
    )

    # Normalize horizontal position to [0, 1].
    if len(leaves) <= 1:
        x_norm = {
            node: 0.5
            for node in nodes
        }
    else:
        lo = min(x.values())
        hi = max(x.values())
        span = hi - lo

        x_norm = {
            node: (
                0.5
                if span == 0
                else (x[node] - lo) / span
            )
            for node in nodes
        }

    edge_list = sorted(
        (parent, child)
        for child, parent in parents.items()
    )

    child_count = {
        node: len(children.get(node, []))
        for node in nodes
    }

    max_depth = max(depth.values())

    degree_hist = Counter(
        child_count.values()
    )

    return {
        "nodes": nodes,
        "root": root,
        "parents": parents,
        "children": children,
        "depth": depth,
        "x": x_norm,
        "leaves": leaves,
        "internals": internals,
        "edges": edge_list,
        "max_depth": max_depth,
        "degree_hist": dict(sorted(degree_hist.items())),
    }


def draw_tree(ax, method, tree, annotate=False):
    # Parent -> child edges.
    for parent, child in tree["edges"]:
        ax.plot(
            [
                tree["x"][parent],
                tree["x"][child],
            ],
            [
                tree["depth"][parent],
                tree["depth"][child],
            ],
            linewidth=1.15,
            zorder=1,
        )

    leaves = tree["leaves"]
    internals = tree["internals"]
    root = tree["root"]

    if leaves:
        ax.scatter(
            [tree["x"][n] for n in leaves],
            [tree["depth"][n] for n in leaves],
            marker="o",
            s=34,
            zorder=3,
            label="Leaf",
        )

    if internals:
        ax.scatter(
            [tree["x"][n] for n in internals],
            [tree["depth"][n] for n in internals],
            marker="s",
            s=30,
            zorder=3,
            label="Internal",
        )

    ax.scatter(
        [tree["x"][root]],
        [tree["depth"][root]],
        marker="*",
        s=95,
        zorder=4,
        label="Root",
    )

    if annotate:
        for node in sorted(tree["nodes"]):
            ax.annotate(
                str(node),
                (
                    tree["x"][node],
                    tree["depth"][node],
                ),
                xytext=(3, 2),
                textcoords="offset points",
                fontsize=6.5,
                zorder=5,
            )

    ax.set_title(
        (
            f"{DISPLAY_NAMES[method]}\n"
            f"{len(tree['nodes'])} nodes, "
            f"{len(tree['edges'])} logical arcs, "
            f"max depth {tree['max_depth']}"
        ),
        fontsize=11.5,
        fontweight="bold",
    )

    ax.set_xlim(-0.06, 1.06)
    ax.set_xticks([])
    ax.set_xlabel(
        "horizontal placement for readability only",
        fontsize=8.5,
    )


def render_figure(trees, annotate=False):
    global_max_depth = max(
        t["max_depth"]
        for t in trees.values()
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(15.6, 5.4),
        sharey=True,
    )

    for ax, method in zip(
        axes,
        METHOD_ORDER,
    ):
        draw_tree(
            ax,
            method,
            trees[method],
            annotate=annotate,
        )

        ax.set_ylim(
            global_max_depth + 0.6,
            -0.6,
        )

        ax.set_yticks(
            range(global_max_depth + 1)
        )

        ax.grid(
            axis="y",
            alpha=0.18,
        )

    axes[0].set_ylabel(
        "Topological depth from root",
    )

    legend_handles = [
        Line2D(
            [],
            [],
            marker="*",
            linestyle="None",
            markersize=10,
            label="Root",
        ),
        Line2D(
            [],
            [],
            marker="s",
            linestyle="None",
            markersize=7,
            label="Internal node",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=7,
            label="Leaf",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )

    title = (
        "Sample 69: validated abstract Join-Tree hierarchies "
        "(display persistence threshold = 3.0)"
    )

    if annotate:
        title += " — audit view with NodeId labels"

    fig.suptitle(
        title,
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    fig.text(
        0.5,
        0.925,
        (
            "Vertical position = hierarchy depth from the validated root; "
            "horizontal position is a deterministic tidy-tree layout and "
            "has no physical or scalar meaning."
        ),
        ha="center",
        fontsize=9.5,
    )

    fig.subplots_adjust(
        left=0.055,
        right=0.99,
        bottom=0.14,
        top=0.83,
        wspace=0.15,
    )

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-node-id-audit-view",
        action="store_true",
        help=(
            "Do not write the second debug figure "
            "with NodeId labels."
        ),
    )
    args = parser.parse_args()

    node_rows = read_csv(NODES_CSV)
    arc_rows = read_csv(ARCS_CSV)

    present_node_methods = sorted(
        set(r["method"] for r in node_rows)
    )

    present_arc_methods = sorted(
        set(r["method"] for r in arc_rows)
    )

    expected = sorted(METHOD_ORDER)

    if present_node_methods != expected:
        raise RuntimeError(
            f"unexpected node methods: {present_node_methods}"
        )

    if present_arc_methods != expected:
        raise RuntimeError(
            f"unexpected arc methods: {present_arc_methods}"
        )

    trees = {
        method: validate_and_build(
            method,
            node_rows,
            arc_rows,
        )
        for method in METHOD_ORDER
    }

    # ---------------------------------------------------------------
    # Main abstract hierarchy figure
    # ---------------------------------------------------------------

    fig = render_figure(
        trees,
        annotate=False,
    )

    png = (
        OUTDIR
        / "sample_069_mt_t3_abstract_hierarchy.png"
    )

    pdf = (
        OUTDIR
        / "sample_069_mt_t3_abstract_hierarchy.pdf"
    )

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ---------------------------------------------------------------
    # Optional NodeId audit view
    # ---------------------------------------------------------------

    debug_png = None

    if not args.skip_node_id_audit_view:
        fig = render_figure(
            trees,
            annotate=True,
        )

        debug_png = (
            OUTDIR
            / "sample_069_mt_t3_abstract_hierarchy_node_ids.png"
        )

        fig.savefig(
            debug_png,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

    # ---------------------------------------------------------------
    # Save deterministic layout coordinates for audit/provenance
    # ---------------------------------------------------------------

    layout_csv = (
        BASE
        / "sample69_mt_abstract_layout.csv"
    )

    layout_fields = [
        "method",
        "node_id",
        "role",
        "parent_node_id",
        "child_count",
        "depth",
        "layout_x",
    ]

    layout_rows = []

    for method in METHOD_ORDER:
        tree = trees[method]

        for node in sorted(tree["nodes"]):
            if node == tree["root"]:
                role = "root"
            elif node in tree["leaves"]:
                role = "leaf"
            else:
                role = "internal"

            layout_rows.append({
                "method": method,
                "node_id": node,
                "role": role,
                "parent_node_id":
                    tree["parents"].get(node, ""),
                "child_count":
                    len(tree["children"].get(node, [])),
                "depth":
                    tree["depth"][node],
                "layout_x":
                    f"{tree['x'][node]:.12g}",
            })

    with layout_csv.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=layout_fields,
        )
        writer.writeheader()
        writer.writerows(layout_rows)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    summary = (
        BASE
        / "sample69_mt_abstract_hierarchy_summary.txt"
    )

    lines = []

    def emit(s=""):
        lines.append(s)
        print(s)

    emit(
        "SAMPLE-69 ABSTRACT JOIN-TREE HIERARCHY RENDER"
    )
    emit("=" * 92)
    emit()
    emit("display persistence threshold: 3.0 (already frozen)")
    emit("vertical coordinate: topological depth from validated root")
    emit(
        "horizontal coordinate: deterministic readability layout only"
    )
    emit(
        "raw TTK Scalar / original wind_speed used for placement: NO"
    )
    emit()

    for method in METHOD_ORDER:
        tree = trees[method]

        emit(DISPLAY_NAMES[method])
        emit("-" * 92)
        emit(
            f"  root NodeId: {tree['root']}"
        )
        emit(
            f"  nodes: {len(tree['nodes'])}"
        )
        emit(
            f"  logical arcs: {len(tree['edges'])}"
        )
        emit(
            f"  leaves: {len(tree['leaves'])}"
        )
        emit(
            f"  internal non-root nodes: "
            f"{len(tree['internals'])}"
        )
        emit(
            f"  maximum root-to-node depth: "
            f"{tree['max_depth']}"
        )
        emit(
            f"  child-count histogram: "
            f"{tree['degree_hist']}"
        )
        emit()

    emit("OUTPUTS")
    emit("-" * 92)
    emit(f"  main PNG: {png}")
    emit(f"  main PDF: {pdf}")

    if debug_png is not None:
        emit(f"  NodeId audit PNG: {debug_png}")

    emit(f"  layout CSV: {layout_csv}")
    emit(f"  summary: {summary}")

    summary.write_text(
        "\n".join(lines) + "\n"
    )

    print()
    print(
        "ABSTRACT JOIN-TREE HIERARCHY RENDER: COMPLETE"
    )


if __name__ == "__main__":
    main()
