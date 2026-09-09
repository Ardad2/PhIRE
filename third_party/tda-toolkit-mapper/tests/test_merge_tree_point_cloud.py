import numpy as np
import pytest


def test_point_cloud_merge_tree_smoke():
    merge_tree = pytest.importorskip("tda_toolkit.merge_tree")

    X = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [0.4, 0.0],
            [1.2, 1.0],
            [1.4, 1.1],
            [1.6, 1.0],
        ],
        dtype=float,
    )

    tree, coords_2d, values = merge_tree.get_merge_tree_graph_point_cloud(
        X,
        function="eccentricity",
        n_neighbors=3,
    )

    assert coords_2d.shape == (len(X), 2)
    assert values.shape == (len(X),)
    assert tree.number_of_nodes() > 0
    assert tree.graph["backend"] == "gudhi"
    assert tree.graph["knn_algorithm"] == "auto"
    assert tree.graph["metric"] == "euclidean"
    assert all("idx" in tree.nodes[n] for n in tree.nodes)


def test_point_cloud_merge_tree_plots_degenerate_lenses():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    merge_tree = pytest.importorskip("tda_toolkit.merge_tree")

    cases = [
        (np.arange(8, dtype=float).reshape(-1, 1), "coordinate", 0),
        (np.column_stack([np.zeros(8), np.arange(8, dtype=float)]), "coordinate", 0),
        (np.ones((8, 3), dtype=float), "norm", 0),
    ]

    for X, function, function_dim in cases:
        tree, coords_2d, values = merge_tree.get_merge_tree_graph_point_cloud(
            X,
            function=function,
            function_dim=function_dim,
            n_neighbors=3,
        )
        assert tree.number_of_nodes() > 0
        assert all(u != v for u, v in tree.edges())

        fig, ax = plt.subplots(figsize=(5, 4))
        merge_tree.plot_merge_tree_graph_point_cloud(
            tree,
            coords_2d,
            values,
            overlay=False,
            ax=ax,
        )
        plt.close(fig)
