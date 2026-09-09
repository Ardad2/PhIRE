import numpy as np

import tda_toolkit.mapper as mapper_mod
from tda_toolkit.mapper import (
    build_mapper_clusterer,
    compute_mapper_lens,
    mapper_graph_to_networkx,
    plot_mapper_graph,
    preprocess_points,
    project_points,
    run_mapper_pipeline,
    save_mapper_graph_json,
    summarize_mapper_graph,
)


def test_compute_mapper_lens_shapes():
    X = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 2.0],
            [2.0, 1.0, 0.0],
            [3.0, 1.5, 1.0],
        ]
    )

    assert compute_mapper_lens(X, lens="coordinate", lens_dim=1).shape == (4, 1)
    assert compute_mapper_lens(X, lens="pca", n_components=2).shape == (4, 2)
    assert compute_mapper_lens(X, lens="eccentricity").shape == (4, 1)
    assert compute_mapper_lens(X, lens="norm").shape == (4, 1)
    assert compute_mapper_lens(X, lens="knn_distance", n_neighbors=3).shape == (4, 1)
    assert compute_mapper_lens(X, lens="density", n_neighbors=3).shape == (4, 1)


def test_preprocess_and_projection():
    X = np.array(
        [
            [0.0, 10.0, 100.0],
            [1.0, 11.0, 101.0],
            [2.0, 12.0, 102.0],
            [3.0, 13.0, 103.0],
        ]
    )

    scaled = preprocess_points(X, scale="standard")
    assert scaled.shape == X.shape
    assert np.allclose(np.mean(scaled, axis=0), 0.0)

    projected = project_points(scaled, projection="pca", n_components=2)
    assert projected.shape == (4, 2)


def test_build_mapper_clusterers():
    assert build_mapper_clusterer("dbscan").__class__.__name__ == "DBSCAN"
    assert build_mapper_clusterer("kmeans", n_clusters=3).__class__.__name__ == "KMeans"
    assert build_mapper_clusterer("agglomerative", n_clusters=3).__class__.__name__ == "AgglomerativeClustering"


def test_summarize_mapper_graph():
    graph = {
        "nodes": {"cube0_cluster0": [0, 1, 2], "cube1_cluster0": [2, 3]},
        "links": {"cube0_cluster0": ["cube1_cluster0"], "cube1_cluster0": ["cube0_cluster0"]},
    }
    summary = summarize_mapper_graph(graph)
    assert summary["num_nodes"] == 2
    assert summary["num_edges"] == 1
    assert summary["num_points_covered"] == 4
    assert summary["largest_node_size"] == 3


def test_mapper_graph_export_and_networkx(tmp_path):
    graph = {
        "nodes": {"cube0_cluster0": [0, 1, 2], "cube1_cluster0": [2, 3]},
        "links": {"cube0_cluster0": ["cube1_cluster0"], "cube1_cluster0": ["cube0_cluster0"]},
    }
    result = mapper_mod.MapperResult(
        graph=graph,
        mapper=None,
        lens=np.array([[0.0], [0.1], [0.4], [0.8]]),
        clustering_data=np.zeros((4, 2)),
        points=np.zeros((4, 5)),
        lens_name="pca",
        projection="pca",
        clusterer_name="dbscan",
    )

    G = mapper_graph_to_networkx(graph)
    assert G.number_of_nodes() == 2
    assert G.number_of_edges() == 1

    out_json = tmp_path / "mapper.json"
    save_mapper_graph_json(result, str(out_json))
    assert out_json.exists()

    ax = plot_mapper_graph(result)
    assert ax.get_title().startswith("Mapper Graph")


def test_run_mapper_pipeline_with_fake_kmapper(monkeypatch):
    captured = {}

    class FakeCover:
        def __init__(self, n_cubes, perc_overlap):
            self.n_cubes = n_cubes
            self.perc_overlap = perc_overlap

    class FakeKeplerMapper:
        def __init__(self, verbose=0):
            self.verbose = verbose

        def map(self, lens_values, clustering_data, clusterer=None, cover=None):
            captured["lens_shape"] = lens_values.shape
            captured["clustering_shape"] = clustering_data.shape
            captured["clusterer_name"] = clusterer.__class__.__name__
            captured["n_cubes"] = cover.n_cubes
            return {"nodes": {"node0": [0, 1]}, "links": {}}

        def visualize(self, graph, path_html):
            return None

    class FakeKM:
        KeplerMapper = FakeKeplerMapper
        Cover = FakeCover

    monkeypatch.setattr(mapper_mod, "km", FakeKM)

    X = np.random.RandomState(0).rand(12, 5)
    result = run_mapper_pipeline(
        X,
        n_cubes=7,
        overlap=0.2,
        lens="pca",
        lens_components=2,
        scale="standard",
        projection="pca",
        projection_components=3,
        clusterer="kmeans",
        n_clusters=4,
    )

    assert result.lens.shape == (12, 2)
    assert result.clustering_data.shape == (12, 3)
    assert captured["lens_shape"] == (12, 2)
    assert captured["clustering_shape"] == (12, 3)
    assert captured["clusterer_name"] == "KMeans"
    assert captured["n_cubes"] == 7
