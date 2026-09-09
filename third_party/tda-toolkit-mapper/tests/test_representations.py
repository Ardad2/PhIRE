import numpy as np

from tda_toolkit.cluster import cluster_feature_matrix
from tda_toolkit.representations import betti_curve, persistence_image, summary_features, vectorize_diagram


SAMPLE_DIAGRAM = [
    (0, (0.0, 1.0)),
    (0, (0.2, 0.7)),
    (1, (0.3, 0.9)),
    (1, (0.5, 1.1)),
]


def test_summary_features_shape():
    vec = summary_features(SAMPLE_DIAGRAM, dims=[0, 1])
    assert vec.shape == (8,)
    assert np.all(np.isfinite(vec))


def test_betti_curve_shape():
    t, curve = betti_curve(SAMPLE_DIAGRAM, dim=1, num_bins=64)
    assert t.shape == (64,)
    assert curve.shape == (64,)
    assert np.all(curve >= 0.0)


def test_persistence_image_shape():
    img = persistence_image(SAMPLE_DIAGRAM, dim=1, resolution=(16, 20), sigma=0.1)
    assert img.shape == (16, 20)
    assert np.all(np.isfinite(img))


def test_vectorize_diagram_methods():
    summary = vectorize_diagram(SAMPLE_DIAGRAM, method="summary")
    curve = vectorize_diagram(SAMPLE_DIAGRAM, method="betti_curve", dim=1, num_bins=32)
    pimg = vectorize_diagram(SAMPLE_DIAGRAM, method="persistence_image", dim=1, resolution=(8, 8))
    assert summary.ndim == 1
    assert curve.shape == (32,)
    assert pimg.shape == (64,)


def test_cluster_feature_matrix():
    X = np.array(
        [
            [0.0, 1.0, 0.5],
            [0.1, 0.9, 0.6],
            [5.0, 4.9, 5.1],
            [4.8, 5.2, 5.0],
        ]
    )
    labels = cluster_feature_matrix(X, n_clusters=2, method="kmeans", random_state=0)
    assert labels.shape == (4,)
