import numpy as np
import pytest

from tda_toolkit import analyze
from tda_toolkit.engine import analyze_persistence
from tda_toolkit.models import normalize_input


def test_normalize_input_point_cloud_inferred():
    data = np.random.rand(20, 3)
    normalized = normalize_input(data)
    assert normalized.kind == "point_cloud"
    assert normalized.points.shape == (20, 3)


def test_normalize_input_scalar_field_explicit():
    field = np.random.rand(16, 16)
    normalized = normalize_input(field, kind="scalar_field")
    assert normalized.kind == "scalar_field"
    assert normalized.field.shape == (16, 16)


def test_normalize_input_graph_dict():
    graph_data = {"edges": np.array([[0, 1], [1, 2], [2, 0]], dtype=int)}
    normalized = normalize_input(graph_data, kind="graph")
    assert normalized.kind == "graph"
    assert normalized.num_nodes == 3
    assert normalized.edges.shape == (3, 2)


@pytest.mark.parametrize(
    "data,kind,max_dim",
    [
        (np.random.rand(30, 2), "point_cloud", 1),
        (np.random.rand(12, 12), "scalar_field", 1),
        ({"edges": np.array([[0, 1], [1, 2], [2, 0]], dtype=int)}, "graph", 2),
    ],
)
def test_analyze_persistence_smoke(data, kind, max_dim):
    try:
        result = analyze_persistence(data, kind=kind, max_dim=max_dim)
    except ImportError:
        pytest.skip("Gudhi not installed")
    assert result.backend == "gudhi"
    assert result.kind == kind
    assert isinstance(result.diagram, list)


def test_analyze_alias_and_summary():
    try:
        result = analyze(np.random.rand(24, 2), kind="point_cloud", max_dim=1)
    except ImportError:
        pytest.skip("Gudhi not installed")
    summary = result.summary()
    assert summary["kind"] == "point_cloud"
    assert "num_pairs" in summary
    assert result.input_data is not None


def test_result_vectorize_summary_shape():
    try:
        result = analyze(np.random.rand(20, 2), kind="point_cloud", max_dim=1)
    except ImportError:
        pytest.skip("Gudhi not installed")
    vec = result.vectorize(method="summary")
    assert vec.ndim == 1
