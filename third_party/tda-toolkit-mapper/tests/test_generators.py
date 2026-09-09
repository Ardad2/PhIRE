import numpy as np
import pytest

from tda_toolkit.generators import (
    GeneratorMapping,
    compute_point_cloud_generator_mappings,
    compute_scalar_field_generator_mappings,
    select_generator_mapping,
)


def test_select_generator_mapping_by_index():
    mappings = [
        GeneratorMapping(0, 1, 0.0, 1.0, 1.0, (0, 1), (0, 1, 2), (0, 1, 2), ((0, 1), (0, 2), (1, 2))),
        GeneratorMapping(1, 1, 0.2, 0.8, 0.6, (2, 3), (2, 3, 4), (2, 3, 4), ((2, 3), (2, 4), (3, 4))),
    ]
    selected = select_generator_mapping(mappings, pair_index=1)
    assert selected.pair_index == 1
    assert selected.persistence == pytest.approx(0.6)


def test_select_generator_mapping_index_error():
    mappings = [GeneratorMapping(0, 0, 0.0, 1.0, 1.0, (0,), (1,), (0, 1), tuple())]
    with pytest.raises(IndexError):
        select_generator_mapping(mappings, pair_index=3)


def test_point_cloud_generator_smoke():
    X = np.random.rand(25, 2)
    try:
        _, mappings = compute_point_cloud_generator_mappings(X, dim=1, max_dim=2)
    except ImportError:
        pytest.skip("Gudhi not installed")
    assert isinstance(mappings, list)


def test_point_cloud_generator_has_representative_edge_metadata():
    X = np.random.rand(30, 2)
    try:
        _, mappings = compute_point_cloud_generator_mappings(X, dim=1, max_dim=2)
    except ImportError:
        pytest.skip("Gudhi not installed")
    for mapping in mappings:
        assert isinstance(mapping.pair_edges, tuple)
        assert isinstance(mapping.representative_edges, tuple)
        for u, v in mapping.edges:
            assert isinstance(u, int)
            assert isinstance(v, int)


def test_scalar_field_h1_generator_mapping_has_hole_support():
    field = np.ones((24, 24), dtype=float)
    field[4:20, 4:20] = 0.5
    field[9:15, 9:15] = 2.0
    try:
        _, mappings = compute_scalar_field_generator_mappings(field, dim=1, direction=1)
    except ImportError:
        pytest.skip("Gudhi not installed")
    assert len(mappings) >= 1
    top = mappings[0]
    assert top.dimension == 1
    assert len(top.pixel_indices) > 0
    assert len(top.boundary_points) > 0
