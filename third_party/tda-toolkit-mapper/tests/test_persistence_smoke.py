import numpy as np
import pytest


def test_rips_smoke():
    try:
        from tda_toolkit.persistence import compute_rips_persistence
    except Exception:
        pytest.skip("Gudhi not installed")
    X = np.random.rand(20, 2)
    try:
        st = compute_rips_persistence(X, max_dim=1)
    except ImportError:
        pytest.skip("Gudhi not installed")
    assert st is not None
