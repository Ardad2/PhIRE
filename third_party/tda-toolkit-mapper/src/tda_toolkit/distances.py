from __future__ import annotations

import numpy as np
from typing import Callable, List

def distance_matrix(items: List, dist_func: Callable) -> np.ndarray:
    n = len(items)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = float(dist_func(items[i], items[j]))
    return D
