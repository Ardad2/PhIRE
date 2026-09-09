from __future__ import annotations

import numpy as np

def save_distance_matrix(D: np.ndarray, file_path: str) -> None:
    np.save(file_path, D)


def load_distance_matrix(file_path: str) -> np.ndarray:
    return np.load(file_path)
