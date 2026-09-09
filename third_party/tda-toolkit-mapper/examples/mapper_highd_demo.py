from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.datasets import make_swiss_roll

from tda_toolkit.mapper import run_mapper_pipeline, summarize_mapper_graph, visualize_mapper_graph


def main() -> None:
    points, color = make_swiss_roll(n_samples=1200, noise=0.05, random_state=7)

    # Lift the swiss roll into a higher-dimensional ambient space so we can
    # experiment with a lens that is not tied to raw coordinate axes.
    highd_points = points
    for power in (2, 3):
        highd_points = np.hstack([highd_points, points[:, :2] ** power])

    result = run_mapper_pipeline(
        highd_points,
        n_cubes=18,
        overlap=0.35,
        lens="pca",
        lens_components=2,
        scale="standard",
        projection="pca",
        projection_components=3,
        clusterer="dbscan",
        dbscan_eps=0.7,
        dbscan_min_samples=6,
    )

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "mapper_highd_demo.html"
    visualize_mapper_graph(result.mapper, result.graph, path_html=str(out_path))

    print("Wrote:", out_path)
    print("Color proxy range:", float(color.min()), "to", float(color.max()))
    print("Summary:", summarize_mapper_graph(result.graph))


if __name__ == "__main__":
    main()
