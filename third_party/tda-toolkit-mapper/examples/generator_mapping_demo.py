import numpy as np
import matplotlib.pyplot as plt

from tda_toolkit.generators import (
    compute_point_cloud_generator_mappings,
    explore_point_cloud_generators,
    plot_point_cloud_generator_mapping,
)


def main():
    rng = np.random.default_rng(0)
    angles = rng.uniform(0, 2 * np.pi, size=200)
    radii = 1.0 + 0.05 * rng.standard_normal(200)
    X = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    _, mappings = compute_point_cloud_generator_mappings(X, dim=1, max_dim=2)
    if not mappings:
        print("No finite H1 generators found.")
        return

    # Static save for the top generator.
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_point_cloud_generator_mapping(X, mappings[0], ax=ax)
    fig.savefig("generator_top.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Saved: generator_top.png")

    # Interactive explorer: click on diagram points to inspect mappings.
    explore_point_cloud_generators(X, mappings)


if __name__ == "__main__":
    main()
