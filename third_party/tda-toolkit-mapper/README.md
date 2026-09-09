# tda-toolkit

A lightweight Topological Data Analysis (TDA) toolkit for point clouds,
high-dimensional arrays, scalar fields, graphs, and quick local visual
experiments.

The project is intentionally modular: use the Python API in notebooks or
research scripts, run one-off computations from the CLI, or launch the local UI
to upload data and inspect Mapper graphs, persistence diagrams, and merge trees
under `localhost`.

## Features

- Rips and cubical persistence with plotting helpers
- Merge-tree construction (via `gudhi` ~~`topopy` + `nglpy`~~) and topological profiles, point-cloud merge trees built from scalar lenses on exact sklearn kNN graphs
- Mapper graphs via `KeplerMapper` with configurable lenses, covers, clustering, JSON export, PNG export, and interactive HTML export
- Diagram distances, vector summaries, and clustering utilities
- Generator-to-data mapping helpers for point clouds, scalar fields, and graphs
- A Flask UI for uploading or linking numeric datasets and visualizing TDA descriptors locally

## Install

For the current Mapper, persistence, merge-tree, and UI workflow:

```bash
pip install -e ."[persistence,mapper,ui]"
```

For core persistence and scripting only:

```bash
pip install -e ."[persistence]"
```

For every optional extra declared by the package:

```bash
pip install -e ."[full]"
```

## Local UI

Start the app:

```bash
tda-toolkit-ui
```

Then open:

```text
http://127.0.0.1:5000/
```

The UI accepts uploaded files, local paths, or remote URLs for numeric `.csv`,
`.txt`, and `.npy` data. A sample higher-dimensional dataset is available at `examples/test_data.csv`.

Supported UI modes:

- `point_cloud`: compute persistence diagrams, Mapper graphs, and point-cloud merge trees
- `scalar_field`: compute cubical persistence and scalar-field merge trees
- `graph`: compute graph persistence from an edge list

## Input Data

Point-cloud and high-dimensional data should be a numeric matrix with shape
`(n_samples, n_features)`. Each row is one point or observation, and each column
is one coordinate, feature, embedding dimension, or descriptor.

Scalar fields should be a 1D or 2D numeric array, such as an image-like grid.
Graph inputs should be edge lists with at least two columns.

## Python Quickstart

### Persistence

```python
import numpy as np
from tda_toolkit.persistence import compute_rips_persistence, plot_persistence_diagram

X = np.random.rand(100, 2)
simplex_tree = compute_rips_persistence(X, max_dim=1)
plot_persistence_diagram(simplex_tree, dimension=1, title="H1 persistence")
```

### Unified API

```python
import numpy as np
from tda_toolkit import analyze

X = np.random.rand(100, 3)
point_result = analyze(X, kind="point_cloud", max_dim=2)
print(point_result.summary())
point_result.plot_diagram(dimension=1, title="Point-cloud H1")

field = np.random.rand(32, 32)
field_result = analyze(field, kind="scalar_field")

edges = np.array([[0, 1], [1, 2], [2, 0]])
graph_result = analyze({"edges": edges}, kind="graph", max_dim=2)
```

### Mapper On High-Dimensional Data

```python
from sklearn.datasets import make_swiss_roll
from tda_toolkit.mapper import (
    plot_mapper_graph,
    run_mapper_pipeline,
    save_mapper_graph_json,
    summarize_mapper_graph,
)

X, _ = make_swiss_roll(n_samples=1000, noise=0.05, random_state=7)

result = run_mapper_pipeline(
    X,
    lens="pca",
    lens_components=2,
    scale="standard",
    projection="pca",
    projection_components=3,
    clusterer="dbscan",
    dbscan_eps=0.7,
    dbscan_min_samples=6,
    n_cubes=18,
    overlap=0.35,
)

print(summarize_mapper_graph(result.graph))
plot_mapper_graph(result, path_png="mapper_graph.png")
save_mapper_graph_json(result, "mapper_graph.json")
```

Useful Mapper lens choices:

- `coordinate`: inspect one original feature directly
- `pca`: use leading principal directions as the cover coordinates
- `eccentricity`: emphasize distance from the dataset center
- `norm`: emphasize radial structure
- `knn_distance`: highlight sparse regions and potential outliers
- `density`: highlight dense regions

### Point-Cloud Merge Trees

A merge tree needs a scalar function. For point clouds, this toolkit computes a
scalar value on each point, builds an exact sklearn kNN graph in the original
feature space, and then uses a `gudhi` lower-star filtration to recover the H0
merge structure over that graph.

```python
import matplotlib.pyplot as plt
import numpy as np
from tda_toolkit.merge_tree import (
    get_merge_tree_graph_point_cloud,
    plot_merge_tree_graph_point_cloud,
)

X = np.random.rand(120, 5)

tree, coords_2d, values = get_merge_tree_graph_point_cloud(
    X,
    function="eccentricity",
    n_neighbors=8,
    backend="gudhi",
    knn_algorithm="auto",
    metric="euclidean",
)

fig, ax = plt.subplots(figsize=(9, 6))
plot_merge_tree_graph_point_cloud(tree, coords_2d, values, overlay=False, ax=ax)
plt.show()
```

The 2D coordinates returned by `get_merge_tree_graph_point_cloud` are for
visualization only. The kNN graph is built on the original high-dimensional
array.

## Descriptor Representations

Persistence diagrams can be vectorized into simple feature matrices for
clustering or downstream machine-learning experiments.

```python
import numpy as np
from tda_toolkit.cluster import cluster_feature_matrix
from tda_toolkit.representations import vectorize_diagram

diagrams = [point_result.diagram, field_result.diagram, graph_result.diagram]
features = np.vstack([vectorize_diagram(d, method="summary") for d in diagrams])
labels = cluster_feature_matrix(features, n_clusters=2, method="kmeans")
print(labels)
```

## Generator Mapping

Generator mapping helps connect a persistence feature back to the original data
that produced it.

```python
import numpy as np
from tda_toolkit.generators import (
    compute_point_cloud_generator_mappings,
    explore_point_cloud_generators,
)

X = np.random.rand(120, 2)
_, mappings = compute_point_cloud_generator_mappings(X, dim=1, max_dim=2)

top_generator = mappings[0]
explore_point_cloud_generators(X, mappings)
```

## CLI

Compute Rips persistence from a point-cloud CSV and save a diagram:

```bash
tda-toolkit rips --points examples/data.csv --dim 1 --out h1.png
```

Compute cubical persistence from a scalar field:

```bash
tda-toolkit cubical --field field.npy --out field_pd.png
```

Run Mapper on high-dimensional data and export HTML, JSON, and PNG outputs:

```bash
tda-toolkit mapper \
  --points data.npy \
  --lens pca \
  --lens-components 2 \
  --scale standard \
  --projection pca \
  --projection-components 3 \
  --clusterer dbscan \
  --dbscan-eps 0.7 \
  --dbscan-min-samples 6 \
  --png-out mapper_graph.png \
  --json-out mapper_graph.json \
  --out mapper_graph.html
```

Highlight a point-cloud generator:

```bash
tda-toolkit generator-points --points examples/data.csv --dim 1 --pair-index 0 --out gen0.png
```

Open the interactive generator explorer:

```bash
tda-toolkit generator-points --points examples/data.csv --dim 1 --interactive
```

Highlight a scalar-field generator:

```bash
tda-toolkit generator-field --field field.npy --dim 1 --pair-index 0 --out field_h1.png
```
