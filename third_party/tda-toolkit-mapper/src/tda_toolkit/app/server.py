from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import io
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Optional
from urllib.parse import urlparse
from urllib.request import urlopen
from uuid import uuid4

MPL_DIR = Path(tempfile.gettempdir()) / "tda_toolkit_mpl"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_DIR))

from flask import Flask, render_template, request, send_from_directory, url_for
import numpy as np

from ..engine import analyze


ARTIFACT_ROOT = Path(tempfile.gettempdir()) / "tda_toolkit_ui"
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass
class FormDefaults:
    source_mode: str = "upload"
    data_kind: str = "point_cloud"
    delimiter: str = "comma"
    skip_header: bool = True
    max_dim: int = 2
    compute_mapper: bool = True
    compute_merge_tree: bool = False
    n_cubes: int = 18
    overlap: float = 0.35
    lens: str = "pca"
    lens_components: int = 2
    lens_neighbors: int = 15
    scale: str = "standard"
    projection: str = "pca"
    projection_components: int = 3
    clusterer: str = "dbscan"
    dbscan_eps: float = 0.7
    dbscan_min_samples: int = 6
    n_clusters: int = 8
    merge_direction: int = 1


def _bool_from_form(key: str) -> bool:
    return request.form.get(key) in {"on", "true", "1", "yes"}


def _read_defaults() -> FormDefaults:
    return FormDefaults(
        source_mode=request.form.get("source_mode", "upload"),
        data_kind=request.form.get("data_kind", "point_cloud"),
        delimiter=request.form.get("delimiter", "comma"),
        skip_header=_bool_from_form("skip_header"),
        max_dim=int(request.form.get("max_dim", 2)),
        compute_mapper=_bool_from_form("compute_mapper"),
        compute_merge_tree=_bool_from_form("compute_merge_tree"),
        n_cubes=int(request.form.get("n_cubes", 18)),
        overlap=float(request.form.get("overlap", 0.35)),
        lens=request.form.get("lens", "pca"),
        lens_components=int(request.form.get("lens_components", 2)),
        lens_neighbors=int(request.form.get("lens_neighbors", 15)),
        scale=request.form.get("scale", "standard"),
        projection=request.form.get("projection", "pca"),
        projection_components=int(request.form.get("projection_components", 3)),
        clusterer=request.form.get("clusterer", "dbscan"),
        dbscan_eps=float(request.form.get("dbscan_eps", 0.7)),
        dbscan_min_samples=int(request.form.get("dbscan_min_samples", 6)),
        n_clusters=int(request.form.get("n_clusters", 8)),
        merge_direction=int(request.form.get("merge_direction", 1)),
    )


def _delimiter_value(name: str) -> Optional[str]:
    return {
        "comma": ",",
        "tab": "\t",
        "space": None,
        "semicolon": ";",
    }.get(name, ",")


def _fig_to_data_uri(fig) -> str:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _clean_numeric_array(data: np.ndarray, kind: str) -> np.ndarray:
    array = np.asarray(data, dtype=float)
    if array.ndim == 0:
        raise ValueError("The loaded file did not contain a numeric array.")
    if kind == "point_cloud":
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise ValueError("Point-cloud input must be a 2D numeric matrix.")
        mask = ~np.isnan(array).any(axis=1)
        array = array[mask]
    elif kind == "scalar_field":
        if array.ndim not in (1, 2):
            raise ValueError("Scalar-field input must be 1D or 2D.")
        array = np.squeeze(array)
        array = np.nan_to_num(array, nan=0.0)
    elif kind == "graph":
        if array.ndim != 2 or array.shape[1] < 2:
            raise ValueError("Graph input must have at least two columns representing edges.")
        array = array[:, :2]
        mask = ~np.isnan(array).any(axis=1)
        array = array[mask]
    else:
        raise ValueError(f"Unsupported kind: {kind}")

    if array.size == 0:
        raise ValueError("No usable numeric rows were found after cleaning the input.")
    return array


def _load_array_from_bytes(payload: bytes, *, kind: str, delimiter_name: str, skip_header: bool, source_name: str) -> np.ndarray:
    suffix = Path(source_name).suffix.lower()
    if suffix == ".npy":
        array = np.load(io.BytesIO(payload))
    else:
        delimiter = _delimiter_value(delimiter_name)
        array = np.genfromtxt(io.BytesIO(payload), delimiter=delimiter, skip_header=1 if skip_header else 0)
    return _clean_numeric_array(array, kind=kind)


def _load_array_from_source(source_mode: str, kind: str, delimiter_name: str, skip_header: bool) -> tuple[np.ndarray, str]:
    if source_mode == "upload":
        uploaded = request.files.get("data_file")
        if uploaded is None or uploaded.filename == "":
            raise ValueError("Please upload a CSV or NPY file.")
        return _load_array_from_bytes(
            uploaded.read(),
            kind=kind,
            delimiter_name=delimiter_name,
            skip_header=skip_header,
            source_name=uploaded.filename,
        ), uploaded.filename

    if source_mode == "path":
        raw_path = request.form.get("data_path", "").strip()
        if not raw_path:
            raise ValueError("Please provide a local file path.")
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix.lower() == ".npy":
            array = np.load(path)
            return _clean_numeric_array(array, kind=kind), str(path)
        delimiter = _delimiter_value(delimiter_name)
        array = np.genfromtxt(path, delimiter=delimiter, skip_header=1 if skip_header else 0)
        return _clean_numeric_array(array, kind=kind), str(path)

    if source_mode == "url":
        raw_url = request.form.get("data_url", "").strip()
        if not raw_url:
            raise ValueError("Please provide a remote CSV or NPY URL.")
        parsed = urlparse(raw_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("URL sources must start with http:// or https://")
        with urlopen(raw_url, timeout=20) as response:
            payload = response.read()
        source_name = Path(parsed.path).name or "remote.csv"
        return _load_array_from_bytes(
            payload,
            kind=kind,
            delimiter_name=delimiter_name,
            skip_header=skip_header,
            source_name=source_name,
        ), raw_url

    raise ValueError(f"Unsupported source mode: {source_mode}")


def _render_persistence_diagram(diagram: list[tuple[int, tuple[float, float]]]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    finite = [(dim, birth, death) for dim, (birth, death) in diagram if np.isfinite(death)]
    infinite = [(dim, birth) for dim, (birth, death) in diagram if not np.isfinite(death)]
    colors = ["#0f766e", "#ea580c", "#2563eb", "#7c3aed"]

    if finite:
        max_val = max(max(b, d) for _, b, d in finite)
    else:
        max_val = 1.0

    for dim, color in enumerate(colors):
        pts = [(b, d) for pair_dim, b, d in finite if pair_dim == dim]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=35, alpha=0.85, color=color, label=f"H{dim}")

    if infinite:
        inf_y = max_val * 1.05 if max_val > 0 else 1.0
        for dim, color in enumerate(colors):
            xs = [birth for pair_dim, birth in infinite if pair_dim == dim]
            if xs:
                ax.scatter(xs, [inf_y] * len(xs), marker="^", s=55, color=color, alpha=0.85)
        ax.axhline(inf_y, color="#64748b", linestyle="--", linewidth=1)
        ax.text(0.02, inf_y, "inf", color="#475569", fontsize=10, va="bottom")
        max_val = inf_y

    ax.plot([0, max_val], [0, max_val], linestyle="--", color="#94a3b8", linewidth=1)
    ax.set_title("Persistence Diagram")
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    if ax.collections:
        ax.legend(frameon=False, loc="lower right")
    return _fig_to_data_uri(fig)


def _render_merge_tree(array: np.ndarray, direction: int) -> tuple[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..merge_tree import (
        get_merge_tree_graph_1d,
        get_merge_tree_graph_2d,
        plot_merge_tree_graph_1d,
        plot_merge_tree_graph_2d,
    )

    if array.ndim == 1:
        G, coords, values = get_merge_tree_graph_1d(array, direction=direction)
        fig, ax = plt.subplots(figsize=(10, 5.5))
        plot_merge_tree_graph_1d(
            G,
            coords,
            values,
            overlay=False,
            ax=ax,
            direction=direction,
        )
        title = "1D Merge Tree"
        return _fig_to_data_uri(fig), title

    if array.ndim == 2:
        G, coords, values = get_merge_tree_graph_2d(array, direction=direction)
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        plot_merge_tree_graph_2d(G, coords, values, overlay=True, ax=ax)
        title = "2D Merge Tree Overlay"
        return _fig_to_data_uri(fig), title

    raise ValueError("Merge trees are currently supported for 1D and 2D scalar fields only.")


def _render_point_cloud_merge_tree(
    points: np.ndarray,
    *,
    direction: int,
    function: str,
    function_dim: int,
    n_neighbors: int,
) -> tuple[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..merge_tree import get_merge_tree_graph_point_cloud, plot_merge_tree_graph_point_cloud

    tree, coords_2d, values = get_merge_tree_graph_point_cloud(
        points,
        function=function,
        function_dim=function_dim,
        n_neighbors=n_neighbors,
        direction=direction,
    )
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    plot_merge_tree_graph_point_cloud(
        tree,
        coords_2d,
        values,
        overlay=False,
        ax=ax,
        direction=direction,
    )
    title = f"Point-Cloud Merge Tree ({function})"
    return _fig_to_data_uri(fig), title


def _top_pairs(diagram: list[tuple[int, tuple[float, float]]], limit: int = 8) -> list[Dict[str, Any]]:
    rows = []
    for dim, (birth, death) in diagram:
        persistence = float("inf") if not np.isfinite(death) else float(death - birth)
        rows.append(
            {
                "dimension": dim,
                "birth": float(birth),
                "death": None if not np.isfinite(death) else float(death),
                "persistence": None if not np.isfinite(persistence) else persistence,
            }
        )
    rows.sort(key=lambda row: row["persistence"] if row["persistence"] is not None else float("inf"), reverse=True)
    return rows[:limit]


def _run_analysis(defaults: FormDefaults) -> Dict[str, Any]:
    data, source_label = _load_array_from_source(
        defaults.source_mode,
        defaults.data_kind,
        defaults.delimiter,
        defaults.skip_header,
    )
    result = analyze(data, kind=defaults.data_kind, max_dim=defaults.max_dim)

    run_id = uuid4().hex[:12]
    run_dir = ARTIFACT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "source_label": source_label,
        "shape": list(data.shape),
        "summary": result.summary(),
        "persistence_image": _render_persistence_diagram(result.diagram),
        "top_pairs": _top_pairs(result.diagram),
        "mapper": None,
        "merge_tree": None,
    }

    if defaults.compute_mapper and defaults.data_kind == "point_cloud":
        from ..mapper import (
            mapper_graph_has_nodes,
            plot_mapper_graph,
            run_mapper_pipeline,
            save_mapper_graph_json,
            summarize_mapper_graph,
            visualize_mapper_graph,
        )

        mapper_result = run_mapper_pipeline(
            data,
            n_cubes=defaults.n_cubes,
            overlap=defaults.overlap,
            lens=defaults.lens,
            lens_components=defaults.lens_components,
            lens_neighbors=defaults.lens_neighbors,
            scale=defaults.scale,
            projection=defaults.projection,
            projection_components=defaults.projection_components,
            clusterer=defaults.clusterer,
            dbscan_eps=defaults.dbscan_eps,
            dbscan_min_samples=defaults.dbscan_min_samples,
            n_clusters=defaults.n_clusters,
        )
        mapper_summary = summarize_mapper_graph(mapper_result.graph)
        payload["mapper"] = {
            "summary": mapper_summary,
            "image": None,
            "json_url": None,
            "html_url": None,
            "warning": None,
            "suggestions": [],
        }
        if mapper_graph_has_nodes(mapper_result.graph):
            mapper_png = plot_mapper_graph(mapper_result)
            mapper_json_path = save_mapper_graph_json(mapper_result, str(run_dir / "mapper_graph.json"))
            mapper_html_path = run_dir / "mapper_graph.html"
            visualize_mapper_graph(mapper_result.mapper, mapper_result.graph, path_html=str(mapper_html_path))
            payload["mapper"].update(
                {
                    "image": _fig_to_data_uri(mapper_png.figure),
                    "json_url": url_for("artifact_file", run_id=run_id, filename=Path(mapper_json_path).name),
                    "html_url": url_for("artifact_file", run_id=run_id, filename=mapper_html_path.name),
                }
            )
        else:
            payload["mapper"].update(
                {
                    "warning": (
                        "The Mapper run completed, but the current parameters produced 0 nodes. "
                        "Persistence is still valid; only the Mapper view needs retuning."
                    ),
                    "suggestions": [
                        f"Increase DBSCAN eps above {defaults.dbscan_eps:.2f}",
                        f"Lower min_samples below {defaults.dbscan_min_samples}",
                        f"Increase overlap above {defaults.overlap:.2f}",
                        "Try clusterer = kmeans or agglomerative",
                        "Use a lower-dimensional PCA lens or fewer intervals",
                    ],
                }
            )

    if defaults.compute_merge_tree and defaults.data_kind == "scalar_field":
        merge_image, merge_title = _render_merge_tree(np.asarray(data), defaults.merge_direction)
        payload["merge_tree"] = {
            "title": merge_title,
            "image": merge_image,
        }
    elif defaults.compute_merge_tree and defaults.data_kind == "point_cloud":
        merge_image, merge_title = _render_point_cloud_merge_tree(
            np.asarray(data),
            direction=defaults.merge_direction,
            function=defaults.lens,
            function_dim=max(0, defaults.lens_components - 1),
            n_neighbors=defaults.lens_neighbors,
        )
        payload["merge_tree"] = {
            "title": merge_title,
            "image": merge_image,
        }

    return payload


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    @app.get("/")
    def index():
        defaults = FormDefaults()
        return render_template("index.html", defaults=asdict(defaults), results=None, error=None)

    @app.post("/")
    def compute():
        defaults = _read_defaults()
        try:
            results = _run_analysis(defaults)
            return render_template("index.html", defaults=asdict(defaults), results=results, error=None)
        except Exception as exc:
            return render_template("index.html", defaults=asdict(defaults), results=None, error=str(exc))

    @app.get("/artifacts/<run_id>/<path:filename>")
    def artifact_file(run_id: str, filename: str):
        return send_from_directory(ARTIFACT_ROOT / run_id, filename)

    return app


def main() -> None:
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
