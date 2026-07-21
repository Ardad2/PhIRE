#!/usr/bin/env python3
"""
Lean qualitative-panel generator for the Candidate C paper story.

Shows only:
  GT | CNN | GAN | UV-expanded-2688 | Candidate C-expanded-2688

Second row:
  |CNN-GT| | |GAN-GT| | |UV-GT| | |C-GT| | |C-UV|

Run from the PhIRE scripts directory:

  cd /home/adadhwal/PhIRE/scripts
  PYTHONNOUSERSITE=1 /usr/bin/python3 generate_visual_inspection_panels_lean.py --auto

Or choose samples explicitly:

  PYTHONNOUSERSITE=1 /usr/bin/python3 generate_visual_inspection_panels_lean.py \
    --samples 6,18,20,25,62,68,79,92,162,163

Outputs:
  ttk_runs_fixed/visual_inspection_lean/index.html
  ttk_runs_fixed/visual_inspection_lean/selected_samples.csv
  ttk_runs_fixed/visual_inspection_lean/panels_full/*.png
  ttk_runs_fixed/visual_inspection_lean/panels_crop/*.png
"""

from __future__ import annotations

import argparse
import csv
import html
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def repo_root() -> Path:
    here = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    candidates = [
        here.parent if here.name == "scripts" else here,
        cwd.parent if cwd.name == "scripts" else cwd,
        here,
        cwd,
    ]
    for root in candidates:
        if (root / "ttk_runs_fixed").exists() or (root / "data_out").exists():
            return root
    raise FileNotFoundError("Could not locate the PhIRE repository root.")


ROOT = repo_root()
OUTDIR = ROOT / "ttk_runs_fixed" / "visual_inspection_lean"
FULL_DIR = OUTDIR / "panels_full"
CROP_DIR = OUTDIR / "panels_crop"


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


CNN_DIR = first_existing(
    ROOT / "data_out_fixed" / "wind_mrhr_cnn",
    ROOT / "data_out" / "wind_mrhr_cnn",
)
GAN_DIR = first_existing(
    ROOT / "data_out_fixed" / "wind_mrhr_gan",
    ROOT / "data_out" / "wind_mrhr_gan",
)
UV_DIR = ROOT / "data_out" / "wind_finetune_candidateUV_expanded2688"
C_DIR = ROOT / "data_out" / "wind_finetune_candidateC_expanded2688"

UV_TOPOLOGY = (
    ROOT / "ttk_runs_fixed" / "topology_finetuning"
    / "candidateUV_expanded2688_topology"
    / "candidateUV_expanded2688_topology_comparison.csv"
)
C_TOPOLOGY = (
    ROOT / "ttk_runs_fixed" / "topology_finetuning"
    / "candidateC_expanded2688_topology"
    / "candidateC_expanded2688_topology_comparison.csv"
)

FALLBACK_SAMPLES = [6, 18, 20, 25, 62, 68, 79, 92, 162, 163]


@dataclass
class Arrays:
    gt: np.ndarray
    cnn: np.ndarray
    gan: np.ndarray
    uv: np.ndarray
    cand_c: np.ndarray
    base_pos: dict[int, int]
    uv_pos: dict[int, int]
    c_pos: dict[int, int]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sample_id(row: dict[str, str]) -> int:
    for key in ("sample_idx", "sample_id", "sample", "id"):
        value = str(row.get(key, "")).strip()
        if value:
            return int(float(value))
    raise ValueError("No sample identifier found.")


def numeric(row: dict[str, str], aliases: Iterable[str]) -> float:
    for key in aliases:
        if key in row and str(row[key]).strip():
            try:
                return float(row[key])
            except ValueError:
                pass
    return float("nan")


def candidate_metric(row: dict[str, str], stem: str, method: str) -> float:
    aliases = [
        f"{stem}_{method}",
        f"{stem}_{method.lower()}",
        f"{stem}_candidate",
    ]
    return numeric(row, aliases)


def load_topology_rows(path: Path, method: str) -> dict[int, dict[str, float]]:
    output: dict[int, dict[str, float]] = {}
    for row in read_csv(path):
        try:
            sid = sample_id(row)
        except Exception:
            continue
        output[sid] = {
            "pd_cnn": numeric(row, ["pd_distance_cnn"]),
            "pd_gan": numeric(row, ["pd_distance_gan"]),
            "mt_cnn": numeric(row, ["mt_distance_cnn"]),
            "mt_gan": numeric(row, ["mt_distance_gan"]),
            "pd_method": candidate_metric(row, "pd_distance", method),
            "mt_method": candidate_metric(row, "mt_distance", method),
        }
    return output


def build_topology_table() -> dict[int, dict[str, float]]:
    c_rows = load_topology_rows(C_TOPOLOGY, "candidateC_expanded2688")
    uv_rows = load_topology_rows(UV_TOPOLOGY, "candidateUV_expanded2688")
    all_ids = sorted(set(c_rows) | set(uv_rows))
    table: dict[int, dict[str, float]] = {}
    for sid in all_ids:
        c = c_rows.get(sid, {})
        uv = uv_rows.get(sid, {})
        table[sid] = {
            "pd_cnn": c.get("pd_cnn", uv.get("pd_cnn", np.nan)),
            "pd_gan": c.get("pd_gan", uv.get("pd_gan", np.nan)),
            "mt_cnn": c.get("mt_cnn", uv.get("mt_cnn", np.nan)),
            "mt_gan": c.get("mt_gan", uv.get("mt_gan", np.nan)),
            "pd_uv": uv.get("pd_method", np.nan),
            "mt_uv": uv.get("mt_method", np.nan),
            "pd_c": c.get("pd_method", np.nan),
            "mt_c": c.get("mt_method", np.nan),
        }
    return table


def finite(*values: float) -> bool:
    return all(np.isfinite(value) for value in values)


def add_ranked(
    selected: list[tuple[int, str]],
    candidates: list[tuple[float, int]],
    label: str,
    count: int,
) -> None:
    present = {sid for sid, _ in selected}
    for _, sid in sorted(candidates, reverse=True):
        if sid in present:
            continue
        selected.append((sid, label))
        present.add(sid)
        if sum(1 for _, category in selected if category == label) >= count:
            return


def auto_select(table: dict[int, dict[str, float]], per_group: int = 2) -> list[tuple[int, str]]:
    if not table:
        return [(sid, "fallback curated sample") for sid in FALLBACK_SAMPLES]

    selected: list[tuple[int, str]] = []
    strong_pd: list[tuple[float, int]] = []
    pd_better_mt_worse: list[tuple[float, int]] = []
    recovered_mt_gan: list[tuple[float, int]] = []
    c_beats_uv_pd: list[tuple[float, int]] = []
    topology_cnn_controls: list[tuple[float, int]] = []

    for sid, row in table.items():
        pd_cnn, pd_gan, pd_uv, pd_c = row["pd_cnn"], row["pd_gan"], row["pd_uv"], row["pd_c"]
        mt_cnn, mt_gan, mt_uv, mt_c = row["mt_cnn"], row["mt_gan"], row["mt_uv"], row["mt_c"]

        if finite(pd_cnn, pd_c):
            gain_cnn = pd_cnn - pd_c
            if finite(pd_uv):
                gain_uv = pd_uv - pd_c
                strong_pd.append((min(gain_cnn, gain_uv), sid))
                c_beats_uv_pd.append((gain_uv, sid))
            else:
                strong_pd.append((gain_cnn, sid))

        if finite(pd_cnn, pd_c, mt_cnn, mt_c) and pd_c < pd_cnn and mt_c > mt_cnn:
            pd_better_mt_worse.append(((pd_cnn - pd_c) + (mt_c - mt_cnn), sid))

        if finite(mt_cnn, mt_gan, mt_c) and mt_gan < mt_cnn and mt_c < min(mt_cnn, mt_gan):
            recovered_mt_gan.append((min(mt_cnn, mt_gan) - mt_c, sid))

        if finite(pd_cnn, pd_gan, mt_cnn, mt_gan) and pd_cnn < pd_gan and mt_cnn < mt_gan:
            topology_cnn_controls.append(((pd_gan - pd_cnn) + (mt_gan - mt_cnn), sid))

    add_ranked(selected, strong_pd, "largest Candidate C PD gain", per_group)
    add_ranked(selected, c_beats_uv_pd, "Candidate C strongly beats UV on PD", per_group)
    add_ranked(selected, pd_better_mt_worse, "PD improves while MT worsens", per_group)
    add_ranked(selected, recovered_mt_gan, "Candidate C recovers an MT-GAN case", per_group)
    add_ranked(selected, topology_cnn_controls, "rare CNN topology control", per_group)

    if len(selected) < 6:
        present = {sid for sid, _ in selected}
        for sid in FALLBACK_SAMPLES:
            if sid not in present:
                selected.append((sid, "fallback curated sample"))
                present.add(sid)
            if len(selected) >= 10:
                break

    return selected[:10]


def load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing required array: {path}")
    return np.load(path, mmap_mode="r")


def index_map(directory: Path, length: int) -> dict[int, int]:
    path = directory / "idx.npy"
    idx = np.load(path) if path.exists() else np.arange(length)
    return {int(value): pos for pos, value in enumerate(idx.tolist())}


def load_arrays() -> Arrays:
    gt = load_array(CNN_DIR / "dataGT.npy")
    cnn = load_array(CNN_DIR / "dataSR.npy")
    gan = load_array(GAN_DIR / "dataSR.npy")
    uv = load_array(UV_DIR / "dataSR.npy")
    cand_c = load_array(C_DIR / "dataSR.npy")
    return Arrays(
        gt=gt,
        cnn=cnn,
        gan=gan,
        uv=uv,
        cand_c=cand_c,
        base_pos=index_map(CNN_DIR, len(gt)),
        uv_pos=index_map(UV_DIR, len(uv)),
        c_pos=index_map(C_DIR, len(cand_c)),
    )


def speed(field: np.ndarray) -> np.ndarray:
    field = np.asarray(field)
    if field.ndim == 3 and field.shape[-1] == 2:
        return np.sqrt(np.square(field[..., 0]) + np.square(field[..., 1]))
    if field.ndim == 3 and field.shape[-1] == 1:
        return field[..., 0]
    if field.ndim == 2:
        return field
    raise ValueError(f"Unexpected field shape: {field.shape}")


def metric_text(row: dict[str, float]) -> str:
    def value(key: str) -> str:
        v = row.get(key, np.nan)
        return f"{v:.3f}" if np.isfinite(v) else "—"
    return (
        "PD ↓  CNN {0} | GAN {1} | UV {2} | C {3}\n"
        "MT ↓  CNN {4} | GAN {5} | UV {6} | C {7}"
    ).format(
        value("pd_cnn"), value("pd_gan"), value("pd_uv"), value("pd_c"),
        value("mt_cnn"), value("mt_gan"), value("mt_uv"), value("mt_c"),
    )


def crop_field(field: np.ndarray, crop: tuple[int, int, int, int] | None) -> np.ndarray:
    if crop is None:
        return field
    y0, y1, x0, x1 = crop
    return field[y0:y1, x0:x1]


def make_panel(
    sid: int,
    category: str,
    arrays: Arrays,
    topo: dict[str, float],
    crop: tuple[int, int, int, int] | None,
    output: Path,
) -> None:
    for mapping, name in [
        (arrays.base_pos, "baseline"),
        (arrays.uv_pos, "UV-2688"),
        (arrays.c_pos, "C-2688"),
    ]:
        if sid not in mapping:
            raise KeyError(f"Sample {sid} is missing from {name} idx.npy")

    i = arrays.base_pos[sid]
    iu = arrays.uv_pos[sid]
    ic = arrays.c_pos[sid]

    fields = {
        "GT": crop_field(speed(arrays.gt[i]), crop),
        "CNN": crop_field(speed(arrays.cnn[i]), crop),
        "GAN": crop_field(speed(arrays.gan[i]), crop),
        "UV-2688": crop_field(speed(arrays.uv[iu]), crop),
        "C-2688": crop_field(speed(arrays.cand_c[ic]), crop),
    }

    gt = fields["GT"]
    errors = {
        "|CNN-GT|": np.abs(fields["CNN"] - gt),
        "|GAN-GT|": np.abs(fields["GAN"] - gt),
        "|UV-GT|": np.abs(fields["UV-2688"] - gt),
        "|C-GT|": np.abs(fields["C-2688"] - gt),
        "|C-UV|": np.abs(fields["C-2688"] - fields["UV-2688"]),
    }

    speed_stack = np.stack(list(fields.values()))
    vmin = float(np.nanpercentile(speed_stack, 1.0))
    vmax = float(np.nanpercentile(speed_stack, 99.0))
    error_stack = np.stack(list(errors.values()))
    error_max = float(np.nanpercentile(error_stack, 99.0))
    if not np.isfinite(error_max) or error_max <= 0:
        error_max = 1.0

    fig, axes = plt.subplots(2, 5, figsize=(15.5, 6.2), constrained_layout=True)

    image = None
    for ax, (label, field) in zip(axes[0], fields.items()):
        image = ax.imshow(field, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(label, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    error_image = None
    for ax, (label, field) in zip(axes[1], errors.items()):
        error_image = ax.imshow(field, origin="lower", vmin=0.0, vmax=error_max)
        ax.set_title(label, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    view_name = "full field" if crop is None else f"crop y={crop[0]}:{crop[1]}, x={crop[2]}:{crop[3]}"
    fig.suptitle(
        f"Sample {sid} — {category} — {view_name}\n{metric_text(topo)}",
        fontsize=12,
    )
    if image is not None:
        fig.colorbar(image, ax=axes[0, :], shrink=0.80, pad=0.01, label="Wind speed")
    if error_image is not None:
        fig.colorbar(error_image, ax=axes[1, :], shrink=0.80, pad=0.01, label="Absolute difference")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_manifest(rows: list[dict[str, object]]) -> None:
    path = OUTDIR / "selected_samples.csv"
    fields = [
        "sample_idx", "category",
        "pd_cnn", "pd_gan", "pd_uv", "pd_c",
        "mt_cnn", "mt_gan", "mt_uv", "mt_c",
        "full_panel", "crop_panel",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_index(rows: list[dict[str, object]]) -> None:
    cards = []
    for row in rows:
        sid = int(row["sample_idx"])
        category = html.escape(str(row["category"]))
        full = html.escape(str(row["full_panel"]))
        crop = html.escape(str(row["crop_panel"]))
        cards.append(
            f"""
            <section class="card">
              <h2>Sample {sid}: {category}</h2>
              <p><a href="{full}">Full field</a> · <a href="{crop}">160×160 crop</a></p>
              <img src="{crop}" alt="Sample {sid} crop panel">
            </section>
            """
        )

    content = "\n".join(cards)
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Lean Candidate C visual inspection</title>
<style>
body {{ font-family: sans-serif; margin: 24px; background: #f7f7f7; }}
.card {{ background: white; padding: 16px; margin-bottom: 24px; border: 1px solid #ddd; }}
img {{ width: 100%; height: auto; display: block; }}
h1, h2 {{ margin-top: 0; }}
</style>
</head>
<body>
<h1>Lean Candidate C visual inspection</h1>
<p>GT, CNN, GAN, UV-expanded-2688, and Candidate C-expanded-2688 only.</p>
{content}
</body>
</html>
"""
    (OUTDIR / "index.html").write_text(document, encoding="utf-8")


def parse_samples(text: str) -> list[int]:
    return sorted({int(value.strip()) for value in text.split(",") if value.strip()})


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--auto", action="store_true", help="Automatically choose diverse, informative samples.")
    group.add_argument("--samples", default="", help="Comma-separated sample IDs.")
    parser.add_argument("--per-group", type=int, default=2, help="Automatic selections per category (default: 2).")
    parser.add_argument("--crop-size", type=int, default=160, help="Top-left square crop size (default: 160).")
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    CROP_DIR.mkdir(parents=True, exist_ok=True)

    topology = build_topology_table()
    if args.auto:
        selected = auto_select(topology, per_group=max(1, args.per_group))
    else:
        selected = [(sid, "manually selected") for sid in parse_samples(args.samples)]

    if not selected:
        raise RuntimeError("No samples selected.")

    print("Selected samples:")
    for sid, category in selected:
        print(f"  {sid}: {category}")

    arrays = load_arrays()
    manifest: list[dict[str, object]] = []
    crop_size = max(1, args.crop_size)
    crop = (0, crop_size, 0, crop_size)

    for sid, category in selected:
        topo = topology.get(sid, {})
        full_path = FULL_DIR / f"sample_{sid:03d}_full.png"
        crop_path = CROP_DIR / f"sample_{sid:03d}_crop.png"
        make_panel(sid, category, arrays, topo, None, full_path)
        make_panel(sid, category, arrays, topo, crop, crop_path)

        row: dict[str, object] = {
            "sample_idx": sid,
            "category": category,
            "full_panel": full_path.relative_to(OUTDIR).as_posix(),
            "crop_panel": crop_path.relative_to(OUTDIR).as_posix(),
        }
        row.update(topo)
        manifest.append(row)

    write_manifest(manifest)
    write_index(manifest)

    print(f"Wrote {len(manifest)} full panels to {FULL_DIR}")
    print(f"Wrote {len(manifest)} crop panels to {CROP_DIR}")
    print(f"Wrote {OUTDIR / 'selected_samples.csv'}")
    print(f"Wrote {OUTDIR / 'index.html'}")


if __name__ == "__main__":
    main()
