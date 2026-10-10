#!/usr/bin/env python3
"""
Render a provenance-safe sample-69 MT cost spatial diagnostic.

Selection is frozen BEFORE rendering:
    top 8 unmatched branches per method and per side
    ranked only by exact TTK nonmatching_cost.

Panels:
    row 1: CNN   | GT deletions | SR insertions
    row 2: UV    | GT deletions | SR insertions
    row 3: F1    | GT deletions | SR insertions

Important:
    - circles/x marks are the two critical endpoints of a persistence branch.
    - NO line is drawn between endpoints because that would falsely imply a
      spatial superarc trajectory.
    - positions come from the validated numerical -> authoritative C-order
      VertexId bridge, not nearest-neighbor heuristics.

Also renders a separate stacked squared-objective component chart:
    relabel + GT deletion + SR insertion.

Expected source arrays:
    data_out_fixed/wind_mrhr_cnn/{idx,dataGT,dataSR}.npy
    data_out/wind_finetune_candidateUV_expanded2688/{idx,dataGT,dataSR}.npy
    data_out/wind_finetune_candidateF_grad_E2_low_expanded2688/{idx,dataGT,dataSR}.npy
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

SAMPLE = 69
PATCH = 160
TOPK = 8

ROOT = Path.home() / "PhIRE"
AUDIT = Path.home() / "phire_runtime_audit_20260809_221548"
W22 = Path(os.environ.get("W22", AUDIT / "recompute_pd_w22"))

SAMPLE_BASE = (
    W22
    / "corrected_pd_mt"
    / "discordance_visuals"
    / "sample_069"
)
ATTR = (
    SAMPLE_BASE
    / "ttk_matching_host96"
    / "cost_attribution"
)
FIG = SAMPLE_BASE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

METHOD_DIRS = {
    "cnn": ROOT / "data_out_fixed/wind_mrhr_cnn",
    "uv": ROOT / "data_out/wind_finetune_candidateUV_expanded2688",
    "f1": ROOT / "data_out/wind_finetune_candidateF_grad_E2_low_expanded2688",
}

DISPLAY = {
    "cnn": "CNN",
    "uv": "Matched UV control",
    "f1": "Candidate F1",
}


def read_csv(path: Path) -> List[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def find_sample_position(idx: np.ndarray, n_samples: int) -> int:
    flat = np.asarray(idx).reshape(-1)

    hits = np.where(flat.astype(np.int64) == SAMPLE)[0]
    if len(hits) == 1:
        pos = int(hits[0])
        if pos >= n_samples:
            raise RuntimeError(
                f"idx says sample {SAMPLE} is position {pos}, "
                f"but data has only {n_samples} samples"
            )
        return pos

    if len(hits) > 1:
        raise RuntimeError(
            f"idx contains sample {SAMPLE} more than once: {hits.tolist()}"
        )

    # Fallback only if the array is clearly directly indexed by sample number.
    if n_samples > SAMPLE:
        print(
            f"WARNING: sample {SAMPLE} not found in idx.npy; "
            f"using direct position {SAMPLE}"
        )
        return SAMPLE

    raise RuntimeError(f"Could not locate sample {SAMPLE}")


def to_speed(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x)

    if a.ndim == 2:
        speed = a.astype(np.float64, copy=False)
    elif a.ndim == 3 and a.shape[-1] == 2:
        speed = np.sqrt(
            a[..., 0].astype(np.float64) ** 2
            + a[..., 1].astype(np.float64) ** 2
        )
    else:
        raise RuntimeError(f"Unexpected field shape {a.shape}")

    if speed.shape[0] < PATCH or speed.shape[1] < PATCH:
        raise RuntimeError(
            f"Field too small for {PATCH}x{PATCH} crop: {speed.shape}"
        )

    out = np.ascontiguousarray(speed[:PATCH, :PATCH])

    if not np.isfinite(out).all():
        raise RuntimeError("Non-finite values in speed field")

    return out


def load_method(directory: Path) -> Tuple[np.ndarray, np.ndarray]:
    idx_p = directory / "idx.npy"
    gt_p = directory / "dataGT.npy"
    sr_p = directory / "dataSR.npy"

    for p in (idx_p, gt_p, sr_p):
        if not p.is_file():
            raise FileNotFoundError(p)

    idx = np.load(idx_p)
    gt = np.load(gt_p, mmap_mode="r")
    sr = np.load(sr_p, mmap_mode="r")

    if len(gt) != len(sr):
        raise RuntimeError(
            f"{directory}: GT/SR sample counts differ {len(gt)} vs {len(sr)}"
        )

    pos = find_sample_position(idx, len(gt))
    return to_speed(gt[pos]), to_speed(sr[pos])


def validate_spatial_audit() -> None:
    p = ATTR / "sample69_unmatched_spatial_mapping_validation.csv"
    rows = read_csv(p)

    if len(rows) != 3:
        raise RuntimeError(f"{p}: expected 3 rows")

    for r in rows:
        if int(r["all_node_ids_resolve_to_original_NodeId_row"]) != 1:
            raise RuntimeError(f"{r['method']}: node mapping validation failed")
        if int(r["all_origin_ids_resolve_to_original_NodeId_row"]) != 1:
            raise RuntimeError(f"{r['method']}: origin mapping validation failed")


def select_top(method: str, side: str) -> Tuple[List[dict], float, float]:
    p = ATTR / f"{method}_unmatched_branches_spatial.csv"
    rows = [r for r in read_csv(p) if r["side"] == side]

    rows.sort(key=lambda r: float(r["nonmatching_cost"]), reverse=True)

    if len(rows) < TOPK:
        raise RuntimeError(
            f"{method}/{side}: only {len(rows)} rows, need {TOPK}"
        )

    selected = rows[:TOPK]
    side_total = math.fsum(float(r["nonmatching_cost"]) for r in rows)
    selected_total = math.fsum(
        float(r["nonmatching_cost"]) for r in selected
    )

    for r in selected:
        for c in (
            "node_corder_x",
            "node_corder_y",
            "origin_corder_x",
            "origin_corder_y",
        ):
            v = int(round(float(r[c])))
            if not (0 <= v < PATCH):
                raise RuntimeError(
                    f"{method}/{side}: coordinate {c}={v} outside patch"
                )

    return selected, selected_total, side_total


def overlay_endpoints(ax, selected: List[dict]) -> None:
    # Separate markers distinguish the persistence branch's two critical
    # endpoints. They are intentionally NOT connected with a line.
    nx = [float(r["node_corder_x"]) for r in selected]
    ny = [float(r["node_corder_y"]) for r in selected]
    ox = [float(r["origin_corder_x"]) for r in selected]
    oy = [float(r["origin_corder_y"]) for r in selected]

    ax.scatter(nx, ny, marker="x", s=55, label="branch node")
    ax.scatter(ox, oy, marker="o", s=32, label="paired origin")

    # Label only the node endpoint with the exact cost rank.
    for rank, r in enumerate(selected, 1):
        ax.text(
            float(r["node_corder_x"]) + 1.2,
            float(r["node_corder_y"]) + 1.2,
            str(rank),
            fontsize=7,
        )


def main() -> int:
    validate_spatial_audit()

    fields: Dict[str, Dict[str, np.ndarray]] = {}

    for method, directory in METHOD_DIRS.items():
        gt, sr = load_method(directory)
        fields[method] = {"gt": gt, "sr": sr}

    # Authoritative GT identity is a hard prerequisite.
    gt_ref = fields["cnn"]["gt"]

    for method in ("uv", "f1"):
        gt = fields[method]["gt"]
        if not np.array_equal(gt_ref, gt):
            maxdiff = float(np.max(np.abs(gt_ref - gt)))
            raise RuntimeError(
                f"GT mismatch CNN vs {method}: maxdiff={maxdiff}"
            )

    print("AUTHORITATIVE GT IDENTITY: PASS")
    print("CNN == UV == F1 GT exactly for sample 69")

    all_fields = [
        gt_ref,
        fields["cnn"]["sr"],
        fields["uv"]["sr"],
        fields["f1"]["sr"],
    ]
    vmin = min(float(x.min()) for x in all_fields)
    vmax = max(float(x.max()) for x in all_fields)

    selection_rows = []
    selection_summary = {}

    for method in ("cnn", "uv", "f1"):
        for side in ("1_delete", "2_insert"):
            selected, selected_total, side_total = select_top(method, side)
            pct = 100.0 * selected_total / side_total

            selection_summary[(method, side)] = (
                selected, selected_total, side_total, pct
            )

            for rank, r in enumerate(selected, 1):
                selection_rows.append({
                    "method": method,
                    "side": side,
                    "rank_within_side": rank,
                    "nonmatching_cost": float(r["nonmatching_cost"]),
                    "node_id": int(r["node_id"]),
                    "origin_id": int(r["origin_id"]),
                    "node_corder_x": int(float(r["node_corder_x"])),
                    "node_corder_y": int(float(r["node_corder_y"])),
                    "origin_corder_x": int(float(r["origin_corder_x"])),
                    "origin_corder_y": int(float(r["origin_corder_y"])),
                })

    selection_csv = ATTR / "sample69_spatial_diagnostic_selection_top8.csv"
    with selection_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=list(selection_rows[0].keys())
        )
        w.writeheader()
        w.writerows(selection_rows)

    # ------------------------------------------------------------------
    # Figure 1: spatial diagnostic
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        3, 2, figsize=(12, 17), constrained_layout=True
    )

    for row, method in enumerate(("cnn", "uv", "f1")):
        delete_sel, delete_top, delete_total, delete_pct = (
            selection_summary[(method, "1_delete")]
        )
        insert_sel, insert_top, insert_total, insert_pct = (
            selection_summary[(method, "2_insert")]
        )

        ax = axes[row, 0]
        im = ax.imshow(gt_ref, vmin=vmin, vmax=vmax)
        overlay_endpoints(ax, delete_sel)
        ax.set_title(
            f"{DISPLAY[method]} — GT branches deleted\n"
            f"top {TOPK}: {delete_top:.3f}/{delete_total:.3f} "
            f"({delete_pct:.1f}% of deletion cost)"
        )
        ax.set_xlim(-0.5, PATCH - 0.5)
        ax.set_ylim(PATCH - 0.5, -0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        ax = axes[row, 1]
        im = ax.imshow(fields[method]["sr"], vmin=vmin, vmax=vmax)
        overlay_endpoints(ax, insert_sel)
        ax.set_title(
            f"{DISPLAY[method]} — SR branches inserted\n"
            f"top {TOPK}: {insert_top:.3f}/{insert_total:.3f} "
            f"({insert_pct:.1f}% of insertion cost)"
        )
        ax.set_xlim(-0.5, PATCH - 0.5)
        ax.set_ylim(PATCH - 0.5, -0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    # One legend is enough; it describes endpoint marker semantics.
    axes[0, 0].legend(loc="upper right")

    cbar = fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        shrink=0.72,
        pad=0.02,
    )
    cbar.set_label("Wind speed")

    fig.suptitle(
        "Sample 69 — exact TTK unmatched-branch spatial diagnostic\n"
        "Markers show persistence-branch endpoints; no spatial arc path is implied",
        fontsize=14,
    )

    spatial_png = FIG / "sample_069_mt_unmatched_branch_spatial_diagnostic.png"
    spatial_pdf = FIG / "sample_069_mt_unmatched_branch_spatial_diagnostic.pdf"
    fig.savefig(spatial_png, dpi=220)
    fig.savefig(spatial_pdf)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: exact squared-objective component chart
    # ------------------------------------------------------------------
    summary_rows = {}

    for method in ("cnn", "uv", "f1"):
        p = ATTR / f"{method}_cost_attribution_summary.csv"
        rr = read_csv(p)
        if len(rr) != 1:
            raise RuntimeError(f"{p}: expected one row")
        summary_rows[method] = rr[0]

    labels = [DISPLAY[m] for m in ("cnn", "uv", "f1")]
    relabel = np.array([
        float(summary_rows[m]["relabel_cost_sum"])
        for m in ("cnn", "uv", "f1")
    ])
    deletion = np.array([
        float(summary_rows[m]["tree1_delete_cost_sum"])
        for m in ("cnn", "uv", "f1")
    ])
    insertion = np.array([
        float(summary_rows[m]["tree2_insert_cost_sum"])
        for m in ("cnn", "uv", "f1")
    ])

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.bar(x, relabel, label="real-real relabel")
    ax.bar(x, deletion, bottom=relabel, label="GT deletion")
    ax.bar(x, insertion, bottom=relabel + deletion, label="SR insertion")

    ax.set_xticks(x, labels)
    ax.set_ylabel("Squared TTK merge-tree objective")
    ax.set_title("Sample 69 — exact TTK cost decomposition")
    ax.legend()

    component_png = FIG / "sample_069_mt_cost_components.png"
    component_pdf = FIG / "sample_069_mt_cost_components.pdf"
    fig.savefig(component_png, dpi=220, bbox_inches="tight")
    fig.savefig(component_pdf, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Text summary
    # ------------------------------------------------------------------
    text = []
    text.append("SAMPLE-69 MT SPATIAL DIAGNOSTIC — FROZEN TOP-8 PER SIDE")
    text.append("=" * 96)
    text.append("Selection rule: top 8 by exact TTK nonmatching cost, before rendering.")
    text.append("")
    text.append("Authoritative GT identity: PASS")
    text.append("")

    for method in ("cnn", "uv", "f1"):
        text.append(DISPLAY[method])
        text.append("-" * 96)

        for side, name in (
            ("1_delete", "GT deletion"),
            ("2_insert", "SR insertion"),
        ):
            _, top_total, side_total, pct = selection_summary[(method, side)]
            text.append(
                f"{name:14s}: top8={top_total:.10f} / "
                f"total={side_total:.10f} ({pct:.3f}%)"
            )

        text.append("")

    text.extend([
        f"selection CSV: {selection_csv}",
        f"spatial PNG:   {spatial_png}",
        f"spatial PDF:   {spatial_pdf}",
        f"component PNG: {component_png}",
        f"component PDF: {component_pdf}",
        "",
        "IMPORTANT: endpoint markers are NOT spatial superarc trajectories.",
    ])

    summary_txt = ATTR / "sample69_spatial_diagnostic_summary.txt"
    summary_txt.write_text("\n".join(text) + "\n")

    print()
    print("\n".join(text))
    print()
    print("SAMPLE-69 MT SPATIAL DIAGNOSTIC: COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
