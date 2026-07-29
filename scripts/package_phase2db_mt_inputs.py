#!/usr/bin/env python3
"""
Package the 21 Phase 2D-B scalar VTI inputs for ParaView/TTK batch rendering.

Run on Spark from the PhIRE repository root:
    python3 scripts/package_phase2db_mt_inputs.py

Output:
    /tmp/phase2db_mt_batch_inputs.tar.gz
"""

from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path


REPO = Path.cwd().resolve()
STAGING = Path("/tmp/phase2db_mt_batch_inputs")
ARCHIVE = Path("/tmp/phase2db_mt_batch_inputs.tar.gz")

SPEED_RANGES = {
    1: [0.004631552681959958, 23.696232135200372],
    2: [0.0015429837994549844, 49.69148981604192],
    3: [0.004337943606259474, 24.774924928978656],
    5: [0.021458192169549858, 37.62227177383977],
}

FIGURES = {
    1: (120, ["GT", "cnn", "gan", "candidate_c", "f3_grad_crit", "f2_grad_levelset_e2", "uv_e2"]),
    2: (34, ["GT", "bicubic", "cnn", "gan"]),
    3: (119, ["GT", "cnn", "f3_grad_crit", "uv_e2"]),
    5: (30, ["GT", "cnn", "candidate_c", "f3_grad_crit", "f2_grad_levelset_e2", "uv_e2"]),
}

LABELS = {
    "GT": "Ground Truth",
    "bicubic": "Bicubic",
    "cnn": "CNN",
    "gan": "GAN",
    "candidate_c": "Candidate C",
    "f3_grad_crit": "F3: Grad+Crit",
    "f2_grad_levelset_e2": "F2: Grad+Levelset+E2",
    "uv_e2": "UV+E2",
}

PREFERRED_PATH_TOKEN = {
    "GT": "candidateC_expanded2688_vti",
    "cnn": "vtk_inputs",
    "gan": "vtk_inputs",
    "candidate_c": "candidateC_expanded2688_topology_vti",
    "f3_grad_crit": "candidateF_grad_crit_expanded2688_topology_vti",
    "f2_grad_levelset_e2": "candidateF_grad_levelset_E2_low_expanded2688_topology_vti",
    "uv_e2": "candidateUV_plus_E2_tf_lowlambda_expanded2688_topology_vti",
}

ALIASES = {
    "candidate_c": "candidateC_expanded2688",
    "f3_grad_crit": "candidateF_grad_crit_expanded2688",
    "f2_grad_levelset_e2": "candidateF_grad_levelset_E2_low_expanded2688",
    "uv_e2": "candidateUV_plus_E2_tf_lowlambda_expanded2688",
}


def scalar_filename(method_id: str, sample_idx: int) -> str:
    suffix = f"s{sample_idx}_speed_p160_x0_y0.vti"
    if method_id == "GT":
        return f"candidateC_expanded2688_GT_{suffix}"
    if method_id == "cnn":
        return f"cnn_SR_{suffix}"
    if method_id == "gan":
        return f"gan_SR_{suffix}"
    if method_id == "bicubic":
        return f"bicubic_SR_{suffix}_mt_port_2.vti"
    return f"{ALIASES[method_id]}_SR_{suffix}"


def source_vtu_path(method_id: str, sample_idx: int) -> str:
    suffix = f"s{sample_idx}_speed_p160_x0_y0_mt_port_1.vtu"
    if method_id == "GT":
        return (
            "ttk_runs_fixed/topology_finetuning/"
            "candidateC_expanded2688_topology/mt/GT/"
            f"candidateC_expanded2688_GT_{suffix}"
        )
    if method_id == "cnn":
        return f"ttk_runs_fixed/cnn/mt/cnn_SR_{suffix}"
    if method_id == "gan":
        return f"ttk_runs_fixed/gan/mt/gan_SR_{suffix}"
    if method_id == "bicubic":
        return f"ttk_runs_fixed/bicubic/mt/SR/bicubic_SR_{suffix}"
    alias = ALIASES[method_id]
    return (
        "ttk_runs_fixed/topology_finetuning/"
        f"{alias}_topology/mt/SR/{alias}_SR_{suffix}"
    )


def resolve_scalar_vti(method_id: str, sample_idx: int) -> Path:
    if method_id == "bicubic":
        path = REPO / (
            "ttk_runs_fixed/bicubic/mt/SR/"
            f"bicubic_SR_s{sample_idx}_speed_p160_x0_y0_mt_port_2.vti"
        )
        if not path.is_file():
            raise SystemExit(f"Missing Bicubic scalar source: {path}")
        return path

    filename = scalar_filename(method_id, sample_idx)
    candidates = []
    for path in REPO.rglob(filename):
        rel = path.relative_to(REPO).as_posix()
        if "superlevel_topology/" in rel:
            continue
        if rel.endswith("_mt_port_2.vti"):
            continue
        candidates.append(path)

    token = PREFERRED_PATH_TOKEN[method_id]
    preferred = [p for p in candidates if token in p.relative_to(REPO).as_posix()]
    selected_pool = preferred or candidates

    if len(selected_pool) != 1:
        detail = "\n".join(f"  - {p.relative_to(REPO)}" for p in selected_pool or candidates)
        raise SystemExit(
            f"Expected exactly one scalar VTI for figure input {method_id=} {sample_idx=} "
            f"filename={filename!r}; found {len(selected_pool)}.\n{detail}"
        )
    return selected_pool[0]


def main() -> None:
    if not (REPO / "ttk_runs_fixed").is_dir():
        raise SystemExit("Run this script from the PhIRE repository root.")

    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)

    rows = []
    for figure_id, (sample_idx, methods) in FIGURES.items():
        figure_dir = STAGING / "inputs" / f"figure_{figure_id:02d}"
        figure_dir.mkdir(parents=True, exist_ok=True)

        for method_id in methods:
            source = resolve_scalar_vti(method_id, sample_idx)
            copied_rel = Path("inputs") / f"figure_{figure_id:02d}" / f"{method_id}.vti"
            destination = STAGING / copied_rel
            shutil.copy2(source, destination)

            vtu_rel = source_vtu_path(method_id, sample_idx)
            vtu_abs = REPO / vtu_rel
            if not vtu_abs.is_file():
                raise SystemExit(f"Missing audited merge-tree arc source: {vtu_abs}")

            rows.append(
                {
                    "figure_id": figure_id,
                    "sample_idx": sample_idx,
                    "method_id": method_id,
                    "display_label": LABELS[method_id],
                    "source_vti_package_path": copied_rel.as_posix(),
                    "source_vti_repo_path": source.relative_to(REPO).as_posix(),
                    "source_vtu_path": vtu_rel,
                    "speed_vmin": SPEED_RANGES[figure_id][0],
                    "speed_vmax": SPEED_RANGES[figure_id][1],
                    "persistence_threshold": 11.0,
                    "arc_sampling": 10,
                    "arc_line_size": 3,
                    "point_size": 5,
                    "camera_or_view_id": f"figure_{figure_id:02d}_view_v1",
                }
            )
            print(
                f"figure={figure_id} sample={sample_idx:3d} method={method_id:<24} "
                f"source={source.relative_to(REPO)}"
            )

    manifest = {
        "schema_version": 1,
        "repository": "Ardad2/PhIRE",
        "panel_count": len(rows),
        "panels": rows,
    }
    (STAGING / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    if ARCHIVE.exists():
        ARCHIVE.unlink()
    with tarfile.open(ARCHIVE, "w:gz") as tar:
        tar.add(STAGING, arcname=STAGING.name)

    assert len(rows) == 21, len(rows)
    print()
    print(f"PACKAGED {len(rows)} PANEL INPUTS")
    print(f"Archive: {ARCHIVE}")
    print(f"Bytes:   {ARCHIVE.stat().st_size}")


if __name__ == "__main__":
    main()
