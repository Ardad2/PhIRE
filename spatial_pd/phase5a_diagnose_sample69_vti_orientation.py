#!/usr/bin/env python3
"""
Phase 5A Step 3A — exact orientation matrix.

Compare the newly generated canonical C-order sample-69 VTI fields against the
historical exact TTK-input port-2 VTI fields in BOTH raw and transposed
orientation.

No files are modified.
"""

from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_speed(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    r = vtk.vtkXMLImageDataReader()
    r.SetFileName(str(path))
    r.Update()
    img = r.GetOutput()
    dims = img.GetDimensions()
    arr = img.GetPointData().GetArray("wind_speed")
    if arr is None:
        raise RuntimeError(f"{path}: missing wind_speed")
    flat = np.asarray(vtk_to_numpy(arr))
    W, H, Z = dims
    if Z != 1:
        raise RuntimeError(f"{path}: unexpected dims {dims}")
    a = flat.reshape(H, W, order="C")
    return a, dims


def comp(a, b):
    if a.shape != b.shape:
        return {
            "same_shape": False,
            "equal": False,
            "max_abs_diff": None,
        }
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {
        "same_shape": True,
        "equal": bool(np.array_equal(a, b)),
        "max_abs_diff": float(d.max()) if d.size else 0.0,
        "mean_abs_diff": float(d.mean()) if d.size else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase5",
        default=str(Path.home() / "PhIRE/spatial_pd/phase5a_sample69_canonical_pd"),
    )
    ap.add_argument(
        "--phire",
        default=str(Path.home() / "PhIRE"),
    )
    ap.add_argument(
        "--out",
        default=str(Path.home() / "PhIRE/spatial_pd/phase5a_sample69_orientation_diagnostic"),
    )
    args = ap.parse_args()

    root = Path(args.phire).expanduser().resolve()
    phase5 = Path(args.phase5).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    vti = phase5 / "vti"

    def one(pattern):
        hits = sorted(vti.glob(pattern))
        if len(hits) != 1:
            raise RuntimeError(f"{pattern}: expected one hit, got {hits}")
        return hits[0]

    new_paths = {
        "GT": one("phase5_GT_s69_speed_p160_x0_y0.vti"),
        "CNN_SR": one("phase5_CNN_SR_s69_speed_p160_x0_y0.vti"),
        "UV_SR": one("phase5_UV_SR_s69_speed_p160_x0_y0.vti"),
        "F1_SR": one("phase5_F1_SR_s69_speed_p160_x0_y0.vti"),
    }

    old_paths = {
        "CNN_GT": root / "ttk_runs_fixed/cnn/mt/cnn_GT_s69_speed_p160_x0_y0_mt_port_2.vti",
        "CNN_SR": root / "ttk_runs_fixed/cnn/mt/cnn_SR_s69_speed_p160_x0_y0_mt_port_2.vti",
        "UV_GT": root / "ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_topology/mt/GT/candidateUV_expanded2688_GT_s69_speed_p160_x0_y0_mt_port_2.vti",
        "UV_SR": root / "ttk_runs_fixed/topology_finetuning/candidateUV_expanded2688_topology/mt/SR/candidateUV_expanded2688_SR_s69_speed_p160_x0_y0_mt_port_2.vti",
        "F1_GT": root / "ttk_runs_fixed/topology_finetuning/candidateF_grad_E2_low_expanded2688_topology/mt/GT/candidateF_grad_E2_low_expanded2688_GT_s69_speed_p160_x0_y0_mt_port_2.vti",
        "F1_SR": root / "ttk_runs_fixed/topology_finetuning/candidateF_grad_E2_low_expanded2688_topology/mt/SR/candidateF_grad_E2_low_expanded2688_SR_s69_speed_p160_x0_y0_mt_port_2.vti",
    }

    new = {}
    old = {}
    metadata = {}

    for k, p in new_paths.items():
        a, dims = read_speed(p)
        new[k] = a
        metadata[f"new_{k}"] = {
            "path": str(p),
            "sha256": sha256(p),
            "dims": list(dims),
            "dtype": str(a.dtype),
            "min": float(a.min()),
            "max": float(a.max()),
        }

    for k, p in old_paths.items():
        a, dims = read_speed(p)
        old[k] = a
        metadata[f"historical_{k}"] = {
            "path": str(p),
            "sha256": sha256(p),
            "dims": list(dims),
            "dtype": str(a.dtype),
            "min": float(a.min()),
            "max": float(a.max()),
        }

    tests = []

    def add(new_name, old_name):
        a = new[new_name]
        b = old[old_name]
        tests.append({
            "new": new_name,
            "historical": old_name,
            "raw": comp(a, b),
            "transpose": comp(a, b.T),
        })

    for gt_name in ("CNN_GT", "UV_GT", "F1_GT"):
        add("GT", gt_name)

    add("CNN_SR", "CNN_SR")
    add("UV_SR", "UV_SR")
    add("F1_SR", "F1_SR")

    # Historical GT cross-track relationships as an extra provenance matrix.
    hist_gt = {}
    for a_name in ("CNN_GT", "UV_GT", "F1_GT"):
        hist_gt[a_name] = {}
        for b_name in ("CNN_GT", "UV_GT", "F1_GT"):
            hist_gt[a_name][b_name] = {
                "raw": comp(old[a_name], old[b_name]),
                "transpose_b": comp(old[a_name], old[b_name].T),
            }

    report = {
        "sample": 69,
        "metadata": metadata,
        "new_vs_historical": tests,
        "historical_gt_matrix": hist_gt,
    }

    out_json = out / "phase5a_step3a_orientation_matrix.json"
    out_json.write_text(json.dumps(report, indent=2))

    print("===== PHASE 5A STEP 3A — EXACT ORIENTATION MATRIX =====")
    for t in tests:
        print()
        print(f'NEW {t["new"]} vs HISTORICAL {t["historical"]}')
        print(
            "  raw       :",
            "equal=", t["raw"]["equal"],
            "max_abs_diff=", t["raw"]["max_abs_diff"],
            "mean_abs_diff=", t["raw"].get("mean_abs_diff"),
        )
        print(
            "  transpose :",
            "equal=", t["transpose"]["equal"],
            "max_abs_diff=", t["transpose"]["max_abs_diff"],
            "mean_abs_diff=", t["transpose"].get("mean_abs_diff"),
        )

    print()
    print("===== HISTORICAL GT CROSS-TRACK MATRIX =====")
    for a_name in ("CNN_GT", "UV_GT", "F1_GT"):
        for b_name in ("CNN_GT", "UV_GT", "F1_GT"):
            x = hist_gt[a_name][b_name]
            print(
                f"{a_name:6s} vs {b_name:6s} | "
                f"raw={x['raw']['equal']} "
                f"transpose_b={x['transpose_b']['equal']}"
            )

    print()
    print("Wrote:", out_json)


if __name__ == "__main__":
    main()
