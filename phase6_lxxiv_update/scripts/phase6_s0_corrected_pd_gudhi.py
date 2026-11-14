#!/usr/bin/env python3
"""Phase 6, single-sample corrected persistence-distance validation.

Reuse the archive's canonical TTK VTU parser and GUDHI conventions. Never
modify TTK inputs, model checkpoints, prior metrics, or archival audit files.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import platform
import sys

HISTORICAL_CNN_S0_DB = 3.0914716720581055
HISTORICAL_CNN_S0_W2INF = 17.467813434330861
HISTORICAL_CNN_S0_COUNTS = {"gt_d0": 967, "sr_d0": 703, "gt_d1": 871, "sr_d1": 632}
ABS_TOL = 1e-10
REL_TOL = 1e-12
METHOD = "candidateF_grad_pairzero_expanded2688"
PILOT_NAME = METHOD + "_s0_extraction_pilot_20261014"
AUDIT_NAME = METHOD + "_s0_corrected_pd_audit_20261014"


def require(cond, message):
    if not cond:
        raise RuntimeError(message)


def exactly_one(folder: Path, pattern: str) -> Path:
    choices = sorted(folder.glob(pattern))
    require(len(choices) == 1, f"Expected exactly one file for {folder / pattern}; found {len(choices)}: {choices}")
    require(choices[0].is_file() and choices[0].stat().st_size > 0, f"Missing or empty file: {choices[0]}")
    return choices[0].resolve()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def collect_pilot_artifacts(pilot: Path) -> dict[str, Path]:
    items: dict[str, Path] = {}
    for kind in ("GT", "SR"):
        stem = f"{METHOD}_{kind}_s0_speed_p160_x0_y0"
        items[f"{kind}_vti"] = exactly_one(pilot / "vti", stem + ".vti")
        items[f"{kind}_pd0"] = exactly_one(pilot / "pd", stem + "_pd_port_0.vtu")
        for port, suffix in ((0, ".vtu"), (1, ".vtu"), (2, ".vti")):
            items[f"{kind}_mt{port}"] = exactly_one(pilot / "mt", stem + f"_mt_port_{port}" + suffix)
    return items


def validate_diagrams(diagrams, label: str, np) -> tuple[dict[int, object], dict[str, int]]:
    arrays = {}
    counts = {}
    for dim in (0, 1):
        value = np.asarray(diagrams[dim], dtype=float)
        require(value.ndim == 2 and value.shape[1] == 2, f"{label} D{dim} must be an N x 2 array; got {value.shape}")
        require(np.isfinite(value).all(), f"{label} D{dim} includes nonfinite birth/death values")
        require(np.all(value[:, 1] >= value[:, 0]), f"{label} D{dim} has death earlier than birth")
        arrays[dim] = value
        counts[f"{label.lower()}_d{dim}"] = int(len(value))
    return arrays, counts


def dist_report(gt, sr, gudhi, wasserstein_distance, np):
    dB = {}
    w2inf = {}
    w22 = {}
    for dim in (0, 1):
        a, b = gt[dim], sr[dim]
        dB[dim] = float(gudhi.bottleneck_distance(a, b, e=0.0))
        w2inf[dim] = float(wasserstein_distance(a, b, matching=False, order=2.0,
                                              internal_p=np.inf, keep_essential_parts=False))
        w22[dim] = float(wasserstein_distance(a, b, matching=False, order=2.0,
                                            internal_p=2.0, keep_essential_parts=False))
    result = {
        "dB_D0": dB[0], "dB_D1": dB[1], "dB_all": max(dB.values()),
        "W2inf_D0": w2inf[0], "W2inf_D1": w2inf[1],
        "W2inf_all": math.hypot(w2inf[0], w2inf[1]),
        "W22_D0": w22[0], "W22_D1": w22[1],
        "W22_all": math.hypot(w22[0], w22[1]),
    }
    require(all(math.isfinite(x) for x in result.values()), f"Nonfinite corrected distance: {result}")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path.home() / "PhIRE")
    p.add_argument("--audit-root", type=Path, default=Path.home() / "phire_runtime_audit_20260809_221548")
    p.add_argument("--pilot-root", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--preflight-only", action="store_true", help="Check paths and hashes; do not import GUDHI or write output")
    args = p.parse_args()

    repo = args.repo_root.resolve()
    pilot = (args.pilot_root or repo / "ttk_runs_fixed/topology_finetuning" / PILOT_NAME).resolve()
    out_dir = (args.out_dir or repo / "ttk_runs_fixed/topology_finetuning" / AUDIT_NAME).resolve()
    require(repo.is_dir(), f"Repository not found: {repo}")
    require(pilot.is_dir(), f"Pilot directory not found: {pilot}")
    parser_file = args.audit_root.resolve() / "recompute_pd" / "canonical_pd_pilot.py"
    require(parser_file.is_file(), f"Authoritative canonical PD parser not found: {parser_file}")
    historical = repo / "ttk_runs_fixed/cnn/pd"
    historical_gt = exactly_one(historical, "cnn_GT_s0_speed_p160_x0_y0_pd_port_0.vtu")
    historical_sr = exactly_one(historical, "cnn_SR_s0_speed_p160_x0_y0_pd_port_0.vtu")
    artifacts = collect_pilot_artifacts(pilot)
    read_paths = dict(artifacts, historical_cnn_GT_pd=historical_gt, historical_cnn_SR_pd=historical_sr,
                      archived_canonical_parser=parser_file)
    manifest = {key: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
                for key, path in sorted(read_paths.items())}

    print("[PREFLIGHT] Canonical parser and historical CNN sample-0 PDs found")
    print(f"[PREFLIGHT] {len(artifacts)} pilot artifacts (2 VTI + 2 PD + 6 MT) present and nonempty")
    print("[PREFLIGHT] SHA256 manifest prepared in memory")
    if args.preflight_only:
        print("[PREFLIGHT] PASS (no corrected metrics computed, no files written)")
        return 0

    # Reuse the exact canonical reader from the historical audit. The read_pd()
    # function returns a {0: N x 2, 1: N x 2} finite-diagram mapping plus metadata.
    sys.path.insert(0, str(parser_file.parent))
    canonical = importlib.import_module("canonical_pd_pilot")
    import numpy as np
    import gudhi
    from gudhi.wasserstein import wasserstein_distance

    hist_gt_raw, _ = canonical.read_pd(historical_gt)
    hist_sr_raw, _ = canonical.read_pd(historical_sr)
    hist_gt, counts_gt = validate_diagrams(hist_gt_raw, "GT", np)
    hist_sr, counts_sr = validate_diagrams(hist_sr_raw, "SR", np)
    hist_counts = dict(counts_gt, **counts_sr)
    require(hist_counts == HISTORICAL_CNN_S0_COUNTS,
            f"Historical CNN sample-0 cardinalities disagree with frozen audit: {hist_counts}")
    hist_dist = dist_report(hist_gt, hist_sr, gudhi, wasserstein_distance, np)
    require(math.isclose(hist_dist["dB_all"], HISTORICAL_CNN_S0_DB, rel_tol=REL_TOL, abs_tol=ABS_TOL),
            f"Historical CNN sample-0 bottleneck regression failed: {hist_dist['dB_all']}")
    require(math.isclose(hist_dist["W2inf_all"], HISTORICAL_CNN_S0_W2INF, rel_tol=REL_TOL, abs_tol=ABS_TOL),
            f"Historical CNN sample-0 W2inf regression failed: {hist_dist['W2inf_all']}")
    print(f"[REGRESSION] PASS CNN sample-0 dB={hist_dist['dB_all']:.12g} W2inf={hist_dist['W2inf_all']:.12g}")

    new_gt_raw, _ = canonical.read_pd(artifacts["GT_pd0"])
    new_sr_raw, _ = canonical.read_pd(artifacts["SR_pd0"])
    new_gt, gt_counts = validate_diagrams(new_gt_raw, "GT", np)
    new_sr, sr_counts = validate_diagrams(new_sr_raw, "SR", np)
    new_counts = dict(gt_counts, **sr_counts)
    result = dist_report(new_gt, new_sr, gudhi, wasserstein_distance, np)
    print(f"[NEW MODEL] D0 GT/SR={new_counts['gt_d0']}/{new_counts['sr_d0']} "
          f"D1 GT/SR={new_counts['gt_d1']}/{new_counts['sr_d1']}")
    print(f"[NEW MODEL] dB_all={result['dB_all']:.12g}")
    print(f"[NEW MODEL] W2inf_all={result['W2inf_all']:.12g}")
    print(f"[NEW MODEL] W22_all={result['W22_all']:.12g}")

    payload = {
        "status": "PASS",
        "method": METHOD,
        "sample_idx": 0,
        "scope": "corrected PD distances on TTK-extracted finite D0/D1; NOT a merge-tree or spatial audit",
        "patch": "160x160 top-left (x0=0,y0=0), active scalar wind_speed",
        "distance_definitions": {
            "dB": "exact GUDHI bottleneck e=0, L_infinity ground metric, aggregate=max(D0,D1)",
            "W2inf": "GUDHI Wasserstein order=2 internal_p=inf, aggregate=hypot(D0,D1)",
            "W22": "GUDHI Wasserstein order=2 internal_p=2, aggregate=hypot(D0,D1)",
            "essential_global_pair": "excluded by the archived canonical read_pd parser",
        },
        "historical_cnn_s0_regression": {"status": "PASS", "counts": hist_counts, "distances": hist_dist,
                                         "expected_dB_all": HISTORICAL_CNN_S0_DB,
                                         "expected_W2inf_all": HISTORICAL_CNN_S0_W2INF},
        "gradient_only_s0": {"counts": new_counts, "distances": result},
        "environment": {"python": sys.version, "architecture": platform.machine(),
                        "numpy": np.__version__, "gudhi": gudhi.__version__},
        "source_sha256": manifest,
    }
    require(not (out_dir / "phase6_s0_corrected_pd.json").exists(),
            "Output JSON already exists; refusing to replace archived run. Use a new --out-dir for a fresh run.")
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "phase6_s0_corrected_pd.json"
    with dest.open("x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(f"[OUTPUT] {dest}")
    print("[AUDIT] PASS: historical GUDHI reference reproduced and gradient-only sample-0 corrected PD values saved")
    print("[AUDIT] Merge-tree and spatial-locality distances NOT computed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[AUDIT] FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
