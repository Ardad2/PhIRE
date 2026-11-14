#!/usr/bin/env python3
"""Phase 6 / LXXIX: canonical five-model, all-168 corrected GUDHI PD sweep.

Runs on Spark in micromamba environment `gudhi-audit` from ~/PhIRE.
The accepted Part LXXVII GUDHI pilot's parse() and distance() are imported unchanged.
No TTK execution or input-file modification. Results in a separate LXXIX folder.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

N = 168
NEW_METHODS = {
    "grad": "candidateF_grad_pairzero_expanded2688",
    "cv": "candidateF_grad_CV_only_expanded2688",
    "pers": "candidateF_grad_PERS_only_expanded2688",
}
ARMS = ("cnn", "grad", "cv", "pers", "f1")
FAC = ("grad", "cv", "pers", "f1")
METRICS = ("dB", "W2inf", "W22")
COLS = ["sample", "arm", "method", "gt_sha256", "sr_sha256", "D0_gt", "D1_gt",
        "D0_sr", "D1_sr", *METRICS]
FROZEN = Path("spatial_pd/phase5e_all168_canonical_pd/pd")
NEWROOT = Path("ttk_runs_fixed/topology_finetuning/phase6_lxxvii_canonical_pd_newarms_20261014")
OUT = Path("ttk_runs_fixed/topology_finetuning/phase6_lxxix_corrected_pd_gudhi_20261014")
PILOT_SCRIPT = Path("phase6_lxxvii_update/scripts/phase6_pilot_pd_gudhi_gate.py")
DRIVER = Path("phase6_lxxvii_update/scripts/phase6_canonical_newarms_pd.py")
COMPAT = Path("phase6_lxxviii_hotfix_20261014/scripts/phase6_lxxvii_pd_xml_compat.py")
CANONICAL = Path.home() / "phire_runtime_audit_20260809_221548/recompute_pd/canonical_pd_pilot.py"


def need(condition, msg):
    if not condition:
        raise RuntimeError(msg)


def sha_file(p: Path):
    need(p.is_file() and p.stat().st_size > 0, f"File absent or empty: {p}")
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def location(root: Path, arm: str, idx: int):
    stem = f"s{idx}_speed_p160_x0_y0_pd_port_0.vtu"
    if arm == "gt":
        return root / FROZEN / f"phase5_GT_{stem}"
    if arm in ("cnn", "f1"):
        key = "CNN_SR" if arm == "cnn" else "F1_SR"
        return root / FROZEN / f"phase5_{key}_{stem}"
    method = NEW_METHODS[arm]
    return root / NEWROOT / arm / "pd" / f"{method}_SR_{stem}"


def pd_path(root: Path, arm: str, idx: int):
    stem = f"s{idx}_speed_p160_x0_y0_pd_port_0.vtu"
    if arm == "gt":
        return root / FROZEN / f"phase5_GT_{stem}"
    return location(root, arm, idx)


def load_pilot(root: Path):
    gate_path = root / NEWROOT / "pilot_verified.json"
    gate = json.loads(gate_path.read_text())
    need(gate.get("format") == "phase6-lxxvii-pilot-verified-v1", "Unexpected pilot gate format")
    return gate


def source_manifest(root: Path):
    gate = load_pilot(root)
    hashes = {}
    for arm in ("gt", *ARMS):
        for idx in range(N):
            p = pd_path(root, arm, idx)
            hashes[f"{arm}:{idx}"] = sha_file(p)
    need(hashes["gt:0"] == gate["gt_sha256"], "Frozen GT s0 SHA differs from pilot")
    for arm in NEW_METHODS:
        need(hashes[f"{arm}:0"] == gate["pd_sha256"][arm], f"{arm} s0 SHA differs from pilot")
        stored = json.loads((root / NEWROOT / arm / "input_manifest.json").read_text())
        need(stored.get("format") == "phase6-lxxvii-sr-pd-v1", f"Invalid {arm} extraction manifest")
        need(stored.get("method") == NEW_METHODS[arm], f"Method mismatch in {arm} manifest")
        need(stored.get("sample_positions") == list(range(N)), f"Sample ordering mismatch {arm}")
        need(stored.get("driver_sha256") == sha_file(root / DRIVER), f"Driver SHA mismatch {arm}")
    need(gate["parser_sha256"] == sha_file(CANONICAL), "Canonical parser changed since GUDHI pilot")
    checks = {str(p): sha_file(p) for p in [root / PILOT_SCRIPT, root / DRIVER, root / COMPAT,
                                           root / "phase6_lxxix_update/scripts/phase6_corrected_pd_full_sweep.py",
                                           CANONICAL]}
    payload = {
        "format": "phase6-lxxix-canonical-gudhi-sweep-v1",
        "n_samples": N,
        "arms": list(ARMS),
        "metrics": list(METRICS),
        "reused_phase5e": ["gt", "cnn", "f1"],
        "reference_pilot_gate": gate,
        "input_sha256": hashes,
        "script_sha256": checks,
    }
    return payload


def loader(root):
    # Same validated implementation as the LXXVII GUDHI pilot. No new distance math.
    import numpy  # noqa: F401
    import gudhi  # noqa: F401
    from gudhi.wasserstein import wasserstein_distance  # noqa: F401
    spec = importlib.util.spec_from_file_location("phase6_pilot_gate", root / PILOT_SCRIPT)
    need(spec is not None and spec.loader is not None, "Cannot load pilot")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    parser_spec = importlib.util.spec_from_file_location("canonical_pd_parser", CANONICAL)
    need(parser_spec is not None and parser_spec.loader is not None, "Cannot load canonical parser")
    canonical = importlib.util.module_from_spec(parser_spec)
    parser_spec.loader.exec_module(canonical)
    return m, canonical.read_pd


def parse(m, read_pd, p):
    d = m.parse(read_pd, p)
    for dim in (0, 1):
        need(d[dim].ndim == 2 and d[dim].shape[1] == 2, f"D{dim} shape {p}")
    return d


def calculate(m, gt, sr):
    result = m.distance(gt, sr)
    vals = [float(result[k]) for k in METRICS]
    need(all(math.isfinite(v) and v >= 0 for v in vals), "Nonfinite/negative GUDHI distance")
    need(vals[0] <= vals[1] + 1e-7 and vals[1] <= vals[2] + 1e-7,
         f"Bottleneck/Wasserstein monotonicity failed: {vals}")
    return result


def key(row):
    return (int(row["sample"]), row["arm"])


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        r = csv.DictReader(fh)
        need(r.fieldnames == COLS, f"Wrong CSV columns in {path}")
        return list(r)


def atomic_write_csv(path, rows):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def check_rows(rows, hashes):
    seen = set()
    for row in rows:
        i, arm = key(row)
        need(0 <= i < N and arm in ARMS, f"Bad CSV key {i}:{arm}")
        need((i, arm) not in seen, f"Duplicate CSV key {i}:{arm}")
        seen.add((i, arm))
        expected_method = "pretrained_CNN" if arm == "cnn" else "full_F1" if arm == "f1" else NEW_METHODS[arm]
        need(row["method"] == expected_method, f"Wrong method label {arm}:{i}")
        need(row["gt_sha256"] == hashes[f"gt:{i}"], f"GT SHA drift in CSV {arm}:{i}")
        need(row["sr_sha256"] == hashes[f"{arm}:{i}"], f"SR SHA drift in CSV {arm}:{i}")
        for field in ("D0_gt", "D1_gt", "D0_sr", "D1_sr"):
            need(int(row[field]) >= 0, f"Bad pair count in {arm}:{i}")
        d = [float(row[k]) for k in METRICS]
        need(all(math.isfinite(v) and v >= 0 for v in d), f"Nonfinite metric in {arm}:{i}")
        need(d[0] <= d[1] + 1e-7 and d[1] <= d[2] + 1e-7, f"Monotonicity in {arm}:{i}")
    for i in range(N):
        have = {arm for x, arm in seen if x == i}
        need(not have or have == set(ARMS), f"Incomplete sample group {i}: {have}")
    return seen


def pilot_regression(row, gate):
    arm = row["arm"]
    if int(row["sample"]) != 0 or arm not in NEW_METHODS:
        return
    for metric in METRICS:
        a, b = float(row[metric]), float(gate["metrics_s0"][arm][metric])
        need(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8),
             f"Pilot regression failure: {arm} {metric}: sweep={a}, pilot={b}")


def summarize(rows, root: Path, out: Path):
    grouped = {arm: {int(r["sample"]): r for r in rows if r["arm"] == arm} for arm in ARMS}
    need(all(len(grouped[arm]) == N for arm in ARMS), "Cannot summarize partial sweep")
    results = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for m in METRICS:
            a = [float(grouped[arm][i][m]) for i in range(N)]
            results[arm][m] = {"mean": statistics.mean(a), "median": statistics.median(a),
                               "std_sample": statistics.stdev(a), "min": min(a), "max": max(a),
                               "better_than_cnn_count": None if arm == "cnn" else sum(
                                   v < float(grouped["cnn"][i][m]) for i, v in enumerate(a))}
    effects = {}
    for m in METRICS:
        # Matches the LXXII fixed 2x2: grad=0/0, cv=1/0, pers=0/1, f1=1/1.
        diff = {"cv": [], "pers": [], "interaction": []}
        for i in range(N):
            z = {a: float(grouped[a][i][m]) for a in FAC}
            diff["cv"].append(((z["cv"] - z["grad"]) + (z["f1"] - z["pers"])) / 2)
            diff["pers"].append(((z["pers"] - z["grad"]) + (z["f1"] - z["cv"])) / 2)
            diff["interaction"].append(z["f1"] - z["cv"] - z["pers"] + z["grad"])
        effects[m] = {k: {"mean": statistics.mean(v), "median": statistics.median(v),
                          "negative_count": sum(x < 0 for x in v), "positive_count": sum(x > 0 for x in v)}
                      for k, v in diff.items()}
    obj = {"format": "phase6-lxxix-corrected-pd-summary-v1", "n": N,
           "baseline_note": "Phase5E canonical CNN/F1, not the older differently oriented CNN PD archive",
           "statistics_note": "Descriptive per-field statistics; 168 consecutive hours are temporally correlated. No significance tests or multi-seed replication.",
           "methods": results, "two_by_two_effects": effects}
    (out / "summary.json").write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    lines = ["# Phase 6 full corrected persistence-diagram results", "",
             "168 aligned benchmark fields; frozen canonical Phase 5E GT / CNN / F1 diagrams;",
             "three new extracted SR arms. Recalculated with canonical TTK point parser and GUDHI.",
             "Lower values are better. All statistics descriptive (consecutive hourly samples).", "",
             "| Arm | dB mean | W2inf mean | W22 mean |", "|---|---:|---:|---:|"]
    for arm in ARMS:
        lines.append("| " + arm + " | " + " | ".join(f"{results[arm][m]['mean']:.6f}" for m in METRICS) + " |")
    lines.extend(["", "## 2×2 factorial means", "", "Negative main effects improve lower-is-better distance.",
                  "Interaction is signed and does not alone imply improvement.", "",
                  "| Metric | CV main effect | PERS main effect | Interaction |", "|---|---:|---:|---:|"])
    for m in METRICS:
        lines.append("| " + m + " | " + " | ".join(f"{effects[m][k]['mean']:.6f}" for k in ("cv","pers","interaction")) + " |")
    lines.extend(["", "## Caveats", "", "- No merge-tree or spatial-locality metric is computed here.",
                  "- No formal independence assumption across consecutive hourly samples.",
                  "- Confirm F1 source continuity with the 2×2 experiment documentation before manuscript attribution.", ""])
    (out / "summary.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--mode", choices=["preflight", "sweep", "verify"], required=True)
    args = ap.parse_args()
    root = args.repo_root.resolve()
    need((root / FROZEN).is_dir(), "Missing Phase5E canonical archive")
    payload = source_manifest(root)
    hashes = payload["input_sha256"]
    pilot = payload["reference_pilot_gate"]
    print("[SOURCE] PASS: 1008 input PD VTUs (168 GT + 840 SR) hashed, including all 504 new-arm SR files", flush=True)
    print("[PILOT] PASS: frozen GT and all three sample-0 SR hashes match independent GUDHI gate", flush=True)
    m, read_pd = loader(root)
    # Read and recompute the three new-arm pilot entries before any output is created.
    gt0 = parse(m, read_pd, pd_path(root, "gt", 0))
    for arm in NEW_METHODS:
        sr0 = parse(m, read_pd, pd_path(root, arm, 0))
        metrics = calculate(m, gt0, sr0)
        for metric in METRICS:
            need(math.isclose(metrics[metric], float(pilot["metrics_s0"][arm][metric]), rel_tol=1e-10, abs_tol=1e-8),
                 f"GUDHI s0 {arm} regression for {metric}")
        print(f"[S0] PASS {arm}: " + " ".join(f"{k}={metrics[k]:.12g}" for k in METRICS), flush=True)
    out = root / OUT
    if args.mode == "preflight":
        print("[PREFLIGHT] PASS: no files written; corrected GUDHI s0 regression and full input inventory passed", flush=True)
        return
    need(out.exists() or args.mode == "sweep", "No sweep results exist for verification")
    if args.mode == "sweep":
        out.mkdir(parents=True, exist_ok=True)
        manifest_file = out / "manifest.json"
        if manifest_file.exists():
            need(json.loads(manifest_file.read_text()) == payload, "Existing output manifest differs; refusing to reuse")
        else:
            need(not any(out.iterdir()), "Output directory exists without manifest; stop")
            manifest_file.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        need(json.loads((out / "manifest.json").read_text()) == payload,
             "Output manifest differs from current input hashes")
    csv_file = out / "per_sample_corrected_pd.csv"
    rows = read_csv(csv_file)
    seen = check_rows(rows, hashes)
    for row in rows:
        pilot_regression(row, pilot)
    for i in range(N):
        if (i, "cnn") in seen:
            continue
        if args.mode == "verify":
            raise RuntimeError(f"Missing sample {i} in CSV (verify is read-only)")
        gt = gt0 if i == 0 else parse(m, read_pd, pd_path(root, "gt", i))
        group = []
        for arm in ARMS:
            sr = parse(m, read_pd, pd_path(root, arm, i))
            d = calculate(m, gt, sr)
            row = {"sample": i, "arm": arm,
                   "method": "pretrained_CNN" if arm == "cnn" else "full_F1" if arm == "f1" else NEW_METHODS[arm],
                   "gt_sha256": hashes[f"gt:{i}"], "sr_sha256": hashes[f"{arm}:{i}"],
                   "D0_gt": len(gt[0]), "D1_gt": len(gt[1]),
                   "D0_sr": len(sr[0]), "D1_sr": len(sr[1]), **d}
            pilot_regression(row, pilot)
            group.append(row)
        rows.extend(group)
        atomic_write_csv(csv_file, rows)  # checkpoint after each complete five-model sample
        if (i + 1) % 10 == 0 or i == N - 1:
            print(f"[SWEEP] PASS {i+1}/{N} complete five-arm samples ({len(rows)} rows)", flush=True)
    final = read_csv(csv_file)
    check_rows(final, hashes)
    need(len(final) == 5 * N, f"Expected 840 rows; found {len(final)}")
    for row in final:
        pilot_regression(row, pilot)
    print("[AUDIT] PASS: 840 unique (sample,arm) rows; 504 new-arm GUDHI comparisons; 336 Phase5E CNN/F1 controls", flush=True)
    if args.mode == "sweep":
        summarize(final, root, out)
        digest = sha_file(csv_file)
        (out / "results_sha256.txt").write_text(f"{digest}  {csv_file.name}\n")
        print(f"[OUTPUT] {out}", flush=True)
        print("[SWEEP-GATE] PASS: CSV, summary and SHA256 generated", flush=True)
    else:
        recorded = (out / "results_sha256.txt").read_text().split()[0]
        need(sha_file(csv_file) == recorded, "Results CSV hash differs from frozen sweep")
        print("[VERIFY-GATE] PASS: all input hashes, rows, pilot regressions, invariants and CSV SHA256", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[LXXIX] ERROR {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        raise SystemExit(1)
