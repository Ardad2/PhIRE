#!/usr/bin/env python3
"""
All-168 PD descriptor compatibility study:
PhIRE/TTK corrected PDs vs colleague tda-toolkit GUDHI CubicalComplex PDs.

This script assumes the sample-69 Phase-A same-diagram metric-parity audit has
already passed. It therefore focuses on the scientifically distinct Phase B:
independent descriptor construction from the same authoritative scalar fields.

For each sample 0..167 and each method:
    CNN
    matched L_uv-only control (UV)
    Candidate F1

it records:
    - authoritative frozen TTK dB / W2inf / W22
    - independently constructed colleague/GUDHI cubical dB / W2inf / W22
    - cubical finite D0/D1 cardinalities
    - absolute / relative cross-backend differences

It then summarizes:
    - method means/medians under both backends
    - per-method Pearson/Spearman TTK-vs-cubical association
    - pairwise method-winner agreement
    - exact three-method ranking agreement per sample
    - the predeclared visual cases 78,71,80,63,69

Important:
    Different numerical values are expected because TTK and CubicalComplex
    construct persistence on different filtered-complex/cell conventions.
    The primary robustness question is whether method rankings and conclusions
    are stable.

Resume-safe:
    one CSV row per (sample, method), flushed after every completed sample.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import gudhi as gd
from gudhi.wasserstein import wasserstein_distance
from scipy.stats import pearsonr, spearmanr


ROOT_DEFAULT = Path.home() / "PhIRE"
AUDIT_DEFAULT = Path.home() / "phire_runtime_audit_20260809_221548"
W22_DEFAULT = AUDIT_DEFAULT / "recompute_pd_w22"
N_SAMPLES = 168
PATCH = 160

METHODS = ("cnn", "uv", "f1")
DISPLAY = {
    "cnn": "CNN",
    "uv": "Matched UV control",
    "f1": "Candidate F1",
}
RUN_NAME = {
    "cnn": "cnn",
    "uv": "topology_finetuning/candidateUV_expanded2688_topology",
    "f1": (
        "topology_finetuning/"
        "candidateF_grad_E2_low_expanded2688_topology"
    ),
}
FIELD_DIRS_REL = {
    "cnn": "data_out_fixed/wind_mrhr_cnn",
    "uv": "data_out/wind_finetune_candidateUV_expanded2688",
    "f1": "data_out/wind_finetune_candidateF_grad_E2_low_expanded2688",
}
VISUAL_SAMPLES = (78, 71, 80, 63, 69)

METRICS = ("dB", "W2inf", "W22")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    p.add_argument("--w22", type=Path, default=W22_DEFAULT)
    p.add_argument(
        "--toolkit-root",
        type=Path,
        default=ROOT_DEFAULT / "third_party" / "tda-toolkit-mapper",
    )
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--progress-every", type=int, default=1)
    return p.parse_args()


def import_colleague(toolkit_root: Path):
    src = toolkit_root / "src"
    pkg = src / "tda_toolkit"
    if not pkg.is_dir():
        raise RuntimeError(f"Missing colleague package: {pkg}")

    sys.path.insert(0, str(src))
    import tda_toolkit  # type: ignore
    from tda_toolkit.persistence import compute_cubical_persistence  # type: ignore

    return tda_toolkit, compute_cubical_persistence


def finite_pd_from_gudhi(persistence):
    out = {0: [], 1: []}
    for dim, pair in persistence:
        if dim not in (0, 1):
            continue
        b = float(pair[0])
        d = float(pair[1])
        if math.isfinite(b) and math.isfinite(d):
            out[dim].append((b, d))

    def arr(vals):
        if not vals:
            return np.empty((0, 2), dtype=np.float64)
        return np.asarray(vals, dtype=np.float64).reshape((-1, 2))

    return {0: arr(out[0]), 1: arr(out[1])}


def exact_bottleneck(a, b):
    return float(gd.bottleneck_distance(a, b, e=0.0))


def exact_w2(a, b, internal_p):
    return float(
        wasserstein_distance(
            a,
            b,
            matching=False,
            order=2.0,
            internal_p=internal_p,
            keep_essential_parts=False,
        )
    )


def metrics(gt, sr):
    db0 = exact_bottleneck(gt[0], sr[0])
    db1 = exact_bottleneck(gt[1], sr[1])

    wi0 = exact_w2(gt[0], sr[0], np.inf)
    wi1 = exact_w2(gt[1], sr[1], np.inf)

    w20 = exact_w2(gt[0], sr[0], 2.0)
    w21 = exact_w2(gt[1], sr[1], 2.0)

    return {
        "dB_D0": db0,
        "dB_D1": db1,
        "dB": max(db0, db1),
        "W2inf_D0": wi0,
        "W2inf_D1": wi1,
        "W2inf": math.hypot(wi0, wi1),
        "W22_D0": w20,
        "W22_D1": w21,
        "W22": math.hypot(w20, w21),
    }


def speed_field(x):
    a = np.asarray(x)

    if a.ndim == 2:
        s = a.astype(np.float64, copy=False)
    elif a.ndim == 3 and a.shape[-1] == 2:
        s = np.sqrt(
            a[..., 0].astype(np.float64) ** 2
            + a[..., 1].astype(np.float64) ** 2
        )
    else:
        raise RuntimeError(f"Unexpected field shape: {a.shape}")

    if s.shape[0] < PATCH or s.shape[1] < PATCH:
        raise RuntimeError(f"Field too small: {s.shape}")

    out = np.ascontiguousarray(s[:PATCH, :PATCH])

    if not np.isfinite(out).all():
        raise RuntimeError("Non-finite scalar field")

    return out


class FieldStore:
    def __init__(self, root: Path):
        self.data = {}

        for method in METHODS:
            d = root / FIELD_DIRS_REL[method]
            idx = np.asarray(np.load(d / "idx.npy")).reshape(-1).astype(np.int64)
            gt = np.load(d / "dataGT.npy", mmap_mode="r")
            sr = np.load(d / "dataSR.npy", mmap_mode="r")

            if len(gt) != len(sr):
                raise RuntimeError(f"{method}: GT/SR count mismatch")

            positions = defaultdict(list)
            for pos, sample in enumerate(idx):
                positions[int(sample)].append(pos)

            mapping = {}
            for sample in range(N_SAMPLES):
                hits = positions.get(sample, [])
                if len(hits) == 1:
                    mapping[sample] = hits[0]
                elif not hits and len(gt) > sample:
                    # Historical arrays are usually direct sample order.
                    mapping[sample] = sample
                else:
                    raise RuntimeError(
                        f"{method}: sample {sample} maps to {hits}"
                    )

            self.data[method] = {
                "gt": gt,
                "sr": sr,
                "pos": mapping,
            }

    def get(self, method: str, side: str, sample: int):
        rec = self.data[method]
        pos = rec["pos"][sample]
        return speed_field(rec[side.lower()][pos])


def load_frozen(w22_csv: Path):
    with w22_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    out = {}

    for method in METHODS:
        run = RUN_NAME[method]

        hits = [
            r for r in rows
            if r["run"] == run
        ]

        if len(hits) != N_SAMPLES:
            raise RuntimeError(
                f"{method}: expected {N_SAMPLES} frozen rows, got {len(hits)}"
            )

        for r in hits:
            s = int(r.get("sample", r.get("sample_idx", "-1")))
            out[(s, method)] = {
                "dB": float(r["bottleneck_all"]),
                "W2inf": float(r["w2inf_all"]),
                "W22": float(r["w22_all"]),
            }

    if len(out) != N_SAMPLES * len(METHODS):
        raise RuntimeError(f"Frozen key count mismatch: {len(out)}")

    return out


FIELDNAMES = [
    "sample",
    "method",
    "display_name",
    "cubical_gt_D0_count",
    "cubical_gt_D1_count",
    "cubical_sr_D0_count",
    "cubical_sr_D1_count",
    "ttk_dB",
    "cubical_dB",
    "delta_dB_cubical_minus_ttk",
    "relative_delta_dB_percent",
    "ttk_W2inf",
    "cubical_W2inf",
    "delta_W2inf_cubical_minus_ttk",
    "relative_delta_W2inf_percent",
    "ttk_W22",
    "cubical_W22",
    "delta_W22_cubical_minus_ttk",
    "relative_delta_W22_percent",
]


def load_recorded(path: Path):
    if not path.is_file() or path.stat().st_size == 0:
        return {}

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    out = {}
    for r in rows:
        key = (int(r["sample"]), r["method"])
        if key in out:
            raise RuntimeError(f"Duplicate output key: {key}")
        out[key] = r
    return out


def append_rows(path: Path, rows: List[dict]):
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            w.writeheader()
        w.writerows(rows)
        f.flush()


def float_rows(path: Path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != N_SAMPLES * len(METHODS):
        raise RuntimeError(
            f"Expected {N_SAMPLES*len(METHODS)} final rows, got {len(rows)}"
        )

    return rows


def safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(pearsonr(x, y).statistic), float(spearmanr(x, y).statistic)


def winner(a, b, tol=1e-12):
    if a < b - tol:
        return "A"
    if b < a - tol:
        return "B"
    return "tie"


def summarize(out_csv: Path, outdir: Path):
    rows = float_rows(out_csv)
    by = {(int(r["sample"]), r["method"]): r for r in rows}

    summary_rows = []
    text = []

    text.append("ALL-168 TTK vs COLLEAGUE/GUDHI CUBICAL PD DESCRIPTOR STUDY")
    text.append("=" * 104)
    text.append("Lower distance is better.")
    text.append("")

    # Means / medians / correlations.
    text.append("METHOD-LEVEL BACKEND COMPARISON")
    text.append("-" * 104)

    for method in METHODS:
        text.append(DISPLAY[method])

        for metric in METRICS:
            tx = [
                float(by[(s, method)][f"ttk_{metric}"])
                for s in range(N_SAMPLES)
            ]
            cx = [
                float(by[(s, method)][f"cubical_{metric}"])
                for s in range(N_SAMPLES)
            ]

            pr, sr = safe_corr(tx, cx)

            mean_t = float(np.mean(tx))
            mean_c = float(np.mean(cx))
            median_t = float(np.median(tx))
            median_c = float(np.median(cx))
            mean_abs = float(np.mean(np.abs(np.asarray(cx) - np.asarray(tx))))

            summary_rows.append({
                "section": "method_backend",
                "method": method,
                "metric": metric,
                "ttk_mean": mean_t,
                "cubical_mean": mean_c,
                "ttk_median": median_t,
                "cubical_median": median_c,
                "mean_abs_backend_difference": mean_abs,
                "pearson": pr,
                "spearman": sr,
            })

            text.append(
                f"  {metric:6s}: "
                f"mean TTK={mean_t:.8g}, cubical={mean_c:.8g}; "
                f"median TTK={median_t:.8g}, cubical={median_c:.8g}; "
                f"Pearson={pr:+.4f}, Spearman={sr:+.4f}"
            )

        text.append("")

    # Pairwise winner agreement.
    text.append("PAIRWISE METHOD-WINNER AGREEMENT: TTK vs CUBICAL")
    text.append("-" * 104)

    pairs = (
        ("f1", "cnn"),
        ("f1", "uv"),
        ("uv", "cnn"),
    )

    for metric in METRICS:
        for a, b in pairs:
            agree = 0
            disagree = 0
            ties = 0

            for s in range(N_SAMPLES):
                tw = winner(
                    float(by[(s, a)][f"ttk_{metric}"]),
                    float(by[(s, b)][f"ttk_{metric}"]),
                )
                cw = winner(
                    float(by[(s, a)][f"cubical_{metric}"]),
                    float(by[(s, b)][f"cubical_{metric}"]),
                )

                if tw == "tie" or cw == "tie":
                    ties += 1
                elif tw == cw:
                    agree += 1
                else:
                    disagree += 1

            denom = agree + disagree
            frac = agree / denom if denom else float("nan")

            summary_rows.append({
                "section": "pairwise_winner",
                "method": f"{a}_vs_{b}",
                "metric": metric,
                "ttk_mean": "",
                "cubical_mean": "",
                "ttk_median": "",
                "cubical_median": "",
                "mean_abs_backend_difference": "",
                "pearson": "",
                "spearman": "",
                "agreement_count": agree,
                "disagreement_count": disagree,
                "ties": ties,
                "agreement_fraction": frac,
            })

            text.append(
                f"{metric:6s} {DISPLAY[a]} vs {DISPLAY[b]}: "
                f"agree={agree}, disagree={disagree}, ties={ties}, "
                f"agreement={100*frac:.2f}%"
            )

    text.append("")

    # Exact 3-method ranking agreement.
    text.append("EXACT THREE-METHOD RANKING AGREEMENT PER SAMPLE")
    text.append("-" * 104)

    for metric in METRICS:
        same = 0
        for s in range(N_SAMPLES):
            trank = tuple(sorted(
                METHODS,
                key=lambda m: float(by[(s, m)][f"ttk_{metric}"]),
            ))
            crank = tuple(sorted(
                METHODS,
                key=lambda m: float(by[(s, m)][f"cubical_{metric}"]),
            ))
            same += int(trank == crank)

        frac = same / N_SAMPLES
        text.append(
            f"{metric:6s}: {same}/{N_SAMPLES} = {100*frac:.2f}% exact ranking agreement"
        )
        summary_rows.append({
            "section": "exact_ranking",
            "method": "all3",
            "metric": metric,
            "agreement_count": same,
            "agreement_fraction": frac,
        })

    text.append("")

    # Predeclared visual samples.
    text.append("PREDECLARED VISUAL CASES")
    text.append("-" * 104)

    visual_csv_rows = []

    for s in VISUAL_SAMPLES:
        text.append(f"sample {s}")

        for metric in METRICS:
            tcnn = float(by[(s, "cnn")][f"ttk_{metric}"])
            tf1 = float(by[(s, "f1")][f"ttk_{metric}"])
            ccnn = float(by[(s, "cnn")][f"cubical_{metric}"])
            cf1 = float(by[(s, "f1")][f"cubical_{metric}"])

            tgain = 100.0 * (tcnn - tf1) / tcnn
            cgain = 100.0 * (ccnn - cf1) / ccnn

            text.append(
                f"  {metric:6s}: F1-vs-CNN gain "
                f"TTK={tgain:+.2f}%, cubical={cgain:+.2f}%"
            )

            visual_csv_rows.append({
                "sample": s,
                "metric": metric,
                "ttk_cnn": tcnn,
                "ttk_f1": tf1,
                "ttk_f1_vs_cnn_gain_percent": tgain,
                "cubical_cnn": ccnn,
                "cubical_f1": cf1,
                "cubical_f1_vs_cnn_gain_percent": cgain,
            })

        text.append("")

    visual_csv = outdir / "predeclared_visual_cases_ttk_vs_cubical.csv"
    with visual_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(visual_csv_rows[0].keys()))
        w.writeheader()
        w.writerows(visual_csv_rows)

    # Normalize heterogeneous summary-row schema.
    all_fields = []
    seen = set()
    for r in summary_rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                all_fields.append(k)

    summary_csv = outdir / "all168_backend_consistency_summary.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    summary_txt = outdir / "all168_backend_consistency_summary.txt"
    summary_txt.write_text("\n".join(text) + "\n")

    print()
    print("\n".join(text))
    print()
    print("Summary CSV:", summary_csv)
    print("Summary TXT:", summary_txt)
    print("Visual cases:", visual_csv)


def main():
    args = parse_args()

    root = args.root.expanduser().resolve()
    w22 = args.w22.expanduser().resolve()
    toolkit_root = args.toolkit_root.expanduser().resolve()

    if args.out is None:
        outdir = w22 / "pd_colleague_compatibility" / "all168_phaseB"
    else:
        outdir = args.out.expanduser().resolve()

    outdir.mkdir(parents=True, exist_ok=True)

    toolkit, colleague_cubical = import_colleague(toolkit_root)

    frozen_csv = w22 / "w22_full_sweep.csv"
    if not frozen_csv.is_file():
        raise FileNotFoundError(frozen_csv)

    print("ALL-168 PD DESCRIPTOR COMPATIBILITY")
    print("=" * 104)
    print("Python:", sys.executable)
    print("NumPy:", np.__version__)
    print("GUDHI:", gd.__version__)
    print("colleague toolkit:", toolkit.__file__)
    print("colleague version:", getattr(toolkit, "__version__", "unknown"))
    print("frozen TTK metric source:", frozen_csv)
    print("output:", outdir)
    print()

    store = FieldStore(root)
    frozen = load_frozen(frozen_csv)

    out_csv = outdir / "all168_ttk_vs_colleague_cubical_per_sample.csv"
    recorded = load_recorded(out_csv)

    start = time.time()
    completed_now = 0

    for sample in range(N_SAMPLES):
        missing = [
            method for method in METHODS
            if (sample, method) not in recorded
        ]

        if not missing:
            continue

        # Authoritative GT identity across all three method stores.
        gt_cnn = store.get("cnn", "gt", sample)
        gt_uv = store.get("uv", "gt", sample)
        gt_f1 = store.get("f1", "gt", sample)

        if not np.array_equal(gt_cnn, gt_uv):
            raise RuntimeError(f"sample {sample}: CNN GT != UV GT")
        if not np.array_equal(gt_cnn, gt_f1):
            raise RuntimeError(f"sample {sample}: CNN GT != F1 GT")

        gt_cc = colleague_cubical(gt_cnn)
        gt_pd = finite_pd_from_gudhi(gt_cc.persistence())

        rows = []

        for method in missing:
            sr = store.get(method, "sr", sample)
            sr_cc = colleague_cubical(sr)
            sr_pd = finite_pd_from_gudhi(sr_cc.persistence())
            cm = metrics(gt_pd, sr_pd)
            tm = frozen[(sample, method)]

            row = {
                "sample": sample,
                "method": method,
                "display_name": DISPLAY[method],
                "cubical_gt_D0_count": len(gt_pd[0]),
                "cubical_gt_D1_count": len(gt_pd[1]),
                "cubical_sr_D0_count": len(sr_pd[0]),
                "cubical_sr_D1_count": len(sr_pd[1]),
            }

            for metric in METRICS:
                tv = tm[metric]
                cv = cm[metric]
                delta = cv - tv
                rel = 100.0 * delta / tv if tv else float("nan")

                row[f"ttk_{metric}"] = tv
                row[f"cubical_{metric}"] = cv
                row[f"delta_{metric}_cubical_minus_ttk"] = delta
                row[f"relative_delta_{metric}_percent"] = rel

            rows.append(row)
            recorded[(sample, method)] = row

        append_rows(out_csv, rows)
        completed_now += len(rows)

        if (
            sample % args.progress_every == 0
            or sample == N_SAMPLES - 1
        ):
            elapsed = time.time() - start
            print(
                f"sample={sample:3d}/{N_SAMPLES-1} "
                f"rows_now={completed_now} "
                f"rows_total={len(recorded)}/{N_SAMPLES*len(METHODS)} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    final_rows = load_recorded(out_csv)
    expected = N_SAMPLES * len(METHODS)

    if len(final_rows) != expected:
        raise RuntimeError(
            f"Incomplete output: {len(final_rows)}/{expected}"
        )

    summarize(out_csv, outdir)

    print()
    print("=" * 104)
    print("ALL-168 PHASE-B PD DESCRIPTOR COMPATIBILITY: COMPLETE")
    print("=" * 104)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
