#!/usr/bin/env python3
"""Read-only source audit and isolated descriptive per-field report for Phase 6 LXXIX.

No GUDHI or TTK runs. Exact method pairing on sample indices 0..167.
The 168 consecutive hourly fields are temporally dependent; no inferential test.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import sys

ARMS = ("cnn", "grad", "cv", "pers", "f1")
METRICS = ("dB", "W2inf", "W22")
N = 168
SOURCE = Path("ttk_runs_fixed/topology_finetuning/phase6_lxxix_corrected_pd_gudhi_20261014")
DEST = Path("ttk_runs_fixed/topology_finetuning/phase6_lxxx_paired_effect_audit_20261014")


def require(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def average(a):
    return st.mean(a)


def basic(a):
    return {"mean": average(a), "median": st.median(a),
            "min": min(a), "max": max(a), "std_sample": st.stdev(a)}


def check_and_read(root):
    folder = root / SOURCE
    paths = [folder / x for x in ("per_sample_corrected_pd.csv", "summary.json", "manifest.json", "results_sha256.txt")]
    for p in paths:
        require(p.is_file() and p.stat().st_size > 0, f"Missing/empty {p}")
    csv_path, summary_path, manifest_path, hash_path = paths
    fields = hash_path.read_text().split()
    require(len(fields) == 2 and fields[1] == csv_path.name, "Unexpected LXXIX checksum manifest")
    require(sha(csv_path) == fields[0], "LXXIX CSV SHA-256 mismatch")
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    require(int(summary.get("n", -1)) == N, "Wrong summary sample count")
    require(manifest.get("format") == "phase6-lxxix-canonical-gudhi-sweep-v1", "Wrong LXXIX manifest format")
    require(int(manifest.get("n_samples", -1)) == N, "Wrong manifest sample count")
    require(manifest.get("arms") == list(ARMS), "Wrong arms in source manifest")
    rows = {}
    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames and all(x in reader.fieldnames for x in ("sample", "arm", *METRICS)), "Missing required CSV columns")
        for row in reader:
            i, arm = int(row["sample"]), row["arm"]
            require(0 <= i < N and arm in ARMS, f"Bad key {i}/{arm}")
            require((i, arm) not in rows, f"Duplicate {i}/{arm}")
            arr = {k: float(row[k]) for k in METRICS}
            require(all(math.isfinite(x) and x >= 0 for x in arr.values()), f"Nonfinite/negative distance {i}/{arm}")
            require(arr["dB"] <= arr["W2inf"] + 1e-7 <= arr["W22"] + 2e-7, f"Distance ordering {i}/{arm}")
            # Compare input hashes against the frozen manifest, not filenames alone.
            hashes = manifest["input_sha256"]
            require(row["gt_sha256"] == hashes[f"gt:{i}"], f"GT hash mismatch {i}/{arm}")
            require(row["sr_sha256"] == hashes[f"{arm}:{i}"], f"SR hash mismatch {i}/{arm}")
            rows[(i, arm)] = arr
    require(len(rows) == N * len(ARMS), f"Expected {N*len(ARMS)} rows, found {len(rows)}")
    for i in range(N):
        require(all((i, arm) in rows for arm in ARMS), f"Incomplete sample {i}")
    for arm in ARMS:
        for metric in METRICS:
            actual = average([rows[(i, arm)][metric] for i in range(N)])
            prior = float(summary["methods"][arm][metric]["mean"])
            require(math.isclose(actual, prior, rel_tol=0, abs_tol=1e-10),
                    f"Summary mean mismatch {arm}/{metric}")
    return rows, summary, sha(csv_path)


def audit(rows, summary):
    out = {"format": "phase6-lxxx-paired-audit-v1", "n": N,
           "analysis": "Descriptive only; consecutive hourly samples, one checkpoint per method.",
           "metrics": {}}
    lines = ["# Phase 6 paired corrected persistence-distance audit", "",
             "Source: frozen LXXIX five-arm per-sample GUDHI CSV. All 168 sample indices matched.",
             "Lower distances are better. No independent seed or statistical significance claims.", ""]
    for m in METRICS:
        values = {a: [rows[(i, a)][m] for i in range(N)] for a in ARMS}
        winners = Counter()
        for i in range(N):
            v = min(values[a][i] for a in ARMS)
            eq = [a for a in ARMS if math.isclose(values[a][i], v, rel_tol=0, abs_tol=1e-12)]
            if len(eq) == 1:
                winners[eq[0]] += 1
            else:
                winners["tie"] += 1
        # Factorial effects match the original Phase 6 design with gradient-only as 00.
        cv = [((values["cv"][i]-values["grad"][i])+(values["f1"][i]-values["pers"][i]))/2 for i in range(N)]
        pers = [((values["pers"][i]-values["grad"][i])+(values["f1"][i]-values["cv"][i]))/2 for i in range(N)]
        interaction = [values["f1"][i]-values["cv"][i]-values["pers"][i]+values["grad"][i] for i in range(N)]
        effects = {"cv": cv, "pers": pers, "interaction": interaction}
        for eff, arr in effects.items():
            prior = summary["two_by_two_effects"][m][eff]["mean"]
            require(math.isclose(average(arr), float(prior), rel_tol=0, abs_tol=1e-10),
                    f"Factorial effect mismatch {m}/{eff}")
        pairs = {}
        for a, b in (("grad", "cnn"), ("cv", "grad"), ("pers", "grad"), ("f1", "grad"),
                     ("f1", "cv"), ("f1", "pers"), ("pers", "cv"),
                     ("cv", "cnn"), ("pers", "cnn"), ("f1", "cnn")):
            diffs = [values[a][i] - values[b][i] for i in range(N)]
            win = [i for i, x in enumerate(diffs) if x < -1e-12]
            lose = [i for i, x in enumerate(diffs) if x > 1e-12]
            tie = N-len(win)-len(lose)
            pairs[f"{a}_minus_{b}"] = {**basic(diffs), "better_count": len(win),
                                       "worse_count": len(lose), "tie_count": tie,
                                       "largest_improvements": sorted(range(N), key=lambda i: (diffs[i], i))[:5],
                                       "largest_regressions": sorted(range(N), key=lambda i: (-diffs[i], i))[:5]}
        dayblocks = {}
        for eff, arr in effects.items():
            dayblocks[eff] = [average(arr[i:i+24]) for i in range(0, N, 24)]
        out["metrics"][m] = {"means": {a: average(values[a]) for a in ARMS},
                             "best_model_counts": {x: winners[x] for x in (*ARMS, "tie")},
                             "paired_comparisons": pairs,
                             "factorial": {k: {**basic(v), "negative_count": sum(x < 0 for x in v),
                                                "positive_count": sum(x > 0 for x in v),
                                                "zero_count": sum(x == 0 for x in v),
                                                "contiguous_24h_block_means": dayblocks[k]}
                                           for k, v in effects.items()}}
        lines += [f"## {m}", "", "| Arm | Mean | Best-of-five count |", "|---|---:|---:|"]
        for a in ARMS:
            lines.append(f"| {a} | {average(values[a]):.6f} | {winners[a]} |")
        lines.append(f"| ties | — | {winners['tie']} |")
        lines += ["", "| Comparison (A minus B) | Mean difference | Median | A better / worse / tied |",
                  "|---|---:|---:|---:|"]
        for name, p in pairs.items():
            lines.append(f"| {name} | {p['mean']:+.6f} | {p['median']:+.6f} | {p['better_count']} / {p['worse_count']} / {p['tie_count']} |")
        lines += ["", "| Factorial effect | Mean | Negative / positive / zero fields | 24-hour block means |",
                  "|---|---:|---:|---|"]
        for name, p in out["metrics"][m]["factorial"].items():
            block = ", ".join(f"{x:+.3f}" for x in p["contiguous_24h_block_means"])
            lines.append(f"| {name} | {p['mean']:+.6f} | {p['negative_count']} / {p['positive_count']} / {p['zero_count']} | {block} |")
        lines.append("")
    lines += ["## Interpretation boundaries", "",
              "- Each value is a matched same-sample descriptive comparison; the fields are successive hours and are not independent replicates.",
              "- Mean advantage alone does not prove systematic superiority; consult wins/losses, medians, and 24-hour blocks.",
              "- A positive CV×PERS interaction in a lower-is-better metric means less-than-additive combination relative to the gradient-only 00 cell, not by itself failure of the combined model.",
              "- F1's historical source/checkpoint continuity with the 2×2 training protocol remains a separate provenance gate.",
              "- Unconstrained persistence distances do not establish correct spatial feature locations or merge-tree agreement.", ""]
    return out, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--mode", required=True, choices=["preflight", "report"])
    args = ap.parse_args()
    root = args.repo_root.resolve()
    rows, summary, csv_sha = check_and_read(root)
    obj, md = audit(rows, summary)
    print(f"[SOURCE] PASS: 840 matched rows, 168 indices, all source hashes and LXXIX means/effects verified; CSV_SHA256={csv_sha}")
    for m in METRICS:
        data = obj["metrics"][m]
        winners = data["best_model_counts"]
        print(f"[PAIRED] {m} best_counts={winners}  pers_vs_f1={data['paired_comparisons']['f1_minus_pers']['better_count']}/168 F1_wins")
    if args.mode == "preflight":
        print("[LXXX] PREFLIGHT PASS: read-only; no data files modified or report created")
        return
    dest = root / DEST
    dest.mkdir(parents=True, exist_ok=True)
    expected = dest / "source_csv_sha256.txt"
    if expected.exists():
        require(expected.read_text().strip() == csv_sha, "Existing LXXX reports refer to a different source CSV")
    else:
        require(not any(dest.iterdir()), "Output folder already populated without source hash; abort")
        expected.write_text(csv_sha + "\n")
    (dest / "paired_audit.json").write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    (dest / "paired_audit.md").write_text(md)
    with (dest / "SHA256SUMS").open("w") as output:
        for fn in ("source_csv_sha256.txt", "paired_audit.json", "paired_audit.md"):
            output.write(f"{sha(dest/fn)}  {fn}\n")
    print(f"[LXXX] REPORT PASS: isolated paired audit saved to {dest}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[LXXX] ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
