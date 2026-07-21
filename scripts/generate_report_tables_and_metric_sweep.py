#!/usr/bin/env python3
import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

THRESHOLDS = [0.03, 0.05, 0.075, 0.10]
ABLT = {0.05: "0p05", 0.075: "0p075"}

TABLE_DOMAIN_METRICS = [
    ("wpd_bias", "WPD bias |.|", "abs_lower", "physics_domain"),
    ("wpd_mae", "WPD MAE", "lower", "physics_domain"),
    ("wpd_rmse", "WPD RMSE", "lower", "physics_domain"),
    ("wpd_w1", "WPD Wasserstein-1", "lower", "physics_domain"),
    ("psd_log_l2", "PSD log-L2", "lower", "physics_domain"),
    ("psd_slope_abs_delta", "PSD slope |Delta|", "lower", "physics_domain"),
    ("grad_mae", "Gradient MAE", "lower", "physics_domain"),
    ("grad_w1", "Gradient Wasserstein-1", "lower", "physics_domain"),
    ("grad_kurtosis_abs_delta", "Gradient kurtosis |Delta|", "lower", "physics_domain"),
    ("exceed_frac_abs_delta_t5", "Exceedance |Delta|, s>5", "lower", "physics_domain"),
    ("exceed_frac_abs_delta_t10", "Exceedance |Delta|, s>10", "lower", "physics_domain"),
    ("exceed_frac_abs_delta_t15", "Exceedance |Delta|, s>15", "lower", "physics_domain"),
    ("exceed_frac_abs_delta_p90", "Exceedance |Delta|, p90", "lower", "physics_domain"),
    ("exceed_frac_abs_delta_p95", "Exceedance |Delta|, p95", "lower", "physics_domain"),
    ("exceed_frac_abs_delta_p99", "Exceedance |Delta|, p99", "lower", "physics_domain"),
]

RANKABLE_METRICS = [
    ("psnr", "PSNR_uv", "higher", "standard"),
    ("ssim", "SSIM_speed", "higher", "standard"),
    ("mt_distance", "Merge tree distance", "lower", "topology"),
    ("pd_distance", "Bottleneck PD distance", "lower", "topology"),
    ("wpd_bias", "WPD bias", "abs_lower", "physics_domain"),
    ("wpd_mae", "WPD MAE", "lower", "physics_domain"),
    ("wpd_rmse", "WPD RMSE", "lower", "physics_domain"),
    ("wpd_w1", "WPD Wasserstein-1", "lower", "physics_domain"),
    ("psd_log_l2", "PSD log-L2", "lower", "physics_domain"),
    ("psd_slope_delta", "PSD slope signed delta", "abs_lower", "physics_domain"),
    ("psd_slope_abs_delta", "PSD slope absolute delta", "lower", "physics_domain"),
    ("grad_mae", "Gradient MAE", "lower", "physics_domain"),
    ("grad_w1", "Gradient Wasserstein-1", "lower", "physics_domain"),
    ("grad_kurtosis_delta", "Gradient kurtosis signed delta", "abs_lower", "physics_domain"),
    ("grad_kurtosis_abs_delta", "Gradient kurtosis absolute delta", "lower", "physics_domain"),
    ("exceed_frac_delta_t5", "Exceedance signed delta, s>5", "abs_lower", "physics_domain"),
    ("exceed_frac_abs_delta_t5", "Exceedance absolute delta, s>5", "lower", "physics_domain"),
    ("exceed_frac_delta_t10", "Exceedance signed delta, s>10", "abs_lower", "physics_domain"),
    ("exceed_frac_abs_delta_t10", "Exceedance absolute delta, s>10", "lower", "physics_domain"),
    ("exceed_frac_delta_t15", "Exceedance signed delta, s>15", "abs_lower", "physics_domain"),
    ("exceed_frac_abs_delta_t15", "Exceedance absolute delta, s>15", "lower", "physics_domain"),
    ("exceed_frac_delta_p90", "Exceedance signed delta, p90", "abs_lower", "physics_domain"),
    ("exceed_frac_abs_delta_p90", "Exceedance absolute delta, p90", "lower", "physics_domain"),
    ("exceed_frac_delta_p95", "Exceedance signed delta, p95", "abs_lower", "physics_domain"),
    ("exceed_frac_abs_delta_p95", "Exceedance absolute delta, p95", "lower", "physics_domain"),
    ("exceed_frac_delta_p99", "Exceedance signed delta, p99", "abs_lower", "physics_domain"),
    ("exceed_frac_abs_delta_p99", "Exceedance absolute delta, p99", "lower", "physics_domain"),
]

PHYSICS_GROUP_COMPONENTS = [
    ("wpd_rmse", "lower"),
    ("wpd_mae", "lower"),
    ("psd_log_l2", "lower"),
    ("grad_mae", "lower"),
]

def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def fnum(x):
    if x is None or x == "":
        return math.nan
    return float(x)

def winner(cnn, gan, criterion, tol=1e-12):
    cv = abs(cnn) if criterion == "abs_lower" else cnn
    gv = abs(gan) if criterion == "abs_lower" else gan
    if math.isclose(cv, gv, rel_tol=0.0, abs_tol=tol):
        return "tie"
    if criterion == "higher":
        return "cnn" if cv > gv else "gan"
    return "cnn" if cv < gv else "gan"

def score_value(x, criterion):
    return abs(x) if criterion == "abs_lower" else x

def paired(rows):
    out = {}
    for row in rows:
        idx = int(row["sample_idx"])
        out.setdefault(idx, {})[row["method"].lower()] = row
    missing = [idx for idx, v in out.items() if set(v) != {"cnn", "gan"}]
    if missing:
        raise ValueError(f"Missing CNN/GAN pair for sample(s): {missing[:10]}")
    return dict(sorted(out.items()))

def counts(vals):
    c = Counter(vals)
    return c.get("cnn", 0), c.get("gan", 0), c.get("tie", 0)

def pct(n, d):
    return "na" if d == 0 else f"{100*n/d:.1f}%"

def safe_mean(vals):
    vals = [v for v in vals if not math.isnan(v)]
    return "" if not vals else sum(vals) / len(vals)

def safe_min(vals):
    vals = [v for v in vals if not math.isnan(v)]
    return "" if not vals else min(vals)

def safe_max(vals):
    vals = [v for v in vals if not math.isnan(v)]
    return "" if not vals else max(vals)

def std(vals):
    vals = [v for v in vals if not math.isnan(v)]
    if len(vals) < 2:
        return ""
    m = sum(vals) / len(vals)
    return math.sqrt(sum((x-m)**2 for x in vals) / (len(vals) - 1))

def table_near_ties(pairs):
    out = []
    for t in THRESHOLDS:
        n = sum(abs(fnum(v["cnn"]["ssim"]) - fnum(v["gan"]["ssim"])) <= t for v in pairs.values())
        interp = "Too small" if t == 0.03 else "Robustness check" if t == 0.05 else "Primary regime" if t == 0.075 else "Broader but looser"
        out.append({"epsilon": t, "near_tie_count": n, "interpretation": interp})
    return out

def physics_group_winner(crow, grow):
    ws = []
    for col, crit in PHYSICS_GROUP_COMPONENTS:
        ws.append(winner(fnum(crow[col]), fnum(grow[col]), crit))
    c = Counter(ws)
    if c["cnn"] > c["gan"]:
        return "cnn"
    if c["gan"] > c["cnn"]:
        return "gan"
    return "tie"

def table_all_selector_counts(pairs):
    categories = {
        "PSNR_uv": [],
        "SSIM": [],
        "Merge tree (MT)": [],
        "Bottleneck PD": [],
        "Physics group majority": [],
    }
    mt_gan_samples = []
    for idx, v in pairs.items():
        c, g = v["cnn"], v["gan"]
        categories["PSNR_uv"].append(winner(fnum(c["psnr"]), fnum(g["psnr"]), "higher"))
        categories["SSIM"].append(winner(fnum(c["ssim"]), fnum(g["ssim"]), "higher"))
        mtw = winner(fnum(c["mt_distance"]), fnum(g["mt_distance"]), "lower")
        categories["Merge tree (MT)"].append(mtw)
        if mtw == "gan":
            mt_gan_samples.append(idx)
        categories["Bottleneck PD"].append(winner(fnum(c["pd_distance"]), fnum(g["pd_distance"]), "lower"))
        categories["Physics group majority"].append(physics_group_winner(c, g))
    rows = []
    for name, winners in categories.items():
        cnn, gan, tie = counts(winners)
        rows.append({"selector_or_validator": name, "cnn_wins": cnn, "gan_wins": gan, "ties_or_unavailable": tie})
    return rows, mt_gan_samples

def table_metric_winner_counts(pairs, metrics):
    rows = []
    for col, label, crit, category in metrics:
        if col not in next(iter(pairs.values()))["cnn"]:
            continue
        ws = []
        cnn_vals, gan_vals, deltas = [], [], []
        gan_samples = []
        for idx, v in pairs.items():
            cv, gv = fnum(v["cnn"][col]), fnum(v["gan"][col])
            w = winner(cv, gv, crit)
            ws.append(w)
            if w == "gan":
                gan_samples.append(idx)
            cnn_vals.append(cv)
            gan_vals.append(gv)
            deltas.append(cv - gv)
        cnn, gan, tie = counts(ws)
        rows.append({
            "metric": label,
            "source_column": col,
            "category": category,
            "criterion": crit,
            "cnn_wins": cnn,
            "gan_wins": gan,
            "ties": tie,
            "mean_cnn": safe_mean(cnn_vals),
            "mean_gan": safe_mean(gan_vals),
            "mean_cnn_minus_gan": safe_mean(deltas),
            "gan_win_samples": " ".join(map(str, gan_samples)),
        })
    return rows

def metric_sweep_long(pairs, metrics):
    rows = []
    for idx, v in pairs.items():
        for col, label, crit, category in metrics:
            if col not in v["cnn"]:
                continue
            cv, gv = fnum(v["cnn"][col]), fnum(v["gan"][col])
            rows.append({
                "sample_idx": idx,
                "metric": label,
                "source_column": col,
                "category": category,
                "criterion": crit,
                "cnn_value": cv,
                "gan_value": gv,
                "cnn_score_for_winner": score_value(cv, crit),
                "gan_score_for_winner": score_value(gv, crit),
                "delta_cnn_minus_gan": cv - gv,
                "winner": winner(cv, gv, crit),
            })
    return rows

def metric_sweep_wide(long_rows):
    by = defaultdict(dict)
    for r in long_rows:
        idx = r["sample_idx"]
        slug = r["source_column"]
        by[idx]["sample_idx"] = idx
        by[idx][f"{slug}_cnn"] = r["cnn_value"]
        by[idx][f"{slug}_gan"] = r["gan_value"]
        by[idx][f"{slug}_delta_cnn_minus_gan"] = r["delta_cnn_minus_gan"]
        by[idx][f"{slug}_winner"] = r["winner"]
    keys = ["sample_idx"]
    for r in long_rows:
        slug = r["source_column"]
        for suffix in ["cnn", "gan", "delta_cnn_minus_gan", "winner"]:
            k = f"{slug}_{suffix}"
            if k not in keys:
                keys.append(k)
    return [by[idx] for idx in sorted(by)], keys

def raw_numeric_long(rows):
    exclude = {"method", "sample_idx", "key"}
    numeric_cols = []
    for col in rows[0].keys():
        if col in exclude:
            continue
        ok = True
        for r in rows:
            try:
                fnum(r[col])
            except Exception:
                ok = False
                break
        if ok:
            numeric_cols.append(col)
    out = []
    for r in rows:
        for col in numeric_cols:
            out.append({"method": r["method"], "sample_idx": r["sample_idx"], "metric_column": col, "value": r[col]})
    return out, numeric_cols

def unranked_columns(rows, ranked):
    ranked_cols = {m[0] for m in ranked}
    exclude = {"method", "sample_idx", "key"}
    out = []
    for col in rows[0].keys():
        if col in exclude or col in ranked_cols:
            continue
        reason = "GT/SR/reference value or duplicate raw component; use corresponding error or abs_delta column for winner comparisons"
        out.append({"source_column": col, "reason_not_ranked": reason})
    return out

def table_ablation(ablation_dir):
    rows = []
    for eps, tag in ABLT.items():
        p = Path(ablation_dir) / f"selector_ablation_threshold_{tag}.csv"
        if not p.exists():
            continue
        data = read_csv(p)
        row = {"epsilon": eps}
        for desc, prefix, col in [
            ("mt_lr", "MT", "agree_mt_winner_lr_group"),
            ("mt_extreme", "MT", "agree_mt_winner_extreme_group"),
            ("mt_physics", "MT", "agree_mt_winner_physics_group"),
            ("pd_lr", "PD", "agree_pd_winner_lr_group"),
            ("pd_extreme", "PD", "agree_pd_winner_extreme_group"),
            ("pd_physics", "PD", "agree_pd_winner_physics_group"),
        ]:
            vals = [r[col] for r in data if r.get(col, "na") != "na"]
            n = sum(str(v) == "1" for v in vals)
            d = len(vals)
            row[f"{desc}_num"] = n
            row[f"{desc}_den"] = d
            row[f"{desc}_pct"] = pct(n, d)
        rows.append(row)
    return rows

def validator_majority(row):
    vals = [row.get("winner_lr_group"), row.get("winner_extreme_group"), row.get("winner_physics_group")]
    vals = [v for v in vals if v in {"cnn", "gan"}]
    c = Counter(vals)
    if c["cnn"] > c["gan"]:
        return "cnn"
    if c["gan"] > c["cnn"]:
        return "gan"
    return "tie"

def table_validator_counts(ablation_dir):
    rows = []
    for eps, tag in ABLT.items():
        p = Path(ablation_dir) / f"selector_ablation_threshold_{tag}.csv"
        if not p.exists():
            continue
        data = read_csv(p)
        for name, col in [("LR group", "winner_lr_group"), ("Extreme group", "winner_extreme_group"), ("Physics group", "winner_physics_group")]:
            c = Counter(r.get(col, "") for r in data)
            rows.append({"epsilon": eps, "validator": name, "cnn_wins": c["cnn"], "gan_wins": c["gan"], "ties": c["tie"], "majority_cnn_gan": ""})
        maj = Counter(validator_majority(r) for r in data)
        rows.append({"epsilon": eps, "validator": "3-validator majority", "cnn_wins": maj["cnn"], "gan_wins": maj["gan"], "ties": maj["tie"], "majority_cnn_gan": f"{maj['cnn']}/{maj['gan']}"})
    return rows

def opposite_cases(ablation_dir):
    rows = []
    for eps, tag in ABLT.items():
        p = Path(ablation_dir) / f"selector_ablation_threshold_{tag}.csv"
        if not p.exists():
            continue
        for r in read_csv(p):
            ssim_w = winner(fnum(r["ssim_cnn"]), fnum(r["ssim_gan"]), "higher")
            mt_w = r["winner_mt"]
            if ssim_w != mt_w:
                rows.append({
                    "epsilon": eps,
                    "sample_idx": int(r["sample_idx"]),
                    "ssim_winner": ssim_w,
                    "mt_winner": mt_w,
                    "pd_winner": r["winner_pd"],
                    "validator_majority": validator_majority(r),
                    "winner_lr_group": r.get("winner_lr_group", ""),
                    "winner_extreme_group": r.get("winner_extreme_group", ""),
                    "winner_physics_group": r.get("winner_physics_group", ""),
                    "mt_support_count": sum(1 for k in ["winner_lr_group", "winner_extreme_group", "winner_physics_group"] if r.get(k) == mt_w),
                })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged-csv", default="ttk_runs_fixed/combined/psnr_topology_physics_merged.csv")
    ap.add_argument("--selector-ablation-dir", default="ttk_runs_fixed/selector_ablation_full")
    ap.add_argument("--out-dir", default="ttk_runs_fixed/report_tables")
    args = ap.parse_args()
    rows = read_csv(args.merged_csv)
    pairs = paired(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    near = table_near_ties(pairs)
    all_counts, mt_gan_samples = table_all_selector_counts(pairs)
    domain_counts = table_metric_winner_counts(pairs, TABLE_DOMAIN_METRICS)
    all_metric_counts = table_metric_winner_counts(pairs, RANKABLE_METRICS)
    long_rows = metric_sweep_long(pairs, RANKABLE_METRICS)
    wide_rows, wide_fields = metric_sweep_wide(long_rows)
    raw_rows, numeric_cols = raw_numeric_long(rows)
    unranked = unranked_columns(rows, RANKABLE_METRICS)

    write_csv(out / "table_near_tie_counts.csv", near)
    write_csv(out / "table_all_sample_selector_winner_counts.csv", all_counts)
    write_csv(out / "table_domain_metric_winner_counts.csv", domain_counts)
    write_csv(out / "metric_winner_summary_all_rankable.csv", all_metric_counts)
    write_csv(out / "metric_sweep_all_samples_long.csv", long_rows)
    write_csv(out / "metric_sweep_all_samples_wide.csv", wide_rows, wide_fields)
    write_csv(out / "raw_numeric_measurements_long.csv", raw_rows)
    write_csv(out / "unranked_source_columns.csv", unranked)
    write_csv(out / "mt_gan_diagnostic_samples.csv", [{"sample_idx": s} for s in mt_gan_samples])

    if Path(args.selector_ablation_dir).exists():
        write_csv(out / "table_selector_ablation_agreement.csv", table_ablation(args.selector_ablation_dir))
        write_csv(out / "table_near_tie_validator_winners.csv", table_validator_counts(args.selector_ablation_dir))
        write_csv(out / "table_opposite_direction_cases.csv", opposite_cases(args.selector_ablation_dir))

    note = []
    note.append("# Report table and metric sweep generation\n")
    note.append(f"Input merged CSV: `{args.merged_csv}`\n")
    note.append(f"Selector ablation directory: `{args.selector_ablation_dir}`\n")
    note.append(f"Output directory: `{args.out_dir}`\n")
    note.append(f"Paired samples checked: {len(pairs)}\n")
    note.append(f"Numeric source columns exported in raw long form: {len(numeric_cols)}\n")
    note.append("\nGenerated CSVs:\n")
    for p in sorted(out.glob("*.csv")):
        note.append(f"- `{p.name}`\n")
    (out / "README_report_tables.md").write_text("".join(note))
    print(f"Generated report tables and metric sweep CSVs in {out}")

if __name__ == "__main__":
    main()
