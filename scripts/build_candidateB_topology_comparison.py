#!/usr/bin/env python3
"""Build topology comparison CSV and markdown report for Candidate B.

Reads:
  - Candidate B phase_c_results.csv  (from run_candidateB_topology_pipeline.sh)
  - Baseline combined phase_c_results.csv  (CNN and GAN, pre-computed)
  - Candidate B idx.npy  (to map array positions to TFRecord sample IDs)

Writes:
  <out-dir>/candidateB_pd_mt_distances.csv
  <out-dir>/candidateB_topology_comparison.csv
  <report-path>  docs/topology_finetuning_candidateB_topology_eval.md

IMPORTANT: Existing CNN/GAN topology outputs are never overwritten.

Usage (from repo root):
  python3 scripts/build_candidateB_topology_comparison.py \\
      --candidateB-results  ttk_runs_fixed/topology_finetuning/candidateB_topology/phase_c_final/phase_c_results.csv \\
      --baseline-results    ttk_runs_fixed/combined/phase_c_results.csv \\
      --candidateB-idx      data_out/wind_finetune_pilot_candidateB/idx.npy \\
      --out-dir             ttk_runs_fixed/topology_finetuning/candidateB_topology \\
      --report-path         docs/topology_finetuning_candidateB_topology_eval.md
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent

# The 20 samples where GAN beats CNN on MT distance in the baseline.
# Computed from ttk_runs_fixed/combined/phase_c_results.csv.
# Re-validated at runtime if the baseline CSV is available.
_KNOWN_MT_GAN_WINS = {
    6, 8, 12, 16, 17, 18, 19, 20, 25, 48,
    62, 63, 65, 68, 77, 79, 80, 82, 92, 154,
}

ADJACENT_CLUSTER_SAMPLES = [10, 11, 12, 13, 90, 91, 92, 93]
RARE_CONTROL_SAMPLES     = [162, 163]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--candidateB-results', required=True,
                    help='phase_c_results.csv from Candidate B TTK run')
    ap.add_argument('--baseline-results', required=True,
                    help='Combined baseline phase_c_results.csv (CNN + GAN)')
    ap.add_argument('--candidateB-idx', required=True,
                    help='idx.npy from Candidate B inference output')
    ap.add_argument('--out-dir', required=True,
                    help='Output directory for CSVs')
    ap.add_argument('--report-path',
                    default='docs/topology_finetuning_candidateB_topology_eval.md',
                    help='Markdown report path')
    return ap.parse_args()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_KEY_RE = re.compile(r'_s(\d+)_')

def _parse_pos_idx(key: str) -> Optional[int]:
    """Extract the positional array index N from a key like 'cnn_sN_speed_p160_x0_y0'."""
    m = _KEY_RE.search(key)
    return int(m.group(1)) if m else None


def _f(x: object) -> float:
    try:
        v = float(str(x).strip())
        return float('nan') if math.isnan(v) else v
    except (TypeError, ValueError):
        return float('nan')


def _load_phase_c(path: Path, method_filter: Optional[str] = None,
) -> Dict[int, Dict[str, float]]:
    """Return {pos_idx: {pd_distance, mt_distance}} from a phase_c_results.csv."""
    result: Dict[int, Dict[str, float]] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            method = row.get('method', '').strip()
            if method_filter and method.lower() != method_filter.lower():
                continue
            key = row.get('key', '').strip()
            pos = _parse_pos_idx(key)
            if pos is None:
                continue
            pd = _f(row.get('pd_distance', ''))
            mt = _f(row.get('mt_distance', ''))
            result[pos] = {'pd_distance': pd, 'mt_distance': mt}
    return result


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _fmt(v: float, d: int = 4) -> str:
    return f'{v:.{d}f}' if not math.isnan(v) else 'N/A'


def _winner(*pairs: Tuple[str, float]) -> str:
    """Return name of method with lowest (best) distance value."""
    valid = [(n, v) for n, v in pairs if not math.isnan(v)]
    if not valid:
        return 'N/A'
    return min(valid, key=lambda x: x[1])[0]


def _relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    cand_results_p  = Path(args.candidateB_results)
    baseline_p      = Path(args.baseline_results)
    cand_idx_p      = Path(args.candidateB_idx)
    out_dir         = Path(args.out_dir)
    report_path     = Path(args.report_path)

    # Validate inputs
    for p in (cand_results_p, baseline_p, cand_idx_p):
        if not p.exists():
            sys.exit(f'[error] Required input not found: {p}')

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load candidate B idx.npy: pos_idx → actual TFRecord sample_idx
    # -----------------------------------------------------------------------
    cand_idx = np.load(cand_idx_p, allow_pickle=False).astype(int).ravel()
    pos_to_sample: Dict[int, int] = {i: int(cand_idx[i]) for i in range(len(cand_idx))}
    print(f'Loaded Candidate B idx.npy: {len(cand_idx)} samples, '
          f'range [{cand_idx.min()}–{cand_idx.max()}]')

    # -----------------------------------------------------------------------
    # Load Candidate B distances: pos_idx → {pd, mt}
    # -----------------------------------------------------------------------
    cand_dists = _load_phase_c(cand_results_p, method_filter='candidateB')
    print(f'Candidate B topology entries: {len(cand_dists)} '
          f'(PD valid: {sum(1 for v in cand_dists.values() if not math.isnan(v["pd_distance"]))}, '
          f'MT valid: {sum(1 for v in cand_dists.values() if not math.isnan(v["mt_distance"]))})')

    # -----------------------------------------------------------------------
    # Load baseline CNN and GAN distances: pos_idx → {pd, mt}
    # -----------------------------------------------------------------------
    cnn_dists = _load_phase_c(baseline_p, method_filter='cnn')
    gan_dists = _load_phase_c(baseline_p, method_filter='gan')
    print(f'Baseline CNN entries: {len(cnn_dists)}, GAN entries: {len(gan_dists)}')

    # -----------------------------------------------------------------------
    # Compute MT-GAN win set from loaded data (validates against known set)
    # -----------------------------------------------------------------------
    mt_gan_wins: set = set()
    for pos in cnn_dists:
        if pos in gan_dists:
            cnn_mt = cnn_dists[pos]['mt_distance']
            gan_mt = gan_dists[pos]['mt_distance']
            if not (math.isnan(cnn_mt) or math.isnan(gan_mt)) and gan_mt < cnn_mt:
                mt_gan_wins.add(pos)
    if mt_gan_wins != _KNOWN_MT_GAN_WINS:
        print(f'[note] MT-GAN wins from baseline: {sorted(mt_gan_wins)} '
              f'(hardcoded set has {len(_KNOWN_MT_GAN_WINS)} entries)')
    else:
        print(f'[ok] MT-GAN wins validated: {len(mt_gan_wins)} samples')

    # -----------------------------------------------------------------------
    # Build per-sample distance CSV (Candidate B only)
    # -----------------------------------------------------------------------
    cand_only_rows: List[Dict] = []
    for pos in sorted(cand_dists):
        sample_idx = pos_to_sample.get(pos, pos)
        d = cand_dists[pos]
        cand_only_rows.append({
            'sample_idx':              sample_idx,
            'pos_idx':                 pos,
            'pd_distance_candidateB':  d['pd_distance'],
            'mt_distance_candidateB':  d['mt_distance'],
        })
    cand_only_csv = out_dir / 'candidateB_pd_mt_distances.csv'
    _write_csv(cand_only_csv, cand_only_rows)
    print(f'Written: {cand_only_csv}')

    # -----------------------------------------------------------------------
    # Build merged comparison CSV (CNN + GAN + Candidate B)
    # -----------------------------------------------------------------------
    all_pos = sorted(set(cnn_dists) | set(gan_dists) | set(cand_dists))
    comp_rows: List[Dict] = []

    for pos in all_pos:
        sample_idx = pos_to_sample.get(pos, pos)
        cnn_pd = cnn_dists.get(pos, {}).get('pd_distance', float('nan'))
        cnn_mt = cnn_dists.get(pos, {}).get('mt_distance', float('nan'))
        gan_pd = gan_dists.get(pos, {}).get('pd_distance', float('nan'))
        gan_mt = gan_dists.get(pos, {}).get('mt_distance', float('nan'))
        cb_pd  = cand_dists.get(pos, {}).get('pd_distance', float('nan'))
        cb_mt  = cand_dists.get(pos, {}).get('mt_distance', float('nan'))

        pd_winner_before = _winner(('cnn', cnn_pd), ('gan', gan_pd))
        pd_winner_after  = _winner(('cnn', cnn_pd), ('gan', gan_pd), ('candidateB', cb_pd))
        mt_winner_before = _winner(('cnn', cnn_mt), ('gan', gan_mt))
        mt_winner_after  = _winner(('cnn', cnn_mt), ('gan', gan_mt), ('candidateB', cb_mt))

        comp_rows.append({
            'sample_idx':                    sample_idx,
            'pos_idx':                       pos,
            'pd_distance_cnn':               cnn_pd,
            'pd_distance_gan':               gan_pd,
            'pd_distance_candidateB':        cb_pd,
            'mt_distance_cnn':               cnn_mt,
            'mt_distance_gan':               gan_mt,
            'mt_distance_candidateB':        cb_mt,
            'pd_winner_before':              pd_winner_before,
            'pd_winner_after_candidateB':    pd_winner_after,
            'mt_winner_before':              mt_winner_before,
            'mt_winner_after_candidateB':    mt_winner_after,
            'was_mt_gan_win_before':         int(pos in mt_gan_wins),
        })

    comp_csv = out_dir / 'candidateB_topology_comparison.csv'
    _write_csv(comp_csv, comp_rows)
    print(f'Written: {comp_csv}')

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    def _mean(vals):
        v = [x for x in vals if not math.isnan(x)]
        return sum(v) / len(v) if v else float('nan')

    def _n_wins(method_vals, other_vals_list):
        wins = 0
        for mv, ovs in zip(method_vals, other_vals_list):
            if math.isnan(mv):
                continue
            if all((math.isnan(ov) or mv < ov) for ov in ovs):
                wins += 1
        return wins

    pd_cnn  = [r['pd_distance_cnn']        for r in comp_rows]
    pd_gan  = [r['pd_distance_gan']        for r in comp_rows]
    pd_cb   = [r['pd_distance_candidateB'] for r in comp_rows]
    mt_cnn  = [r['mt_distance_cnn']        for r in comp_rows]
    mt_gan  = [r['mt_distance_gan']        for r in comp_rows]
    mt_cb   = [r['mt_distance_candidateB'] for r in comp_rows]

    n_valid = len(comp_rows)

    pd_cnn_mean  = _mean(pd_cnn)
    pd_gan_mean  = _mean(pd_gan)
    pd_cb_mean   = _mean(pd_cb)
    mt_cnn_mean  = _mean(mt_cnn)
    mt_gan_mean  = _mean(mt_gan)
    mt_cb_mean   = _mean(mt_cb)

    # Candidate B vs CNN per-sample comparisons
    n_pd_cb_better_cnn = sum(
        1 for i in range(n_valid)
        if not (math.isnan(pd_cb[i]) or math.isnan(pd_cnn[i])) and pd_cb[i] < pd_cnn[i]
    )
    n_mt_cb_better_cnn = sum(
        1 for i in range(n_valid)
        if not (math.isnan(mt_cb[i]) or math.isnan(mt_cnn[i])) and mt_cb[i] < mt_cnn[i]
    )
    n_pd_cb_beats_gan = sum(
        1 for i in range(n_valid)
        if not (math.isnan(pd_cb[i]) or math.isnan(pd_gan[i])) and pd_cb[i] < pd_gan[i]
    )
    n_mt_cb_beats_gan = sum(
        1 for i in range(n_valid)
        if not (math.isnan(mt_cb[i]) or math.isnan(mt_gan[i])) and mt_cb[i] < mt_gan[i]
    )

    # MT-GAN wins: how many are now taken over by Candidate B?
    mt_gan_now_cb = sum(
        1 for r in comp_rows
        if r['was_mt_gan_win_before'] and r['mt_winner_after_candidateB'] == 'candidateB'
    )
    mt_gan_still_gan = sum(
        1 for r in comp_rows
        if r['was_mt_gan_win_before'] and r['mt_winner_after_candidateB'] == 'gan'
    )
    mt_gan_now_cnn = sum(
        1 for r in comp_rows
        if r['was_mt_gan_win_before'] and r['mt_winner_after_candidateB'] == 'cnn'
    )

    print(f'\nSummary:')
    print(f'  PD: CNN={_fmt(pd_cnn_mean)}, GAN={_fmt(pd_gan_mean)}, CandB={_fmt(pd_cb_mean)}')
    print(f'  MT: CNN={_fmt(mt_cnn_mean)}, GAN={_fmt(mt_gan_mean)}, CandB={_fmt(mt_cb_mean)}')
    print(f'  PD CandB < CNN: {n_pd_cb_better_cnn}/{n_valid}')
    print(f'  MT CandB < CNN: {n_mt_cb_better_cnn}/{n_valid}')
    print(f'  PD CandB < GAN: {n_pd_cb_beats_gan}/{n_valid}')
    print(f'  MT-GAN wins → CandB: {mt_gan_now_cb}/{len(mt_gan_wins)}  still GAN: {mt_gan_still_gan}  now CNN: {mt_gan_now_cnn}')

    # -----------------------------------------------------------------------
    # Adjacent and rare-control detail tables
    # -----------------------------------------------------------------------
    focus_samples = ADJACENT_CLUSTER_SAMPLES + RARE_CONTROL_SAMPLES

    # Build lookup by sample_idx
    by_sample: Dict[int, Dict] = {r['sample_idx']: r for r in comp_rows}

    def _focus_row(si: int) -> Optional[Dict]:
        return by_sample.get(si)

    # -----------------------------------------------------------------------
    # Markdown report
    # -----------------------------------------------------------------------
    import datetime
    md = io.StringIO()

    md.write('# Candidate B topology evaluation\n\n')
    md.write(f'**Generated:** {datetime.date.today().isoformat()}\n\n')
    md.write('**Important:** This report covers **true topology metrics** '
             '(PD bottleneck distance, MT Wasserstein distance) computed by TTK. '
             'These are distinct from the component-count *proxy* metrics in '
             '`topology_finetuning_candidateB_eval.md`.\n\n')
    md.write('**Candidate B:** `lambda_speed=0.01, lambda_grad=0.05, '
             'lambda_wpd=0.0, lambda_levelset=0.25, lr=1e-5, 3 epochs`\n\n')
    md.write(f'**Samples evaluated:** {n_valid}\n\n')
    md.write(f'**MT-GAN baseline wins (before Candidate B):** {len(mt_gan_wins)} samples — '
             f'{sorted(mt_gan_wins)}\n\n')

    md.write('## Summary statistics\n\n')
    md.write('| Metric | CNN (baseline) | GAN (baseline) | Candidate B |\n')
    md.write('|---|---|---|---|\n')
    md.write(f'| PD distance (mean) | {_fmt(pd_cnn_mean)} | {_fmt(pd_gan_mean)} | {_fmt(pd_cb_mean)} |\n')
    md.write(f'| MT distance (mean) | {_fmt(mt_cnn_mean)} | {_fmt(mt_gan_mean)} | {_fmt(mt_cb_mean)} |\n')
    md.write(f'| PD wins (vs CNN) | — | '
             f'{sum(1 for i in range(n_valid) if not (math.isnan(pd_gan[i]) or math.isnan(pd_cnn[i])) and pd_gan[i] < pd_cnn[i])} | '
             f'{n_pd_cb_better_cnn} |\n')
    md.write(f'| MT wins (vs CNN) | — | '
             f'{sum(1 for i in range(n_valid) if not (math.isnan(mt_gan[i]) or math.isnan(mt_cnn[i])) and mt_gan[i] < mt_cnn[i])} | '
             f'{n_mt_cb_better_cnn} |\n')
    md.write(f'| PD beats GAN | — | — | {n_pd_cb_beats_gan} |\n')
    md.write(f'| MT beats GAN | — | — | {n_mt_cb_beats_gan} |\n\n')

    md.write('## 6 Key evaluation questions\n\n')

    # Q1
    md.write('### Q1. Does Candidate B improve PD distance relative to baseline CNN?\n\n')
    delta_pd = pd_cb_mean - pd_cnn_mean
    direction_pd = '▲ worse' if delta_pd > 0 else '▼ better'
    md.write(f'- Mean PD: CNN={_fmt(pd_cnn_mean)}, CandidateB={_fmt(pd_cb_mean)}, '
             f'Δ={_fmt(delta_pd)} ({direction_pd})\n')
    md.write(f'- Candidate B has lower PD on **{n_pd_cb_better_cnn}/{n_valid}** samples.\n\n')

    # Q2
    md.write('### Q2. Does Candidate B ever beat GAN on PD distance?\n\n')
    md.write(f'- Candidate B beats GAN on PD for **{n_pd_cb_beats_gan}/{n_valid}** samples.\n')
    md.write(f'- Mean PD: GAN={_fmt(pd_gan_mean)}, CandidateB={_fmt(pd_cb_mean)}. '
             f'Δ={_fmt(pd_cb_mean - pd_gan_mean)} '
             f'({"▲ worse" if pd_cb_mean > pd_gan_mean else "▼ better"} than GAN on average)\n\n')

    # Q3
    md.write('### Q3. Does Candidate B improve MT distance relative to baseline CNN?\n\n')
    delta_mt = mt_cb_mean - mt_cnn_mean
    direction_mt = '▲ worse' if delta_mt > 0 else '▼ better'
    md.write(f'- Mean MT: CNN={_fmt(mt_cnn_mean)}, CandidateB={_fmt(mt_cb_mean)}, '
             f'Δ={_fmt(delta_mt)} ({direction_mt})\n')
    md.write(f'- Candidate B has lower MT on **{n_mt_cb_better_cnn}/{n_valid}** samples.\n\n')

    # Q4
    md.write('### Q4. Does Candidate B change any of the original 20 MT-GAN cases?\n\n')
    md.write(f'The 20 samples where GAN originally beat CNN on MT distance: '
             f'{sorted(mt_gan_wins)}\n\n')
    md.write(f'| Outcome | Count |\n|---|---|\n')
    md.write(f'| MT winner stays GAN | {mt_gan_still_gan} |\n')
    md.write(f'| MT winner changes to CandidateB | {mt_gan_now_cb} |\n')
    md.write(f'| MT winner changes to CNN | {mt_gan_now_cnn} |\n\n')
    if mt_gan_now_cb > 0 or mt_gan_now_cnn > 0:
        md.write('Per-sample detail for changed cases:\n\n')
        md.write('| sample_idx | MT CNN | MT GAN | MT CandB | winner before | winner after |\n')
        md.write('|---|---|---|---|---|---|\n')
        for r in comp_rows:
            if r['was_mt_gan_win_before'] and r['mt_winner_after_candidateB'] != 'gan':
                md.write(f'| {r["sample_idx"]} '
                         f'| {_fmt(r["mt_distance_cnn"])} '
                         f'| {_fmt(r["mt_distance_gan"])} '
                         f'| {_fmt(r["mt_distance_candidateB"])} '
                         f'| {r["mt_winner_before"]} '
                         f'| {r["mt_winner_after_candidateB"]} |\n')
        md.write('\n')
    else:
        md.write('No MT-GAN wins changed — GAN retains all 20 topology advantages.\n\n')

    # Q5: adjacent clusters
    md.write('### Q5. What happens in adjacent clusters (samples 10–13 and 90–93)?\n\n')
    md.write('| sample_idx | PD CNN | PD GAN | PD CandB | MT CNN | MT GAN | MT CandB | PD winner → | MT winner → |\n')
    md.write('|---|---|---|---|---|---|---|---|---|\n')
    for si in ADJACENT_CLUSTER_SAMPLES:
        r = _focus_row(si)
        if r is None:
            md.write(f'| {si} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |\n')
            continue
        md.write(f'| {si} '
                 f'| {_fmt(r["pd_distance_cnn"])} '
                 f'| {_fmt(r["pd_distance_gan"])} '
                 f'| {_fmt(r["pd_distance_candidateB"])} '
                 f'| {_fmt(r["mt_distance_cnn"])} '
                 f'| {_fmt(r["mt_distance_gan"])} '
                 f'| {_fmt(r["mt_distance_candidateB"])} '
                 f'| {r["pd_winner_before"]} → {r["pd_winner_after_candidateB"]} '
                 f'| {r["mt_winner_before"]} → {r["mt_winner_after_candidateB"]} |\n')
    md.write('\n')
    # Note which of these are in the MT-GAN-wins set
    adj_in_mt_gan = [si for si in ADJACENT_CLUSTER_SAMPLES if si in mt_gan_wins]
    if adj_in_mt_gan:
        md.write(f'Note: samples {adj_in_mt_gan} are in the MT-GAN-wins set (GAN beat CNN on MT '
                 f'in baseline).\n\n')

    # Q6: rare controls
    md.write('### Q6. What happens in rare topology-CNN controls (samples 162 and 163)?\n\n')
    md.write('These samples were originally topology-CNN controls (CNN wins on topology metrics). '
             'Candidate B should not reverse this.\n\n')
    md.write('| sample_idx | PD CNN | PD GAN | PD CandB | MT CNN | MT GAN | MT CandB | PD winner → | MT winner → |\n')
    md.write('|---|---|---|---|---|---|---|---|---|\n')
    for si in RARE_CONTROL_SAMPLES:
        r = _focus_row(si)
        if r is None:
            md.write(f'| {si} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |\n')
            continue
        md.write(f'| {si} '
                 f'| {_fmt(r["pd_distance_cnn"])} '
                 f'| {_fmt(r["pd_distance_gan"])} '
                 f'| {_fmt(r["pd_distance_candidateB"])} '
                 f'| {_fmt(r["mt_distance_cnn"])} '
                 f'| {_fmt(r["mt_distance_gan"])} '
                 f'| {_fmt(r["mt_distance_candidateB"])} '
                 f'| {r["pd_winner_before"]} → {r["pd_winner_after_candidateB"]} '
                 f'| {r["mt_winner_before"]} → {r["mt_winner_after_candidateB"]} |\n')
    md.write('\n')

    # Per-method PD/MT winner distribution
    md.write('## Winner distribution\n\n')
    for dist_name, w_before_col, w_after_col in [
        ('PD', 'pd_winner_before', 'pd_winner_after_candidateB'),
        ('MT', 'mt_winner_before', 'mt_winner_after_candidateB'),
    ]:
        before_counts: Dict[str, int] = {}
        after_counts:  Dict[str, int] = {}
        for r in comp_rows:
            before_counts[r[w_before_col]] = before_counts.get(r[w_before_col], 0) + 1
            after_counts[r[w_after_col]]   = after_counts.get(r[w_after_col], 0) + 1
        md.write(f'### {dist_name} distance winners\n\n')
        md.write(f'| Method | Wins before CandB | Wins after CandB |\n|---|---|---|\n')
        all_methods = sorted(set(before_counts) | set(after_counts))
        for m in all_methods:
            md.write(f'| {m} | {before_counts.get(m,0)} | {after_counts.get(m,0)} |\n')
        md.write('\n')

    # Full per-sample table for focus samples
    md.write('## Full per-sample detail: focus samples\n\n')
    focus_of_interest = sorted(set(focus_samples) | mt_gan_wins)
    md.write(f'Includes adjacent-cluster samples, rare controls, '
             f'and the 20 MT-GAN baseline wins.\n\n')
    md.write('| sample | PD CNN | PD GAN | PD CandB | MT CNN | MT GAN | MT CandB | '
             'MT-GAN-win? | PD after | MT after |\n')
    md.write('|---|---|---|---|---|---|---|---|---|---|\n')
    for si in sorted(focus_of_interest):
        r = _focus_row(si)
        if r is None:
            continue
        is_mt_gan = '✓' if r['was_mt_gan_win_before'] else ''
        md.write(f'| {si} '
                 f'| {_fmt(r["pd_distance_cnn"])} '
                 f'| {_fmt(r["pd_distance_gan"])} '
                 f'| {_fmt(r["pd_distance_candidateB"])} '
                 f'| {_fmt(r["mt_distance_cnn"])} '
                 f'| {_fmt(r["mt_distance_gan"])} '
                 f'| {_fmt(r["mt_distance_candidateB"])} '
                 f'| {is_mt_gan} '
                 f'| {r["pd_winner_after_candidateB"]} '
                 f'| {r["mt_winner_after_candidateB"]} |\n')
    md.write('\n')

    md.write('## Output files\n\n')
    md.write(f'- `{_relpath(cand_only_csv)}` — Candidate B PD/MT per sample\n')
    md.write(f'- `{_relpath(comp_csv)}` — three-way comparison (CNN, GAN, CandidateB)\n')
    md.write(f'- `{_relpath(report_path)}` — this report\n\n')

    md.write('## Notes\n\n')
    md.write('- PD distance = bottleneck distance between persistence diagrams '
             '(computed by `ttkBottleneckDistance`).\n')
    md.write('- MT distance = Wasserstein-type merge tree distance '
             '(computed by `ttkMergeTreeDistanceMatrix`).\n')
    md.write('- Both use 160×160 speed patches at (x0=0, y0=0), '
             'consistent with the CNN/GAN baseline.\n')
    md.write('- Lower distance = SR is topologically closer to GT. '
             'Lower is better for both metrics.\n')
    md.write('- **Component-count proxy metrics** (counts of connected components in '
             'speed superlevel sets) are reported separately in '
             '`topology_finetuning_candidateB_eval.md` and do not require TTK.\n')
    md.write('- Candidate B was fine-tuned for 3 epochs from the pretrained CNN checkpoint. '
             'Results may improve with longer training.\n')

    report_path.write_text(md.getvalue(), encoding='utf-8')
    print(f'Written: {report_path}')

    print('\n======================================')
    print('Topology comparison complete.')
    print(f'  Per-sample distances : {cand_only_csv}')
    print(f'  3-way comparison     : {comp_csv}')
    print(f'  Report               : {report_path}')
    print('======================================')


if __name__ == '__main__':
    main()
