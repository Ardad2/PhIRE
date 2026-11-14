#!/usr/bin/env python3
"""Compare Phase 6 gradient-only and archived F1 cheap-evaluation reports.

Pure Python 3 standard library; reads two Markdown report summary tables.
Outputs report-level mean comparisons, not per-field paired statistics.
"""
import argparse
import math
import re
from pathlib import Path

KEYS = [
    "PSNRuv (dB)", "Speed MAE (m/s)", "Speed RMSE (m/s)",
    "WPD MAE (m³/s³)", "WPD Wasserstein-1", "Gradient MAE",
    "Gradient W1", "PSD log-L2", "PSD slope abs Δ",
    "Component-count curve L1", "SSIM on speed"
]


def parse_report(path: Path):
    raw = path.read_text(encoding='utf-8')
    m = re.search(r'\*\*Candidate:\*\*\s*`([^`]+)`', raw)
    if not m:
        raise ValueError(f'Candidate name not found: {path}')
    if '## Summary metrics table' not in raw:
        raise ValueError(f'Summary metrics table not found: {path}')
    block = raw.split('## Summary metrics table', 1)[1].split('\n## ', 1)[0]
    data = {}
    for line in block.splitlines():
        if not line.startswith('|'):
            continue
        parts = [x.strip() for x in line.strip().strip('|').split('|')]
        if len(parts) != 5 or parts[0] in ('Metric', '---') or parts[0].startswith('---'):
            continue
        vals = []
        for value in parts[1:]:
            try:
                vals.append(float(value))
            except ValueError:
                vals.append(math.nan)
        data[parts[0]] = vals
    return m.group(1), data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--control', type=Path, required=True)
    ap.add_argument('--f1', type=Path, required=True)
    args = ap.parse_args()
    c_name, c_data = parse_report(args.control)
    f_name, f_data = parse_report(args.f1)
    print(f'Control: {c_name} ({args.control})')
    print(f'F1:      {f_name} ({args.f1})')
    print('Mean comparison only; candidate topology distances are not included.')
    print('| Metric | CNN | Gradient-only | F1 | Control−F1 |')
    print('|---|---:|---:|---:|---:|')
    for metric in KEYS:
        if metric not in c_data or metric not in f_data:
            print(f'| {metric} | MISSING | MISSING | MISSING | MISSING |')
            continue
        cc, cv = c_data[metric][1], c_data[metric][3]
        fc, fv = f_data[metric][1], f_data[metric][3]
        if math.isfinite(cc) and math.isfinite(fc) and not math.isclose(cc,fc,abs_tol=1e-4):
            raise ValueError(f'CNN baseline disagreement on {metric}: control={cc} F1={fc}')
        def fmt(v):
            return f'{v:.4f}' if math.isfinite(v) else 'N/A'
        diff = cv - fv
        print(f'| {metric} | {fmt(cc)} | {fmt(cv)} | {fmt(fv)} | {fmt(diff)} |')


if __name__ == '__main__':
    main()
