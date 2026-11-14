#!/usr/bin/env python3
"""Phase 5E presentation figures from the released frozen summary CSVs.

Run from the PhIRE repository root:
  python3 scripts/plot_phase5_spatial_summary.py --out-dir phase6_figures

Uses ONLY the Phase5E all168 macro and closeout summaries. Does not
change the audited spatial matching calculations or frozen manuscript data.
"""
import argparse
import csv
from pathlib import Path
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
MACRO_DEFAULT = ROOT / 'spatial_pd/phase5e_all168_locality_analysis/all168_locality_macro_summary.csv'
HIST_DEFAULT = ROOT / 'spatial_pd/phase5e_all168_closeout_summary/stable_radius_histogram.csv'
RADII = ['4.0', '8.0', '16.0', '32.0', '64.0', '96.0', '128.0', 'inf']
METHODS = [('CNN', 'Pretrained CNN'), ('UV', 'Reconstruction-only FT'), ('F1', 'Topology-inspired F1')]


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def plot_curves(rows, field, ylabel, title, stem, outdir, scale=1.0):
    xs = list(range(len(RADII)))
    fig, ax = plt.subplots(figsize=(8.1, 4.8))
    for method, label in METHODS:
        d = {r['radius']: r for r in rows if r['method'] == method}
        assert all(k in d for k in RADII), f'Missing radius for {method}'
        vals = [float(d[r][field]) * scale for r in RADII]
        assert all(math.isfinite(x) for x in vals)
        ax.plot(xs, vals, marker='o', linewidth=2, markersize=5, label=label)
    ax.set_xticks(xs, ['4', '8', '16', '32', '64', '96', '128', '$\\infty$'])
    ax.set_xlabel('Spatial admissibility radius $r$ (grid pixels; categorical axis)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / f'{stem}.png', dpi=240)
    fig.savefig(outdir / f'{stem}.svg')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--macro', type=Path, default=MACRO_DEFAULT)
    ap.add_argument('--hist', type=Path, default=HIST_DEFAULT)
    ap.add_argument('--out-dir', type=Path, default=ROOT / 'phase6_figures')
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.macro)
    assert len(rows) == 24, f'Expected 3 methods x 8 radii, got {len(rows)}'
    assert all(int(r['n']) == 168 for r in rows), 'Expected 168 fields per curve cell'
    plot_curves(rows, 'mean_constrained_w22_all',
                'Mean locality-constrained persistence cost',
                'Cost of persistence matching under spatial locality',
                'phase5_constrained_cost', args.out_dir)
    plot_curves(rows, 'mean_gt_persistence_coverage_all',
                'Mean GT persistence-mass coverage (%)',
                'Ground-truth structure represented by local correspondences',
                'phase5_gt_coverage', args.out_dir, scale=100)
    plot_curves(rows, 'mean_sr_persistence_coverage_all',
                'Mean SR persistence-mass coverage (%)',
                'Fraction of predicted persistence matched locally',
                'phase5_sr_coverage', args.out_dir, scale=100)

    hist = read_csv(args.hist)
    labels = [r['first_stable_radius'] for r in hist]
    counts = [int(r['count']) for r in hist]
    assert sum(counts) == 168, f'Histogram does not sum to 168: {sum(counts)}'
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    bars = ax.bar(range(len(labels)), counts)
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, n + 1.5, str(n), ha='center', fontsize=10)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlabel('First radius of sustained F1 cost advantage over both baselines')
    ax.set_ylabel('Number of benchmark fields')
    ax.set_title('Stable joint cost advantage across spatial scales (n=168)')
    ax.set_ylim(0, max(counts)*1.18)
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)
    fig.tight_layout()
    fig.savefig(args.out_dir / 'phase5_stable_advantage.png', dpi=240)
    fig.savefig(args.out_dir / 'phase5_stable_advantage.svg')
    plt.close(fig)
    print(f'PASS: verified {len(rows)} macro cells and {sum(counts)} histogram samples.')
    print(f'Saved 4 PNG + 4 SVG to {args.out_dir}')


if __name__ == '__main__':
    main()
