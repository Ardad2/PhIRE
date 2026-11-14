#!/usr/bin/env python3
"""Compare Phase6 gradient-only, CV-only and historical F1 scalar reports.

Only Python standard library. Values are rounded by source reports. Lower is
better except PSNR. This is aggregate descriptive comparison, not paired
per-field testing and not a PD/MT or spatial-locality audit.
"""
import argparse
from pathlib import Path

KEY_METRICS = (
    'PSNRuv (dB)', 'Speed MAE (m/s)', 'Speed RMSE (m/s)',
    'WPD MAE (m³/s³)', 'WPD Wasserstein-1',
    'Gradient MAE', 'Gradient W1', 'PSD log-L2',
    'PSD slope abs Δ', 'Exceedance abs Δ s>5',
    'Exceedance abs Δ s>10', 'Exceedance abs Δ s>15',
    'Component-count curve L1',
)

def parse_report(path):
    text = Path(path).read_text(encoding='utf-8')
    rows = {}
    in_table = False
    for line in text.splitlines():
        if line.startswith('## Summary metrics table'):
            in_table = True
            continue
        if in_table and line.startswith('## '):
            break
        if not in_table or not line.startswith('|'):
            continue
        fields = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(fields) != 5 or fields[0] in ('Metric', '---'):
            continue
        try:
            # columns are: Bicubic, CNN, GAN, Candidate.
            values = [float(v) for v in fields[1:]]
        except ValueError:
            continue
        rows[fields[0]] = values
    if not rows:
        raise ValueError(f'No numeric 5-column table found in {path}')
    return rows

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grad-only', default='docs/topology_finetuning_candidateF_grad_pairzero_expanded2688_eval.md')
    p.add_argument('--cv-only', default='docs/topology_finetuning_candidateF_grad_CV_only_expanded2688_eval.md')
    p.add_argument('--f1', default='docs/topology_finetuning_candidateF_grad_E2_low_expanded2688_eval.md')
    args = p.parse_args()
    grad, cv, f1 = [parse_report(getattr(args, name)) for name in ('grad_only', 'cv_only', 'f1')]
    print('Metric | CNN | Gradient-only | Grad+CV | F1 | CV minus gradient-only')
    print('---|---:|---:|---:|---:|---:')
    for k in KEY_METRICS:
        if not all(k in d for d in (grad, cv, f1)):
            raise ValueError(f'Missing key metric {k!r} from a source report')
        cnn = cv[k][1]
        g = grad[k][3]
        c = cv[k][3]
        f = f1[k][3]
        if abs(grad[k][1]-cnn) > 0.0005 or abs(f1[k][1]-cnn) > 0.0005:
            raise ValueError(f'CNN anchor mismatch for {k}: {grad[k][1]}, {cnn}, {f1[k][1]}')
        print(f'{k} | {cnn:.4f} | {g:.4f} | {c:.4f} | {f:.4f} | {c-g:+.4f}')
    print('\nNote: summary means only; no new PD/MT/spatial-locality computation. SSIM unavailable in these reports.')

if __name__ == '__main__':
    main()
