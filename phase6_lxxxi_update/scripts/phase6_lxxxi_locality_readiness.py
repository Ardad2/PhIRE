#!/usr/bin/env python3
"""Phase 6 LXXXI: read-only inventory and Phase 5E locality source discovery.

This is NOT the locality solver and does not attribute model checkpoints to training
recipes. No experiment output changes and no dependency on the older TTK VTK reader.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import re
import sys

EXPECTED_CSV_SHA256='3d53189ec69cc2f89c5561159d0768c973351d7b9998deceeaef42ebb6b379ea'
ARMS=('cnn','grad','cv','pers','f1')
NEW={'grad':'candidateF_grad_pairzero_expanded2688','cv':'candidateF_grad_CV_only_expanded2688','pers':'candidateF_grad_PERS_only_expanded2688'}
ROOT5=Path('spatial_pd/phase5e_all168_canonical_pd/pd')
ROOT6=Path('ttk_runs_fixed/topology_finetuning/phase6_lxxvii_canonical_pd_newarms_20261014')
ROOT9=Path('ttk_runs_fixed/topology_finetuning/phase6_lxxix_corrected_pd_gudhi_20261014')
ROOT10=Path('ttk_runs_fixed/topology_finetuning/phase6_lxxx_paired_effect_audit_20261014')


def need(condition, message):
    if not condition:raise RuntimeError(message)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def inventory(folder,prefix):
    need(folder.is_dir(),f'Missing directory: {folder}')
    files=list(folder.glob('*.vtu'))
    matched={}
    regex=re.compile(re.escape(prefix)+r's(\d+)_speed_p160_x0_y0_pd_port_0\.vtu$')
    for path in files:
        match=regex.fullmatch(path.name)
        if match:
            i=int(match.group(1))
            need(i not in matched, f'Duplicate index {i} for {prefix}')
            matched[i]=path
    need(sorted(matched)==list(range(168)),f'Incomplete {prefix}: found={len(matched)}, missing={sorted(set(range(168))-set(matched))[:12]}')
    need(all(p.stat().st_size>0 for p in matched.values()),f'Empty file in {prefix}')
    return matched


def source_checks(root):
    f=root/ROOT9/'per_sample_corrected_pd.csv'
    need(f.is_file(),'Missing LXXIX CSV')
    sha=digest(f)
    need(sha==EXPECTED_CSV_SHA256,'Frozen LXXIX CSV SHA-256 differs from user-confirmed value')
    with f.open(newline='') as h:rows=list(csv.DictReader(h))
    keys=Counter((int(x['sample']),x['arm']) for x in rows)
    need(len(rows)==840 and all(v==1 for v in keys.values()),'Not 840 unique pairs')
    need(set(keys)=={(i,a) for i in range(168) for a in ARMS},'Missing LXXIX rows')
    print(f'[SOURCE] PASS: 840 uniquely matched LXXIX rows, CSV_SHA256={sha}',flush=True)
    report=root/ROOT10/'paired_audit.md'
    sha_manifest=root/ROOT10/'SHA256SUMS'
    need(report.is_file() and sha_manifest.is_file(),'Missing paired audit report or SHA256SUMS')
    for line in sha_manifest.read_text().splitlines():
        if not line.strip():continue
        d,_,p=line.partition('  ')
        need(len(d)==64 and p and digest(root/ROOT10/p)==d,f'LXXX checksum mismatch: {p}')
    print('[PAIRED] PASS: LXXX report and declared audit artifacts match their SHA256SUMS',flush=True)


def topology_inventory(root):
    for key in ('GT','CNN_SR','F1_SR'):
        inventory(root/ROOT5,f'phase5_{key}_')
        print(f'[PHASE5E] PASS {key}: 168 canonical PD VTUs indexed 0..167',flush=True)
    for key,method in NEW.items():
        pd_dir=root/ROOT6/key/'pd'
        inventory(pd_dir,f'{method}_SR_')
        vti_dir=root/ROOT6/key/'vti'
        need(vti_dir.is_dir(),f'Missing {vti_dir}')
        num=len(list(vti_dir.glob(f'{method}_SR_s*_speed_p160_x0_y0.vti')))
        need(num==168,f'{key}: found {num}, not 168 SR VTIs')
        print(f'[NEW] PASS {key}: 168 PD VTUs and 168 canonical SR VTIs',flush=True)
    pilot=root/ROOT6/'pilot_verified.json'
    need(pilot.is_file() and json.loads(pilot.read_text()).get('format')=='phase6-lxxvii-pilot-verified-v1','Missing canonical LXXVII pilot gate')
    print('[PILOT] PASS: known Phase 6 sample-0 continuity gate is present',flush=True)


def find_sources(root):
    """Find candidate scripts to inspect, not an assertion that one is the canonical solver."""
    found=[]
    for folder in ('spatial_pd','scripts'):
        base=root/folder
        if not base.is_dir():continue
        for p in base.rglob('*'):
            if not p.is_file() or p.suffix not in ('.py','.sh') or p.stat().st_size>2_000_000:continue
            name=p.name.lower()
            if not any(k in name for k in ('phase5e','locality','sparse','matching')):continue
            try:s=p.read_text(errors='replace')
            except OSError:continue
            radius=all(str(v) in s for v in (4,8,16,32,64,96,128))
            savings=('saving' in s.lower() or 'min_weight_full_bipartite_matching' in s)
            flags=sorted(set(re.findall(r'--[a-z][a-z0-9-]+',s)))
            score=(3 if 'phase5e' in name else 0)+(3 if 'locality' in name else 0)+(2 if radius else 0)+(2 if savings else 0)
            found.append((score,p.relative_to(root),radius,savings,flags[:20],digest(p)))
    found.sort(key=lambda z:(-z[0],str(z[1])))
    print(f'[SOLVER-SOURCES] discovered {len(found)} candidate source scripts; this is discovery, NOT solver parity',flush=True)
    for score,path,radius,savings,flags,sha in found[:18]:
        print(f'[SOURCE-CANDIDATE] score={score} path={path} radius_grid_text={radius} savings_text={savings} sha256={sha} cli_flags={",".join(flags)}',flush=True)
    if not found:print('[SOURCE-CANDIDATE] WARNING: no obvious locality script found; inspect checkout before porting',flush=True)
    return found


def f1_readiness(root):
    p=root/'data_out/wind_finetune_candidateF_grad_E2_low_expanded2688'
    ref=root/'data_out_fixed/wind_mrhr_cnn'
    if not p.is_dir():
        print(f'[F1] UNVERIFIED: expected source array folder absent: {p}',flush=True)
        return
    for fname in ('idx.npy','dataGT.npy','dataSR.npy'):
        f=p/fname
        need(f.is_file() and f.stat().st_size>0,f'Missing F1 array: {f}')
        print(f'[F1] PRESENT {fname} bytes={f.stat().st_size} path={f}',flush=True)
    if (ref/'idx.npy').is_file() and (ref/'dataGT.npy').is_file():
        print('[F1] Reference benchmark arrays available; F1 training-run lineage still requires actual saved logs/options and checkpoint hashes',flush=True)
    mod=root/'models_fixed/topology_finetuning/wind_finetune_candidateF_grad_E2_low_expanded2688'
    paths=list(mod.rglob('*.index')) if mod.exists() else []
    print(f'[F1] checkpoint_index_files={len(paths)} paths={[str(x.relative_to(root)) for x in paths[:8]]}',flush=True)
    print('[F1-LINEAGE] NOT CLOSED: confirming F1 PD and source arrays exist does not prove a matched optimizer/data/seed path',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=Path.cwd())
    parser.add_argument('--mode',choices=['preflight','sources'],default='preflight')
    args=parser.parse_args()
    root=args.repo_root.resolve()
    need(root.is_dir(),f'Missing root: {root}')
    if args.mode=='preflight':
        source_checks(root)
        topology_inventory(root)
        f1_readiness(root)
    find_sources(root)
    print(f'[LXXXI] PASS mode={args.mode}: read-only; no spatial distances calculated and no experimental files modified',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        print(f'[LXXXI] ERROR {type(e).__name__}: {e}',file=sys.stderr)
        sys.exit(1)
