#!/usr/bin/env python3
"""Phase 6 LXXXIV: read-only, checksum-gated spatial closeout and F1 evidence inventory.

Uses final LXXXIII CSVs as data; never recomputes or changes the spatial solver.
Does not certify historical F1 training random seeds or optimization lineage.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,statistics,sys
from pathlib import Path

SRC=Path('ttk_runs_fixed/topology_finetuning/phase6_lxxxiii_spatial_all168_20261014')
OUT=Path('ttk_runs_fixed/topology_finetuning/phase6_lxxxiv_spatial_closeout_20261014')
METHODS=('cnn','grad','cv','pers','f1')
RADII=('4.0','8.0','16.0','32.0','64.0','96.0','128.0','inf')
FIELDS=('constrained_w22_all','gt_persistence_coverage_all','sr_persistence_coverage_all','locality_gap_all')


def need(cond,msg):
    if not cond: raise RuntimeError(msg)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda:f.read(1048576),b''):h.update(buf)
    return h.hexdigest()


def load(root):
    p=root/SRC
    manifest=p/'SHA256SUMS'
    need(manifest.exists(),'Missing full LXXXIII SHA256SUMS')
    pairs=[]
    for line in manifest.read_text().splitlines():
        digest,name=line.split('  ',1)
        need('/' not in name and name not in ('.','..'),'Invalid file name in SHA256SUMS')
        need(sha(p/name)==digest,f'Changed/missing frozen source: {name}')
        pairs.append(name)
    need(len(pairs)==7 and len(set(pairs))==7,'Expected exactly seven frozen source hashes')
    with (p/'all168_locality_aggregate.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    data={}
    for row in rows:
        k=(int(row['sample']),row['arm'],row['radius'])
        need(k not in data,f'Duplicate sample/model/radius {k}')
        need(k[0] in range(168) and k[1] in METHODS and k[2] in RADII,f'Invalid key {k}')
        vals={}
        for field in FIELDS:
            vals[field]=float(row[field]);need(math.isfinite(vals[field]),f'Nonfinite {k}/{field}')
        need(0<=vals['gt_persistence_coverage_all']<=1 and 0<=vals['sr_persistence_coverage_all']<=1,f'Invalid coverage {k}')
        data[k]=vals
    expected={(s,a,r) for s in range(168) for a in METHODS for r in RADII}
    need(set(data)==expected,f'Invalid data cardinality {len(data)}, expected {len(expected)}')
    with (p/'all168_macro_summary.csv').open(newline='') as f:
        macro=list(csv.DictReader(f))
    need(len(macro)==40,'Expected 40 macro rows')
    seen=set()
    for row in macro:
        k=(row['arm'],row['radius']);need(k not in seen,f'Duplicate macro {k}');seen.add(k)
        for field in FIELDS:
            got=statistics.mean(data[s,k[0],k[1]][field] for s in range(168))
            exp=float(row['mean_'+field])
            need(math.isclose(got,exp,abs_tol=1e-9,rel_tol=1e-11),f'Mean mismatch {k}/{field}')
    need(seen=={(a,r) for a in METHODS for r in RADII},'Macro grid incomplete')
    print('[SOURCE] PASS: 7 LXXXIII checksums; 6720 unique aggregate entries; 40 means recalculated')
    return data,{n:sha(p/n) for n in pairs}


def summarize(data):
    lines=['# Phase 6 LXXXIV — full spatial paired closeout','',
           '168 consecutive hourly samples; original Phase5E sparse matching; lower constrained W22 is better.',
           'GT/SR coverage is persistence-mass coverage, **not** localization error or geographic distance.','',
           '## Radius-dependent cost and GT/SR coverage','',
           '| Radius | Best mean cost | Mean cost | Gradient cost | PERS cost | F1 cost | Gradient GT cov | PERS GT cov | F1 GT cov |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    grouped={}
    for r in RADII:
        a={arm:{field:statistics.mean(data[s,arm,r][field] for s in range(168)) for field in FIELDS} for arm in METHODS}
        grouped[r]=a
        winner=min(METHODS,key=lambda arm:a[arm]['constrained_w22_all'])
        cost=lambda arm:a[arm]['constrained_w22_all']
        cover=lambda arm:a[arm]['gt_persistence_coverage_all']
        lines.append(f"| {r} | {winner} | {cost(winner):.6f} | {cost('grad'):.6f} | {cost('pers'):.6f} | {cost('f1'):.6f} | {cover('grad'):.6f} | {cover('pers'):.6f} | {cover('f1'):.6f} |")
    lines.extend(['','## Pairwise effects: Gradient versus PERS and F1 versus PERS','',
      '| Radius | Contrast | Mean cost A−B | Median cost A−B | A lower / B lower / ties | Mean GT coverage A−B |',
      '|---|---|---:|---:|---:|---:|'])
    for r in RADII:
        for armA,armB in (('grad','pers'),('f1','pers')):
            diffs=[data[s,armA,r]['constrained_w22_all']-data[s,armB,r]['constrained_w22_all'] for s in range(168)]
            covers=[data[s,armA,r]['gt_persistence_coverage_all']-data[s,armB,r]['gt_persistence_coverage_all'] for s in range(168)]
            lines.append(f"| {r} | {armA}−{armB} | {statistics.mean(diffs):+.6f} | {statistics.median(diffs):+.6f} | {sum(x<0 for x in diffs)} / {sum(x>0 for x in diffs)} / {sum(x==0 for x in diffs)} | {statistics.mean(covers):+.6f} |")
    lines.extend(['','## Within-benchmark factorial contrasts by radius','',
                  'The zero cell is gradient-only, not pretrained CNN. Negative cost effects improve lower-is-better W22. Positive coverage effects raise matched GT persistence mass.','',
                  '| Radius | Outcome | CV effect | PERS effect | Interaction |',
                  '|---|---|---:|---:|---:|'])
    for r in RADII:
        for field in ('constrained_w22_all','gt_persistence_coverage_all'):
            comps={'cv':[],'pers':[],'interaction':[]}
            for s in range(168):
                g,c,p,f=(data[s,a,r][field] for a in ('grad','cv','pers','f1'))
                comps['cv'].append(((c-g)+(f-p))/2)
                comps['pers'].append(((p-g)+(f-c))/2)
                comps['interaction'].append(f-c-p+g)
            lines.append('| {} | {} | {} | {} | {} |'.format(r,'cost W22' if field=='constrained_w22_all' else 'GT coverage',*(f'{statistics.mean(comps[t]):+.6f}' for t in ('cv','pers','interaction'))))
    lines.extend(['','## 24-hour descriptive blocks: Gradient versus PERS constrained cost','',
                  '| Radius | Seven chronological 24-hour mean differences (gradient−PERS) |',
                  '|---|---|'])
    for r in ('4.0','16.0','32.0','inf'):
        seven=[statistics.mean(data[s,'grad',r]['constrained_w22_all']-data[s,'pers',r]['constrained_w22_all'] for s in range(24*b,24*(b+1))) for b in range(7)]
        lines.append('| {} | {} |'.format(r,', '.join(f'{x:+.4f}' for x in seven)))
    lines.extend(['','**Interpretation limits:** Single training instance for each configuration; temporal dependence between hourly fields; F1 optimizer, checkpoint and random-seed lineage unclosed. Original PD/MT and Phase5E sources unchanged. Merge-tree distances for the three new arms remain pending.',''])
    print('[ANALYSIS] PASS: all eight radii; paired costs/coverage, factorial contrasts, seven 24-hour blocks')
    return '\n'.join(lines),grouped


def inventory_f1(root):
    names=('logs/wind_finetune_candidateF_grad_E2_low_expanded2688.log',
           'logs/candidateF_grad_E2_low_expanded2688_batch.log',
           'docs/topology_finetuning_candidateF_grad_E2_low_expanded2688_eval.md')
    out=['# F1 historical lineage: evidence inventory (not certification)','',
         'This inventory proves existence and records hashes; it does not prove matched initialization, optimizer seeds, training-code version or identical stochastic trajectories.','']
    for name in names:
        p=root/name
        if p.is_file():
            out.append(f'- PRESENT `{name}`: bytes={p.stat().st_size}, SHA256=`{sha(p)}`')
            print('[F1-EVIDENCE] PRESENT '+name)
        else:
            out.append(f'- MISSING `{name}`')
            print('[F1-EVIDENCE] MISSING '+name)
    model=root/'models_fixed/topology_finetuning/wind_finetune_candidateF_grad_E2_low_expanded2688'
    for label in ('cnn00001','cnn00002','cnn00003','cnn'):
        p=model/label/'cnn.index'
        if p.is_file():
            out.append(f'- CHECKPOINT_INDEX `{p.relative_to(root)}`: bytes={p.stat().st_size}, SHA256=`{sha(p)}`')
        else:out.append(f'- CHECKPOINT_INDEX MISSING `{p.relative_to(root)}`')
    out+=['','To close attribution, review original log command/flags, fixed training-set digest, repaired E2 constraints, initial pretrained checkpoint, runner/code revision, optimizer path, and random-seed documentation against the three new arms. Do not label these independently identical merely because archived checkpoints or the spatial metrics agree.','']
    print('[F1-LINEAGE] NOT CLOSED: historical artifact inventory only')
    return '\n'.join(out)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',required=True,choices=('preflight','report','verify'))
    ap.add_argument('--repo-root',type=Path,default=Path.cwd());args=ap.parse_args()
    root=args.repo_root.resolve();data,sources=load(root)
    if args.mode=='preflight':
        print('[PREFLIGHT] PASS: read-only LXXXIII spatial source audit');return
    p=root/OUT
    if args.mode=='report':
        need(not p.exists(),f'Output exists; no overwrite allowed: {p}. Use --mode verify')
        report,_=summarize(data)
        lineage=inventory_f1(root)
        p.mkdir(parents=True,exist_ok=False)
        (p/'spatial_closeout.md').write_text(report)
        (p/'f1_evidence_inventory.md').write_text(lineage)
        (p/'source_sha256.json').write_text(json.dumps(sources,sort_keys=True,indent=2)+'\n')
        names=('spatial_closeout.md','f1_evidence_inventory.md','source_sha256.json')
        (p/'SHA256SUMS').write_text(''.join(f'{sha(p/n)}  {n}\n' for n in names))
        print(f'[REPORT] PASS: isolated output {p}; source unchanged; F1 lineage still open')
    elif args.mode=='verify':
        need((p/'SHA256SUMS').is_file(),'Report does not exist')
        report,_=summarize(data)
        need((p/'spatial_closeout.md').read_text()==report,'Spatial closeout report differs from recomputation')
        recorded=json.loads((p/'source_sha256.json').read_text())
        need(recorded==sources,'Recorded source hashes changed')
        lines=(p/'SHA256SUMS').read_text().splitlines()
        need(len(lines)==3,'Expected exactly three checksum entries')
        for line in lines:
            expected,name=line.split('  ',1)
            need(sha(p/name)==expected,f'Corrupted output {name}')
        print('[VERIFY-GATE] PASS: all LXXXIII inputs, spatial arithmetic, output hashes and provenance inventory file checks')

if __name__=='__main__':
    try:main()
    except Exception as ex:
        print(f'[LXXXIV] ERROR {type(ex).__name__}: {ex}',file=sys.stderr)
        raise SystemExit(1)
