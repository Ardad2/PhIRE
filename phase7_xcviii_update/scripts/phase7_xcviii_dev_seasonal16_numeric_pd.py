#!/usr/bin/env python3
"""Phase 7 XCVIII: corrected GUDHI distances on 16 frozen development fields.

Reuses the exact Phase6 canonical parser and numeric distance functions through
verified original XCIII, checks original XCIII 4-field regression for all 24
comparisons, and verifies 112 original XCVII diagrams. No TTK/TF/WTK, no locked
regions, no overwrite. Requires gudhi-audit micromamba environment.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
import tempfile

N=16
ARMS=('cnn','gan','grad','cv','pers','f1')
ALL=('gt',*ARMS)
METRICS=('dB','W2inf','W22')
SAMPLES_OLD_TO_NEW={0:0,1:4,2:8,3:12}
SUB=Path('ttk_runs_fixed/topology_finetuning')
XCVII=SUB/'phase7_xcvii_dev_seasonal16_pd_20261018'
XCIII=SUB/'phase7_xciii_dev_corrected_pd_20261016'
OUT=SUB/'phase7_xcviii_dev_seasonal16_corrected_pd_20261018'
ORIGINAL_XCIII=Path('phase7_xciii_update/scripts/phase7_xciii_dev_corrected_pd.py')
ORIGINAL_XCVII=Path('phase7_xcvii_update/phase7_xcvii_update/scripts/phase7_xcvii_dev_seasonal16_pd.py')
# Also support source tree assembled directly from the local ZIP, provided SHA matches.
ALTERNATE_XCVII=Path('phase7_xcvii_update/scripts/phase7_xcvii_dev_seasonal16_pd.py')
PIN_XCIII='daff0159dcb6f3734eb5f1ffdd7f1e120eb887750f94daf91ddf5bdad3905cf5'
PIN_XCVII='af0187383ab667495689541e393b0980b083d4334993d87887b75af42efb1954'
SELF=Path('phase7_xcviii_update/scripts/phase7_xcviii_dev_seasonal16_numeric_pd.py')
COLS=('sample','region','season_utc','hour_offset','wtk_index','arm','gt_sha256','sr_sha256',
      'D0_gt','D1_gt','D0_sr','D1_sr',*METRICS)
BLOCK_COLS=('region','season_utc','arm','n_fields',*METRICS)
OUTPUT_FILES=('per_field_corrected_pd.csv','region_week_corrected_pd.csv','numeric_manifest.json','numeric_report.md')


def need(condition, message):
    if not condition: raise RuntimeError(message)


def sha(path):
    p=Path(path)
    need(p.is_file() and p.stat().st_size > 0,f'Missing/empty source: {p}')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def import_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    need(spec is not None and spec.loader is not None,f'Unable to load module {path}')
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def pd(arm,i):
    return f'pd/dev_{arm}_s{i}_speed_p160_x0_y0_pd_port_0.vtu'


def verified_sources(root):
    xciii_path=root/ORIGINAL_XCIII
    candidates=(root/ORIGINAL_XCVII,root/ALTERNATE_XCVII)
    xcvii_path=next((p for p in candidates if p.is_file()),None)
    need(sha(xciii_path)==PIN_XCIII,f'Original XCIII source hash mismatch {xciii_path}')
    need(xcvii_path is not None and sha(xcvii_path)==PIN_XCVII,'Original XCVII source hash mismatch')
    old=import_module(xciii_path,'trusted_xciii_numeric')
    extracted=import_module(xcvii_path,'trusted_xcvii_extraction')
    # XCIII source_gate reruns historical Phase6 regression against the
    # independent, frozen LXXIX sample-0 CSV and opens original XCII PDs only.
    pilot,decoder,old_fields,old_source=old.source_gate(root,regression=True)
    old.verify_results(root,pilot,decoder,old_fields,old_source)
    old_digests=old.verified_hashes(root/XCIII,
                  ['per_field_corrected_pd.csv','numeric_manifest.json','numeric_report.md'])
    with (root/XCIII/'per_field_corrected_pd.csv').open(newline='') as f:
        reader=csv.DictReader(f)
        need(tuple(reader.fieldnames or ())==old.COLS,'Archived XCIII CSV schema changed')
        old_rows=list(reader)
    need(len(old_rows)==24,'Archived XCIII row count changed')
    old_lookup={(int(r['sample']),r['arm']):r for r in old_rows}
    need(len(old_lookup)==24,'Duplicate keys in archived XCIII scores')
    fields,lineage,arrays,oldhash=extracted.sources(root,load=False)
    # Verify all 226 VTI/PD/report hashes, then check manifest identity.
    digests=extracted.check_sum(root/XCVII,extracted.paths())
    obj=json.loads((root/XCVII/'extraction_manifest.json').read_text())
    need(obj['schema_version']==1 and obj['format']=='phase7-xcvii-canonical-dev-seasonal16-pd-v1' and
         obj['scope']=='DEVELOPMENT_ONLY' and obj['n_fields']==N and obj['arms']==list(ALL) and
         obj['support']=='speed_p160_x0_y0' and obj['no_numerical_distance_yet'] is True,
         'XCVII scope/protocol mismatch')
    need(obj['fields']==fields and obj['source_sha256']==lineage,'XCVII source lineage changed')
    details=obj['details']
    need(len(details)==112,'XCVII source diagram count changed')
    seen=set()
    for e in details:
        key=(e['field_index'],e['arm'])
        need(key not in seen and e['arm'] in ALL and 0<=e['field_index']<N,
             f'Duplicate or invalid diagram key {key}')
        seen.add(key)
        need(e['pd_file']==pd(e['arm'],e['field_index']),'Diagram name mismatch')
    need(seen=={(i,a) for i in range(N) for a in ALL},'Missing diagram(s)')
    # Bitwise source equivalence for four previously numerically audited fields.
    for old_i,new_i in SAMPLES_OLD_TO_NEW.items():
        for arm in ALL:
            expect=old_source['source_pd_sha256'][f'{arm}:{old_i}']
            need(digests[pd(arm,new_i)]==expect,
                 f'Four-field diagram regression failed {arm} old={old_i} new={new_i}')
    need([f['region'] for f in fields]==['gulf_inland']*8+['southwest_borderland']*8,
         'Locked or out-of-order region')
    need([int(f['wtk_index']) for f in fields]==
         [27048,27132,29184,29268,31392,31476,33600,33684]*2,
         'Seasonal indices altered')
    need([f['offset_hours'] for f in fields]==[0,84]*8,
         'Hour offset changed')
    source={'format':'phase7-xcviii-numerical-sources-v1',
            'XCVII_manifest_sha256':sha(root/XCVII/'extraction_manifest.json'),
            'XCVII_SHA256SUMS_sha256':sha(root/XCVII/'SHA256SUMS'),
            'XCIII_numeric_manifest_sha256':sha(root/XCIII/'numeric_manifest.json'),
            'XCIII_SHA256SUMS_sha256':sha(root/XCIII/'SHA256SUMS'),
            'XCVII_script_sha256':PIN_XCVII,
            'XCIII_script_sha256':PIN_XCIII,
            'original_phase6_pilot_sha256':sha(root/old.PILOT),
            'canonical_parser_sha256':sha(old.CANONICAL),
            'this_script_sha256':sha(root/SELF),
            'all_112_pd_sha256':{f'{a}:{i}':digests[pd(a,i)] for i in range(N) for a in ALL},
            'XCIII_old_source_sha256':old_source,
            'XCIII_report_checksum':old_digests['numeric_report.md']}
    print('[SOURCE] PASS: XCVII 226 checksums, XCI frozen split, XCVI lineage, XCIII numerical regression',flush=True)
    print('[OLD-4] PASS: 28 bitwise diagrams and all 24 original corrected GUDHI scores available',flush=True)
    print('[DEV-SCOPE] PASS: 16 development-only fields, 2010 4 weeks × 2 hours × 2 regions; no external data',flush=True)
    return old,pilot,decoder,fields,source,old_lookup


def fmt(x):
    return format(float(x),'.17g')


def compute_rows(root,pilot,decoder,fields,source,old_lookup,progress):
    rows=[]
    for i in range(N):
        g=pilot.parse(decoder.read_pd,root/XCVII/pd('gt',i))
        need(set(g)=={0,1},f'Unrecognized GT homology dimension field {i}')
        for arm in ARMS:
            s=pilot.parse(decoder.read_pd,root/XCVII/pd(arm,i))
            need(set(s)=={0,1},f'Unrecognized homology dimension field {i} {arm}')
            for d in (g,s):
                for dim in (0,1):
                    need(d[dim].ndim==2 and d[dim].shape[1]==2,
                         f'Invalid decoded pair array field={i} arm={arm} dim={dim}')
                    need(math.isfinite(d[dim].sum()),f'Nonfinite PD array {i}/{arm}')
            scores=pilot.distance(g,s)
            vals=[float(scores[k]) for k in METRICS]
            need(all(math.isfinite(v) and v>=0 for v in vals),
                 f'Invalid distance field={i} arm={arm}: {vals}')
            need(vals[0]<=vals[1]+1e-7 and vals[1]<=vals[2]+1e-7,
                 f'Invalid distance inequality field={i} arm={arm}: {vals}')
            f=fields[i]
            row={'sample':str(i),'region':f['region'],'season_utc':f['anchor_utc'],
                 'hour_offset':str(f['offset_hours']),'wtk_index':str(f['wtk_index']),
                 'arm':arm,'gt_sha256':source['all_112_pd_sha256'][f'gt:{i}'],
                 'sr_sha256':source['all_112_pd_sha256'][f'{arm}:{i}'],
                 'D0_gt':str(len(g[0])),'D1_gt':str(len(g[1])),
                 'D0_sr':str(len(s[0])),'D1_sr':str(len(s[1])),
                 **{k:fmt(scores[k]) for k in METRICS}}
            rows.append(row)
            if i in SAMPLES_OLD_TO_NEW.values():
                old_idx=next(k for k,v in SAMPLES_OLD_TO_NEW.items() if v==i)
                frozen=old_lookup[(old_idx,arm)]
                for key in (*METRICS,'D0_gt','D1_gt','D0_sr','D1_sr'):
                    actual=float(row[key]) if key in METRICS else int(row[key])
                    expected=float(frozen[key]) if key in METRICS else int(frozen[key])
                    need(math.isclose(actual,expected,rel_tol=1e-11,abs_tol=1e-8),
                         f'XCIII original score regression failed field={i} {arm} {key} {actual} vs {expected}')
            if progress:
                print(f'[PD-NUMERIC] {len(rows):03d}/096 region={f["region"]} s{i} arm={arm} '
                      f'dB={vals[0]:.6f} W2inf={vals[1]:.6f} W22={vals[2]:.6f}',flush=True)
    need(len(rows)==96,'Wrong number of scored model-to-GT diagrams')
    return rows


def check_rows(rows,fields,source):
    need(len(rows)==96,'Incorrect number of CSV rows')
    seen=set()
    for r in rows:
        need(tuple(r)==COLS,'CSV order/columns drift')
        i=int(r['sample']);a=r['arm']
        need(i in range(N) and a in ARMS and (i,a) not in seen,'Duplicate/invalid numerical key')
        seen.add((i,a));f=fields[i]
        need(r['region']==f['region'] and r['season_utc']==f['anchor_utc'] and
             r['hour_offset']==str(f['offset_hours']) and r['wtk_index']==str(f['wtk_index']),
             'Field temporal/regional identity mismatch')
        need(r['gt_sha256']==source['all_112_pd_sha256'][f'gt:{i}'] and
             r['sr_sha256']==source['all_112_pd_sha256'][f'{a}:{i}'],'Diagram source hash drift')
        need(all(int(r[k])>=0 for k in ('D0_gt','D1_gt','D0_sr','D1_sr')),'Invalid pair count')
        d=[float(r[k]) for k in METRICS]
        need(all(math.isfinite(v) and v>=0 for v in d) and
             d[0]<=d[1]+1e-7 and d[1]<=d[2]+1e-7,'Invalid metric values')
    need(seen=={(i,a) for i in range(N) for a in ARMS},'Missing scored rows')


def grouped(rows,fields):
    group_rows=[]
    for region in ('gulf_inland','southwest_borderland'):
        for season in ('2010-02-01','2010-05-01','2010-08-01','2010-11-01'):
            for arm in ARMS:
                matches=[r for r in rows if r['region']==region and
                         r['season_utc'].startswith(season) and r['arm']==arm]
                need(len(matches)==2,f'Expected two observations {region}/{season}/{arm}')
                group_rows.append({'region':region,'season_utc':season,'arm':arm,'n_fields':'2',
                     **{m:fmt(statistics.mean(float(r[m]) for r in matches)) for m in METRICS}})
    need(len(group_rows)==48,'Wrong region-week group count')
    return group_rows


def summary_report(rows,blocks):
    means={arm:{k:statistics.mean(float(r[k]) for r in rows if r['arm']==arm)
                for k in METRICS} for arm in ARMS}
    lines=['# Phase 7 XCVIII — 16-field corrected persistence distances (development only)','',
           '16 predeclared 2010 source fields (two development regions × four weeks × offsets 0 h / 84 h).',
           '112 numerically decoded TTK port-0 diagrams; 96 model-to-GT comparisons via the **unchanged Phase6 GUDHI functions**.',
           '**Two snapshots per region-week are engineering checks, not independent observations or a generalization benchmark.**','',
           '| Model | dB mean ↓ | W2inf mean ↓ | W22 mean ↓ |','|---|---:|---:|---:|']
    for arm in ARMS:
        lines.append(f'| {arm} | {means[arm]["dB"]:.6f} | {means[arm]["W2inf"]:.6f} | {means[arm]["W22"]:.6f} |')
    lines+=['','## Region–week summaries','',
            'Each entry aggregates **two predeclared fields**, 0 h and 84 h; do not interpret 16 hours as independent replications.','',
            '| Region | Week start | Model | dB ↓ | W2inf ↓ | W22 ↓ |',
            '|---|---|---|---:|---:|---:|']
    for b in blocks:
        lines.append(f'| {b["region"]} | {b["season_utc"]} | {b["arm"]} | '
                     f'{float(b["dB"]):.6f} | {float(b["W2inf"]):.6f} | {float(b["W22"]):.6f} |')
    lines+=['','## Important boundaries','',
             '- Topology is for the primary **160×160 top-left scalar-speed crop**, not the full 500×500 wind field.',
             '- The four XCIII fields map to XCVIII field indices 0, 4, 8, 12; 24 archived distance values and feature counts must agree.',
             '- Same trusted Phase6 corrected numeric decoder and GUDHI distance implementation, rechecked against frozen 2007 sample-0.',
             '- Verification independently repeats numerical computation but **uses the same algorithm**, not a separate GUDHI implementation.',
             '- No Phase5E spatial-locality measures, merge trees, independent hourly uncertainty, training or external locked tests here.',
             '- Original pretrained PhIRE geographic/temporal training membership remains unknown.', '']
    return '\n'.join(lines),means


def write_csv(path,rows,columns):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns)
        w.writeheader();w.writerows(rows)


def read_csv(path,columns):
    with path.open(newline='') as f:
        r=csv.DictReader(f)
        need(tuple(r.fieldnames or ())==columns,f'CSV columns drifted {path}')
        return list(r)


def seal(root,rows,fields,source):
    out=root/OUT
    need(not out.exists(),f'Output exists; refusing overwrite: {out}')
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'.staging-',dir=out.parent))
    try:
        check_rows(rows,fields,source)
        blocks=grouped(rows,fields)
        report,means=summary_report(rows,blocks)
        write_csv(stage/OUTPUT_FILES[0],rows,COLS)
        write_csv(stage/OUTPUT_FILES[1],blocks,BLOCK_COLS)
        obj={'format':'phase7-xcviii-corrected-pd-seasonal16-v1','scope':'DEVELOPMENT_ONLY',
             'n_fields':N,'n_diagrams':112,'n_model_pairs':96,'arms':list(ARMS),
             'metrics':list(METRICS),'source_sha256':source,'fields':fields,
             'mean_descriptive':means,'region_week_n':8,
             'verification_method':'recompute same trusted Phase6 decoder and GUDHI implementation'}
        (stage/OUTPUT_FILES[2]).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
        (stage/OUTPUT_FILES[3]).write_text(report)
        (stage/'SHA256SUMS').write_text(''.join(f'{sha(stage/f)}  {f}\n' for f in OUTPUT_FILES))
        need(not out.exists(),'Output collision appeared during staging')
        stage.rename(out)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True)
        raise
    print(f'[COMPUTE-GATE] PASS: 112 decoded diagrams, 96 corrected model comparisons, '
          f'48 region-week rows, 24 XCIII score regressions; sealed={out}',flush=True)


def verify(root,pilot,decoder,fields,source,old_lookup):
    out=root/OUT
    need(out.is_dir(),f'No XCVIII results: {out}')
    digest=import_module(root/ORIGINAL_XCIII,'trusted_xciii_for_verify')
    digest.verified_hashes(out,OUTPUT_FILES)
    obj=json.loads((out/OUTPUT_FILES[2]).read_text())
    need(obj['format']=='phase7-xcviii-corrected-pd-seasonal16-v1' and obj['scope']=='DEVELOPMENT_ONLY'
         and obj['n_fields']==16 and obj['n_diagrams']==112 and obj['n_model_pairs']==96 and
         obj['arms']==list(ARMS) and obj['metrics']==list(METRICS) and obj['region_week_n']==8 and
         obj['source_sha256']==source and obj['fields']==fields,'Output provenance changed')
    rows=read_csv(out/OUTPUT_FILES[0],COLS)
    check_rows(rows,fields,source)
    # Full numerical recomputation is required, not just report and file hashes.
    recomputed=compute_rows(root,pilot,decoder,fields,source,old_lookup,progress=False)
    for stored,calc in zip(rows,recomputed):
        for key in COLS:
            if key in METRICS:
                need(math.isclose(float(stored[key]),float(calc[key]),rel_tol=1e-11,abs_tol=1e-8),
                     f'Numeric mismatch field={calc["sample"]} arm={calc["arm"]} metric={key}')
            else:
                need(stored[key]==calc[key],f'Metadata/feature count drift {key}')
    blocks=grouped(recomputed,fields)
    saved_blocks=read_csv(out/OUTPUT_FILES[1],BLOCK_COLS)
    need(saved_blocks==blocks,'Region/week aggregates differ from recomputation')
    report,means=summary_report(recomputed,blocks)
    need((out/OUTPUT_FILES[3]).read_text()==report and obj['mean_descriptive']==means,
         'Report/means differ from recomputation')
    print('[VERIFY-GATE] PASS: XCI/XCVII/XCIII SHA256 lineage, all 112 PDs decoded, '
          '96 distances recalculated, 24 XCIII regressions, 4 output hashes; no external scores',flush=True)


def selftest():
    import numpy as np
    import gudhi
    from gudhi.wasserstein import wasserstein_distance
    g=np.asarray([[0.,2.],[1.,2.2]],dtype=float)
    need(gudhi.bottleneck_distance(g,g,e=0.)==0,'GUDHI identity bottleneck failed')
    need(wasserstein_distance(g,g,matching=False,order=2.,internal_p=2.,keep_essential_parts=False)==0,
         'GUDHI identity W22 failed')
    need(list(SAMPLES_OLD_TO_NEW.values())==[0,4,8,12] and len(OUTPUT_FILES)==4,
         'Four-field mapping / output schema changed')
    need(hashlib.sha256(b'abc').hexdigest()==
         'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
         'SHA256 identity failed')
    print(f'[SELFTEST] PASS: GUDHI {getattr(gudhi,"__version__","installed")}, metric identity, legacy mapping and hashing',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=('selftest','preflight','compute','verify'))
    parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
    args=parser.parse_args()
    if args.mode=='selftest':selftest();return
    root=args.repo_root.resolve();out=root/OUT
    old,pilot,decoder,fields,source,old_lookup=verified_sources(root)
    if args.mode=='preflight':
        need(not out.exists(),f'Output exists, refusing overwrite: {out}')
        print('[PREFLIGHT] PASS: historic Phase6 regression, old XCIII scores, 112 XCVII PDs, '
              '16 dev-only fields, output absent; no writes',flush=True)
    elif args.mode=='compute':
        need(not out.exists(),f'Output exists, refusing overwrite: {out}')
        rows=compute_rows(root,pilot,decoder,fields,source,old_lookup,progress=True)
        seal(root,rows,fields,source)
    else:
        verify(root,pilot,decoder,fields,source,old_lookup)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(f'[XCVIII] ERROR {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        sys.exit(1)
