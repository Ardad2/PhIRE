#!/usr/bin/env python3
"""Phase 7 XCIII: numerical GUDHI audit of frozen development-only XCII diagrams.

Reuses the Phase 6 LXXVII exact corrected read_pd/parse/distance functions.
Preflight includes a Phase 6 LXXIX historical sample-0 five-arm regression.
No TTK, TensorFlow, HSDS, training, or locked-region access. No overwrite.
Run on Spark in micromamba environment `gudhi-audit`, with PYTHONNOUSERSITE=1.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import sys
import tempfile

N = 4
ARMS = ("cnn", "gan", "grad", "cv", "pers", "f1")
ALL = ("gt", *ARMS)
METRICS = ("dB", "W2inf", "W22")
XCII = Path('ttk_runs_fixed/topology_finetuning/phase7_xcii_dev_canonical_pd_20261015')
XCI = Path('ttk_runs_fixed/topology_finetuning/phase7_xci_frozen_evaluation_protocol_20261015')
OUT = Path('ttk_runs_fixed/topology_finetuning/phase7_xciii_dev_corrected_pd_20261016')
PILOT = Path('phase6_lxxvii_update/scripts/phase6_pilot_pd_gudhi_gate.py')
LEGACY = Path('ttk_runs_fixed/topology_finetuning/phase6_lxxix_corrected_pd_gudhi_20261014')
LEGACY_CSV_SHA256 = '3d53189ec69cc2f89c5561159d0768c973351d7b9998deceeaef42ebb6b379ea'
LEGACY_PDS = Path('spatial_pd/phase5e_all168_canonical_pd/pd')
NEW_PDS = Path('ttk_runs_fixed/topology_finetuning/phase6_lxxvii_canonical_pd_newarms_20261014')
CANONICAL = Path.home() / 'phire_runtime_audit_20260809_221548/recompute_pd/canonical_pd_pilot.py'
SELF = Path('phase7_xciii_update/scripts/phase7_xciii_dev_corrected_pd.py')
COLS = ('sample', 'region', 'preview', 'arm', 'gt_sha256', 'sr_sha256',
        'D0_gt', 'D1_gt', 'D0_sr', 'D1_sr', *METRICS)


def need(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def sha(p):
    need(p.is_file() and p.stat().st_size > 0, f'Missing or empty source: {p}')
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def verified_hashes(folder, required):
    manifest = folder / 'SHA256SUMS'
    need(manifest.is_file(), f'Missing SHA256SUMS: {folder}')
    entries = {}
    for line in manifest.read_text().splitlines():
        match = re.fullmatch(r'([a-f0-9]{64})  (.+)', line)
        need(match is not None, f'Invalid SHA256SUMS record in {folder}')
        digest, name = match.groups()
        path = (folder / name).resolve()
        need(path.is_relative_to(folder.resolve()) and name not in entries,
             f'Unsafe/duplicate SHA256SUMS member: {name}')
        need(sha(path) == digest, f'Source checksum differs: {path}')
        entries[name] = digest
    need(set(entries) == set(required), f'Unexpected SHA256SUMS inventory in {folder}: {set(entries)^set(required)}')
    return entries


def pd_name(arm, idx):
    return f'pd/dev_{arm}_s{idx}_speed_p160_x0_y0_pd_port_0.vtu'


def vti_name(arm, idx):
    return f'vti/dev_{arm}_s{idx}_speed_p160_x0_y0.vti'


def module_at(path, name):
    need(path.is_file(), f'Missing trusted module {path}')
    spec = importlib.util.spec_from_file_location(name, path)
    need(spec is not None and spec.loader is not None, f'Cannot load {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_pd(root, arm):
    stem = 's0_speed_p160_x0_y0_pd_port_0.vtu'
    if arm == 'gt':
        return root / LEGACY_PDS / ('phase5_GT_' + stem)
    if arm in ('cnn', 'f1'):
        return root / LEGACY_PDS / (('phase5_CNN_SR_' if arm == 'cnn' else 'phase5_F1_SR_') + stem)
    methods = {'grad':'candidateF_grad_pairzero_expanded2688',
               'cv':'candidateF_grad_CV_only_expanded2688',
               'pers':'candidateF_grad_PERS_only_expanded2688'}
    return root / NEW_PDS / arm / 'pd' / f'{methods[arm]}_SR_{stem}'


def legacy_regression(root, gate, decoder, pilot):
    # This tests the *exact* archived numeric functions on the old, independent 2007 inputs.
    hist = root / LEGACY / 'per_sample_corrected_pd.csv'
    need(sha(hist) == LEGACY_CSV_SHA256, 'Historical LXXIX corrected-PD CSV SHA drift')
    with hist.open(newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if int(r['sample']) == 0]
    need(len(rows) == 5 and {r['arm'] for r in rows} == {'cnn','grad','cv','pers','f1'},
         'Historical corrected-PD sample-0 reference incomplete')
    by_arm = {r['arm']:r for r in rows}
    gt_path = reference_pd(root, 'gt')
    g = pilot.parse(decoder.read_pd, gt_path)
    for arm in by_arm:
        s = pilot.parse(decoder.read_pd, reference_pd(root, arm))
        measured = pilot.distance(g,s)
        for key in METRICS:
            expected = float(by_arm[arm][key]); actual = float(measured[key])
            need(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8),
                 f'Historical LXXIX regression failed {arm}:{key}: actual={actual}, original={expected}')
        print(f'[LEGACY-REGRESSION] PASS s0 {arm} dB={measured["dB"]:.8g} '
              f'W2inf={measured["W2inf"]:.8g} W22={measured["W22"]:.8g}', flush=True)
    return {'csv_sha256':sha(hist),'sample0_arms':sorted(by_arm),
            'gt_sha256':sha(gt_path),'parser_sha256':sha(CANONICAL)}


def source_gate(root, regression=True):
    need(root.is_dir(), f'Missing PhIRE root: {root}')
    frozen = json.loads((root / XCI / 'evaluation_protocol.json').read_text())
    need(frozen['status'] == 'FROZEN_BEFORE_EXTERNAL_INFERENCE', 'XCI protocol is not frozen')
    geo = frozen['protocol']['geographic_roles']
    need(geo['development'] == ['gulf_inland','southwest_borderland'] and
         geo['locked_external_test'] == ['upper_midwest','pacific_inland'], 'Region roles drifted')
    need(frozen['protocol']['models']['order'] == list(ARMS), 'Model set drifted')
    need(frozen['protocol']['metric_definitions']['canonical_topology']['primary_crop'] ==
         {'x0':0,'y0':0,'height':160,'width':160}, 'Primary crop drifted')
    verified_hashes(root / XCI, ['evaluation_protocol.json','evaluation_protocol.md'])
    files = [*(vti_name(a,i) for i in range(N) for a in ALL),
             *(pd_name(a,i) for i in range(N) for a in ALL),
             'extraction_manifest.json','extraction_report.md']
    digests = verified_hashes(root / XCII, files)
    extraction = json.loads((root / XCII / 'extraction_manifest.json').read_text())
    need(extraction['format'] == 'phase7-xcii-canonical-dev-pd-v1' and extraction['mode'] == 'DEVELOPMENT_ONLY',
         'XCII extraction mode mismatch')
    need(extraction['n_fields'] == N and extraction['arms'] == list(ALL) and
         extraction['support'] == 'speed_p160_x0_y0', 'XCII source scope mismatch')
    fields = extraction['fields']
    expected_previews = ['preview_gulf_inland_20100201.npz','preview_gulf_inland_20100801.npz',
                         'preview_southwest_borderland_20100201.npz',
                         'preview_southwest_borderland_20100801.npz']
    need([f['name'] for f in fields] == expected_previews, 'XCII field ordering changed')
    need(len(extraction['details']) == 28, 'XCII extracted sample inventory changed')
    need({(e['field_index'],e['arm'],e['pd_file'],e['vti_file']) for e in extraction['details']} ==
         {(i,a,pd_name(a,i),vti_name(a,i)) for i in range(N) for a in ALL},
         'XCII extraction manifest incomplete or mislabeled')
    trial = json.loads((root / NEW_PDS / 'pilot_verified.json').read_text())
    need(trial['format'] == 'phase6-lxxvii-pilot-verified-v1' and sha(CANONICAL) == trial['parser_sha256'],
         'Original canonical numerical PD parser changed since Phase 6')
    pilot_path = root / PILOT
    pilot = module_at(pilot_path,'trusted_phase6_gudhi_pilot')
    decoder = module_at(CANONICAL,'trusted_phase6_pd_parser')
    need(all(callable(getattr(pilot,f,None)) for f in ('parse','distance')) and
         callable(getattr(decoder,'read_pd',None)), 'Trusted Phase6 numeric API unavailable')
    print('[SOURCE] PASS: 58 XCII output hashes, frozen XCI protocol, 28 development-only PDs', flush=True)
    print(f'[PARSER] PASS: Phase 6 canonical decoder SHA256={sha(CANONICAL)}', flush=True)
    legacy = legacy_regression(root,trial,decoder,pilot) if regression else None
    source = {'format':'phase7-xciii-numerical-source-v1',
              'xcII_manifest_sha256':sha(root / XCII / 'extraction_manifest.json'),
              'xci_protocol_sha256':sha(root / XCI / 'evaluation_protocol.json'),
              'source_pd_sha256':{f'{a}:{i}':digests[pd_name(a,i)] for i in range(N) for a in ALL},
              'trusted_pilot_script_sha256':sha(pilot_path),
              'trusted_parser_sha256':sha(CANONICAL),
              'trusted_phase6_gate_sha256':sha(root / NEW_PDS / 'pilot_verified.json'),
              'this_script_sha256':sha(root / SELF),
              'legacy_regression':legacy}
    return pilot,decoder,fields,source


def fmt_float(x):
    return format(float(x), '.17g')


def make_rows(root,pilot,decoder,fields,source,progress=True):
    rows = []
    for i in range(N):
        g = pilot.parse(decoder.read_pd,root / XCII / pd_name('gt',i))
        for arm in ARMS:
            s = pilot.parse(decoder.read_pd,root / XCII / pd_name(arm,i))
            d = pilot.distance(g,s)
            vals = [float(d[k]) for k in METRICS]
            need(all(math.isfinite(v) and v >= 0 for v in vals),f'Invalid GUDHI cost field={i} arm={arm}')
            need(vals[0] <= vals[1]+1e-7 and vals[1] <= vals[2]+1e-7,
                 f'GUDHI metric bound failed field={i} arm={arm}: {vals}')
            for dim in (0,1):
                need(g[dim].ndim == 2 and s[dim].ndim == 2 and
                     g[dim].shape[1] == 2 and s[dim].shape[1] == 2,
                     f'Malformed diagram field={i} arm={arm} dimension={dim}')
            record = {'sample':str(i),'region':fields[i]['region'],
                      'preview':fields[i]['name'],'arm':arm,
                      'gt_sha256':source['source_pd_sha256'][f'gt:{i}'],
                      'sr_sha256':source['source_pd_sha256'][f'{arm}:{i}'],
                      'D0_gt':str(len(g[0])),'D1_gt':str(len(g[1])),
                      'D0_sr':str(len(s[0])),'D1_sr':str(len(s[1])),
                      **{k:fmt_float(d[k]) for k in METRICS}}
            rows.append(record)
            if progress:
                print(f'[PD-NUMERIC] {len(rows):02d}/24 region={fields[i]["region"]} sample={i} '
                      f'arm={arm} dB={d["dB"]:.6f} W2inf={d["W2inf"]:.6f} W22={d["W22"]:.6f}',flush=True)
    need(len(rows)==24, 'Unexpected numeric row count')
    return rows


def validate_rows(rows, fields, source):
    need(len(rows)==24, f'Expected 24 rows, got {len(rows)}')
    have=set()
    for row in rows:
        need(set(row)==set(COLS), 'CSV field names differ')
        i,arm=int(row['sample']),row['arm']
        need(0 <= i < N and arm in ARMS and (i,arm) not in have, 'Duplicate/invalid numeric key')
        have.add((i,arm))
        need(row['region']==fields[i]['region'] and row['preview']==fields[i]['name'],
             'Wrong region or sample identity')
        need(row['gt_sha256']==source['source_pd_sha256'][f'gt:{i}'] and
             row['sr_sha256']==source['source_pd_sha256'][f'{arm}:{i}'], 'Numeric source hash changed')
        need(all(int(row[k])>=0 for k in ('D0_gt','D1_gt','D0_sr','D1_sr')), 'Negative pair count')
        vals=[float(row[k]) for k in METRICS]
        need(all(math.isfinite(v) and v>=0 for v in vals), 'Invalid distance')
        need(vals[0]<=vals[1]+1e-7 and vals[1]<=vals[2]+1e-7, 'Metric bound failed')
    need(have=={(i,a) for i in range(N) for a in ARMS}, 'Missing numeric rows')


def report(rows):
    summary={a:{m:statistics.mean(float(r[m]) for r in rows if r['arm']==a) for m in METRICS}
             for a in ARMS}
    lines=['# Phase 7 XCIII — corrected persistence distances (development-only)', '',
           'Four selected fields from Gulf inland / Southwest borderland, February and August 2010.',
           'Corrected Phase 6 canonical TTK decoder and GUDHI metrics; unchanged XCI protocol.',
           '**These are descriptive smoke-test results on 4 development fields, not a generalization conclusion.**','',
           '| Model | dB mean ↓ | W2inf mean ↓ | W22 mean ↓ |','|---|---:|---:|---:|']
    for a in ARMS:
        lines.append(f'| {a} | {summary[a]["dB"]:.6f} | {summary[a]["W2inf"]:.6f} | {summary[a]["W22"]:.6f} |')
    lines.extend(['','## Sample-level results','',
                  '| Region | Sample | Model | dB ↓ | W2inf ↓ | W22 ↓ |', '|---|---:|---|---:|---:|---:|'])
    for r in rows:
        lines.append(f'| {r["region"]} | {r["sample"]} | {r["arm"]} | {float(r["dB"]):.6f} | '
                     f'{float(r["W2inf"]):.6f} | {float(r["W22"]):.6f} |')
    lines.extend(['','## Boundaries','',
                  '- Distances are from 160×160 top-left patches only, not full 500×500 topology.',
                  '- Diagram decomposition: H0/H1 combined using the original Phase 6 functions.',
                  '- Numeric verification reruns the same validated implementation; it is not a second independent algorithm.',
                  '- No locality scores, merge-tree scores, new training or external-test fields in this workpack.',
                  '- Four fields are neither independent seasonal replicates nor a geographic generalization benchmark.', ''])
    return '\n'.join(lines)+'\n',summary


def save_results(root, rows, fields, source):
    out=root/OUT
    need(not out.exists(), f'XCIII output exists; never overwrite: {out}')
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'.staging-',dir=out.parent))
    try:
        csvpath=stage/'per_field_corrected_pd.csv'
        with csvpath.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=COLS)
            writer.writeheader(); writer.writerows(rows)
        md,summary=report(rows)
        obj={'format':'phase7-xciii-corrected-pd-v1','status':'DEVELOPMENT_ONLY',
             'n_fields':N,'n_model_rows':len(rows),'arms':list(ARMS),
             'metrics':list(METRICS),'sources':source,'fields':fields,'means_descriptive':summary,
             'verification_method':'same trusted Phase6 numeric implementation recomputed on verify'}
        (stage/'numeric_manifest.json').write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
        (stage/'numeric_report.md').write_text(md)
        filenames=('per_field_corrected_pd.csv','numeric_manifest.json','numeric_report.md')
        (stage/'SHA256SUMS').write_text(''.join(f'{sha(stage/f)}  {f}\n' for f in filenames))
        need(not out.exists(),'Output appeared during staging')
        stage.rename(out)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True)
        raise
    print(f'[COMPUTE-GATE] PASS: 24 corrected numeric model-vs-GT comparisons, '
          f'3 metrics each, isolated SHA-indexed output={out}',flush=True)


def verify_results(root, pilot, decoder, fields, source):
    out=root/OUT
    need(out.is_dir(),f'No XCIII numeric results at {out}')
    verified_hashes(out,['per_field_corrected_pd.csv','numeric_manifest.json','numeric_report.md'])
    obj=json.loads((out/'numeric_manifest.json').read_text())
    need(obj['format']=='phase7-xciii-corrected-pd-v1' and obj['status']=='DEVELOPMENT_ONLY',
         'Unexpected numeric manifest format/status')
    need(obj['sources']==source and obj['fields']==fields and obj['arms']==list(ARMS),
         'Frozen provenance/input changed since numeric computation')
    with (out/'per_field_corrected_pd.csv').open(newline='') as stream:
        reader=csv.DictReader(stream)
        need(tuple(reader.fieldnames or ())==COLS,'Numeric CSV columns differ')
        rows=list(reader)
    validate_rows(rows,fields,source)
    # Stronger than a mere hash gate: re-decode 28 VTUs and redo all 24 GUDHI comparisons.
    recalculated=make_rows(root,pilot,decoder,fields,source,progress=False)
    for stored, recomputed in zip(rows,recalculated):
        need((stored['sample'],stored['arm'])==(recomputed['sample'],recomputed['arm']),
             'Numeric row order changed')
        for k in COLS:
            if k in METRICS:
                need(math.isclose(float(stored[k]),float(recomputed[k]),rel_tol=1e-11,abs_tol=1e-8),
                     f'GUDHI recomputation differs {stored["arm"]}:{stored["sample"]}:{k}')
            else:
                need(stored[k]==recomputed[k],f'Metadata/dimension count drift: {k}')
    md,summary=report(recalculated)
    need((out/'numeric_report.md').read_text()==md,'Markdown report differs from recomputation')
    need(obj['means_descriptive']==summary and obj['n_model_rows']==24 and obj['n_fields']==N and
         obj['metrics']==list(METRICS),'Numeric summary differs from recomputation')
    print('[VERIFY-GATE] PASS: 28 decoded PDs, 24 recomputed GUDHI model comparisons, '
          'original Phase6 regression, all SHA256, no locked-region samples',flush=True)


def selftest():
    # This also confirms the installed GUDHI environment is usable.
    import numpy as np
    import gudhi
    from gudhi.wasserstein import wasserstein_distance
    a=np.asarray([[0.,2.],[1.,1.5]],dtype=float)
    need(gudhi.bottleneck_distance(a,a,e=0.)==0, 'GUDHI identity bottleneck failed')
    need(abs(wasserstein_distance(a,a,order=2.,internal_p=2.,matching=False,
                                  keep_essential_parts=False))<1e-10, 'GUDHI identity W22 failed')
    need(hashlib.sha256(b'abc').hexdigest() ==
         'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
         'Hash regression failed')
    print(f'[SELFTEST] PASS: GUDHI={getattr(gudhi,"__version__","installed")}, identity PD distances and hashing',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',required=True,choices=('selftest','preflight','compute','verify'))
    p.add_argument('--repo-root',type=Path,default=Path('.'))
    args=p.parse_args()
    if args.mode=='selftest':
        selftest();return
    root=args.repo_root.resolve()
    pilot,decoder,fields,source=source_gate(root,regression=True)
    if args.mode=='preflight':
        need(not (root/OUT).exists(),f'Output already exists; use --mode verify: {root/OUT}')
        print('[PREFLIGHT] PASS: no output files, 58 XCII sources, old GUDHI regression, locked-region ban',flush=True)
    elif args.mode=='compute':
        need(not (root/OUT).exists(),f'XCIII output exists; refusing overwrite: {root/OUT}')
        rows=make_rows(root,pilot,decoder,fields,source)
        validate_rows(rows,fields,source)
        save_results(root,rows,fields,source)
    else:
        verify_results(root,pilot,decoder,fields,source)

if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print(f'[XCIII] ERROR {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        sys.exit(1)
