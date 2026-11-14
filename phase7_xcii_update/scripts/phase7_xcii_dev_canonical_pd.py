#!/usr/bin/env python3
"""Phase 7 / XCII: development-only canonical 160x160 persistence extraction.

Run with the established phire-ttk:latest Docker environment. No inference or
GUDHI distance calculation; GT + six XC predictions, 4 development fields.
Modes: selftest, preflight, extract, verify. No overwrite/resume.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT_DEFAULT = Path(__file__).resolve().parents[2]
OUT_REL = Path('ttk_runs_fixed/topology_finetuning/phase7_xcii_dev_canonical_pd_20261015')
XC_REL = Path('ttk_runs_fixed/topology_finetuning/phase7_xc_development_inference_pilot_20261015')
XCI_REL = Path('ttk_runs_fixed/topology_finetuning/phase7_xci_frozen_evaluation_protocol_20261015')
CONVERTER = Path('scripts/convert_phire_to_vti.py')
LEGACY_INPUT = Path('ttk_runs_fixed/topology_finetuning/phase6_lxxvii_canonical_pd_newarms_20261014/grad/input_manifest.json')
MODELS = ('gt', 'cnn', 'gan', 'grad', 'cv', 'pers', 'f1')
DEV_REGIONS = ('gulf_inland', 'southwest_borderland')
EXTERNAL = ('upper_midwest', 'pacific_inland')
CROP = (0, 0, 160, 160)
N = 4


def need(test, message):
    if not test:
        raise RuntimeError(message)


def hash_file(p):
    h = hashlib.sha256()
    with p.open('rb') as file:
        for part in iter(lambda: file.read(2**20), b''):
            h.update(part)
    return h.hexdigest()


def hash_list(folder, exact):
    manifest = folder / 'SHA256SUMS'
    need(manifest.is_file(), f'Missing SHA256SUMS in {folder}')
    records = {}
    for line in manifest.read_text().splitlines():
        match = re.fullmatch(r'([0-9a-f]{64})  ([^\n]+)', line)
        need(match is not None, 'Malformed SHA256SUMS entry')
        digest, name = match.groups()
        p = (folder / name).resolve()
        need(p.is_relative_to(folder.resolve()) and p.is_file(), f'Unsafe/missing entry {name}')
        need(name not in records and hash_file(p) == digest, f'Checksum or duplicate failure {name}')
        records[name] = digest
    need(set(records) == set(exact), f'Expected files differ: {set(exact)^set(records)}')
    return records


def source_gate(root, load_arrays=False):
    import numpy as np
    xc = root / XC_REL
    xci = root / XCI_REL
    hash_list(xc, ['reference_mr.npy', 'reference_gt.npy', 'development_4_paired.tfrecord',
                   *(f'pred_{arm}.npy' for arm in MODELS if arm != 'gt'),
                   'dev_inference_report.json', 'dev_inference_report.md'])
    hash_list(xci, ['evaluation_protocol.json', 'evaluation_protocol.md'])
    frozen = json.loads((xci/'evaluation_protocol.json').read_text())
    plan = frozen['protocol']
    need(frozen['status'] == 'FROZEN_BEFORE_EXTERNAL_INFERENCE', 'XCI protocol not frozen')
    need(plan['geographic_roles']['development'] == list(DEV_REGIONS), 'Development regions changed')
    need(plan['geographic_roles']['locked_external_test'] == list(EXTERNAL), 'Test regions changed')
    need(plan['models']['order'] == list(MODELS[1:]), 'Model order changed')
    need(plan['metric_definitions']['canonical_topology']['primary_crop'] ==
         {'x0':0,'y0':0,'height':160,'width':160}, 'Primary topology support changed')
    need(frozen['sources']['verified_XC_report_sha256'] == hash_file(xc/'dev_inference_report.json'),
         'XC lineage changed from frozen XCI')
    report = json.loads((xc/'dev_inference_report.json').read_text())
    need(report['scope'] == 'DEVELOPMENT_ONLY_FOUR_FIELD_PILOT', 'XC scope mismatch')
    fields = report['sources']['fields']
    need(len(fields) == N and all(f['region'] in DEV_REGIONS for f in fields),
         'Pilot fields contain locked region or changed inventory')
    need([f['name'] for f in fields] == [
         'preview_gulf_inland_20100201.npz','preview_gulf_inland_20100801.npz',
         'preview_southwest_borderland_20100201.npz',
         'preview_southwest_borderland_20100801.npz'], 'Unexpected sample ordering')
    converter = root / CONVERTER
    need(converter.is_file(), f'Converter missing: {converter}')
    old = root / LEGACY_INPUT
    need(old.is_file(), 'Missing prior verified LXXVII source manifest')
    old_manifest = json.loads(old.read_text())
    need(old_manifest['format'] == 'phase6-lxxvii-sr-pd-v1', 'Unexpected legacy manifest')
    need(old_manifest['converter_sha256'] == hash_file(converter),
         'Original LXXVII canonical VTI converter changed; investigate before proceeding')
    hashes = {'XC_report_sha256': hash_file(xc/'dev_inference_report.json'),
              'XCI_protocol_sha256':hash_file(xci/'evaluation_protocol.json'),
              'original_converter_sha256':hash_file(converter),
              'legacy_LXXVII_driver_manifest_sha256':hash_file(old),
              'XC_arrays_sha256':{a:hash_file(xc/('reference_gt.npy' if a=='gt' else f'pred_{a}.npy'))
                                   for a in MODELS},
              'this_script_sha256':hash_file(Path(__file__))}
    arrays = {}
    if load_arrays:
        for arm in MODELS:
            path = xc / ('reference_gt.npy' if arm == 'gt' else f'pred_{arm}.npy')
            a = np.load(path, mmap_mode='r', allow_pickle=False)
            need(a.shape == (N,500,500,2), f'Wrong shape {arm}: {a.shape}')
            need(a.dtype in (np.dtype('float64'), np.dtype('float32')), f'Wrong dtype for {arm}: {a.dtype}')
            need(np.isfinite(a).all(), f'Nonfinite {arm} field')
            arrays[arm] = a
    print('[SCOPE] PASS: 4 development-only fields; no locked-test predictions opened',flush=True)
    print('[SOURCE] PASS: XC/XCI SHA-256; frozen protocol and canonical LXXVII converter continuity',flush=True)
    return hashes, fields, arrays


def top_speed(arr, i):
    import numpy as np
    s = np.asarray(arr[i,:160,:160,:], dtype=np.float32)
    return np.sqrt(s[...,0]**2+s[...,1]**2)


def vti_name(arm, i):
    return f'dev_{arm}_s{i}_speed_p160_x0_y0.vti'


def pd_name(arm,i):
    return f'dev_{arm}_s{i}_speed_p160_x0_y0_pd_port_0.vtu'


def converter_module(root):
    spec = importlib.util.spec_from_file_location('frozen_canonical_vti',root/CONVERTER)
    need(spec is not None and spec.loader is not None, 'Converter module unavailable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_vti(path, reference):
    import numpy as np
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    need(path.is_file() and path.stat().st_size>0, f'Missing VTI: {path}')
    reader = vtk.vtkXMLImageDataReader();reader.SetFileName(str(path));reader.Update()
    image = reader.GetOutput()
    need(image is not None and image.GetDimensions() == (160,160,1), f'Invalid VTI dimensions {path}')
    data = image.GetPointData().GetArray('wind_speed')
    need(data is not None, f'No wind_speed: {path}')
    actual = vtk_to_numpy(data).reshape(160,160)
    need(np.array_equal(actual,reference),f'VTI field values/axis mismatch: {path}; max error={np.max(np.abs(actual-reference))}')


def check_pd(path):
    import vtk
    need(path.is_file() and path.stat().st_size>0, f'Missing PD: {path}')
    r=vtk.vtkXMLUnstructuredGridReader();r.SetFileName(str(path));r.Update()
    g=r.GetOutput()
    need(g is not None and g.GetNumberOfCells()>0 and g.GetNumberOfPoints()>0, f'Invalid/empty PD port0: {path}')
    return {'cells':int(g.GetNumberOfCells()),'points':int(g.GetNumberOfPoints())}


def plan_payload(hashes,fields):
    return {'format':'phase7-xcii-canonical-dev-pd-v1','mode':'DEVELOPMENT_ONLY',
            'n_fields':N,'arms':list(MODELS), 'support':'speed_p160_x0_y0',
            'ttk_command':'ttkPersistenceDiagramCmd -t 1 -i <vti> -a wind_speed -o <prefix>',
            'no_corrected_distances_yet':True,'source_sha256':hashes,
            'fields':fields}


def report_markdown(entries):
    lines=['# Phase 7 XCII — canonical development-only persistence extraction','',
           f'{N*len(MODELS)} TTK port-0 diagrams from {N} original XC development fields × GT/model sources.',
           'No GUDHI distances, locality scores, or external-test inference in this stage.', '',
           '| Arm | 160×160 VTIs | Valid TTK port-0 diagrams |', '|---|---:|---:|']
    for arm in MODELS:
        count=sum(x['arm']==arm for x in entries)
        lines.append(f'| {arm} | {count} | {count} |')
    lines.extend(['','Canonical support: scalar speed in m/s, x=columns, y=rows; converter and TTK CLI identical to verified historical workflow.',
                  'The next gate must independently compute corrected GUDHI dB/W2inf/W22 and frozen Phase5E locality; these extraction results alone do not establish topological rankings.'])
    return '\n'.join(lines)+'\n'


def extract(root,out,hashes,fields,arrays):
    import numpy as np
    need(not out.exists(),f'Refusing existing output (never overwrite): {out}')
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'.staging-',dir=out.parent))
    try:
        (stage/'vti').mkdir();(stage/'pd').mkdir()
        module=converter_module(root)
        details=[]
        for i in range(N):
            for arm in MODELS:
                a=top_speed(arrays[arm],i)
                vti=stage/'vti'/vti_name(arm,i)
                pd=stage/'pd'/pd_name(arm,i)
                module.make_vti_from_scalar(a,'wind_speed',str(vti),ascii=True)
                check_vti(vti,a)
                prefix=stage/'pd'/pd_name(arm,i).replace('_port_0.vtu','')
                t0=time.perf_counter()
                p=subprocess.run(['ttkPersistenceDiagramCmd','-t','1','-i',str(vti),
                                  '-a','wind_speed','-o',str(prefix)],
                                 capture_output=True,text=True,timeout=600)
                need(p.returncode==0,f'TTK failure {arm} s{i}: {(p.stderr+p.stdout)[-2000:]}')
                info=check_pd(pd)
                details.append({'arm':arm,'field_index':i,'vti_file':f'vti/{vti.name}',
                                'pd_file':f'pd/{pd.name}',**info,
                                'ttk_seconds':round(time.perf_counter()-t0,6)})
                print(f'[PD] {len(details):02d}/{N*len(MODELS)} {arm} s{i} cells={info["cells"]} PASS',flush=True)
        payload=plan_payload(hashes,fields)
        payload['details']=details
        (stage/'extraction_manifest.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
        (stage/'extraction_report.md').write_text(report_markdown(details))
        members=[*(f'vti/{vti_name(arm,i)}' for i in range(N) for arm in MODELS),
                 *(f'pd/{pd_name(arm,i)}' for i in range(N) for arm in MODELS),
                 'extraction_manifest.json','extraction_report.md']
        (stage/'SHA256SUMS').write_text(''.join(f'{hash_file(stage/k)}  {k}\n' for k in members))
        need(not out.exists(),'Output collision appeared mid-run')
        stage.rename(out)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True)
        raise
    print(f'[EXTRACT-GATE] PASS: {N*len(MODELS)} canonical VTIs + {N*len(MODELS)} PDs; zero locked-region samples; output={out}',flush=True)


def verify(root,out,hashes,fields,arrays):
    need(out.is_dir(),f'No XCII output: {out}')
    members=[*(f'vti/{vti_name(arm,i)}' for i in range(N) for arm in MODELS),
             *(f'pd/{pd_name(arm,i)}' for i in range(N) for arm in MODELS),
             'extraction_manifest.json','extraction_report.md']
    hash_list(out,members)
    actual=json.loads((out/'extraction_manifest.json').read_text())
    expect=plan_payload(hashes,fields)
    for k,v in expect.items():
        need(actual.get(k)==v,f'Manifest mismatch: {k}')
    rows=actual['details']
    need(len(rows)==N*len(MODELS),'Incorrect number of PD extracts')
    seen=set()
    for e in rows:
        arm,i=e['arm'],e['field_index']
        need(arm in MODELS and type(i)==int and 0<=i<N and (arm,i) not in seen,'PD row keys invalid')
        seen.add((arm,i))
        need(e['vti_file']==f'vti/{vti_name(arm,i)}' and e['pd_file']==f'pd/{pd_name(arm,i)}','Output names drift')
        check_vti(out/e['vti_file'],top_speed(arrays[arm],i))
        info=check_pd(out/e['pd_file'])
        need(info['cells']==e['cells'] and info['points']==e['points'],'PD counts differ')
        need(math.isfinite(e['ttk_seconds']) and e['ttk_seconds']>=0,'Invalid recorded runtime')
    need(seen=={(arm,i) for arm in MODELS for i in range(N)},'Missing expected PD outputs')
    need((out/'extraction_report.md').read_text()==report_markdown(rows),'Human report mismatch')
    print(f'[VERIFY-GATE] PASS: {N*len(MODELS)} canonical VTI/PDs; original converter+input hashes; bitwise field orientation; immutable outputs',flush=True)


def selftest():
    import numpy as np
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    with tempfile.TemporaryDirectory() as t:
        a=np.arange(160*160,dtype=np.float32).reshape(160,160)
        image=vtk.vtkImageData();image.SetDimensions(160,160,1)
        data=numpy_to_vtk(a.ravel(order='C'),deep=True);data.SetName('wind_speed');image.GetPointData().SetScalars(data)
        w=vtk.vtkXMLImageDataWriter();p=Path(t)/'demo.vti';w.SetFileName(str(p));w.SetInputData(image)
        need(w.Write()!=0,'Synthetic VTK output failed')
        check_vti(p,a)
        payload={'format':'test','value':1};test=Path(t)/'test.json';test.write_text(json.dumps(payload))
        need(hash_file(test)==hashlib.sha256(test.read_bytes()).hexdigest(),'SHA selftest')
        test.write_text('tampered')
        need(hash_file(test)!=hashlib.sha256(json.dumps(payload).encode()).hexdigest(),'SHA tamper selftest')
    print('[SELFTEST] PASS: VTK C-order roundtrip, 160² axes, SHA/tamper check',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=('selftest','preflight','extract','verify'),required=True)
    ap.add_argument('--repo-root',type=Path,default=ROOT_DEFAULT)
    a=ap.parse_args()
    if a.mode=='selftest':
        selftest();return
    root=a.repo_root.resolve();out=root/OUT_REL
    hashes,fields,arrays=source_gate(root,load_arrays=a.mode in ('extract','verify'))
    if a.mode=='preflight':
        need(not out.exists(),f'Output already exists; choose --mode verify instead: {out}')
        print('[PREFLIGHT] PASS: frozen XC/XCI and old converter, 4 allowed development samples, no file writes',flush=True)
    elif a.mode=='extract':
        extract(root,out,hashes,fields,arrays)
    else:
        verify(root,out,hashes,fields,arrays)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        print(f'[XCII] ERROR {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        sys.exit(1)
