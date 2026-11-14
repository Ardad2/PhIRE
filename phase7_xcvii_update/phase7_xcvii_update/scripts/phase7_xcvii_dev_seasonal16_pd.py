#!/usr/bin/env python3
"""Phase 7 XCVII: development-only 2010 seasonal16 canonical TTK PD extraction.

Reuses XCII's 28 verified (VTI, PD) pairs bit-for-bit, generates only 84 new
pairs from XCVI's frozen 16 development fields. No model inference, training,
GUDHI scoring, locality matching, or locked-region access. All new numerical
VTI values are checked against the canonical converter's source arrays.
TTK VTU checks are structural only; a corrected numerical gate is REQUIRED.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np

ROOT_DEFAULT=Path(__file__).resolve().parents[2]
ROOT_OUT=Path('ttk_runs_fixed/topology_finetuning')
XCII=ROOT_OUT/'phase7_xcii_dev_canonical_pd_20261015'
XCVI=ROOT_OUT/'phase7_xcvi_dev_seasonal16_inference_20261018'
XC=ROOT_OUT/'phase7_xc_development_inference_pilot_20261015'
XCI=ROOT_OUT/'phase7_xci_frozen_evaluation_protocol_20261015'
OUT=ROOT_OUT/'phase7_xcvii_dev_seasonal16_pd_20261018'
CANONICAL=Path('scripts/convert_phire_to_vti.py')
XCII_SCRIPT=Path('phase7_xcii_update/scripts/phase7_xcii_dev_canonical_pd.py')
COMPAT_SCRIPT=Path('phase7_xcii_xml_compat_20261016/scripts/phase7_xcii_vtk_xml_compat.py')
XCVI_SCRIPT=Path('phase7_xcvi_update/scripts/phase7_xcvi_dev_seasonal16_infer.py')
PINNED={'XCII_script':'743c7642611b6f8281891ffece8b60a6b717a0047042d745daa75cd42caee53b',
        'XCII_compat':'ccf48c3625c41d92e8cd6552e84a1fa8bcdfea015e04cff9fc4c38bd9beff6c2',
        'XCVI_script':'8d0f6ca62cf4f91a4ddc07fb6be16a17489e2c6f2899adac2c7b4a561db0c0f7'}
ARMS=('gt','cnn','gan','grad','cv','pers','f1')
REGIONS=('gulf_inland','southwest_borderland')
OLD_TO_NEW={0:0,1:4,2:8,3:12}
N=16


def need(ok,msg):
    if not ok: raise RuntimeError(msg)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def check_sum(folder, expected):
    base=folder.resolve();f=base/'SHA256SUMS'
    need(f.is_file(),f'Missing SHA256SUMS: {folder}')
    found={}
    for s in f.read_text().splitlines():
        m=re.fullmatch(r'([0-9a-f]{64})  ([^\n]+)',s)
        need(m is not None,f'Invalid checksum line in {folder}')
        digest,name=m.groups();path=(base/name).resolve()
        need(name not in found and path.is_relative_to(base) and path.is_file(),f'Unsafe/missing checksum file {name}')
        need(sha(path)==digest,f'Corrupt file: {path}')
        found[name]=digest
    need(set(found)==set(expected),f'Checksum manifest unexpected members {set(found)^set(expected)}')
    return found


def vti(arm,i):return f'dev_{arm}_s{i}_speed_p160_x0_y0.vti'

def pd(arm,i):return f'dev_{arm}_s{i}_speed_p160_x0_y0_pd_port_0.vtu'

def paths():
    return [*(f'vti/{vti(a,i)}' for i in range(N) for a in ARMS),
            *(f'pd/{pd(a,i)}' for i in range(N) for a in ARMS),
            'extraction_manifest.json','extraction_report.md']


def legacy_paths():
    return [*(f'vti/{vti(a,i)}' for i in range(4) for a in ARMS),
            *(f'pd/{pd(a,i)}' for i in range(4) for a in ARMS),
            'extraction_manifest.json','extraction_report.md']


def xcvi_files():
    return ['reference_mr.npy','reference_gt.npy','development_16_paired.tfrecord',
            *(f'pred_{a}.npy' for a in ARMS if a!='gt'),
            'seasonal16_inference_manifest.json','seasonal16_inference_report.md']


def read_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    need(spec is not None and spec.loader is not None,f'Missing import loader: {path}')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def sources(root, load=False):
    for key,p in [('XCII_script',XCII_SCRIPT),('XCII_compat',COMPAT_SCRIPT),('XCVI_script',XCVI_SCRIPT)]:
        need((root/p).is_file() and sha(root/p)==PINNED[key],f'Original {key} source changed')
    old=root/XCII;recent=root/XCVI;xci=root/XCI
    oldhash=check_sum(old,legacy_paths())
    newhash=check_sum(recent,xcvi_files())
    check_sum(xci,['evaluation_protocol.json','evaluation_protocol.md'])
    xc_old=check_sum(root/XC,['reference_mr.npy','reference_gt.npy','development_4_paired.tfrecord',
                               *(f'pred_{a}.npy' for a in ARMS if a!='gt'),
                               'dev_inference_report.json','dev_inference_report.md'])
    manifest=json.loads((old/'extraction_manifest.json').read_text())
    need(manifest['format']=='phase7-xcii-canonical-dev-pd-v1' and manifest['n_fields']==4 and manifest['arms']==list(ARMS),'Legacy XCII diagram schema changed')
    need(manifest['source_sha256']['compat_launcher_sha256']==PINNED['XCII_compat'],'Original XCII XML-compat provenance changed')
    need(manifest['source_sha256']['XC_report_sha256']==sha(root/XC/'dev_inference_report.json'),'Original XCII / XC report mismatch')
    need(manifest['source_sha256']['original_converter_sha256']==sha(root/CANONICAL),'Canonical converter changed')
    need(len(manifest['details'])==28,'Original XCII extraction count changed')
    rpt=json.loads((recent/'seasonal16_inference_manifest.json').read_text())
    need(rpt['scope']=='DEVELOPMENT_ONLY_SEASONAL16_INFERENCE' and len(rpt['fields'])==N,'XCVI scope changed')
    fields=rpt['fields']
    need([f['region'] for f in fields]==['gulf_inland']*8+['southwest_borderland']*8,'Locked-region access blocked')
    need([int(f['wtk_index']) for f in fields]==[27048,27132,29184,29268,31392,31476,33600,33684]*2,'Time indices changed')
    need([f['offset_hours'] for f in fields]==[0,84]*8,'Hour offsets changed')
    protocol=json.loads((xci/'evaluation_protocol.json').read_text())
    p=protocol['protocol'];need(protocol['status']=='FROZEN_BEFORE_EXTERNAL_INFERENCE','XCI not frozen')
    need(p['geographic_roles']['development']==list(REGIONS) and p['geographic_roles']['locked_external_test']==['upper_midwest','pacific_inland'],'Geographic assignment changed')
    need(p['metric_definitions']['canonical_topology']['primary_crop']=={'x0':0,'y0':0,'height':160,'width':160},'Canonical crop changed')
    need(p['models']['order']==list(ARMS[1:]),'Model set changed')
    need(rpt['source_sha256']['XCI_protocol_sha256']==sha(xci/'evaluation_protocol.json'),'XCVI frozen-protocol hash mismatch')
    need(rpt['source_sha256']['XC_report_sha256']==sha(root/XC/'dev_inference_report.json'),'XCVI original XC ancestry changed')
    lineage={'XCII_manifest':sha(old/'extraction_manifest.json'), 'XCVI_manifest':sha(recent/'seasonal16_inference_manifest.json'),
             'XCI_protocol':sha(xci/'evaluation_protocol.json'), 'canonical_converter':sha(root/CANONICAL),
             'pinned_scripts':PINNED, 'XCVII_script':sha(Path(__file__)),
             'XCII_SHA256SUMS':sha(old/'SHA256SUMS'), 'XCVI_SHA256SUMS':sha(recent/'SHA256SUMS'),
             'XC_SHA256SUMS':sha(root/XC/'SHA256SUMS')}
    print('[SOURCES] PASS: immutable XCII/XCVI/XCI and XC ancestry; canonical source and compatibility scripts pinned',flush=True)
    print('[SCOPE] PASS: 16 predeclared 2010 development fields, four legacy s0..s3 mapped to s0,s4,s8,s12; no external test',flush=True)
    arrays={}
    if load:
        for a in ARMS:
            arr=np.load(recent/('reference_gt.npy' if a=='gt' else f'pred_{a}.npy'),mmap_mode='r',allow_pickle=False)
            need(arr.shape==(16,500,500,2) and arr.dtype in (np.dtype('float32'),np.dtype('float64')) and np.isfinite(arr).all(),f'Invalid XCVI array {a}')
            arrays[a]=arr
            oldarr=np.load(root/XC/('reference_gt.npy' if a=='gt' else f'pred_{a}.npy'),mmap_mode='r',allow_pickle=False)
            need(oldarr.shape==(4,500,500,2),'XC original shape invalid')
            for oi,ni in OLD_TO_NEW.items():
                need(np.array_equal(arr[ni],oldarr[oi]),f'Old four-field numerical prediction drift: {a}, old {oi} / new {ni}')
        print('[REGRESSION] PASS: 28 four-field GT/model source arrays exactly identical to original XC',flush=True)
    return fields,lineage,arrays,oldhash


def top_speed(a,i):
    x=np.asarray(a[i,:160,:160,:],dtype=np.float32)
    return np.sqrt(x[...,0]**2+x[...,1]**2)


def md_report(details):
    lines=['# Phase 7 XCVII — canonical development-only seasonal16 persistence extraction','',
           '16 predeclared 2010 development fields × GT and six frozen models = 112 canonical 160×160 PDs.',
           'The original XCII 28 files were preserved bit-for-bit; 84 new source fields were extracted.',
           'No locked-region inference, GUDHI scores, locality scores, or training.','',
           '| Arm | Reused XCII | Newly extracted | Total |','|---|---:|---:|---:|']
    for a in ARMS:
        entries=[e for e in details if e['arm']==a]
        used=sum(e['origin']=='XCII_REUSED_BITWISE' for e in entries)
        new=len(entries)-used
        lines.append(f'| {a} | {used} | {new} | {len(entries)} |')
    lines+=['','Canonical scalar speed (m/s), top-left (0,0), 160×160, VTK x=columns y=rows.',
            'The unchanged canonical converter generates ASCII Float32 VTIs; the original `ttkPersistenceDiagramCmd -t 1` produces port-0 VTUs.',
            'VTI values/orientation are checked numerically; PD VTU XML is validated *structurally only*.',
            'A subsequent corrected GUDHI numerical decoder/regression gate is required before topological conclusions.',
            'These 16 fields are a two-hour-per-region-week engineering pilot, not the locked 168h/week benchmark.',
            'The original pretrained PhIRE geographic exposure remains unknown.']
    return '\n'.join(lines)+'\n'


def extraction(root,out,fields,lineage,arrays,oldhash):
    need(not out.exists(),f'Output exists, refuse overwrite: {out}')
    conv=read_module(root/CANONICAL,'xcvii_orig_converter')
    xml=read_module(root/COMPAT_SCRIPT,'xcvii_orig_xml')
    old=root/XCII
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+out.name+'.staging-',dir=out.parent))
    try:
        (stage/'vti').mkdir();(stage/'pd').mkdir()
        rows=[];reuse=0;new=0
        for i in range(N):
            for arm in ARMS:
                scalar=top_speed(arrays[arm],i)
                v=stage/'vti'/vti(arm,i);d=stage/'pd'/pd(arm,i)
                old_idx=next((k for k,j in OLD_TO_NEW.items() if j==i),None)
                if old_idx is not None:
                    # Bitwise copy prevalidated against original XCII's immutable manifest.
                    prev_v=old/'vti'/vti(arm,old_idx);prev_d=old/'pd'/pd(arm,old_idx)
                    shutil.copyfile(prev_v,v);shutil.copyfile(prev_d,d)
                    need(sha(v)==oldhash[f'vti/{prev_v.name}'] and sha(d)==oldhash[f'pd/{prev_d.name}'],'Copied XCII hashes differ')
                    origin='XCII_REUSED_BITWISE';seconds=0.0;reuse+=1
                else:
                    conv.make_vti_from_scalar(scalar,'wind_speed',str(v),ascii=True)
                    origin='NEW_TTK';start=time.perf_counter()
                    # Original XCII CLI and single-thread setting, no options changed.
                    pref=stage/'pd'/pd(arm,i).replace('_port_0.vtu','')
                    xml.check_vti_xml(v,scalar)
                    proc=subprocess.run(['ttkPersistenceDiagramCmd','-t','1','-i',str(v),'-a','wind_speed','-o',str(pref)],
                                        text=True,capture_output=True,timeout=600)
                    need(proc.returncode==0,f'TTK failed on {arm} s{i}: {(proc.stdout+proc.stderr)[-1800:]}')
                    seconds=round(time.perf_counter()-start,6);new+=1
                xml.check_vti_xml(v,scalar)
                info=xml.check_pd_xml(d)  # *structural* only, not numerical pairing data
                rows.append({'arm':arm,'field_index':i,'vti_file':'vti/'+v.name,'pd_file':'pd/'+d.name,
                             'origin':origin,**info,'ttk_seconds':seconds})
                print(f'[PD] {len(rows):03d}/112 {arm} s{i} {origin} cells={info["cells"]}',flush=True)
        need(reuse==28 and new==84 and len(rows)==112,'Reuse/generation counts changed')
        report={'schema_version':1,'format':'phase7-xcvii-canonical-dev-seasonal16-pd-v1',
                'scope':'DEVELOPMENT_ONLY','n_fields':N,'arms':list(ARMS),
                'support':'speed_p160_x0_y0','no_numerical_distance_yet':True,
                'ttk_command':'ttkPersistenceDiagramCmd -t 1 -i <vti> -a wind_speed -o <prefix>',
                'source_sha256':lineage,'fields':fields,'details':rows}
        (stage/'extraction_manifest.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
        (stage/'extraction_report.md').write_text(md_report(rows))
        (stage/'SHA256SUMS').write_text(''.join(f'{sha(stage/rel)}  {rel}\n' for rel in paths()))
        need(not out.exists(),'Output collision appeared mid-run')
        stage.rename(out)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True)
        raise
    print(f'[EXTRACT-GATE] PASS: 28 bitwise XCII copies + 84 newly extracted PDs = 112 pairs; no locked region; output={out}',flush=True)


def verify(root,out,fields,lineage,arrays,oldhash):
    need(out.is_dir(),f'Output missing: {out}')
    check_sum(out,paths())
    rep=json.loads((out/'extraction_manifest.json').read_text())
    need(rep['schema_version']==1 and rep['format']=='phase7-xcvii-canonical-dev-seasonal16-pd-v1' and rep['scope']=='DEVELOPMENT_ONLY','Wrong report format/scope')
    need(rep['n_fields']==N and rep['arms']==list(ARMS) and rep['support']=='speed_p160_x0_y0' and rep['no_numerical_distance_yet']==True,'Protocol/mode drift')
    need(rep['source_sha256']==lineage and rep['fields']==fields,'Frozen input lineage/field order changed')
    need(rep['ttk_command']=='ttkPersistenceDiagramCmd -t 1 -i <vti> -a wind_speed -o <prefix>','TTK CLI drift')
    entries=rep['details'];need(len(entries)==112,'Wrong diagram count')
    xml=read_module(root/COMPAT_SCRIPT,'xcvii_verify_xml')
    seen=set();nold=0;nnew=0
    for e in entries:
        arm,i=e['arm'],e['field_index']
        need(arm in ARMS and type(i)==int and 0<=i<N and (i,arm) not in seen,'Duplicate or invalid diagram key')
        seen.add((i,arm))
        need(e['vti_file']==f'vti/{vti(arm,i)}' and e['pd_file']==f'pd/{pd(arm,i)}','File naming drift')
        v=out/e['vti_file'];d=out/e['pd_file']
        xml.check_vti_xml(v,top_speed(arrays[arm],i))
        info=xml.check_pd_xml(d)
        need(info['cells']==e['cells'] and info['points']==e['points'],'PD structure/count modified')
        old_idx=next((k for k,j in OLD_TO_NEW.items() if j==i),None)
        expected_origin='XCII_REUSED_BITWISE' if old_idx is not None else 'NEW_TTK'
        need(e['origin']==expected_origin and math.isfinite(e['ttk_seconds']) and e['ttk_seconds']>=0,'Origin/runtime invalid')
        if old_idx is not None:
            need(sha(v)==oldhash[f'vti/{vti(arm,old_idx)}'] and sha(d)==oldhash[f'pd/{pd(arm,old_idx)}'],'Previously verified XCII file changed')
            need(e['ttk_seconds']==0,'Old diagram must not have new runtime')
            nold+=1
        else:nnew+=1
    need(seen=={(i,a) for i in range(N) for a in ARMS} and nold==28 and nnew==84,'Missing diagrams / wrong mix')
    need((out/'extraction_report.md').read_text()==md_report(entries),'Human report changed')
    print('[VERIFY-GATE] PASS: 112 VTIs/PDs numerically oriented; 28 bitwise XCII regressions, 84 new structural PDs; 226 SHA256, no test region',flush=True)


def selftest():
    need(list(OLD_TO_NEW.items())==[(0,0),(1,4),(2,8),(3,12)],'Historic XC map drift')
    need(len(paths())==226 and len(legacy_paths())==58,'Manifest member count drift')
    need(len({*paths()})==226,'Duplicate members')
    a=np.arange(160*160,dtype=np.float32).reshape(160,160)
    with tempfile.TemporaryDirectory() as t:
        p=Path(t);f=p/'test';f.write_bytes(b'original')
        (p/'SHA256SUMS').write_text(f'{sha(f)}  test\n')
        check_sum(p,['test']);f.write_bytes(b'altered')
        try:check_sum(p,['test'])
        except RuntimeError:pass
        else:raise RuntimeError('Checksum tampering not detected')
    need(float(a[9,17]) != float(a[17,9]),'Transposed coordinate selftest failed')
    print('[SELFTEST] PASS: XC map, 112 diagram names, 226-file SHA manifest schema, corruption detection',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=('selftest','preflight','extract','verify'))
    parser.add_argument('--repo-root',type=Path,default=ROOT_DEFAULT)
    args=parser.parse_args()
    if args.mode=='selftest':selftest();return
    root=args.repo_root.resolve();out=root/OUT
    fields,lineage,arrays,oldhash=sources(root,load=args.mode in ('extract','verify'))
    if args.mode=='preflight':
        need(not out.exists(),f'Output exists; refuse overwrite and use verify: {out}')
        print('[PREFLIGHT] PASS: all lineage checks, only 2 dev regions; 28 reused/84 new planned; no TF/HSDS/writes',flush=True)
    elif args.mode=='extract':extraction(root,out,fields,lineage,arrays,oldhash)
    else:verify(root,out,fields,lineage,arrays,oldhash)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(f'[XCVII] ERROR {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        sys.exit(1)
