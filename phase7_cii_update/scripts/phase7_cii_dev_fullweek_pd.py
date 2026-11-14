#!/usr/bin/env python3
"""Part CII: atomic, resumable, 168-hour development-only canonical TTK extraction.

Exactly 168 hours x (GT + six frozen PhIRE checkpoints) = 1176 VTI/PD pairs.
Existing XCVII pairs for week hours 0 and 84 are preserved bitwise (14 pairs).
The 1162 others use the SHA-pinned original converter and XCII TTK command.
PD VTU parsing here is STRUCTURAL ONLY: corrected numeric decoder required later.
"""
from __future__ import annotations
import argparse, contextlib, fcntl, hashlib, importlib.util, io, json, math, os, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np

BASE=Path('ttk_runs_fixed/topology_finetuning')
CI=BASE/'phase7_ci_dev_gulf_inland_feb2010_inference168_20261018'
XCVII=BASE/'phase7_xcvii_dev_seasonal16_pd_20261018'
XCI=BASE/'phase7_xci_frozen_evaluation_protocol_20261015'
OUT=BASE/'phase7_cii_dev_gulf_inland_feb2010_pd168_20261018'
CI_SCRIPT=Path('phase7_ci_update/scripts/phase7_ci_dev_fullweek_inference.py')
XCVII_SCRIPT=Path('phase7_xcvii_update/phase7_xcvii_update/scripts/phase7_xcvii_dev_seasonal16_pd.py')
XCII_COMPAT=Path('phase7_xcii_xml_compat_20261016/scripts/phase7_xcii_vtk_xml_compat.py')
CONVERTER=Path('scripts/convert_phire_to_vti.py')
EXPECTED={
    'ci_script':'5a5174da40ce975440b6f66b86aeaf96da72942bdc17765d63d8c28419304dba',
    'xcvii_script':'af0187383ab667495689541e393b0980b083d4334993d87887b75af42efb1954',
    'xml_compat':'ccf48c3625c41d92e8cd6552e84a1fa8bcdfea015e04cff9fc4c38bd9beff6c2',
}
ARMS=('gt','cnn','gan','grad','cv','pers','f1')
ANCHORS={0:0,84:1}
N=168
SCHEMA='phase7_cii_gulf_inland_feb2010_pd168_v1'
STAGE_SUFFIX='.staging'

def need(cond,msg):
    if not cond:raise RuntimeError(msg)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def rawsha(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

def jd(obj):return json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n'

def atomic_text(p,txt):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.cii-write-',dir=p.parent)
    try:
        with os.fdopen(fd,'w') as f:f.write(txt);f.flush();os.fsync(f.fileno())
        os.replace(tmp,p)
    finally:Path(tmp).unlink(missing_ok=True)

def module(p,name):
    spec=importlib.util.spec_from_file_location(name,p)
    need(spec and spec.loader,f'Missing source loader {p}')
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m

def check_hashfile(folder,expected_names):
    folder=Path(folder);sums=folder/'SHA256SUMS';need(sums.is_file(),f'Missing {sums}')
    seen={};folderbase=folder.resolve()
    for line in sums.read_text().splitlines():
        mat=re.fullmatch(r'([0-9a-f]{64})  ([^\n]+)',line)
        need(mat is not None,f'Invalid SHA256SUMS line in {sums}')
        digest,name=mat.groups();p=(folder/name).resolve()
        need(name not in seen and p.is_relative_to(folderbase) and p.is_file(),f'Missing/unsafe SHA entry {name}')
        need(sha(p)==digest,f'Hash mismatch: {p}')
        seen[name]=digest
    need(set(seen)==set(expected_names),f'Wrong indexed files: {set(seen)^set(expected_names)}')
    return seen

def vti_name(arm,i):return f'dev_{arm}_s{i}_speed_p160_x0_y0.vti'
def pd_name(arm,i):return f'dev_{arm}_s{i}_speed_p160_x0_y0_pd_port_0.vtu'
def entry_key(i,arm):return f'h{i:03d}_{arm}'
def expected_keys():return [entry_key(i,a) for i in range(N) for a in ARMS]
def output_files():
    return [*(f'vti/{vti_name(a,i)}' for i in range(N) for a in ARMS),
            *(f'pd/{pd_name(a,i)}' for i in range(N) for a in ARMS),
            'extraction_manifest.json','extraction_report.md']

def speed(arr,i):
    # Identical to the archived XCVII implementation: cast uv before sqrt.
    x=np.asarray(arr[i,:160,:160,:],dtype=np.float32)
    return np.sqrt(x[...,0]**2+x[...,1]**2)

def sources(root):
    for label,rel,key in [('CI',CI_SCRIPT,'ci_script'),('XCVII',XCVII_SCRIPT,'xcvii_script'),('XML',XCII_COMPAT,'xml_compat')]:
        need((root/rel).is_file() and sha(root/rel)==EXPECTED[key],f'Archived {label} source hash mismatch')
    ci=module(root/CI_SCRIPT,'cii_archived_ci')
    xcvii=module(root/XCVII_SCRIPT,'cii_archived_xcvii')
    xml=module(root/XCII_COMPAT,'cii_archived_xml')
    need(tuple(xcvii.ARMS)==ARMS and N==168 and ci.N==N and tuple(ci.ARMS)==ARMS[1:],'Source arm or field protocol drift')
    ci_hash=check_hashfile(root/CI,ci.output_names())
    xcvii_hash=check_hashfile(root/XCVII,xcvii.paths())
    cfg=json.loads((root/CI/'run_config.json').read_text())
    m=json.loads((root/CI/'fullweek_inference_manifest.json').read_text())
    pg=json.loads((root/CI/'progress.json').read_text())
    need(cfg['scope']==ci.SCHEMA and cfg['hours']==N and cfg['region']=='gulf_inland' and cfg['week_start_utc']=='2010-02-01T00:00:00','CI week/region changed')
    need(cfg['wtk_first']==27048 and cfg['wtk_last']==27215 and cfg['model_names']==list(ARMS[1:]),'CI frozen indices/models changed')
    need(m['scope']==ci.SCHEMA and m['run_config']==cfg and pg['config']==cfg and set(pg['models'])==set(ARMS[1:]),'CI output manifest/progress mismatch')
    need(m['source_region']=='gulf_inland' and m['source_field_count']==N,'CI source range mismatch')
    need(cfg['source_sha256']['xci_protocol_sha256']==sha(root/XCI/'evaluation_protocol.json'),'XCI hash no longer matches CI')
    xci=json.loads((root/XCI/'evaluation_protocol.json').read_text())
    p=xci['protocol'];need(xci['status']=='FROZEN_BEFORE_EXTERNAL_INFERENCE' and
        p['geographic_roles']['development']==['gulf_inland','southwest_borderland'] and
        p['geographic_roles']['locked_external_test']==['upper_midwest','pacific_inland'],'XCI geographic roles changed')
    need(p['metric_definitions']['canonical_topology']['primary_crop']=={'x0':0,'y0':0,'height':160,'width':160},'XCI crop changed')
    xm=json.loads((root/XCVII/'extraction_manifest.json').read_text())
    need(xm['format']=='phase7-xcvii-canonical-dev-seasonal16-pd-v1' and xm['n_fields']==16 and xm['arms']==list(ARMS),'XCVII source diagram schema changed')
    need(xm['source_sha256']['canonical_converter']==sha(root/CONVERTER),'Canonical converter no longer matches XCVII')
    need(len(xm['details'])==112,'XCVII diagram count mismatch')
    lineage={'CII_script_sha256':sha(Path(__file__).resolve()),
             'CI_script_sha256':sha(root/CI_SCRIPT),'XCVII_script_sha256':sha(root/XCVII_SCRIPT),
             'XCII_xml_compat_sha256':sha(root/XCII_COMPAT),
             'converter_sha256':sha(root/CONVERTER),
             'CI_SHA256SUMS_sha256':sha(root/CI/'SHA256SUMS'),
             'CI_manifest_sha256':sha(root/CI/'fullweek_inference_manifest.json'),
             'XCVII_SHA256SUMS_sha256':sha(root/XCVII/'SHA256SUMS'),
             'XCVII_manifest_sha256':sha(root/XCVII/'extraction_manifest.json'),
             'XCI_protocol_sha256':sha(root/XCI/'evaluation_protocol.json')}
    print('[SOURCES] PASS: CI 13 hashes, XCVII 226 hashes, pinned scripts and frozen XCI protocol',flush=True)
    print('[SCOPE] PASS: Gulf inland Feb 2010 168 development hours; two reused anchors, no external locked region',flush=True)
    return ci,xcvii,xml,lineage,ci_hash,xcvii_hash

def arrays(root):
    arr={}
    for arm in ARMS:
        p=root/CI/('reference_gt.npy' if arm=='gt' else f'pred_{arm}.npy')
        a=np.load(p,mmap_mode='r',allow_pickle=False)
        need(a.shape==(168,500,500,2) and a.dtype.kind=='f',f'Full-week array schema changed: {arm}')
        arr[arm]=a
    return arr

def anchors(root,arr,xcvii,xml):
    oldroot=root/xcvii.XCVI
    for i,old_i in ANCHORS.items():
        for arm in ARMS:
            old=np.load(oldroot/('reference_gt.npy' if arm=='gt' else f'pred_{arm}.npy'),mmap_mode='r',allow_pickle=False)
            need(np.array_equal(arr[arm][i],old[old_i]),f'XCVI anchor numeric prediction drift: hour {i} {arm}')
            with contextlib.redirect_stdout(io.StringIO()):
                xml.check_vti_xml(root/XCVII/'vti'/xcvii.vti(arm,old_i),speed(arr[arm],i))
    print('[ANCHOR] PASS: week hours 0/84 match the 14 bitwise XCVII persistence input arrays',flush=True)

def config(lineage):
    return {'schema_version':1,'scope':SCHEMA,'region':'gulf_inland',
            'week_start_utc':'2010-02-01T00:00:00','hours':N,'wtk_range':[27048,27215],
            'arms':list(ARMS),'source_sha256':lineage,
            'crop':{'x0':0,'y0':0,'height':160,'width':160},
            'ttk_command':'ttkPersistenceDiagramCmd -t 1 -i <vti> -a wind_speed -o <prefix>',
            'anchor_hours':{'0':0,'84':1},'numerical_pd_distances_computed':False}

def read_pair(root,stage,i,arm,arr,xcvii,xml,check_input=True):
    pairdir=stage/'pairs'/entry_key(i,arm)
    need(pairdir.is_dir(),f'Missing stage pair {pairdir}')
    need({x.name for x in pairdir.iterdir()}=={'vti.vti','pd.vtu','pair_manifest.json'},f'Unexpected pair contents {pairdir}')
    m=json.loads((pairdir/'pair_manifest.json').read_text())
    need(set(m)=={'hour','arm','origin','vti_sha256','pd_sha256','speed_sha256','points','cells','ttk_seconds'},'Pair manifest schema changed')
    need(m['hour']==i and m['arm']==arm,'Pair identity changed')
    expect_origin='XCVII_REUSED_BITWISE' if i in ANCHORS else 'NEW_TTK'
    need(m['origin']==expect_origin and type(m['ttk_seconds']) in (int,float) and math.isfinite(m['ttk_seconds']) and m['ttk_seconds']>=0,'Invalid pair provenance')
    need(sha(pairdir/'vti.vti')==m['vti_sha256'] and sha(pairdir/'pd.vtu')==m['pd_sha256'],'Pair hash corruption')
    if check_input:need(rawsha(speed(arr[arm],i))==m['speed_sha256'],'Input speed changed')
    with contextlib.redirect_stdout(io.StringIO()):
        xml.check_vti_xml(pairdir/'vti.vti',speed(arr[arm],i))
        info=xml.check_pd_xml(pairdir/'pd.vtu')
    need(info=={'points':m['points'],'cells':m['cells']},'PD structural changes after sealing')
    if i in ANCHORS:
        oldidx=ANCHORS[i]
        need(m['ttk_seconds']==0 and m['vti_sha256']==sha(root/XCVII/'vti'/xcvii.vti(arm,oldidx)) and
             m['pd_sha256']==sha(root/XCVII/'pd'/xcvii.pd(arm,oldidx)),f'Bitwise anchor drift for hour{i} {arm}')
    return m

def load_progress(root,stage,arr,xcvii,xml,cfg):
    need(stage.is_dir() and (stage/'pairs').is_dir(),'CII stage missing pair directory')
    need(json.loads((stage/'run_config.json').read_text())==cfg,'Stage config or source SHA changed')
    keyset=set(expected_keys());found=set()
    for e in (stage/'pairs').iterdir():
        need(e.is_dir() and e.name in keyset,f'Unexpected/incomplete staging entry {e}; inspect before resuming')
        found.add(e.name)
    need(len(found)==len(list((stage/'pairs').iterdir())),'Duplicate staged paths')
    result=[]
    for i in range(N):
        for a in ARMS:
            if entry_key(i,a) in found:result.append(read_pair(root,stage,i,a,arr,xcvii,xml))
    return result

def make_pair(root,stage,i,arm,arr,xcvii,xml,converter,ttk_bin='ttkPersistenceDiagramCmd'):
    pairs=stage/'pairs';target=pairs/entry_key(i,arm)
    need(not target.exists(),f'Pair already exists {target}')
    scalar=speed(arr[arm],i)
    with tempfile.TemporaryDirectory(prefix='.cii_tmp_',dir=pairs) as tmp:
        t=Path(tmp);v=t/'vti.vti';d=t/'pd.vtu';t0=time.monotonic()
        if i in ANCHORS:
            j=ANCHORS[i]
            shutil.copyfile(root/XCVII/'vti'/xcvii.vti(arm,j),v)
            shutil.copyfile(root/XCVII/'pd'/xcvii.pd(arm,j),d)
            provenance='XCVII_REUSED_BITWISE';runtime=0.0
        else:
            converter.make_vti_from_scalar(scalar,'wind_speed',str(v),ascii=True)
            # Output naming must match the untouched XCII command.
            proc=subprocess.run([ttk_bin,'-t','1','-i',str(v),'-a','wind_speed','-o',str(t/'pd')],
                                capture_output=True,text=True,timeout=600)
            need(proc.returncode==0,f'TTK failed hour{i} {arm}: {(proc.stdout+proc.stderr)[-1200:]}')
            candidate=t/'pd_port_0.vtu';need(candidate.is_file(),f'TTK missing port-0 VTU hour{i} {arm}')
            candidate.rename(d)
            provenance='NEW_TTK';runtime=round(time.monotonic()-t0,6)
        with contextlib.redirect_stdout(io.StringIO()):
            xml.check_vti_xml(v,scalar)
            info=xml.check_pd_xml(d)
        info_m={'hour':i,'arm':arm,'origin':provenance,
                'vti_sha256':sha(v),'pd_sha256':sha(d),'speed_sha256':rawsha(scalar),
                'points':info['points'],'cells':info['cells'],'ttk_seconds':runtime}
        (t/'pair_manifest.json').write_text(jd(info_m))
        need(not target.exists(),'Pair collision before commit')
        t.rename(target)
    print(f'[PD] {i:03d}/167 {arm} {provenance} cells={info["cells"]} ({len(list(pairs.iterdir()))}/1176)',flush=True)
    return info_m

def report(rows):
    lines=['# Phase 7 Part CII — 168-hour canonical TTK persistence extraction','',
           'Frozen Gulf inland 2010-02-01 00:00 through 2010-02-07 23:00 UTC.',
           '**One development region-week; not an external geographic generalization benchmark.**','',
           '168 sequential WTK hours × GT and six fixed model outputs = **1176** canonical VTI/port-0 VTU pairs.',
           'Existing hour 0 and hour 84 inputs/diagrams (XCVII indices 0 and 1) are bitwise reused (14 pairs); **1162** pairs are newly generated.','',
           '| Source | Reused XCVII | Newly extracted | Total |','|---|---:|---:|---:|']
    for a in ARMS:
        subset=[r for r in rows if r['arm']==a]
        reused=sum(r['origin']=='XCVII_REUSED_BITWISE' for r in subset)
        lines.append(f'| {a} | {reused} | {len(subset)-reused} | {len(subset)} |')
    lines+=['','All model/GT scalar-speed crops are top-left 160×160, VTK x=columns, y=rows.',
           'Unmodified canonical ASCII Float32 converter and `ttkPersistenceDiagramCmd -t 1`.',
           '**Only VTI values and PD VTU XML structure are validated here. Compressed TTK numerical persistence pairs are NOT decoded.**',
           'Corrected GUDHI numerical decoding, XCIII/XCVIII anchor regression, and Phase5E locality matching are later steps.',
           'No WTK reads, TensorFlow inference, model changes, locked-region access, or hyperparameter selection.',
           'Original pretrained PhIRE training-geography/time exposure remains unknown.']
    return '\n'.join(lines)+'\n'

def publish(root,stage,out,arr,xcvii,xml,cfg,rows):
    need(not out.exists(),'Completed CII output exists; verify only')
    need(len(rows)==N*len(ARMS),'Cannot publish incomplete diagram set')
    with tempfile.TemporaryDirectory(prefix='.cii_publish_',dir=out.parent) as td:
        final=Path(td);(final/'vti').mkdir();(final/'pd').mkdir()
        manifest_rows=[]
        for i in range(N):
            for arm in ARMS:
                r=read_pair(root,stage,i,arm,arr,xcvii,xml)
                a=f'vti/{vti_name(arm,i)}';b=f'pd/{pd_name(arm,i)}'
                shutil.copyfile(stage/'pairs'/entry_key(i,arm)/'vti.vti',final/a)
                shutil.copyfile(stage/'pairs'/entry_key(i,arm)/'pd.vtu',final/b)
                need(sha(final/a)==r['vti_sha256'] and sha(final/b)==r['pd_sha256'],'Publish copied bytes differ')
                manifest_rows.append({**r,'vti_file':a,'pd_file':b})
        manifest={'schema_version':1,'format':SCHEMA,'run_config':cfg,'pairs':manifest_rows,
                  'total_pairs':1176,'reused_pairs':14,'new_ttk_pairs':1162}
        (final/'extraction_manifest.json').write_text(jd(manifest))
        (final/'extraction_report.md').write_text(report(manifest_rows))
        (final/'SHA256SUMS').write_text(''.join(f'{sha(final/n)}  {n}\n' for n in output_files()))
        check_hashfile(final,output_files())
        need(not out.exists(),'Output appeared while publishing')
        final.rename(out)
    # Archive stage intentionally remains intact for forensic lineage and resume proof.
    # Do not write to stage or previous research outputs after completion.
    print('[EXTRACT-GATE] PASS: 1176 VTI/PD pairs; 14 bitwise XCVII copies, 1162 NEW_TTK; 2354 SHA256 entries; no external region',flush=True)
    print('[OUTPUT]',out,flush=True)

def verify(root,stage,out,arr,xcvii,xml,cfg):
    need(out.is_dir(),'Completed CII output not present')
    check_hashfile(out,output_files())
    m=json.loads((out/'extraction_manifest.json').read_text())
    need(m['schema_version']==1 and m['format']==SCHEMA and m['run_config']==cfg and
         m['total_pairs']==1176 and m['reused_pairs']==14 and m['new_ttk_pairs']==1162,'Completed output schema/lineage changed')
    rows=m['pairs'];need(len(rows)==1176,'Missing manifest entries')
    seen=set();nr=0;nn=0
    for i in range(N):
        for arm in ARMS:
            key=entry_key(i,arm);expected_file_a=f'vti/{vti_name(arm,i)}';expected_file_b=f'pd/{pd_name(arm,i)}'
            r=rows[i*len(ARMS)+ARMS.index(arm)]
            need(r['hour']==i and r['arm']==arm and r['vti_file']==expected_file_a and r['pd_file']==expected_file_b,'Output row identity/order mismatch')
            need(key not in seen,'Duplicate diagram identity');seen.add(key)
            with contextlib.redirect_stdout(io.StringIO()):
                xml.check_vti_xml(out/expected_file_a,speed(arr[arm],i))
                info=xml.check_pd_xml(out/expected_file_b)
            need(info=={'points':r['points'],'cells':r['cells']} and
                 sha(out/expected_file_a)==r['vti_sha256'] and sha(out/expected_file_b)==r['pd_sha256'] and
                 rawsha(speed(arr[arm],i))==r['speed_sha256'],'Numerical VTI input or structural PD corruption')
            if i in ANCHORS:
                need(r['origin']=='XCVII_REUSED_BITWISE' and r['ttk_seconds']==0 and
                     r['vti_sha256']==sha(root/XCVII/'vti'/xcvii.vti(arm,ANCHORS[i])) and
                     r['pd_sha256']==sha(root/XCVII/'pd'/xcvii.pd(arm,ANCHORS[i])),'XCVII anchor drift')
                nr+=1
            else:
                need(r['origin']=='NEW_TTK' and isinstance(r['ttk_seconds'],(int,float)) and
                     math.isfinite(r['ttk_seconds']) and r['ttk_seconds']>=0,'New extraction provenance incorrect')
                nn+=1
    need(nr==14 and nn==1162 and len(seen)==1176,'Diagram inventory wrong')
    need((out/'extraction_report.md').read_text()==report(rows),'Human-readable extraction report changed')
    need({p.name for p in out.iterdir()}=={'vti','pd','extraction_manifest.json','extraction_report.md','SHA256SUMS'},'Unexpected output contents')
    need({p.name for p in (out/'vti').iterdir()}=={vti_name(a,i) for i in range(N) for a in ARMS},'Unexpected VTI members')
    need({p.name for p in (out/'pd').iterdir()}=={pd_name(a,i) for i in range(N) for a in ARMS},'Unexpected PD members')
    print('[VERIFY-GATE] PASS: all 1176 numeric VTI readbacks, 1176 structural PD VTUs, 14 XCVII bitwise anchors, 2354 SHA256 outputs; no locked test',flush=True)

def selftest():
    need(len(expected_keys())==1176 and len(set(expected_keys()))==1176,'Duplicate pair keys')
    need(len(output_files())==2354 and len(set(output_files()))==2354,'Output set size changed')
    need(list(ANCHORS.items())==[(0,0),(84,1)],'Anchor mapping changed')
    a=np.zeros((1,160,160,2),np.float32);a[...,0]=3;a[...,1]=4
    need(np.all(speed(a,0)==5),'Scalar speed conversion error')
    with tempfile.TemporaryDirectory() as td:
        p=Path(td);(p/'a').write_text('good');(p/'SHA256SUMS').write_text(f'{sha(p/"a")}  a\n')
        check_hashfile(p,['a']);(p/'a').write_text('tampered')
        try:check_hashfile(p,['a'])
        except RuntimeError:pass
        else:raise RuntimeError('Hash tampering not detected')
    print('[SELFTEST] PASS: 1176 canonical pair identities, 14 bitwise anchor slots, 2354 checksum entries, speed and tamper guards',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',required=True,choices=('selftest','preflight','extract','verify'))
    p.add_argument('--repo-root',default='/work')
    p.add_argument('--max-new-pairs',type=int,default=None)
    args=p.parse_args()
    if args.mode=='selftest':selftest();return
    need(args.max_new_pairs is None or (args.mode=='extract' and 1<=args.max_new_pairs<=1176),
         '--max-new-pairs only allowed for extract, 1..1176')
    root=Path(args.repo_root).resolve();os.chdir(root)
    ci,xcvii,xml,lineage,_,_=sources(root)
    arr=arrays(root)
    cfg=config(lineage)
    out=root/OUT;stage=out.parent/('.'+out.name+STAGE_SUFFIX)
    if args.mode=='verify':verify(root,stage,out,arr,xcvii,xml,cfg);return
    need(not out.exists(),f'Completed output already exists; run --mode verify: {out}')
    anchors(root,arr,xcvii,xml)
    free=shutil.disk_usage(root).free
    need(free>=8*1024**3,f'Disk available {free/1024**3:.1f} GiB below 8 GiB source-to-VTU safety floor')
    print(f'[BUDGET] disk_free={free/1024**3:.2f}GiB; 1176 VTI/PD pairs (14 bitwise, 1162 new); runtime is not predicted',flush=True)
    if args.mode=='preflight':
        if stage.exists():
            n=len(load_progress(root,stage,arr,xcvii,xml,cfg))
            print(f'[PREFLIGHT] PASS: existing validated stage {n}/1176 pairs; no writes/TTK',flush=True)
        else:print('[PREFLIGHT] PASS: immutable CI/XCVII/XCI lineage, 168 dev hours, no stage yet; no writes/TTK',flush=True)
        return
    out.parent.mkdir(parents=True,exist_ok=True)
    lock=out.parent/('.'+out.name+'.lock')
    with lock.open('a+') as fd:
        fcntl.flock(fd.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        need(not out.exists(),'Completed output appeared during lock')
        if not stage.exists():
            stage.mkdir();(stage/'pairs').mkdir()
            atomic_text(stage/'run_config.json',jd(cfg))
        rows=load_progress(root,stage,arr,xcvii,xml,cfg)
        print(f'[RESUME] PASS: {len(rows)}/1176 atomically sealed VTI/PD pairs',flush=True)
        converter=module(root/CONVERTER,'cii_pinned_canonical_converter')
        existing={entry_key(r['hour'],r['arm']) for r in rows}
        done=0
        for i in range(N):
            for a in ARMS:
                if entry_key(i,a) in existing:continue
                if args.max_new_pairs is not None and done>=args.max_new_pairs:
                    print(f'[PARTIAL-GATE] PASS: {len(rows)}/1176 validated pairs; resume with same --mode extract',flush=True)
                    return
                rows.append(make_pair(root,stage,i,a,arr,xcvii,xml,converter))
                done+=1
        rows=load_progress(root,stage,arr,xcvii,xml,cfg)
        publish(root,stage,out,arr,xcvii,xml,cfg,rows)

if __name__=='__main__':
    try:main()
    except Exception as e:
        print(f'[CII] ERROR {type(e).__name__}: {e}',file=sys.stderr,flush=True)
        sys.exit(1)
