#!/usr/bin/env python3
"""Phase 6 LXXXV: read-only historical-F1 lineage evidence audit.

Produces an inventory and *selected* log lines for human comparison, not an
inference that four optimizations followed identical random trajectories.
Does not run training, import TensorFlow, or mutate any archived source.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT_LOGS = {
    "f1_train": "logs/wind_finetune_candidateF_grad_E2_low_expanded2688.log",
    "f1_batch": "logs/candidateF_grad_E2_low_expanded2688_batch.log",
    "grad_train": "logs/wind_finetune_candidateF_grad_pairzero_expanded2688.log",
    "cv_train": "logs/wind_finetune_candidateF_grad_CV_only_expanded2688.log",
    "pers_train": "logs/wind_finetune_candidateF_grad_PERS_only_expanded2688.log",
    "grad_console": "logs/phase6_grad_pairzero_console.log",
    "cv_console": "logs/phase6_grad_cv_only_console.log",
    "pers_console": "logs/phase6_grad_pers_only_console.log",
}

MODEL = {
    "f1": "candidateF_grad_E2_low_expanded2688",
    "grad": "candidateF_grad_pairzero_expanded2688",
    "cv": "candidateF_grad_CV_only_expanded2688",
    "pers": "candidateF_grad_PERS_only_expanded2688",
}

# Only evidence-rich keywords; redact potentially sensitive long tokens in selected lines.
CATEGORIES = {
    "launch / branch / environment": r"(?:run_candidateF|--variant|--(?:method|candidate)|(?:^|\s)(?:git|commit|branch|revision|python|tensorflow|cuda)(?:\s|:|=))",
    "loss coefficients": r"(?:lambda[_-]?(?:uv|grad|ttkcv|ttkpers|speed|levelset|crit)|ttk(?:cv|pers)|grad_e2_low|grad_(?:cv|pers|pairzero))",
    "training hyperparameters": r"(?:learning[_-]?rate|(?:\blr\b\s*[:=])|(?:batch[_-]?size)|(?:epochs?\b)|(?:global[_-]?step)|(?:optimizer)|(?:seed|random[_-]?state))",
    "input paths / constraints / checkpoint restore": r"(?:wind_MR-HR\.tfrecord|wind_MR.HR\.tfrecord|ttk_pd_critical_pairs_gtvalues|train(?:ing)?_data|restore|checkpoint|pretrain|2688|repaired.*pair)",
    "completion / validation": r"(?:epoch.*(?:done|complete|finished)|validation.*pass|training.*(?:complete|finish)|saved.*(?:model|checkpoint)|run complete|pipeline exit|inference.*complete)",
}
PATTERNS={k:re.compile(v,re.IGNORECASE) for k,v in CATEGORIES.items()}
# Mask common secret-bearing options and key=value tokens; avoid printing entire log files.
SECRET=re.compile(r"(?i)(\b(?:token|password|passwd|secret|api[_-]?key|access[_-]?key|authorization)\b\s*(?:=|:)\s*)\S+")
BEARER=re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._-]+")

FROZEN_SOURCE={
    "training TFRecord":"example_data_topology_expanded_2688/wind_MR-HR.tfrecord",
    "evaluation TFRecord":"example_data_fixed/wind_MR-HR.tfrecord",
    "fixed E2 targets":"ttk_runs_fixed/topology_finetuning/candidateE2_fixed_lowlambda_expanded2688_constraints/ttk_pd_critical_pairs_gtvalues.npz",
    "original CNN init index":"models/wind_mr-hr/trained_cnn/cnn.index",
    "current base runner":"scripts/run_candidateF_expanded2688_finetune.py",
    "current ablation wrapper":"scripts/run_candidateF_cv_pers_ablation.py",
    "F1 evaluation report":"docs/topology_finetuning_candidateF_grad_E2_low_expanded2688_eval.md",
}
OUT=Path("ttk_runs_fixed/topology_finetuning/phase6_lxxxv_f1_lineage_audit_20261015")


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):
            h.update(chunk)
    return h.hexdigest()


def evidence(path, per_category=12):
    """Only snippets matching provenance terms, not entire potentially private logs."""
    records={k:[] for k in CATEGORIES}
    counts={k:0 for k in CATEGORIES}
    if not path.is_file():return records,counts
    with path.open('r', encoding='utf-8', errors='replace') as f:
        for lineno,line in enumerate(f,1):
            line=line.strip().replace('\x00','')
            for cat,regex in PATTERNS.items():
                if regex.search(line):
                    counts[cat]+=1
                    if len(records[cat])<per_category:
                        safe=SECRET.sub(r'\1[REDACTED]',line)
                        safe=BEARER.sub('Bearer [REDACTED]',safe)
                        # Plain command paths/flags only; truncate noisy training warning lines.
                        records[cat].append({'line':lineno,'text':safe[:320]})
    return records,counts


def fingerprint(root,relative,deep=False):
    p=root/relative
    record={'path':relative,'exists':p.is_file()}
    if p.is_file():
        record['bytes']=p.stat().st_size
        if deep or p.stat().st_size<60*1024*1024:
            record['sha256']=sha(p)
        else:record['sha256']='SKIPPED (large file; --deep-hash to include)'
    return record


def gather(root,deep=False):
    logs={}
    for k,v in ROOT_LOGS.items():
        p=root/v
        matches,totals=evidence(p)
        logs[k]={'path':v,'exists':p.is_file(), 'bytes':p.stat().st_size if p.is_file() else None,
                 'sha256':sha(p) if p.is_file() else None,
                 'matched_lines':matches,'match_counts':totals}
    sources={k:fingerprint(root,v,deep) for k,v in FROZEN_SOURCE.items()}
    models={}
    for short,method in MODEL.items():
        base=Path('models_fixed/topology_finetuning')/f'wind_finetune_{method}'
        cand={}
        for ck in ('cnn00001','cnn00002','cnn00003','cnn'):
            idx=str(base/ck/'cnn.index')
            cand[ck]=fingerprint(root,idx,deep)
            prefix=(root/base/ck)
            data=sorted(prefix.glob('cnn.data-*')) if prefix.is_dir() else []
            cand[ck]['checkpoint_data_files']=[fingerprint(root,str(p.relative_to(root)),deep) for p in data]
        arr=Path('data_out')/f'wind_finetune_{method}'
        cand['arrays']={key:fingerprint(root,str(arr/f'{key}.npy'),deep)
                        for key in ('idx','dataGT','dataSR')}
        models[short]=cand
    # Invariant check on *complete bytes*, without loading GB-scale arrays.
    benchmark={}
    for file in ('idx.npy','dataGT.npy'):
        x=root/Path('data_out')/f'wind_finetune_{MODEL["f1"]}'/file
        y=root/Path('data_out')/f'wind_finetune_{MODEL["grad"]}'/file
        if x.is_file() and y.is_file():
            ha,hb=sha(x),sha(y)
            benchmark[file]={'f1_sha256':ha,'grad_sha256':hb,'equal':ha==hb}
        else:
            benchmark[file]={'equal':None,'reason':'source missing'}
    # Hash up to 4 final checkpoint pieces for F1 epoch3 continuity. Not necessarily identical if later model save transformed.
    f1=models['f1']
    a=f1['cnn00003']; b=f1['cnn']
    complete=[]
    for r0,r1 in ((a,b),):
        x=r0.get('sha256'); y=r1.get('sha256')
        complete.append({'part':'cnn.index','equal':x==y if x is not None and y is not None else None})
    result={'schema':1,'root':str(root),'analysis_kind':'evidence inventory, NOT training provenance certification',
            'logs':logs,'sources':sources,'models':models,'benchmark_alignment':benchmark,
            'final_epoch3_index_match':complete[0]['equal']}
    return result


def lines_markdown(data):
    rows=['# Phase 6 LXXXV — F1 historical training-lineage evidence','',
          '**Status: NOT CLOSED.** This is an evidence inventory and a targeted excerpting of logs. '
          'It does not certify identical random seeds, optimizer state or code versions.','',
          '## Proven input files and current source fingerprints','',
          '| Role | Exists | Bytes | SHA256 or reason |','|---|---|---:|---|']
    for label in FROZEN_SOURCE:
        f=data['sources'][label]
        rows.append(f"| {label} (`{f['path']}`) | {f['exists']} | {f.get('bytes','—')} | `{f.get('sha256','MISSING')}` |")
    rows.extend(['','## Full benchmark F1 versus gradient-only array identity',''])
    for name in ('idx.npy','dataGT.npy'):
        v=data['benchmark_alignment'][name]
        rows.append(f"- `{name}`: exact SHA256 equality = **{v['equal']}**.")
    rows.extend(['','## Checkpoint inventory',''])
    for arm in MODEL:
        obj=data['models'][arm]
        rows.append(f'### {arm}')
        for ck in ('cnn00001','cnn00002','cnn00003','cnn'):
            f=obj[ck]
            rows.append(f"- `{f['path']}`: {'PRESENT' if f['exists'] else 'MISSING'}, index SHA256=`{f.get('sha256','n/a')}`; data shards={len(f['checkpoint_data_files'])}")
            for shard in f['checkpoint_data_files']:
                rows.append(f"  - `{shard['path']}` bytes={shard['bytes']} SHA256=`{shard['sha256']}`")
    rows.append(f"\n- F1 final-vs-epoch3 checkpoint **index** SHA256 match: `{data['final_epoch3_index_match']}` (index alone is not the full checkpoint).")
    rows.extend(['','## Targeted saved-log evidence','',
      'Excerpt lines are for manual validation; a missing excerpt does not establish a contrary fact. '
      'Each file SHA256 is stored in the JSON manifest.',''])
    for key in ROOT_LOGS:
        rec=data['logs'][key]
        rows.append(f"### {key}: `{rec['path']}` ({'present' if rec['exists'] else 'missing'})")
        for cat in CATEGORIES:
            xs=rec['matched_lines'][cat]
            rows.append(f"**{cat}:** {rec['match_counts'][cat]} matching lines (first {len(xs)} shown)")
            for item in xs:
                # backtick-safe plain quoted text
                rows.append(f"- L{item['line']}: `{item['text'].replace('`', "'")}`")
        rows.append('')
    rows.extend(['','## Lineage decision, evidence still required','',
      '1. Confirm historical F1 loss weights from its actual log and the original launch flags: UV 1.0, gradient 0.05, CV 0.004, PERS 0.002, all other optional losses disabled. Compare with the three new variant launch logs.',''
      '2. Confirm both historical and new runs used the same 2688-record training TFRecord, 64-pair repaired constraints and original pretrained CNN initialization. Current **file hashes** do not prove what bytes were used at July 2026 training time.',''
      '3. Compare trainer revision/code path and optimizer options. Current source SHA256 only fingerprints the current file; it is **not** the archived training-time source SHA256 unless separately recorded.',''
      '4. Record available random seeds and optimizer-state restoration evidence. If they are missing, keep the four-cell contrasts *descriptive* and avoid strict controlled causal attribution.','',
      '**Never infer training replication equivalence or identical stochastic trajectories from matching benchmark outputs.**',''])
    # clean up escaped line-break from string-concatenated list above if necessary
    return '\n'.join(rows)


def integrity(root):
    gate=root/Path('ttk_runs_fixed/topology_finetuning/phase6_lxxxiv_spatial_closeout_20261014')/'SHA256SUMS'
    if not gate.is_file():raise RuntimeError('LXXXIV verified closeout SHA256SUMS absent')
    count=0
    for line in gate.read_text().splitlines():
        expected,name=line.split('  ',1)
        if Path(name).name!=name:raise RuntimeError('Invalid checksum entry')
        p=gate.parent/name
        if not p.is_file() or sha(p)!=expected:raise RuntimeError('LXXXIV closeout mutated: '+name)
        count+=1
    if count!=3:raise RuntimeError('Expected three LXXXIV checksums')
    print('[SOURCE] PASS: frozen LXXXIV closeout checksums unchanged')


def run():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo-root',type=Path,default=Path.cwd())
    ap.add_argument('--mode',required=True,choices=('preflight','report','verify'))
    ap.add_argument('--deep-hash',action='store_true',help='Hash large TFRecords/checkpoint shards too (slow, and only a current snapshot)')
    args=ap.parse_args();root=args.repo_root.resolve()
    integrity(root)
    dest=root/OUT
    if args.mode=='verify':
        if not dest.is_dir():raise RuntimeError('LXXXV report directory absent')
        names=['f1_training_lineage.md','f1_lineage_manifest.json']
        checks=(dest/'SHA256SUMS').read_text().splitlines()
        if len(checks)!=2:raise RuntimeError('Expected two output checksums')
        for line in checks:
            digest,name=line.split('  ',1)
            if name not in names or sha(dest/name)!=digest:raise RuntimeError('LXXXV output digest mismatch: '+name)
        saved=json.loads((dest/'f1_lineage_manifest.json').read_text())
        latest=gather(root,deep=saved.get('deep_hash',False))
        latest['deep_hash']=saved.get('deep_hash',False)
        if latest!=saved:raise RuntimeError('Source evidence changed since report; preserve old audit and create versioned follow-up')
        if (dest/'f1_training_lineage.md').read_text()!=lines_markdown(saved):raise RuntimeError('Report content does not match manifest')
        print('[VERIFY-GATE] PASS: frozen source, source evidence, report and SHA256 continuity')
        return
    # preflight and report both gather evidence; only report writes.
    data=gather(root,deep=args.deep_hash);data['deep_hash']=bool(args.deep_hash)
    for label,rec in data['logs'].items():
        print(f'[LOG] {label} present={rec["exists"]} bytes={rec["bytes"]}')
    for key in ('idx.npy','dataGT.npy'):
        print(f'[BENCHMARK] {key} F1-vs-grad exact_sha256_equal={data["benchmark_alignment"][key]["equal"]}')
    missing=[k for k in ('f1_train','f1_batch') if not data['logs'][k]['exists']]
    if missing:raise RuntimeError('Missing mandatory historical F1 logs: '+','.join(missing))
    if any(data['benchmark_alignment'][k]['equal'] is not True for k in ('idx.npy','dataGT.npy')):
        raise RuntimeError('F1 vs gradient benchmark index/GT mismatch; stop lineage attribution')
    if args.mode=='preflight':
        print('[PREFLIGHT] PASS: source archived; F1 training logs present; exact benchmark parity. No output files written.')
    else:
        if dest.exists():raise RuntimeError('Report already exists; refusing overwrite: '+str(dest))
        report=lines_markdown(data)
        dest.mkdir(parents=True,exist_ok=False)
        (dest/'f1_training_lineage.md').write_text(report)
        (dest/'f1_lineage_manifest.json').write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
        (dest/'SHA256SUMS').write_text(''.join(f'{sha(dest/n)}  {n}\n' for n in ('f1_training_lineage.md','f1_lineage_manifest.json')))
        print('[REPORT] PASS: immutable evidence report generated in '+str(dest))
    print('[F1-LINEAGE] NOT CLOSED AUTOMATICALLY: review actual training flags, code revision, shared initialization and randomness')

if __name__=='__main__':
    try:run()
    except Exception as exc:
        print(f'[LXXXV] ERROR {type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(1)
