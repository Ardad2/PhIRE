#!/usr/bin/env python3
"""Safely append verified CXIV gradient gate notes; default is dry-run."""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import re
import tempfile

def h(data): return hashlib.sha256(data).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--apply',action='store_true')
    ap.add_argument('--notes',type=Path,default=Path('dataset_generation_and_repair_notes.md'))
    ap.add_argument('--addendum',type=Path,default=Path('phase8_CXIV_vertex_gradient_audit_20261116.md'))
    a=ap.parse_args()
    if not a.notes.is_file() or not a.addendum.is_file():
        ap.error('Missing original notes/addendum: transfer both into the PhIRE repo root')
    original=a.notes.read_bytes()
    addendum=a.addendum.read_bytes()
    base=original.decode('utf-8')
    additional=addendum.decode('utf-8')
    if not additional.startswith('# Part CXIV — Phase 8: TTK critical-vertex surrogate gradient verified'):
        ap.error('Unexpected CXIV addendum content; abort')
    if len(re.findall(r'^# Part CXIII\b',base,re.M))!=1:
        ap.error('Expected exactly one existing Part CXIII; abort')
    if re.search(r'^# Part CXIV\b|^# Part CXV\b',base,re.M):
        ap.error('Part CXIV or later already exists; refusing duplicate/reordering')
    # Audited appends: never replace, trim, or edit original historical bytes.
    amended=original+(b'\n' if not original.endswith(b'\n') else b'')+b'\n---\n\n'+addendum.strip(b'\n')+b'\n'
    print(f'[CXIV-PREFLIGHT] PASS: original SHA256={h(original)}; addendum SHA256={h(addendum)}; '
          f'append bytes={len(amended)-len(original)}; mode={"apply" if a.apply else "dry-run"}',flush=True)
    if not a.apply:return
    backup=a.notes.with_name(a.notes.name+'.before_CXIV_'+h(original)[:12]+'.bak')
    if backup.exists():ap.error(f'Backup exists, refusing overwrite: {backup}')
    with backup.open('xb') as fh:fh.write(original)
    temp=None
    try:
        with tempfile.NamedTemporaryFile(mode='wb',dir=a.notes.parent,
                                         prefix='.'+a.notes.name+'.CXIV-',delete=False) as fh:
            temp=Path(fh.name);fh.write(amended)
        os.chmod(temp,a.notes.stat().st_mode & 0o777)
        if a.notes.read_bytes()!=original:ap.error('Notes changed since check; abort')
        os.replace(temp,a.notes)
    finally:
        if temp is not None and temp.exists():temp.unlink()
    print(f'[CXIV-APPLY] PASS: appended one CXIV; backup={backup}',flush=True)

if __name__=='__main__': main()
