#!/usr/bin/env python3
"""Append the one combined Phase 8 CXIII audit entry safely to the CURRENT notes.

By default validation only; use --apply after reviewing the preview. This is
version-aware and aborts if CXIII or any later Part heading already exists.
Never rewrites an earlier section or replaces the master document.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import re
import hashlib
import tempfile


def h(b):return hashlib.sha256(b).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--apply',action='store_true')
    p.add_argument('--notes',type=Path,default=Path('dataset_generation_and_repair_notes.md'))
    p.add_argument('--addendum',type=Path,default=Path('phase8_cxiii_timing_and_synthetic_audit_20261116.md'))
    a=p.parse_args()
    if not a.notes.is_file() or not a.addendum.is_file():
        p.error('Missing notes/addendum file; transfer the addendum into ~/PhIRE first')
    old=a.notes.read_bytes(); app=a.addendum.read_bytes()
    notes=old.decode('utf-8');ext=app.decode('utf-8')
    if not ext.startswith('# Part CXIII — Phase 8: TTK/GUDHI runtime'):
        p.error('Unexpected addendum heading; no changes made')
    if notes.count('# Part CXII — Phase 8 dynamic persistence-loss pilot:')!=1:
        p.error('Expected exactly one original Part CXII Phase 8 heading; no changes made')
    if re.search(r'^# Part CXIII\b|^# Part CXIV\b|^# Part CXV\b',notes,re.M):
        p.error('Part CXIII or later already exists; abort to avoid duplicate/retroactive edits')
    if notes.count('## November 14, 2026 provenance clarification')!=1:
        p.error('Candidate Dpd provenance note missing/duplicated; no changes made')
    if not notes.rstrip().endswith(('---','*','.',')',']','`','"','\'')):
        print('NOTE: Original document has nonstandard ending; retaining all original bytes.')
    rendered=old+(b'\n' if not old.endswith(b'\n') else b'')+b'\n---\n\n'+app.strip(b'\n')+b'\n'
    print('[AUDIT-PREFLIGHT] PASS: preserves all original content, adds one Part CXIII section')
    print(' original SHA256:',h(old));print(' appended SHA256:',h(app))
    print(' resulting SHA256:',h(rendered));print(' append bytes:',len(rendered)-len(old))
    print(' notes:',a.notes.resolve())
    if not a.apply:return
    # Fail-closed copy with a content-addressed unique backup in the same folder;
    # will never overwrite an existing file.
    backup=a.notes.with_name(a.notes.name+'.pre_phase8_CXIII_'+h(old)[:12]+'.bak')
    if backup.exists():p.error(f'Backup already exists; refusing to overwrite: {backup}')
    with backup.open('xb') as f:f.write(old)
    tmp=None
    try:
        with tempfile.NamedTemporaryFile(mode='wb',dir=a.notes.parent,
                                         prefix='.'+a.notes.name+'.audit-',delete=False) as f:
            tmp=Path(f.name);f.write(rendered)
        os.chmod(tmp,a.notes.stat().st_mode & 0o777)
        if a.notes.read_bytes()!=old:p.error('Document changed during preparation; retry after inspection')
        os.replace(tmp,a.notes)
    finally:
        if tmp is not None and tmp.exists():tmp.unlink()
    print('[AUDIT-APPLY] PASS: appended CXIII; original preserved at',backup)

if __name__=='__main__':main()
