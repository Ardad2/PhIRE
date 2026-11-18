#!/usr/bin/env python3
"""Fail-closed and append-only Part CXV audit to the PhIRE master research notes.

Run from the PhIRE repository root. Dry run by default; --apply creates a
backup and atomically appends the supplied addendum. No historical text is
changed. Requires exactly one existing Part CXIV and no Part CXV or later.
"""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

NOTES = Path('dataset_generation_and_repair_notes.md')
ADDENDUM = Path('phase8_CXV_bridge_and_one_step_audit_20261116.md')


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--apply', action='store_true', help='Append audit (otherwise read-only)')
    args = ap.parse_args()
    if not NOTES.is_file() or not ADDENDUM.is_file():
        raise RuntimeError('Missing notes or CXV addendum in current directory')
    before, addition = NOTES.read_bytes(), ADDENDUM.read_bytes()
    old = before.decode('utf-8')
    audit = addition.decode('utf-8')
    headings = re.findall(r'^# Part ([IVXLCDM]+)\b.*$', old, flags=re.M)
    if headings.count('CXIV') != 1:
        raise RuntimeError('Expected exactly one existing Part CXIV heading')
    if not headings or headings[-1] != 'CXIV':
        raise RuntimeError(f'Expected CXIV to be the last part; got {headings[-1] if headings else "none"}')
    if re.search(r'^# Part CXV\b', old, flags=re.M):
        raise RuntimeError('Part CXV already exists: refusing duplicate')
    if not audit.startswith('# Part CXV — ') or len(re.findall(r'^# Part CXV\b', audit, flags=re.M)) != 1:
        raise RuntimeError('Unexpected addendum heading')
    if re.search(r'^# Part CXVI\b', old, flags=re.M):
        raise RuntimeError('Part CXVI already exists: refusing out-of-order append')
    if not before.endswith(b'\n') or not addition.endswith(b'\n'):
        raise RuntimeError('Both Markdown files must end with a newline')
    if (NOTES.parent / 'phase8_CXV_bridge_and_one_step_audit_20261116.md').resolve() != ADDENDUM.resolve():
        raise RuntimeError('Addendum path mismatch')
    combined = before.rstrip(b'\n') + b'\n\n' + addition
    mode = 'APPLY' if args.apply else 'PREFLIGHT'
    print(f'[CXV-{mode}] PASS: original={sha(before)} addendum={sha(addition)} resulting={sha(combined)} append_bytes={len(combined)-len(before)}', flush=True)
    if not args.apply:
        print('[CXV-PREFLIGHT] Dry run only; no files changed.', flush=True)
        return
    backup = NOTES.with_name(NOTES.name + f'.before_CXV_{sha(before)[:12]}.bak')
    if backup.exists():
        raise RuntimeError(f'Backup path already exists; refusing overwrite: {backup}')
    # Exclusive backup; never overwrite or silently remove existing records.
    fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as f:
        f.write(before)
        f.flush()
        os.fsync(f.fileno())
    tmp = None
    try:
        with tempfile.NamedTemporaryFile('wb', dir=NOTES.parent, prefix='.CXV.', suffix='.tmp', delete=False) as f:
            tmp = Path(f.name)
            f.write(combined)
            f.flush()
            os.fsync(f.fileno())
        shutil.copymode(NOTES, tmp)
        # Double-check against concurrent user updates before replacement.
        if NOTES.read_bytes() != before:
            raise RuntimeError('Notes changed during operation; original remains unchanged')
        os.replace(tmp, NOTES)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()
    if NOTES.read_bytes() != combined:
        raise RuntimeError('Post-write content verification failed')
    print(f'[CXV-APPLY] PASS: Part CXV appended, backup={backup.name}', flush=True)

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'[CXV-ERROR] {exc}', file=sys.stderr)
        sys.exit(1)
