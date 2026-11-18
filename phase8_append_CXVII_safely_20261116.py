#!/usr/bin/env python3
"""Fail-closed, append-only Part CXVII update of the PhIRE master research notes.

Dry-run by default. --apply creates a byte-identical backup and atomically
appends the CXVII addendum. Does not edit prior sections or other files.
Run from ~/PhIRE with the named CXVII addendum in the same directory.
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
ADDENDUM = Path('phase8_CXVII_reconstruction_runtime_audit_20261116.md')


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Write changes (default: read-only preflight)')
    args = parser.parse_args()

    if not NOTES.is_file() or not ADDENDUM.is_file():
        raise RuntimeError('Both the master notes and the CXVII addendum must exist in the current directory')
    before = NOTES.read_bytes()
    addition = ADDENDUM.read_bytes()
    old = before.decode('utf-8')
    add = addition.decode('utf-8')

    # Use only level-1 Part headers, so references to earlier Parts in body text
    # cannot be mistaken for duplicate or out-of-order sections.
    heads = re.findall(r'^# Part ([IVXLCDM]+)\b', old, flags=re.M)
    if len(heads) < 2 or heads[-2:] != ['CXV', 'CXVI']:
        raise RuntimeError(f'Expected existing Part CXV then CXVI at end, found {heads[-3:]}')
    if heads.count('CXVI') != 1 or 'CXVII' in heads:
        raise RuntimeError('Duplicate/out-of-order Part CXVI/CXVII heading detected; no changes made')
    if not add.startswith('# Part CXVII — '):
        raise RuntimeError('Addendum is not headed Part CXVII')
    addheads = re.findall(r'^# Part ([IVXLCDM]+)\b', add, flags=re.M)
    if addheads != ['CXVII']:
        raise RuntimeError('Expected exactly one CXVII heading in the addendum')
    if not before.endswith(b'\n') or not addition.endswith(b'\n'):
        raise RuntimeError('Both Markdown files must end with newline')
    if len(addition) < 1000:
        raise RuntimeError('Addendum unexpectedly small')

    # Append without trimming/changing even the trailing whitespace of the
    # previous notes. Exactly one extra newline separates sections.
    merged = before + b'\n' + addition
    if not merged.startswith(before):
        raise RuntimeError('Prefix preservation failed')
    tag = 'CXVII-APPLY' if args.apply else 'CXVII-PREFLIGHT'
    print(f'[{tag}] PASS: original={sha256(before)} '
          f'addendum={sha256(addition)} resulting={sha256(merged)} '
          f'append_bytes={len(merged)-len(before)}', flush=True)
    if not args.apply:
        print('[CXVII-PREFLIGHT] Dry run only; master notes unchanged.', flush=True)
        return

    # Exclusive backup; do not silently reuse/overwrite a past backup.
    backup = NOTES.with_name(NOTES.name + f'.before_CXVII_{sha256(before)[:12]}.bak')
    fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as f:
        f.write(before)
        f.flush()
        os.fsync(f.fileno())

    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=NOTES.parent,
                                         prefix='.CXVII.', suffix='.tmp', delete=False) as f:
            temp = Path(f.name)
            f.write(merged)
            f.flush()
            os.fsync(f.fileno())
        shutil.copymode(NOTES, temp)
        if NOTES.read_bytes() != before:
            raise RuntimeError('Master notes changed during append; original not overwritten')
        os.replace(temp, NOTES)
    finally:
        if temp is not None and temp.exists():
            temp.unlink()
    if NOTES.read_bytes() != merged:
        raise RuntimeError('Post-write SHA256 verification failed')
    print(f'[CXVII-APPLY] PASS: Part CXVII appended; original backup={backup}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'[CXVII-ERROR] {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
