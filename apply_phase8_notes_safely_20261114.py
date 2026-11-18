#!/usr/bin/env python3
"""Safely apply the two *additive* Phase 8 audit entries to evolving PhIRE notes.

Requires the previously supplied phase8_provenance_and_audit_notes_20261114.patch
in the same directory as dataset_generation_and_repair_notes.md.

Default is validation/dry-run. Pass --apply to create a backup and update the notes.
No model, experimental, source-code, or other file is edited.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import shutil
import sys

NOTES = Path('dataset_generation_and_repair_notes.md')
PATCH = Path('phase8_provenance_and_audit_notes_20261114.patch')
BACKUP = Path('dataset_generation_and_repair_notes.md.pre_phase8_20261114.bak')
ERRATUM = '## November 14, 2026 provenance clarification — active Candidate Dpd run, unrecovered executed source'
PART9 = '# Part IX — Candidate E2-expanded-672 faithful TTK critical-pair expansion'
PART101 = '# Part CXI — Phase 8 dynamic persistence-loss pilot: audit protocol and design record (November 14, 2026)'


def added_blocks(patch_text: str) -> list[str]:
    pieces = re.split(r'(?m)^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@[^\n]*\n', patch_text)
    if len(pieces) != 3:
        raise ValueError(f'Expected exactly 2 unified-diff hunks, got {len(pieces) - 1}.')
    blocks = [''.join(line[1:] for line in hunk.splitlines(keepends=True)
                      if line.startswith('+') and not line.startswith('+++'))
              for hunk in pieces[1:]]
    if ERRATUM not in blocks[0] or PART101 not in blocks[1]:
        raise ValueError('Patch additions do not contain the expected audit headings.')
    return blocks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Create backup and write notes; default only validates')
    args = parser.parse_args()
    if not NOTES.is_file() or not PATCH.is_file():
        parser.error(f'Run from ~/PhIRE with both {NOTES.name} and {PATCH.name} present.')
    source = NOTES.read_text(encoding='utf-8')
    erratum, part101 = added_blocks(PATCH.read_text(encoding='utf-8'))
    if ERRATUM in source or PART101 in source:
        parser.error('One or both audit headings are already present; refusing to create duplicates.')
    marker = '\n---\n\n' + PART9
    if source.count(marker) != 1:
        parser.error(f'Expected exactly one Part IX section boundary; found {source.count(marker)}. Refusing to guess.')
    updated = source.replace(marker, erratum + marker, 1)
    updated = updated + ('' if updated.endswith('\n') else '\n') + part101
    if not updated.endswith('\n'):
        updated += '\n'
    assert updated.count(ERRATUM) == updated.count(PART101) == 1
    print('PASS: Found unique Part IX insertion point; prepared both additions.')
    print(f'Before: {len(source.splitlines()):,} lines; after: {len(updated.splitlines()):,} lines.')
    if not args.apply:
        print('DRY RUN only; notes unchanged. Rerun with --apply to save.')
        return 0
    if BACKUP.exists():
        parser.error(f'Backup already exists: {BACKUP}. Refusing to overwrite it.')
    temp = NOTES.with_name(NOTES.name + '.phase8_pending')
    if temp.exists():
        parser.error(f'Temporary output already exists: {temp}; refusing to overwrite.')
    shutil.copy2(NOTES, BACKUP)
    try:
        temp.write_text(updated, encoding='utf-8')
        temp.replace(NOTES)
    except Exception:
        if temp.exists():
            temp.unlink()
        raise
    print(f'APPLIED: {NOTES}; original saved as {BACKUP}.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
