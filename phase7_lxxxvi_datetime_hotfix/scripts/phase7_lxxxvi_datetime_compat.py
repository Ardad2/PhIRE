#!/usr/bin/env python3
"""Read-only launch adapter for WTK's YYYYMMDDHHMMSS HSDS timestamps.

Loads the frozen original LXXXVI module and changes ONLY the in-memory
parser function.  It never edits source scripts, candidate coordinates,
notes, historical datasets, or model artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / 'phase7_lxxxvi_update/scripts/phase7_region_inventory.py'
EXPECTED_ORIGINAL_SHA256 = '24ba066d5aa3030b32d5ef0ece7ec6ba7c86b9f9c2f7d049319f8ff845f5c57c'
CANDIDATES = ROOT / 'phase7_lxxxvi_update/region_candidates.json'
EXPECTED_CANDIDATE_SHA256 = 'ba0fb5e1236acda290c8220bca1a89587f71ca42cc98ea8f540295834f519b59'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_original():
    for p, expected in ((ORIGINAL, EXPECTED_ORIGINAL_SHA256),
                        (CANDIDATES, EXPECTED_CANDIDATE_SHA256)):
        if not p.is_file():
            raise RuntimeError(f'Missing frozen LXXXVI input: {p}')
        got = digest(p)
        if got != expected:
            raise RuntimeError(f'Frozen input SHA mismatch: {p}; expected={expected}, actual={got}')
    spec = importlib.util.spec_from_file_location('phase7_lxxxvi_original_pinned', ORIGINAL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to import original LXXXVI module: {ORIGINAL}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    original_parser = mod.to_datetime

    def parse_wtk_datetime(raw):
        if hasattr(raw, 'item') and not isinstance(raw, (str, bytes)):
            raw = raw.item()
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8', errors='strict')
        stripped = str(raw).strip()
        if re.fullmatch(r'\d{14}', stripped):
            return datetime.strptime(stripped, '%Y%m%d%H%M%S')
        return original_parser(raw)

    mod.to_datetime = parse_wtk_datetime
    return mod


def main():
    # Original argparse interface is preserved: --mode, --dataset, --out,
    # --buffer, --stride.  We do not intercept or alter these arguments.
    mod = load_original()
    print('[DATETIME-COMPAT] PASS: frozen script and candidate SHA256; in-memory 14-digit WTK timestamp support only', flush=True)
    print(f'[DATETIME-COMPAT] wrapper_sha256={digest(Path(__file__))}', flush=True)
    mod.main()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'[LXXXVI-COMPAT] ERROR {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)
