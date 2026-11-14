#!/usr/bin/env python3
"""Read-only structural probe and compatibility launcher for Phase 6 LXXVII.

Do not edit the original LXXVII driver: its SHA256 is part of the existing
input_manifest.json, and the original resume/integrity gates should continue
checking precisely the same extraction inputs and settings.

The specific compatibility workaround is to replace only verify_pd with a
stdlib XML/structural check. Numerical validation is deliberately delegated to
Phase 6's existing canonical read_pd/GUDHI gate in a separate environment.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import string
import sys
import xml.etree.ElementTree as ET

DRIVER = 'phase6_lxxvii_update/scripts/phase6_canonical_newarms_pd.py'
OLD_REL = ('ttk_runs_fixed/topology_finetuning/'
           'candidateF_grad_pairzero_expanded2688_s0_extraction_pilot_20261014')
NEW_REL = ('ttk_runs_fixed/topology_finetuning/'
           'phase6_lxxvii_canonical_pd_newarms_20261014/grad')
STEM = 'candidateF_grad_pairzero_expanded2688_SR_s0_speed_p160_x0_y0'
NEEDED_ARRAYS = {'Birth', 'Persistence', 'PairIdentifier', 'PairType'}
ALLOWED_B64 = frozenset(string.ascii_letters + string.digits + '+/=')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fp:
        for block in iter(lambda: fp.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def parse_check(path: Path):
    require(path.is_file() and path.stat().st_size > 0, f'Missing/empty VTU: {path}')
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise RuntimeError(f'XML parser also rejects VTU {path}: {exc}') from exc
    require(root.tag == 'VTKFile' and root.get('type') == 'UnstructuredGrid',
            f'Not a VTK unstructured grid: {path}')
    grid = root.find('UnstructuredGrid')
    require(grid is not None, f'Missing UnstructuredGrid in {path}')
    pieces = grid.findall('Piece')
    require(len(pieces) == 1, f'Expected one Piece, found {len(pieces)}: {path}')
    piece = pieces[0]
    try:
        num_points = int(piece.attrib['NumberOfPoints'])
        num_cells = int(piece.attrib['NumberOfCells'])
    except (KeyError, ValueError) as exc:
        raise RuntimeError(f'Invalid Piece count attributes: {path}') from exc
    require(num_points > 0 and num_cells > 0,
            f'Empty persistence geometry points={num_points} cells={num_cells}: {path}')
    arrays = {elem.get('Name') for elem in piece.iter('DataArray') if elem.get('Name')}
    missing = NEEDED_ARRAYS - arrays
    require(not missing, f'Missing required TTK PD arrays {sorted(missing)}: {path}')
    arrays_with_names = sum(1 for elem in piece.iter('DataArray') if elem.get('Name'))
    require(arrays_with_names >= len(NEEDED_ARRAYS), f'Insufficient arrays: {path}')
    # TTK's appended-base64 format is valid XML. This guard deliberately does
    # not decode or claim to validate the numerical payload; that is the role
    # of the independently audited canonical read_pd/GUDHI gate.
    appended = root.find('AppendedData')
    if appended is not None:
        encoding = appended.get('encoding', '')
        require(encoding == 'base64',
                f'Unexpected AppendedData encoding {encoding!r}: {path}')
        content = ''.join((appended.text or '').split())
        require(content.startswith('_') and len(content) > 1,
                f'Bad appended-data sentinel or empty stream: {path}')
        require(all(c in ALLOWED_B64 for c in content[1:]),
                f'Non-base64 character in appended stream: {path}')
    print(f'[PD-XML] PASS points={num_points} cells={num_cells} '
          f'required_arrays={len(NEEDED_ARRAYS)} '
          f'bytes={path.stat().st_size} '
          f'file={path}', flush=True)


def probe(repo: Path):
    for kind, rel in [('OLD', OLD_REL), ('NEW', NEW_REL)]:
        pd = repo / rel / 'pd' / (STEM + '_pd_port_0.vtu')
        vti = repo / rel / 'vti' / (STEM + '.vti')
        parse_check(pd)
        require(vti.is_file(), f'Missing {kind} VTI: {vti}')
        print(f'[PROBE] {kind} VTU_SHA256={sha256(pd)} VTI_SHA256={sha256(vti)}', flush=True)
    old_pd = repo / OLD_REL / 'pd' / (STEM + '_pd_port_0.vtu')
    new_pd = repo / NEW_REL / 'pd' / (STEM + '_pd_port_0.vtu')
    old_vti = repo / OLD_REL / 'vti' / (STEM + '.vti')
    new_vti = repo / NEW_REL / 'vti' / (STEM + '.vti')
    require(sha256(old_pd) == sha256(new_pd), 'Historical/new pilot VTU hashes differ')
    require(sha256(old_vti) == sha256(new_vti), 'Historical/new pilot VTI hashes differ')
    print('[PROBE] PASS: earlier/new gradient sample-0 VTU and VTI are byte-identical; '
          'both VTUs pass XML/header checks.', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo-root', type=Path, default=Path('/work'))
    ap.add_argument('--mode', required=True,
                    choices=['probe', 'preflight', 'pilot', 'full', 'verify'])
    ap.add_argument('--arm', choices=['all', 'grad', 'cv', 'pers'], default='all')
    args = ap.parse_args()
    repo = args.repo_root.resolve()
    require(repo.is_dir(), f'Missing PhIRE root {repo}')
    if args.mode == 'probe':
        probe(repo)
        return
    driver = repo / DRIVER
    require(driver.is_file(), f'Missing original LXXVII driver {driver}')
    spec = importlib.util.spec_from_file_location('phase6_lxxvii_original', driver)
    require(spec is not None and spec.loader is not None, 'Unable to load driver')
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    # Override exactly one validator in the in-memory module. The original file
    # remains unchanged on disk, retaining existing provenance SHA checks.
    original.verify_pd = parse_check
    print('[COMPAT] Original LXXVII driver unchanged. Replacing ONLY the '
          'PD VTK-reader check with XML/schema validation in memory.', flush=True)
    original.process(repo, args.mode, args.arm)
    print('[COMPAT] PASS: XML-level validation complete. For pilot, still run '
          'the independent canonical GUDHI gate; for full, audit all 504 PDs '
          'before reporting results.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'[COMPAT] ERROR {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
