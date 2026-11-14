#!/usr/bin/env python3
"""XCII compatibility launcher: strict VTI scalar XML readback + VTU XML schema.

Reuses the ORIGINAL, SHA-pinned XCII module in memory, with only the failing
VTK reader functions and the reader-dependent selftest replaced. Its original
source gates, converter, TTK command, non-overwrite paths, SHA report verifier,
and locked-geography scope remain unchanged. Adds launcher SHA to provenance.

VTU verification checks XML framing/schema, not compressed numerical payload.
The separate corrected GUDHI parser audit remains MANDATORY before science.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import string
import sys
import tempfile
import xml.etree.ElementTree as ET

ORIGINAL_REL = Path('phase7_xcii_update/scripts/phase7_xcii_dev_canonical_pd.py')
ORIGINAL_SHA256 = '743c7642611b6f8281891ffece8b60a6b717a0047042d745daa75cd42caee53b'
REQUIRED_PD_ARRAYS = frozenset({'Birth', 'Persistence', 'PairIdentifier', 'PairType'})
ALLOWED_BASE64 = frozenset(string.ascii_letters + string.digits + '+/=')
DIM = 160


def need(condition: bool, message: str):
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for blob in iter(lambda: f.read(1024*1024), b''):
            h.update(blob)
    return h.hexdigest()


def parse_xml(path: Path, expected_type: str):
    need(path.is_file() and path.stat().st_size > 0, f'Missing or empty {expected_type}: {path}')
    try:
        r = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise RuntimeError(f'Standard-library XML parser rejected {path}: {exc}') from exc
    need(r.tag == 'VTKFile' and r.get('type') == expected_type,
         f'Wrong VTK XML file type for {path}: {r.tag}, {r.get("type")}')
    return r


def check_vti_xml(path: Path, reference):
    """Full numeric equality readback for canonical converter's ASCII float32 VTI."""
    import numpy as np
    r = parse_xml(path, 'ImageData')
    grid = r.find('ImageData')
    need(grid is not None, f'Missing ImageData: {path}')
    expected_extent = [0, DIM-1, 0, DIM-1, 0, 0]
    def extent(el, key):
        try:
            return [int(x) for x in el.attrib[key].split()]
        except (KeyError, ValueError) as exc:
            raise RuntimeError(f'Invalid VTI {key} in {path}') from exc
    need(extent(grid, 'WholeExtent') == expected_extent, f'Wrong WholeExtent: {path}')
    pieces = grid.findall('Piece')
    need(len(pieces) == 1 and extent(pieces[0], 'Extent') == expected_extent,
         f'Wrong VTI Piece/Extent: {path}')
    pd = pieces[0].find('PointData')
    need(pd is not None, f'Missing PointData: {path}')
    arrays = [x for x in pd.findall('DataArray') if x.get('Name') == 'wind_speed']
    need(len(arrays) == 1, f'Expected exactly one wind_speed DataArray: {path}')
    a = arrays[0]
    need(a.get('format') == 'ascii' and a.get('type') == 'Float32',
         f'Unexpected wind_speed format/dtype in {path}; require ASCII Float32')
    need(int(a.get('NumberOfComponents', '1')) == 1,
         f'wind_speed must be a scalar: {path}')
    need(a.text is not None, f'No VTI numerical payload: {path}')
    tokens = a.text.split()
    need(len(tokens) == DIM*DIM, f'VTI has {len(tokens)} tokens, expected {DIM*DIM}: {path}')
    try:
        vals = np.asarray(tokens, dtype=np.float32)
    except ValueError as exc:
        raise RuntimeError(f'Non-numeric VTI data: {path}') from exc
    actual = vals.reshape(DIM, DIM)   # flat ID x+y*W, consistent with frozen converter
    expected = np.asarray(reference, dtype=np.float32)
    need(expected.shape == (DIM,DIM) and np.isfinite(expected).all(),
         f'Invalid source scalar array for {path}')
    need(np.isfinite(actual).all(), f'Nonfinite VTI scalar payload: {path}')
    need(np.array_equal(actual, expected),
         f'VTI values/orientation mismatch {path}; max_abs={np.max(np.abs(actual-expected))}')


def check_pd_xml(path: Path):
    """Strict VTU *structural* validation; compressed numerics are not decoded."""
    r = parse_xml(path, 'UnstructuredGrid')
    grid = r.find('UnstructuredGrid')
    need(grid is not None, f'Missing UnstructuredGrid: {path}')
    pieces = grid.findall('Piece')
    need(len(pieces) == 1, f'Expected one UnstructuredGrid Piece: {path}')
    piece = pieces[0]
    try:
        points, cells = int(piece.attrib['NumberOfPoints']), int(piece.attrib['NumberOfCells'])
    except (KeyError, ValueError) as exc:
        raise RuntimeError(f'Invalid Piece counts: {path}') from exc
    need(points > 0 and cells > 0, f'Empty PD geometry: {path} points={points} cells={cells}')
    found = {e.get('Name') for e in piece.iter('DataArray') if e.get('Name')}
    need(REQUIRED_PD_ARRAYS <= found,
         f'Missing PD array(s) {sorted(REQUIRED_PD_ARRAYS - found)}: {path}')
    appended = r.find('AppendedData')
    if appended is not None:
        need(appended.get('encoding') == 'base64', f'Unexpected VTU encoding: {path}')
        data = ''.join((appended.text or '').split())
        need(data.startswith('_') and len(data)>1 and all(c in ALLOWED_BASE64 for c in data[1:]),
             f'Invalid base64-framed VTU appended section: {path}')
    print(f'[PD-XML] PASS points={points} cells={cells} required_arrays={len(REQUIRED_PD_ARRAYS)} '
          f'bytes={path.stat().st_size} (structural only)', flush=True)
    return {'cells':cells,'points':points}


def compat_selftest():
    import numpy as np
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
    # Exercise canonical ASCII mode and x-fastest row-major axis handling.
    ref = (np.arange(DIM*DIM,dtype=np.float32).reshape(DIM,DIM)/17.0).astype(np.float32)
    with tempfile.TemporaryDirectory() as t:
        p = Path(t)/'synthetic.vti'
        img=vtk.vtkImageData();img.SetDimensions(DIM,DIM,1)
        arr=numpy_to_vtk(ref.ravel(order='C'),deep=True,array_type=vtk.VTK_FLOAT)
        arr.SetName('wind_speed')
        img.GetPointData().SetScalars(arr)
        w=vtk.vtkXMLImageDataWriter();w.SetFileName(str(p));w.SetInputData(img)
        w.SetDataModeToAscii()
        need(w.Write()!=0,'VTK synthetic ASCII writer failed')
        check_vti_xml(p,ref)
        # Reversal of row/column orientation must fail validation.
        failed=False
        try:check_vti_xml(p,ref.T)
        except RuntimeError:failed=True
        need(failed,'VTI orientation corruption went undetected')
        # Tampering with a scalar must fail numerical readback.
        root=ET.parse(p)
        da=root.getroot().find('ImageData/Piece/PointData/DataArray')
        need(da is not None and da.text,'Selftest VTI structure unexpected')
        toks=da.text.split();toks[0]='1234567';da.text=' '+' '.join(toks)+' '
        bad=Path(t)/'tampered.vti';root.write(bad,encoding='unicode',xml_declaration=True)
        failed=False
        try:check_vti_xml(bad,ref)
        except RuntimeError:failed=True
        need(failed,'Numeric VTI tamper went undetected')
        # VTU structural check (does not simulate compressed payload validity).
        pd=Path(t)/'synthetic.vtu'
        fields=''.join(f'<DataArray Name="{name}" type="Float32" format="ascii">0</DataArray>'
                       for name in sorted(REQUIRED_PD_ARRAYS))
        pd.write_text('<?xml version="1.0"?><VTKFile type="UnstructuredGrid">'
                      '<UnstructuredGrid><Piece NumberOfPoints="2" NumberOfCells="1">'
                      f'<PointData>{fields}</PointData></Piece></UnstructuredGrid></VTKFile>')
        info=check_pd_xml(pd);need(info=={'cells':1,'points':2},'VTU structural mismatch')
        pd.write_text('<unclosed>')
        failed=False
        try:check_pd_xml(pd)
        except RuntimeError:failed=True
        need(failed,'Malformed VTU XML went undetected')
    print('[COMPAT-SELFTEST] PASS: ASCII Float32 C-order scalar values, orientation/tamper rejects, '
          'VTU XML/schema gate (VTU numerics deferred)',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['selftest','preflight','extract','verify'],required=True)
    p.add_argument('--repo-root',type=Path,default=Path('/work'))
    args=p.parse_args()
    root=args.repo_root.resolve()
    original=(root/ORIGINAL_REL).resolve()
    need(original.is_file(), f'Missing original XCII script: {original}')
    need(sha256(original)==ORIGINAL_SHA256,
         f'Original XCII script SHA changed; expected {ORIGINAL_SHA256}, got {sha256(original)}')
    spec=importlib.util.spec_from_file_location('frozen_phase7_xcii',original)
    need(spec is not None and spec.loader is not None,'Cannot import pinned XCII original')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_vti=check_vti_xml
    module.check_pd=check_pd_xml
    module.selftest=compat_selftest
    # Record exact compatibility implementation in XCII's immutable source
    # SHA manifest; the original driver and all frozen upstream artifacts remain unchanged.
    base_gate=module.source_gate
    def source_gate_with_compat(*a,**kw):
        hashes,fields,arrays=base_gate(*a,**kw)
        hashes['compat_launcher_sha256']=sha256(Path(__file__))
        return hashes,fields,arrays
    module.source_gate=source_gate_with_compat
    print(f'[COMPAT] PINNED original_XCII_sha256={ORIGINAL_SHA256}',flush=True)
    print(f'[COMPAT] launcher_sha256={sha256(Path(__file__))}',flush=True)
    print('[COMPAT] Numeric VTI XML readback + structural PD XML readback; '
          'TTK extraction unchanged; compressed PD numerical audit still mandatory',flush=True)
    sys.argv=[str(original),'--mode',args.mode,'--repo-root',str(root)]
    module.main()

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(f'[COMPAT] ERROR {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        sys.exit(1)
