#!/usr/bin/env python3
"""Additive Phase 7 CVIII/CIX/CX source-binding hotfix (2026-10-21).

Loads SHA-pinned ORIGINAL controller unchanged from disk, fixes only CVII
configuration retrieval in a compiled in-memory copy, and invokes original
controller entrypoint. Numerical solvers, historical scripts and model data
are unmodified. No mutation of frozen controller or completed outputs.
"""
from __future__ import annotations
import hashlib, sys, types
from pathlib import Path

ORIGINAL = Path('phase7_cviii_update/scripts/phase7_cviii_cix_2010_dev_topology.py')
EXPECTED_SHA = '027fcea53c67a0123390a8082ede293f4bee0da0ac93383aa3a43612547df79f'
OLD = "cfg=json.loads((pdout/'run_config.json').read_text());mapping=cfg['anchor_hours']"
NEW = """manifest=json.loads((pdout/'extraction_manifest.json').read_text())
    need(manifest.get('format')==cii.SCHEMA and manifest.get('total_pairs')==1176,
         'Bound CVII manifest schema/diagram count mismatch')
    cfg=manifest['run_config']
    need(cfg.get('scope')==cii.SCHEMA and cfg.get('region')==reg and
         cfg.get('week_start_utc')==w and cfg.get('hours')==168,
         'Bound CVII manifest configuration/region-week mismatch')
    mapping=cfg['anchor_hours']
    need(set(mapping)=={'0','84'},'Missing/extra CVII history anchor keys')"""

def digest(data): return hashlib.sha256(data).hexdigest()

def main():
    root=Path(__file__).resolve().parents[2]
    path=root/ORIGINAL
    if not path.is_file():raise RuntimeError('Missing original installed CVIII controller: '+str(path))
    raw=path.read_bytes()
    if digest(raw)!=EXPECTED_SHA:raise RuntimeError('Original controller SHA mismatch; refuse unreviewed code change')
    source=raw.decode('utf-8')
    if source.count(OLD)!=1:raise RuntimeError('Missing or repeated CVII binding template')
    bound=source.replace(OLD,NEW,1)
    compile(bound,str(path)+'::hotfix-read-manifest','exec')
    mod=types.ModuleType('cviii_hotfix_inmemory_original')
    mod.__file__=str(path)
    sys.modules[mod.__name__]=mod
    exec(compile(bound,str(path)+'::hotfix-read-manifest','exec'),mod.__dict__)
    print('[HOTFIX] PASS: original CVIII controller SHA pinned, CVII extraction_manifest.run_config binding corrected in memory',flush=True)
    print('[HOTFIX] Original controller unmodified; patched in-memory sha256='+digest(bound.encode('utf-8')),flush=True)
    mod.main()

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('[HOTFIX] ERROR '+type(e).__name__+': '+str(e),file=sys.stderr,flush=True)
        raise SystemExit(1)
