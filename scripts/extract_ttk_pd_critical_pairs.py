#!/usr/bin/env python3
"""
Extract TTK persistence-diagram critical-pair constraints from VTU files.

Reads GT persistence-diagram VTU files (binary base64+zlib or ASCII inline)
produced by TTK's ttkPersistenceDiagram filter and writes an NPZ file of
birth/death vertex coordinates and target scalar values for each sample.

These constraints are used by run_candidateE_ttkcrit_refiner.py as
fixed topological targets during PyTorch fine-tuning.

Usage
-----
python3 scripts/extract_ttk_pd_critical_pairs.py \
    --pd-dir ttk_runs_fixed/topology_finetuning/candidateD_topology/pd/GT \
    --out-dir ttk_runs_fixed/topology_finetuning/candidateE_constraints \
    --patch 160 \
    --persistence-frac 0.01 \
    --top-k 64

Output NPZ keys
---------------
n_samples           : int — number of VTU files processed
patch_size          : int — grid side (H = W = patch_size)
persistence_frac    : float — threshold used
top_k               : int — max pairs per sample
sample_names        : (N,) object — VTU filenames (without extension)
sample_start        : (N,) int32 — start index into flat pair arrays per sample
sample_count        : (N,) int32 — number of pairs per sample
birth_vid           : (P,) int32 — birth vertex ID (flat index in H×W grid)
death_vid           : (P,) int32 — death vertex ID
birth_val           : (P,) float32 — scalar value at birth vertex
death_val           : (P,) float32 — scalar value at death vertex (birth+pers)
persistence         : (P,) float32 — persistence value
pair_type           : (P,) int32 — TTK pair type (0=min-saddle, 1=saddle-max, -1=inf)
"""

import argparse
import base64
import logging
import struct
import sys
import zlib
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("extract_ttk")

# ── VTU parsers ──────────────────────────────────────────────────────────────


def _parse_vtu_binary(raw: bytes) -> dict:
    """Parse a VTU file with AppendedData encoding='base64' + ZLib compressor."""
    ad_pos = raw.find(b"<AppendedData")
    if ad_pos == -1:
        raise ValueError("No <AppendedData> tag found — is this a binary VTU?")

    underscore_pos = raw.find(b"_", ad_pos)
    if underscore_pos == -1:
        raise ValueError("No underscore data marker in AppendedData section.")

    b64_end = raw.find(b"</AppendedData>", underscore_pos)
    if b64_end == -1:
        b64_end = len(raw)

    b64_data = raw[underscore_pos + 1 : b64_end].strip()
    xml_text = raw[:underscore_pos].decode("latin-1") + "</AppendedData></VTKFile>"
    tree = ET.fromstring(xml_text)

    def get_offset(name: str) -> int:
        for da in tree.iter("DataArray"):
            if da.get("Name") == name:
                return int(da.get("offset", -1))
        raise KeyError(f"DataArray '{name}' not found in VTU XML.")

    def decode_block(off: int, dtype: np.dtype, ncomp: int = 1) -> np.ndarray:
        hdr = base64.b64decode(b64_data[off : off + 24])
        _nblk, _bsz, _lsz, csz = struct.unpack_from("<IIII", hdr)
        data_b64_len = ((csz + 2) // 3) * 4
        arr_bytes = zlib.decompress(
            base64.b64decode(b64_data[off + 24 : off + 24 + data_b64_len])
        )
        arr = np.frombuffer(arr_bytes, dtype=dtype)
        return arr.reshape(-1, ncomp) if ncomp > 1 else arr

    return {
        "vertex_ids": decode_block(get_offset("ttkVertexScalarField"), np.int32),
        "pair_id": decode_block(get_offset("PairIdentifier"), np.int32),
        "pair_type": decode_block(get_offset("PairType"), np.int32),
        "persistence": decode_block(get_offset("Persistence"), np.float32),
        "birth": decode_block(get_offset("Birth"), np.float32),
        "is_finite": decode_block(get_offset("IsFinite"), np.uint8),
        "connectivity": decode_block(get_offset("connectivity"), np.int64),
    }


def _parse_vtu_ascii(root: ET.Element) -> dict:
    """Parse a VTU file where DataArrays are inline ASCII (format='ascii')."""

    def find_da(name: str) -> ET.Element:
        for da in root.iter("DataArray"):
            if da.get("Name") == name:
                return da
        raise KeyError(f"DataArray '{name}' not found.")

    def read_array(name: str, dtype: np.dtype, ncomp: int = 1) -> np.ndarray:
        da = find_da(name)
        data = np.fromstring(da.text, dtype=dtype, sep=" ")
        return data.reshape(-1, ncomp) if ncomp > 1 else data

    return {
        "vertex_ids": read_array("ttkVertexScalarField", np.int32),
        "pair_id": read_array("PairIdentifier", np.int32),
        "pair_type": read_array("PairType", np.int32),
        "persistence": read_array("Persistence", np.float32),
        "birth": read_array("Birth", np.float32),
        "is_finite": read_array("IsFinite", np.uint8),
        "connectivity": read_array("connectivity", np.int64),
    }


def parse_vtu(path: Path) -> dict:
    """Auto-detect binary/ASCII and parse a TTK persistence-diagram VTU file."""
    raw = path.read_bytes()
    if b'encoding="base64"' in raw[:3000]:
        return _parse_vtu_binary(raw)
    # ASCII or raw-binary inline
    root = ET.fromstring(raw.decode("latin-1"))
    # Check if DataArrays have inline text (ASCII)
    for da in root.iter("DataArray"):
        if da.get("format") in ("ascii", None) and da.text and da.text.strip():
            return _parse_vtu_ascii(root)
    raise ValueError(
        f"Unrecognised VTU format in {path.name}. "
        "Expected base64-appended or ASCII-inline DataArrays."
    )


# ── constraint extraction ────────────────────────────────────────────────────


def extract_constraints(
    arrays: dict,
    persistence_frac: float,
    top_k: int,
    patch: int,
) -> dict | None:
    """
    From parsed VTU arrays, select the top-K most persistent finite pairs
    above the persistence threshold.

    Returns a dict with flat arrays, or None if no pairs qualify.
    """
    is_finite = arrays["is_finite"]
    persistence = arrays["persistence"]
    pair_type = arrays["pair_type"]
    birth = arrays["birth"]
    vertex_ids = arrays["vertex_ids"]
    connectivity = arrays["connectivity"]

    # Keep only finite pairs with positive persistence
    mask = (is_finite == 1) & (persistence > 0)
    if mask.sum() == 0:
        log.warning("No finite pairs found in VTU file.")
        return None

    # Persistence threshold: fraction of maximum persistence in this diagram
    max_pers = float(persistence[mask].max())
    thresh = persistence_frac * max_pers
    mask &= persistence >= thresh

    if mask.sum() == 0:
        log.warning("No pairs survive persistence_frac threshold %.4f (max=%.4f).", persistence_frac, max_pers)
        return None

    candidate_idxs = np.where(mask)[0]

    # Sort by persistence descending, keep top_k
    order = np.argsort(-persistence[candidate_idxs])
    selected = candidate_idxs[order[:top_k]]

    n_pairs = len(selected)
    birth_vid = np.empty(n_pairs, dtype=np.int32)
    death_vid = np.empty(n_pairs, dtype=np.int32)
    birth_val = np.empty(n_pairs, dtype=np.float32)
    death_val = np.empty(n_pairs, dtype=np.float32)
    pers_out = np.empty(n_pairs, dtype=np.float32)
    ptype_out = np.empty(n_pairs, dtype=np.int32)

    for j, i in enumerate(selected):
        pt0 = int(connectivity[2 * i])
        pt1 = int(connectivity[2 * i + 1])
        v0 = int(vertex_ids[pt0])
        v1 = int(vertex_ids[pt1])
        b_val = float(birth[i])
        p_val = float(persistence[i])

        birth_vid[j] = v0
        death_vid[j] = v1
        birth_val[j] = b_val
        death_val[j] = b_val + p_val
        pers_out[j] = p_val
        ptype_out[j] = int(pair_type[i])

    return {
        "birth_vid": birth_vid,
        "death_vid": death_vid,
        "birth_val": birth_val,
        "death_val": death_val,
        "persistence": pers_out,
        "pair_type": ptype_out,
    }


# ── main ──────────────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract TTK PD critical-pair constraints from VTU files."
    )
    p.add_argument(
        "--pd-dir",
        required=True,
        type=Path,
        help="Directory containing GT persistence-diagram VTU files.",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory for the constraints NPZ file.",
    )
    p.add_argument(
        "--patch",
        type=int,
        default=160,
        help="Grid side length (H=W). Used to convert vertex IDs to (y, x) coordinates.",
    )
    p.add_argument(
        "--persistence-frac",
        type=float,
        default=0.01,
        help="Minimum persistence as a fraction of each diagram's max persistence (default 0.01).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=64,
        help="Maximum number of pairs to retain per sample (sorted by persistence, default 64).",
    )
    p.add_argument(
        "--glob",
        type=str,
        default="*.vtu",
        help="Glob pattern for VTU files inside --pd-dir (default '*.vtu').",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    vtu_files = sorted(args.pd_dir.glob(args.glob))

    if len(vtu_files) == 0:
        log.warning(
            "No VTU files found in %s (glob=%s). "
            "Falling back to archive VTU files for smoke-test.",
            args.pd_dir,
            args.glob,
        )
        # Fall back to archive GT files for development / CI smoke test
        archive_dir = Path(__file__).resolve().parent.parent / "archive" / "old_ttk_outputs" / "direct" / "pd_vtu"
        vtu_files = sorted(archive_dir.glob("gan_GT_*.vtu"))
        if len(vtu_files) == 0:
            log.error("No fallback VTU files found in %s either. Aborting.", archive_dir)
            sys.exit(1)
        log.info("Using %d archive VTU files as fallback.", len(vtu_files))
    else:
        log.info("Found %d VTU files in %s.", len(vtu_files), args.pd_dir)

    all_names = []
    all_start = []
    all_count = []
    all_birth_vid = []
    all_death_vid = []
    all_birth_val = []
    all_death_val = []
    all_persistence = []
    all_pair_type = []

    flat_offset = 0
    n_processed = 0
    n_skipped = 0

    for vtu_path in vtu_files:
        log.info("  Parsing %s ...", vtu_path.name)
        try:
            arrays = parse_vtu(vtu_path)
        except Exception as exc:
            log.warning("  SKIP %s: %s", vtu_path.name, exc)
            n_skipped += 1
            continue

        constraints = extract_constraints(
            arrays,
            persistence_frac=args.persistence_frac,
            top_k=args.top_k,
            patch=args.patch,
        )
        if constraints is None:
            log.warning("  SKIP %s: no qualifying pairs.", vtu_path.name)
            n_skipped += 1
            continue

        n_pairs = len(constraints["birth_vid"])
        log.info(
            "  %s: %d pairs (pers_thresh=%.4f*max, top_k=%d)",
            vtu_path.stem,
            n_pairs,
            args.persistence_frac,
            args.top_k,
        )

        all_names.append(vtu_path.stem)
        all_start.append(flat_offset)
        all_count.append(n_pairs)
        all_birth_vid.append(constraints["birth_vid"])
        all_death_vid.append(constraints["death_vid"])
        all_birth_val.append(constraints["birth_val"])
        all_death_val.append(constraints["death_val"])
        all_persistence.append(constraints["persistence"])
        all_pair_type.append(constraints["pair_type"])
        flat_offset += n_pairs
        n_processed += 1

    if n_processed == 0:
        log.error("No VTU files successfully processed. Aborting.")
        sys.exit(1)

    # Assemble output arrays
    out = {
        "n_samples": n_processed,
        "patch_size": args.patch,
        "persistence_frac": args.persistence_frac,
        "top_k": args.top_k,
        "sample_names": np.array(all_names, dtype=object),
        "sample_start": np.array(all_start, dtype=np.int32),
        "sample_count": np.array(all_count, dtype=np.int32),
        "birth_vid": np.concatenate(all_birth_vid).astype(np.int32),
        "death_vid": np.concatenate(all_death_vid).astype(np.int32),
        "birth_val": np.concatenate(all_birth_val).astype(np.float32),
        "death_val": np.concatenate(all_death_val).astype(np.float32),
        "persistence": np.concatenate(all_persistence).astype(np.float32),
        "pair_type": np.concatenate(all_pair_type).astype(np.int32),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "ttk_pd_critical_pairs.npz"
    np.savez(out_path, **out)

    log.info("Wrote %s", out_path)
    log.info(
        "Summary: %d samples processed, %d skipped, %d total pairs (mean=%.1f/sample).",
        n_processed,
        n_skipped,
        flat_offset,
        flat_offset / max(n_processed, 1),
    )

    # Print summary table
    print("\n=== Constraint Summary ===")
    print(f"  Output: {out_path}")
    print(f"  Samples processed : {n_processed}")
    print(f"  Total pairs        : {flat_offset}")
    print(f"  Mean pairs/sample  : {flat_offset / max(n_processed, 1):.1f}")
    print(f"  persistence_frac   : {args.persistence_frac}")
    print(f"  top_k              : {args.top_k}")
    print(f"  patch size         : {args.patch}×{args.patch}")
    if len(all_birth_val) > 0:
        bv = np.concatenate(all_birth_val)
        dv = np.concatenate(all_death_val)
        pv = np.concatenate(all_persistence)
        print(f"  Birth val range    : [{bv.min():.4f}, {bv.max():.4f}]")
        print(f"  Death val range    : [{dv.min():.4f}, {dv.max():.4f}]")
        print(f"  Persistence range  : [{pv.min():.4f}, {pv.max():.4f}]")


if __name__ == "__main__":
    main()
