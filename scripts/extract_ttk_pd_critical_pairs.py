#!/usr/bin/env python3
r"""
Extract TTK persistence-diagram critical-pair constraints from VTU files.

Reads GT persistence-diagram VTU files (binary base64+zlib or ASCII inline)
produced by TTK's ttkPersistenceDiagram filter and writes a constraints NPZ
and CSV for use by run_candidateE_ttkcrit_refiner.py.

Sample index is parsed from filenames matching _s(\d+)_ (e.g. candidateD_GT_s113_...).
Results are sorted by numeric sample index and validated against expected counts.

VTU binary format notes
-----------------------
For AppendedData encoding="base64" with vtkZLibDataCompressor, each DataArray
is stored as two adjacent, independently base64-encoded segments:

  [header segment][compressed-data segment]

Header (UInt32 header_type):
  4 bytes : nblk  — number of compressed blocks
  4 bytes : bsz   — uncompressed size per block (except last)
  4 bytes : lsz   — uncompressed size of last block
  4*nblk bytes : csz[0], ..., csz[nblk-1]  — compressed block sizes
  Total: (3 + nblk) * 4 bytes -> ceil((3+nblk)*4/3)*4 base64 chars

Header (UInt64 header_type): same but with 8-byte integers.

Compressed-data: sum(csz) bytes, all blocks concatenated.

The XML DataArray offset attribute is the CHARACTER position in the
AppendedData base64 string where the header segment begins.

Common failure mode: treating the 24-char/16-byte UInt32 header size as
fixed. When nblk > 1, the actual header is larger and the data segment
starts at a different offset, causing zlib to receive garbage bytes.

Usage
-----
python3 scripts/extract_ttk_pd_critical_pairs.py \
    --pd-dir ttk_runs_fixed/topology_finetuning/candidateD_topology/pd/GT \
    --out-dir ttk_runs_fixed/topology_finetuning/candidateE_constraints \
    --patch 160 \
    --persistence-frac 0.01 \
    --top-k 64 \
    --expected-samples 168
"""

import argparse
import base64
import csv
import logging
import re
import struct
import sys
import zlib
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("extract_ttk")

_SAMPLE_IDX_RE = re.compile(r"_s(\d+)_")

# numpy dtype lookup
_VTK_DTYPE = {
    "Int8": np.int8, "UInt8": np.uint8,
    "Int16": np.int16, "UInt16": np.uint16,
    "Int32": np.int32, "UInt32": np.uint32,
    "Int64": np.int64, "UInt64": np.uint64,
    "Float32": np.float32, "Float64": np.float64,
}


# ── VTK Python reader (preferred) ─────────────────────────────────────────────

def _try_vtk_reader(path: Path) -> dict | None:
    """
    Try to read the VTU file with the official VTK Python reader.
    Returns the parsed arrays dict, or None if vtk is unavailable.
    """
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError:
        return None

    try:
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(str(path))
        reader.Update()
        grid = reader.GetOutput()
    except Exception as exc:
        log.warning("[vtk] Failed to read %s: %s", path.name, exc)
        return None

    pd_data = grid.GetPointData()
    cd_data = grid.GetCellData()

    def _get_point(name):
        arr = pd_data.GetArray(name)
        if arr is None:
            raise KeyError(f"Point array '{name}' not found in {path.name}")
        return vtk_to_numpy(arr)

    def _get_cell(name):
        arr = cd_data.GetArray(name)
        if arr is None:
            raise KeyError(f"Cell array '{name}' not found in {path.name}")
        return vtk_to_numpy(arr)

    try:
        connectivity = vtk_to_numpy(grid.GetCells().GetConnectivityArray())
        return {
            "vertex_ids":  _get_point("ttkVertexScalarField").astype(np.int32),
            "pair_id":     _get_cell("PairIdentifier").astype(np.int32),
            "pair_type":   _get_cell("PairType").astype(np.int32),
            "persistence": _get_cell("Persistence").astype(np.float32),
            "birth":       _get_cell("Birth").astype(np.float32),
            "is_finite":   _get_cell("IsFinite").astype(np.uint8),
            "connectivity": connectivity.astype(np.int64),
        }
    except (KeyError, AttributeError) as exc:
        log.warning("[vtk] Array extraction failed for %s: %s", path.name, exc)
        return None


# ── Manual binary VTU parser ───────────────────────────────────────────────────

def _header_int_size(vtk_elem: ET.Element) -> int:
    """Return header integer width in bytes from the VTKFile XML element."""
    ht = vtk_elem.get("header_type", "UInt32")
    return 8 if ht == "UInt64" else 4


def _decode_b64_block(
    b64_data: bytes,
    off: int,
    dtype: np.dtype,
    ncomp: int = 1,
    hint: int = 4,
    name: str = "",
) -> np.ndarray:
    """
    Decode a VTK ZLib-compressed DataArray block at character offset `off`.

    The block consists of two adjacent, independently base64-encoded segments:
      [header segment][compressed-data segment]

    hint  : header integer size in bytes (4 for UInt32, 8 for UInt64).
    name  : DataArray name for error messages.

    This function correctly handles nblk > 1 (multiple compressed blocks) and
    both UInt32 and UInt64 header types.
    """
    fmt = "<I" if hint == 4 else "<Q"

    # ── Step 1: read enough to determine nblk ─────────────────────────────
    # Minimum header for nblk=1 has (3+1)*hint bytes.
    min_hdr_chars = (((3 + 1) * hint + 2) // 3) * 4  # 24 for UInt32, 44 for UInt64

    try:
        partial = base64.b64decode(b64_data[off : off + min_hdr_chars])
    except Exception as exc:
        raise ValueError(
            f"{name}: base64 decode of header failed at offset {off}: {exc}"
        ) from exc

    if len(partial) < hint:
        raise ValueError(
            f"{name}: decoded header too short ({len(partial)} bytes, need {hint})."
        )
    nblk = struct.unpack_from(fmt, partial, 0)[0]

    if nblk == 0 or nblk > 65536:
        raise ValueError(
            f"{name}: suspicious nblk={nblk} at offset {off}. "
            f"header_int_size={hint}. First bytes: {partial[:8].hex()}"
        )

    # ── Step 2: read full header ──────────────────────────────────────────
    full_hdr_bytes = (3 + nblk) * hint
    full_hdr_chars = ((full_hdr_bytes + 2) // 3) * 4

    if full_hdr_chars > min_hdr_chars:
        try:
            hdr = base64.b64decode(b64_data[off : off + full_hdr_chars])
        except Exception as exc:
            raise ValueError(
                f"{name}: base64 decode of extended header failed: {exc}"
            ) from exc
    else:
        hdr = partial

    csz_list = [
        struct.unpack_from(fmt, hdr, (3 + i) * hint)[0] for i in range(nblk)
    ]

    # ── Step 3: read compressed data segment ─────────────────────────────
    data_off = off + full_hdr_chars
    total_csz = sum(csz_list)
    if total_csz == 0:
        raise ValueError(f"{name}: total compressed size is zero at offset {off}.")
    data_chars = ((total_csz + 2) // 3) * 4

    try:
        compressed_all = base64.b64decode(b64_data[data_off : data_off + data_chars])
    except Exception as exc:
        raise ValueError(
            f"{name}: base64 decode of data segment failed at offset {data_off}: {exc}"
        ) from exc

    # ── Step 4: decompress each block ────────────────────────────────────
    out = bytearray()
    p = 0
    for bi, csz in enumerate(csz_list):
        chunk = compressed_all[p : p + csz]
        if len(chunk) < csz:
            raise ValueError(
                f"{name}: block {bi}/{nblk}: expected {csz} compressed bytes, "
                f"got {len(chunk)}. data_off={data_off}, total_csz={total_csz}"
            )
        try:
            out += zlib.decompress(chunk)
        except zlib.error as exc:
            raise ValueError(
                f"{name}: zlib.decompress failed on block {bi}/{nblk} "
                f"(nblk={nblk}, hint={hint}, off={off}, csz={csz}, "
                f"first4={chunk[:4].hex()}): {exc}"
            ) from exc
        p += csz

    arr = np.frombuffer(bytes(out), dtype=dtype)
    return arr.reshape(-1, ncomp) if ncomp > 1 else arr


def _parse_vtu_binary(raw: bytes, path: Path) -> dict:
    """Parse a VTU file with AppendedData encoding='base64' + ZLib compressor."""
    ad_pos = raw.find(b"<AppendedData")
    if ad_pos == -1:
        raise ValueError("No <AppendedData> tag found.")

    underscore_pos = raw.find(b"_", ad_pos)
    if underscore_pos == -1:
        raise ValueError("No underscore data marker in AppendedData section.")

    b64_end = raw.find(b"</AppendedData>", underscore_pos)
    if b64_end == -1:
        b64_end = len(raw)

    # Keep whitespace stripped only at the boundaries; preserve interior layout.
    b64_data = raw[underscore_pos + 1 : b64_end].rstrip()

    # Close open AppendedData for ElementTree
    xml_text = raw[:underscore_pos].decode("latin-1") + "</AppendedData></VTKFile>"
    try:
        tree = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError(f"XML parse error in {path.name}: {exc}") from exc

    vtk_elem = tree  # VTKFile is root
    hint = _header_int_size(vtk_elem)

    def _get_offset(name: str) -> int:
        for da in tree.iter("DataArray"):
            if da.get("Name") == name:
                return int(da.get("offset", -1))
        raise KeyError(f"DataArray '{name}' not found in {path.name}.")

    def _get_dtype(name: str) -> np.dtype:
        for da in tree.iter("DataArray"):
            if da.get("Name") == name:
                return _VTK_DTYPE.get(da.get("type", "Float32"), np.float32)
        return np.float32

    def _get_ncomp(name: str) -> int:
        for da in tree.iter("DataArray"):
            if da.get("Name") == name:
                return int(da.get("NumberOfComponents", 1))
        return 1

    def _decode(name: str) -> np.ndarray:
        off = _get_offset(name)
        dtype = _get_dtype(name)
        ncomp = _get_ncomp(name)
        return _decode_b64_block(b64_data, off, dtype, ncomp, hint=hint, name=name)

    return {
        "vertex_ids":  _decode("ttkVertexScalarField").astype(np.int32),
        "pair_id":     _decode("PairIdentifier").astype(np.int32),
        "pair_type":   _decode("PairType").astype(np.int32),
        "persistence": _decode("Persistence").astype(np.float32),
        "birth":       _decode("Birth").astype(np.float32),
        "is_finite":   _decode("IsFinite").astype(np.uint8),
        "connectivity": _decode("connectivity").astype(np.int64),
    }


def _parse_vtu_ascii(root: ET.Element, path: Path) -> dict:
    """Parse a VTU file where DataArrays are inline ASCII (format='ascii')."""

    def _find_da(name: str) -> ET.Element:
        for da in root.iter("DataArray"):
            if da.get("Name") == name:
                return da
        raise KeyError(f"DataArray '{name}' not found in {path.name}.")

    def _read_array(name: str, dtype: np.dtype, ncomp: int = 1) -> np.ndarray:
        da = _find_da(name)
        data = np.fromstring(da.text, dtype=dtype, sep=" ")
        return data.reshape(-1, ncomp) if ncomp > 1 else data

    return {
        "vertex_ids":  _read_array("ttkVertexScalarField", np.int32),
        "pair_id":     _read_array("PairIdentifier", np.int32),
        "pair_type":   _read_array("PairType", np.int32),
        "persistence": _read_array("Persistence", np.float32),
        "birth":       _read_array("Birth", np.float32),
        "is_finite":   _read_array("IsFinite", np.uint8),
        "connectivity": _read_array("connectivity", np.int64),
    }


def parse_vtu(path: Path) -> dict:
    """Auto-detect format and parse a TTK persistence-diagram VTU file."""
    # Prefer official VTK reader if available
    result = _try_vtk_reader(path)
    if result is not None:
        return result

    # Manual fallback
    raw = path.read_bytes()
    if b'encoding="base64"' in raw[:3000]:
        return _parse_vtu_binary(raw, path)

    root = ET.fromstring(raw.decode("latin-1"))
    for da in root.iter("DataArray"):
        if da.get("format") in ("ascii", None) and da.text and da.text.strip():
            return _parse_vtu_ascii(root, path)

    raise ValueError(
        f"Unrecognised VTU format in {path.name}. "
        "Expected base64-appended or ASCII-inline DataArrays."
    )


def _parse_sample_idx(stem: str) -> int:
    """Extract numeric sample index from filename stem. Returns -1 if not found."""
    m = _SAMPLE_IDX_RE.search(stem)
    return int(m.group(1)) if m else -1


# ── constraint extraction ────────────────────────────────────────────────────


def extract_constraints(
    arrays: dict,
    sample_idx: int,
    sample_name: str,
    persistence_frac: float,
    top_k: int,
    patch: int,
) -> dict | None:
    """
    Select the top-K most persistent finite pairs above the threshold.

    pair_type == -1 (infinite) is excluded. Only finite pairs (is_finite==1)
    with positive persistence are considered.

    persistence = |death_scalar - birth_scalar| (unsigned, as stored by TTK).
    death_val  = birth_val + persistence_raw (for sublevel pairs).

    Returns a dict with flat arrays per pair, or None if no pairs qualify.
    """
    is_finite   = arrays["is_finite"]
    persistence = arrays["persistence"]
    pair_type   = arrays["pair_type"]
    birth       = arrays["birth"]
    vertex_ids  = arrays["vertex_ids"]
    connectivity = arrays["connectivity"]

    # Finite pairs with positive persistence, excluding infinite pairs (type -1)
    mask = (is_finite == 1) & (persistence > 0) & (pair_type != -1)
    if mask.sum() == 0:
        log.warning("  %s: no finite pairs found.", sample_name)
        return None

    max_pers = float(persistence[mask].max())
    thresh   = persistence_frac * max_pers
    mask    &= persistence >= thresh

    if mask.sum() == 0:
        log.warning(
            "  %s: no pairs survive persistence threshold %.4f (max=%.4f).",
            sample_name, thresh, max_pers,
        )
        return None

    candidate_idxs = np.where(mask)[0]
    order    = np.argsort(-persistence[candidate_idxs])
    selected = candidate_idxs[order[:top_k]]

    n_pairs  = len(selected)
    birth_vid  = np.empty(n_pairs, dtype=np.int32)
    death_vid  = np.empty(n_pairs, dtype=np.int32)
    birth_val  = np.empty(n_pairs, dtype=np.float32)
    death_val  = np.empty(n_pairs, dtype=np.float32)
    pers_out   = np.empty(n_pairs, dtype=np.float32)
    ptype_out  = np.empty(n_pairs, dtype=np.int32)
    paid_out   = np.empty(n_pairs, dtype=np.int32)

    for j, i in enumerate(selected):
        pt0 = int(connectivity[2 * i])
        pt1 = int(connectivity[2 * i + 1])
        v0  = int(vertex_ids[pt0])
        v1  = int(vertex_ids[pt1])
        b_val = float(birth[i])
        p_val = float(persistence[i])

        birth_vid[j]  = v0
        death_vid[j]  = v1
        birth_val[j]  = b_val
        death_val[j]  = b_val + p_val
        pers_out[j]   = p_val
        ptype_out[j]  = int(pair_type[i])
        paid_out[j]   = int(arrays["pair_id"][i])

    return {
        "sample_idx":  sample_idx,
        "sample_name": sample_name,
        "birth_vid":   birth_vid,
        "death_vid":   death_vid,
        "birth_val":   birth_val,
        "death_val":   death_val,
        "persistence": pers_out,
        "pair_type":   ptype_out,
        "pair_id":     paid_out,
    }


# ── validation ───────────────────────────────────────────────────────────────


def _validate_constraints(npz_path: Path, patch: int) -> None:
    """Abort if the written NPZ contains invalid values."""
    npz = np.load(npz_path, allow_pickle=True)
    ok = True
    max_vid = patch * patch

    for name in ("birth_vid", "death_vid"):
        arr = npz[name]
        if not np.isfinite(arr.astype(float)).all():
            log.error("VALIDATION FAIL: %s contains non-finite values.", name)
            ok = False
        if (arr < 0).any() or (arr >= max_vid).any():
            log.error(
                "VALIDATION FAIL: %s out of range [0, %d): min=%d max=%d",
                name, max_vid, int(arr.min()), int(arr.max()),
            )
            ok = False

    for name in ("birth_val", "death_val", "persistence"):
        arr = npz[name]
        if not np.isfinite(arr).all():
            log.error("VALIDATION FAIL: %s contains non-finite values.", name)
            ok = False

    pers = npz["persistence"]
    if (pers < 0).any():
        log.error(
            "VALIDATION FAIL: persistence contains negative values (min=%.4f).",
            float(pers.min()),
        )
        ok = False

    pt = npz["pair_type"]
    if (pt == -1).any():
        log.error("VALIDATION FAIL: pair_type contains -1 (infinite pairs not excluded).")
        ok = False

    bvid = npz["birth_vid"]
    dvid = npz["death_vid"]
    for coord_name, coord in [
        ("birth_y", bvid // patch), ("birth_x", bvid % patch),
        ("death_y", dvid // patch), ("death_x", dvid % patch),
    ]:
        if (coord < 0).any() or (coord >= patch).any():
            log.error(
                "VALIDATION FAIL: %s out of range [0, %d): min=%d max=%d",
                coord_name, patch, int(coord.min()), int(coord.max()),
            )
            ok = False

    if not ok:
        log.error("Validation failed. Removing output file and aborting.")
        npz_path.unlink(missing_ok=True)
        sys.exit(1)
    log.info("Validation passed.")


# ── main ──────────────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract TTK PD critical-pair constraints from VTU files."
    )
    p.add_argument(
        "--pd-dir", required=True, type=Path,
        help="Directory containing GT persistence-diagram VTU files.",
    )
    p.add_argument(
        "--out-dir", required=True, type=Path,
        help="Output directory for the constraints NPZ and CSV files.",
    )
    p.add_argument("--patch",            type=int,   default=160)
    p.add_argument("--persistence-frac", type=float, default=0.01)
    p.add_argument("--top-k",            type=int,   default=64)
    p.add_argument("--glob",             type=str,   default="*.vtu")
    p.add_argument(
        "--expected-samples", type=int, default=168,
        help="Expected number of processed samples (default 168). "
             "Aborts if different unless --allow-partial is set.",
    )
    p.add_argument("--allow-partial",   action="store_true",
                   help="Allow fewer than --expected-samples without aborting.")
    p.add_argument(
        "--allow-archive-fallback", action="store_true",
        help="[CI only] Fall back to archive VTU files if --pd-dir is empty. "
             "MUST NOT be used for real Candidate E training.",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    # ── Locate VTU files ────────────────────────────────────────────────────
    vtu_files = sorted(args.pd_dir.glob(args.glob))

    if len(vtu_files) == 0:
        if args.allow_archive_fallback:
            log.warning(
                "[CI] No VTU files in %s — falling back to archive "
                "(--allow-archive-fallback).", args.pd_dir,
            )
            archive_dir = (
                Path(__file__).resolve().parent.parent
                / "archive" / "old_ttk_outputs" / "direct" / "pd_vtu"
            )
            vtu_files = sorted(archive_dir.glob("gan_GT_*.vtu"))
            if len(vtu_files) == 0:
                log.error("No archive VTU files found in %s. Aborting.", archive_dir)
                sys.exit(1)
            log.info("[CI] Using %d archive files.", len(vtu_files))
        else:
            log.error(
                "No VTU files found in %s (glob=%s).\n"
                "Run the TTK pipeline to generate GT persistence diagrams first:\n"
                "  bash scripts/run_candidate_topology_pipeline.sh "
                "--method candidateD --stage pd\n"
                "Or use --allow-archive-fallback for CI smoke tests only.",
                args.pd_dir, args.glob,
            )
            sys.exit(1)
    else:
        log.info("Found %d VTU files in %s.", len(vtu_files), args.pd_dir)

    # ── Process each file ───────────────────────────────────────────────────
    per_sample: list[dict] = []
    n_skipped = 0

    for vtu_path in vtu_files:
        stem = vtu_path.stem
        sidx = _parse_sample_idx(stem)
        if sidx < 0:
            log.warning(
                "  SKIP %s: cannot parse sample index "
                "(expected _s<N>_ pattern).", vtu_path.name,
            )
            n_skipped += 1
            continue

        log.info("  Parsing %s (sample_idx=%d) …", vtu_path.name, sidx)
        try:
            arrays = parse_vtu(vtu_path)
        except Exception as exc:
            log.warning("  SKIP %s: %s", vtu_path.name, exc)
            n_skipped += 1
            continue

        constraints = extract_constraints(
            arrays,
            sample_idx=sidx,
            sample_name=stem,
            persistence_frac=args.persistence_frac,
            top_k=args.top_k,
            patch=args.patch,
        )
        if constraints is None:
            n_skipped += 1
            continue

        log.info(
            "  %s (s%d): %d pairs (pers_frac=%.3f, top_k=%d)",
            stem, sidx, len(constraints["birth_vid"]),
            args.persistence_frac, args.top_k,
        )
        per_sample.append(constraints)

    n_processed = len(per_sample)
    log.info(
        "Processed %d / %d files (%d skipped).",
        n_processed, len(vtu_files), n_skipped,
    )

    # ── Sample count guard ──────────────────────────────────────────────────
    if n_processed == 0:
        log.error("No VTU files successfully processed. Aborting.")
        sys.exit(1)

    if n_processed != args.expected_samples:
        msg = (
            f"Expected {args.expected_samples} samples but got {n_processed}. "
            "Run the full TTK pipeline to produce all GT VTU files, or use "
            "--allow-partial to override."
        )
        if args.allow_partial:
            log.warning(msg)
        else:
            log.error(msg)
            sys.exit(1)

    # ── Sort by numeric sample_idx ──────────────────────────────────────────
    per_sample.sort(key=lambda d: d["sample_idx"])

    # ── Assemble flat arrays ────────────────────────────────────────────────
    flat_offset = 0
    all_sidx  = [];  all_names = []; all_start = []; all_count = []
    all_bvid  = [];  all_dvid  = []; all_bval  = []; all_dval  = []
    all_pers  = [];  all_ptype = []; all_paid  = []

    for d in per_sample:
        n_p = len(d["birth_vid"])
        all_sidx.append(d["sample_idx"])
        all_names.append(d["sample_name"])
        all_start.append(flat_offset)
        all_count.append(n_p)
        all_bvid.append(d["birth_vid"]);  all_dvid.append(d["death_vid"])
        all_bval.append(d["birth_val"]);  all_dval.append(d["death_val"])
        all_pers.append(d["persistence"]); all_ptype.append(d["pair_type"])
        all_paid.append(d["pair_id"])
        flat_offset += n_p

    # ── Write NPZ ───────────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.out_dir / "ttk_pd_critical_pairs.npz"
    np.savez(
        npz_path,
        n_samples    = n_processed,
        patch_size   = args.patch,
        persistence_frac = args.persistence_frac,
        top_k        = args.top_k,
        sample_idx   = np.array(all_sidx,  dtype=np.int32),
        sample_names = np.array(all_names, dtype=object),
        sample_start = np.array(all_start, dtype=np.int32),
        sample_count = np.array(all_count, dtype=np.int32),
        birth_vid    = np.concatenate(all_bvid).astype(np.int32),
        death_vid    = np.concatenate(all_dvid).astype(np.int32),
        birth_val    = np.concatenate(all_bval).astype(np.float32),
        death_val    = np.concatenate(all_dval).astype(np.float32),
        persistence  = np.concatenate(all_pers).astype(np.float32),
        pair_type    = np.concatenate(all_ptype).astype(np.int32),
    )
    log.info("Wrote %s", npz_path)

    # ── Validate ────────────────────────────────────────────────────────────
    _validate_constraints(npz_path, args.patch)

    # ── Write CSV ───────────────────────────────────────────────────────────
    csv_path = args.out_dir / "ttk_pd_critical_pairs.csv"
    W = args.patch
    fieldnames = [
        "sample_idx", "sample_name", "pair_id", "pair_type",
        "birth_vid", "death_vid", "birth_y", "birth_x", "death_y", "death_x",
        "birth_val", "death_val", "persistence",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in per_sample:
            sidx = d["sample_idx"]; sname = d["sample_name"]
            for j in range(len(d["birth_vid"])):
                bv = int(d["birth_vid"][j]); dv = int(d["death_vid"][j])
                writer.writerow({
                    "sample_idx":  sidx,
                    "sample_name": sname,
                    "pair_id":     int(d["pair_id"][j]),
                    "pair_type":   int(d["pair_type"][j]),
                    "birth_vid":   bv,
                    "death_vid":   dv,
                    "birth_y":     bv // W, "birth_x": bv % W,
                    "death_y":     dv // W, "death_x": dv % W,
                    "birth_val":   float(d["birth_val"][j]),
                    "death_val":   float(d["death_val"][j]),
                    "persistence": float(d["persistence"][j]),
                })
    log.info("Wrote %s (%d rows)", csv_path, flat_offset)

    # ── Summary ─────────────────────────────────────────────────────────────
    counts_arr = np.array(all_count)
    bv_all = np.concatenate(all_bval)
    dv_all = np.concatenate(all_dval)
    pv_all = np.concatenate(all_pers)
    sidx_arr = np.array(all_sidx)

    print("\n=== Constraint Summary ===")
    print(f"  Output NPZ        : {npz_path}")
    print(f"  Output CSV        : {csv_path}")
    print(f"  Samples processed : {n_processed}")
    print(f"  Total pairs        : {flat_offset}")
    print(f"  Pairs/sample      : min={counts_arr.min()}  "
          f"mean={counts_arr.mean():.1f}  max={counts_arr.max()}")
    print(f"  persistence_frac   : {args.persistence_frac}")
    print(f"  top_k              : {args.top_k}")
    print(f"  patch size         : {args.patch}×{args.patch}")
    print(f"  Birth val range    : [{bv_all.min():.4f}, {bv_all.max():.4f}]")
    print(f"  Death val range    : [{dv_all.min():.4f}, {dv_all.max():.4f}]")
    print(f"  Persistence range  : [{pv_all.min():.4f}, {pv_all.max():.4f}]")

    # Check for missing sample indices in expected range
    all_present = set(sidx_arr.tolist())
    expected_range = set(range(sidx_arr.min(), sidx_arr.max() + 1))
    missing = sorted(expected_range - all_present)
    if missing:
        print(f"\n  WARNING missing sample indices: {missing[:20]}"
              f"{'...' if len(missing) > 20 else ''}")
    else:
        print(f"  Missing indices    : none (range {sidx_arr.min()}–{sidx_arr.max()} complete)")

    # Print examples for specific sample indices if present
    for target in [6, 18, 25, 80, 162]:
        matches = np.where(sidx_arr == target)[0]
        if len(matches) == 0:
            continue
        pos = matches[0]
        d   = per_sample[pos]
        start = int(all_start[pos])
        count = int(all_count[pos])
        bvids = np.concatenate(all_bvid)[start : start + count]
        pvals = np.concatenate(all_pers)[start : start + count]
        print(
            f"\n  Sample s{target}: {count} pairs, "
            f"pers=[{pvals.min():.3f}, {pvals.max():.3f}], "
            f"birth_vid[0]={bvids[0]} (y={bvids[0]//W}, x={bvids[0]%W})"
        )


if __name__ == "__main__":
    main()
