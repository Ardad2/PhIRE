#!/usr/bin/env python3
"""
Extract TTK persistence-diagram critical-pair constraints from VTU files.

Reads GT persistence-diagram VTU files (binary base64+zlib or ASCII inline)
produced by TTK's ttkPersistenceDiagram filter and writes a constraints NPZ
and CSV for use by run_candidateE_ttkcrit_refiner.py.

Sample index is parsed from filenames matching _s(\d+)_ (e.g. candidateD_GT_s113_...).
Results are sorted by numeric sample index and validated against expected counts.

Usage
-----
python3 scripts/extract_ttk_pd_critical_pairs.py \\
    --pd-dir ttk_runs_fixed/topology_finetuning/candidateD_topology/pd/GT \\
    --out-dir ttk_runs_fixed/topology_finetuning/candidateE_constraints \\
    --patch 160 \\
    --persistence-frac 0.01 \\
    --top-k 64 \\
    --expected-samples 168

Output files
------------
ttk_pd_critical_pairs.npz   — binary constraints (loaded by refiner)
ttk_pd_critical_pairs.csv   — human-readable, one row per pair

NPZ keys
--------
n_samples           : int — number of VTU files processed
patch_size          : int — grid side (H = W = patch_size)
persistence_frac    : float
top_k               : int
sample_idx          : (N,) int32 — numeric sample index from filename
sample_names        : (N,) object — VTU stems sorted by sample_idx
sample_start        : (N,) int32 — start index into flat pair arrays
sample_count        : (N,) int32 — number of pairs per sample
birth_vid           : (P,) int32 — birth vertex ID (flat index in H×W grid)
death_vid           : (P,) int32 — death vertex ID
birth_val           : (P,) float32 — scalar value at birth vertex
death_val           : (P,) float32 — scalar value at death vertex (birth+pers)
persistence         : (P,) float32 — persistence value (= |death_val - birth_val|)
pair_type           : (P,) int32 — TTK pair type (0=min-saddle, 1=saddle-max)
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
    # Close the open <AppendedData> element for ElementTree
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
    root = ET.fromstring(raw.decode("latin-1"))
    for da in root.iter("DataArray"):
        if da.get("format") in ("ascii", None) and da.text and da.text.strip():
            return _parse_vtu_ascii(root)
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

    persistence = birth_val + persistence_raw = death scalar value.
    birth_val and death_val are stored so that:
        death_val = birth_val + persistence_raw  (for sublevel pairs)
    The sign convention is: persistence_raw = abs(death_val - birth_val).

    Returns a dict with flat arrays per pair, or None if no pairs qualify.
    """
    is_finite = arrays["is_finite"]
    persistence = arrays["persistence"]
    pair_type = arrays["pair_type"]
    birth = arrays["birth"]
    vertex_ids = arrays["vertex_ids"]
    connectivity = arrays["connectivity"]

    # Finite pairs with positive persistence (exclude infinite pairs, type -1)
    mask = (is_finite == 1) & (persistence > 0) & (pair_type != -1)
    if mask.sum() == 0:
        log.warning("  %s: no finite pairs found.", sample_name)
        return None

    max_pers = float(persistence[mask].max())
    thresh = persistence_frac * max_pers
    mask &= persistence >= thresh

    if mask.sum() == 0:
        log.warning(
            "  %s: no pairs survive persistence threshold %.4f (max=%.4f).",
            sample_name, thresh, max_pers,
        )
        return None

    candidate_idxs = np.where(mask)[0]
    order = np.argsort(-persistence[candidate_idxs])
    selected = candidate_idxs[order[:top_k]]

    n_pairs = len(selected)
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
        v0 = int(vertex_ids[pt0])
        v1 = int(vertex_ids[pt1])
        b_val = float(birth[i])
        p_val = float(persistence[i])

        # birth_val is the scalar at the birth vertex.
        # death_val = birth_val + persistence (valid for sublevel and superlevel
        # when TTK stores persistence = |death - birth|).
        birth_vid[j] = v0
        death_vid[j] = v1
        birth_val[j] = b_val
        death_val[j] = b_val + p_val
        pers_out[j]  = p_val
        ptype_out[j] = int(pair_type[i])
        paid_out[j]  = int(arrays["pair_id"][i])

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

    for name, arr in [("birth_vid", npz["birth_vid"]), ("death_vid", npz["death_vid"])]:
        if not np.isfinite(arr.astype(float)).all():
            log.error("VALIDATION FAIL: %s contains non-finite values.", name)
            ok = False
        if (arr < 0).any() or (arr >= max_vid).any():
            log.error(
                "VALIDATION FAIL: %s out of range [0, %d): min=%d max=%d",
                name, max_vid, arr.min(), arr.max(),
            )
            ok = False

    for name, arr in [("birth_val", npz["birth_val"]), ("death_val", npz["death_val"]),
                      ("persistence", npz["persistence"])]:
        if not np.isfinite(arr).all():
            log.error("VALIDATION FAIL: %s contains non-finite values.", name)
            ok = False

    pers = npz["persistence"]
    if (pers < 0).any():
        log.error("VALIDATION FAIL: persistence contains negative values (min=%.4f).", pers.min())
        ok = False

    pt = npz["pair_type"]
    if (pt == -1).any():
        log.error("VALIDATION FAIL: pair_type contains -1 (infinite pairs not excluded).")
        ok = False

    bvid = npz["birth_vid"]
    dvid = npz["death_vid"]
    by, bx = bvid // patch, bvid % patch
    dy, dx = dvid // patch, dvid % patch
    for coord_name, coord in [("birth_y", by), ("birth_x", bx), ("death_y", dy), ("death_x", dx)]:
        if (coord < 0).any() or (coord >= patch).any():
            log.error(
                "VALIDATION FAIL: %s out of range [0, %d): min=%d max=%d",
                coord_name, patch, coord.min(), coord.max(),
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
        "--pd-dir",
        required=True,
        type=Path,
        help="Directory containing GT persistence-diagram VTU files.",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory for the constraints NPZ and CSV files.",
    )
    p.add_argument(
        "--patch",
        type=int,
        default=160,
        help="Grid side length (H=W). Used to convert vertex IDs to (y, x).",
    )
    p.add_argument(
        "--persistence-frac",
        type=float,
        default=0.01,
        help="Minimum persistence as fraction of each diagram's max persistence.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=64,
        help="Maximum pairs to retain per sample (sorted by persistence desc).",
    )
    p.add_argument(
        "--glob",
        type=str,
        default="*.vtu",
        help="Glob pattern for VTU files inside --pd-dir.",
    )
    p.add_argument(
        "--expected-samples",
        type=int,
        default=168,
        help="Expected number of successfully processed samples (default 168). "
             "Script aborts if actual count differs unless --allow-partial is set.",
    )
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow fewer than --expected-samples without aborting.",
    )
    p.add_argument(
        "--allow-archive-fallback",
        action="store_true",
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
                "[CI] No VTU files in %s — falling back to archive (--allow-archive-fallback).",
                args.pd_dir,
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
                "  SKIP %s: cannot parse sample index from filename "
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

        n_pairs = len(constraints["birth_vid"])
        log.info(
            "  %s (s%d): %d pairs (pers_frac=%.3f, top_k=%d)",
            stem, sidx, n_pairs, args.persistence_frac, args.top_k,
        )
        per_sample.append(constraints)

    n_processed = len(per_sample)
    log.info(
        "Processed %d / %d files (%d skipped).",
        n_processed, len(vtu_files), n_skipped,
    )

    # ── Sample count check ──────────────────────────────────────────────────
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
    all_sidx    = []
    all_names   = []
    all_start   = []
    all_count   = []
    all_bvid    = []
    all_dvid    = []
    all_bval    = []
    all_dval    = []
    all_pers    = []
    all_ptype   = []
    all_paid    = []

    for d in per_sample:
        n_p = len(d["birth_vid"])
        all_sidx.append(d["sample_idx"])
        all_names.append(d["sample_name"])
        all_start.append(flat_offset)
        all_count.append(n_p)
        all_bvid.append(d["birth_vid"])
        all_dvid.append(d["death_vid"])
        all_bval.append(d["birth_val"])
        all_dval.append(d["death_val"])
        all_pers.append(d["persistence"])
        all_ptype.append(d["pair_type"])
        all_paid.append(d["pair_id"])
        flat_offset += n_p

    # ── Write NPZ ───────────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.out_dir / "ttk_pd_critical_pairs.npz"
    np.savez(
        npz_path,
        n_samples=n_processed,
        patch_size=args.patch,
        persistence_frac=args.persistence_frac,
        top_k=args.top_k,
        sample_idx=np.array(all_sidx,  dtype=np.int32),
        sample_names=np.array(all_names, dtype=object),
        sample_start=np.array(all_start, dtype=np.int32),
        sample_count=np.array(all_count, dtype=np.int32),
        birth_vid=np.concatenate(all_bvid).astype(np.int32),
        death_vid=np.concatenate(all_dvid).astype(np.int32),
        birth_val=np.concatenate(all_bval).astype(np.float32),
        death_val=np.concatenate(all_dval).astype(np.float32),
        persistence=np.concatenate(all_pers).astype(np.float32),
        pair_type=np.concatenate(all_ptype).astype(np.int32),
    )
    log.info("Wrote %s", npz_path)

    # ── Validate ────────────────────────────────────────────────────────────
    _validate_constraints(npz_path, args.patch)

    # ── Write CSV ───────────────────────────────────────────────────────────
    csv_path = args.out_dir / "ttk_pd_critical_pairs.csv"
    W = args.patch
    csv_rows = []
    for d in per_sample:
        sidx = d["sample_idx"]
        sname = d["sample_name"]
        for j in range(len(d["birth_vid"])):
            bv = int(d["birth_vid"][j])
            dv = int(d["death_vid"][j])
            csv_rows.append({
                "sample_idx":  sidx,
                "sample_name": sname,
                "pair_id":     int(d["pair_id"][j]),
                "pair_type":   int(d["pair_type"][j]),
                "birth_vid":   bv,
                "death_vid":   dv,
                "birth_y":     bv // W,
                "birth_x":     bv % W,
                "death_y":     dv // W,
                "death_x":     dv % W,
                "birth_val":   float(d["birth_val"][j]),
                "death_val":   float(d["death_val"][j]),
                "persistence": float(d["persistence"][j]),
            })

    fieldnames = [
        "sample_idx", "sample_name", "pair_id", "pair_type",
        "birth_vid", "death_vid", "birth_y", "birth_x", "death_y", "death_x",
        "birth_val", "death_val", "persistence",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    log.info("Wrote %s (%d rows)", csv_path, len(csv_rows))

    # ── Summary ─────────────────────────────────────────────────────────────
    counts_arr = np.array(all_count)
    bv_all = np.concatenate(all_bval)
    dv_all = np.concatenate(all_dval)
    pv_all = np.concatenate(all_pers)

    print("\n=== Constraint Summary ===")
    print(f"  Output NPZ        : {npz_path}")
    print(f"  Output CSV        : {csv_path}")
    print(f"  Samples processed : {n_processed}")
    print(f"  Total pairs        : {flat_offset}")
    print(f"  Pairs/sample      : min={counts_arr.min()}  mean={counts_arr.mean():.1f}  max={counts_arr.max()}")
    print(f"  persistence_frac   : {args.persistence_frac}")
    print(f"  top_k              : {args.top_k}")
    print(f"  patch size         : {args.patch}×{args.patch}")
    print(f"  Birth val range    : [{bv_all.min():.4f}, {bv_all.max():.4f}]")
    print(f"  Death val range    : [{dv_all.min():.4f}, {dv_all.max():.4f}]")
    print(f"  Persistence range  : [{pv_all.min():.4f}, {pv_all.max():.4f}]")

    # Print examples for specific sample indices if present
    sidx_arr = np.array(all_sidx)
    for target in [6, 18, 25, 80, 162]:
        matches = np.where(sidx_arr == target)[0]
        if len(matches) == 0:
            continue
        pos = matches[0]
        d = per_sample[pos]
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
