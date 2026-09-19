#!/usr/bin/env python3
"""
Phase 5A Step 2b — diagnose sample-69 GT provenance exceptions before any
GT<->SR persistence matching.

Reads outputs from phase5a_extract_sample69_pair_provenance.py.

Goals
-----
1. Confirm the single PD display-geometry failure is confined to a special
   non-analysis cell.
2. Avoid the previous rounded-key/dict-collapse diagnostic.
3. Compare GT pair provenance with exact scalar keys as MULTISETS.
4. Check whether PairIdentifier is stable across CNN / reconstruction-only /
   topology-inspired GT tracks.
5. Report only genuine coordinate/provenance differences after canonicalizing
   the topology-inspired transpose.
6. Identify the extra topology-inspired GT pair.

No scientific matching between GT and SR is performed here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


METHOD_FILES = {
    "pretrained_cnn": "pretrained_cnn_GT_sample69_pair_provenance.csv",
    "reconstruction_only": "reconstruction_only_GT_sample69_pair_provenance.csv",
    "topology_inspired": "topology_inspired_GT_sample69_pair_provenance.csv",
}


def load_csv(path: Path):
    rows = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            rr = dict(r)
            for k in (
                "cell_index", "pair_identifier", "pair_type", "is_finite",
                "birth_critical_type", "death_critical_type",
                "birth_vertex_id_raw", "death_vertex_id_raw",
                "birth_x_raw", "birth_y_raw", "death_x_raw", "death_y_raw",
                "birth_x_canonical", "birth_y_canonical",
                "death_x_canonical", "death_y_canonical",
            ):
                rr[k] = int(rr[k])
            for k in (
                "birth", "death", "persistence",
                "pd_point0_x", "pd_point0_y",
                "pd_point1_x", "pd_point1_y",
            ):
                rr[k] = float(rr[k])
            rows.append(rr)
    return rows


def exact_scalar_key(r):
    # float.hex preserves the exact Python float value recovered from CSV.
    return (
        r["pair_type"],
        r["is_finite"],
        r["birth"].hex(),
        r["persistence"].hex(),
    )


def analysis_rows(rows):
    # Exactly mirrors the corrected finite-PD semantic layer:
    # finite D0/D1 only. Special/global/nonfinite cells are not part of the
    # finite diagram matching.
    return [
        r for r in rows
        if r["is_finite"] == 1 and r["pair_type"] in (0, 1)
    ]


def coord_tuple(r):
    return (
        r["birth_x_canonical"], r["birth_y_canonical"],
        r["death_x_canonical"], r["death_y_canonical"],
    )


def endpoint_record(r):
    return {
        "cell_index": r["cell_index"],
        "pair_identifier": r["pair_identifier"],
        "pair_type": r["pair_type"],
        "birth": r["birth"],
        "death": r["death"],
        "persistence": r["persistence"],
        "birth_critical_type": r["birth_critical_type"],
        "death_critical_type": r["death_critical_type"],
        "birth_vertex_id_raw": r["birth_vertex_id_raw"],
        "death_vertex_id_raw": r["death_vertex_id_raw"],
        "raw_xy": [
            r["birth_x_raw"], r["birth_y_raw"],
            r["death_x_raw"], r["death_y_raw"],
        ],
        "canonical_xy": list(coord_tuple(r)),
    }


def dist2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--provenance",
        default=str(Path.home() / "PhIRE/spatial_pd/phase5a_sample69_provenance"),
    )
    ap.add_argument(
        "--out",
        default=str(Path.home() / "PhIRE/spatial_pd/phase5a_sample69_provenance_diagnostic"),
    )
    args = ap.parse_args()

    prov = Path(args.provenance).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    audit_path = prov / "sample69_pair_provenance_audit.json"
    audit = json.loads(audit_path.read_text())

    rows = {
        m: load_csv(prov / fn)
        for m, fn in METHOD_FILES.items()
    }
    finite = {m: analysis_rows(rs) for m, rs in rows.items()}

    report = {
        "sample": 69,
        "source_audit": str(audit_path),
        "pd_geometry_failures": {},
        "finite_counts": {},
        "exact_scalar_key_duplicates": {},
        "pair_identifier_cross_track": {},
        "cnn_vs_uv": {},
        "cnn_vs_topology_inspired": {},
    }

    # Recover exact display-geometry failure records from Step 2 audit.
    for f in audit["files"]:
        if f["field_kind"] == "GT":
            report["pd_geometry_failures"][f["method"]] = f["pd_geometry_failures"]

    # Multiplicity diagnostics.
    key_maps = {}
    pid_maps = {}
    for m, rs in finite.items():
        km = defaultdict(list)
        pm = defaultdict(list)
        for r in rs:
            km[exact_scalar_key(r)].append(r)
            pm[r["pair_identifier"]].append(r)
        key_maps[m] = km
        pid_maps[m] = pm

        dups = {
            str(k): [endpoint_record(x) for x in vv]
            for k, vv in km.items()
            if len(vv) > 1
        }
        report["finite_counts"][m] = len(rs)
        report["exact_scalar_key_duplicates"][m] = {
            "duplicate_key_count": len(dups),
            "duplicate_row_excess": sum(len(v) - 1 for v in km.values() if len(v) > 1),
            "duplicates": dups,
        }

    # PairIdentifier stability on IDs common to all three finite sets.
    common_pids = set.intersection(*(set(pm.keys()) for pm in pid_maps.values()))
    pid_disagreements = []
    for pid in sorted(common_pids):
        # Require exactly one row per method for this PID.
        if any(len(pid_maps[m][pid]) != 1 for m in pid_maps):
            continue
        recs = {m: pid_maps[m][pid][0] for m in pid_maps}
        scalar_keys = {m: exact_scalar_key(r) for m, r in recs.items()}
        coords = {m: coord_tuple(r) for m, r in recs.items()}
        if len(set(scalar_keys.values())) != 1 or len(set(coords.values())) != 1:
            pid_disagreements.append({
                "pair_identifier": pid,
                "records": {m: endpoint_record(r) for m, r in recs.items()},
            })

    report["pair_identifier_cross_track"] = {
        "common_finite_pair_identifiers": len(common_pids),
        "disagreement_count": len(pid_disagreements),
        "disagreements": pid_disagreements,
    }

    def compare(a_name, b_name):
        A = key_maps[a_name]
        B = key_maps[b_name]
        keys_a = set(A)
        keys_b = set(B)
        common = keys_a & keys_b

        multiplicity_mismatch = []
        unique_coord_mismatch = []
        exact_match_count = 0

        for k in sorted(common, key=str):
            aa = A[k]
            bb = B[k]
            if len(aa) != len(bb):
                multiplicity_mismatch.append({
                    "key": list(k),
                    "a_count": len(aa),
                    "b_count": len(bb),
                    "a_records": [endpoint_record(x) for x in aa],
                    "b_records": [endpoint_record(x) for x in bb],
                })
                continue

            if len(aa) == 1:
                ca = coord_tuple(aa[0])
                cb = coord_tuple(bb[0])
                if ca == cb:
                    exact_match_count += 1
                else:
                    unique_coord_mismatch.append({
                        "key": list(k),
                        "a": endpoint_record(aa[0]),
                        "b": endpoint_record(bb[0]),
                        "birth_displacement": dist2(ca[:2], cb[:2]),
                        "death_displacement": dist2(ca[2:], cb[2:]),
                    })
            else:
                # Duplicate scalar keys: compare canonical coordinate multisets,
                # not arbitrary dict-selected rows.
                ca = Counter(coord_tuple(x) for x in aa)
                cb = Counter(coord_tuple(x) for x in bb)
                if ca == cb:
                    exact_match_count += len(aa)
                else:
                    multiplicity_mismatch.append({
                        "key": list(k),
                        "a_count": len(aa),
                        "b_count": len(bb),
                        "a_coordinate_multiset": {str(x): n for x, n in ca.items()},
                        "b_coordinate_multiset": {str(x): n for x, n in cb.items()},
                        "a_records": [endpoint_record(x) for x in aa],
                        "b_records": [endpoint_record(x) for x in bb],
                    })

        return {
            "a": a_name,
            "b": b_name,
            "finite_rows_a": len(finite[a_name]),
            "finite_rows_b": len(finite[b_name]),
            "exact_scalar_keys_a": len(keys_a),
            "exact_scalar_keys_b": len(keys_b),
            "common_exact_scalar_keys": len(common),
            "a_only_keys": [
                {"key": list(k), "records": [endpoint_record(x) for x in A[k]]}
                for k in sorted(keys_a - keys_b, key=str)
            ],
            "b_only_keys": [
                {"key": list(k), "records": [endpoint_record(x) for x in B[k]]}
                for k in sorted(keys_b - keys_a, key=str)
            ],
            "exact_coordinate_matches_count": exact_match_count,
            "unique_coordinate_mismatch_count": len(unique_coord_mismatch),
            "unique_coordinate_mismatches": unique_coord_mismatch,
            "multiplicity_or_duplicate_mismatch_count": len(multiplicity_mismatch),
            "multiplicity_or_duplicate_mismatches": multiplicity_mismatch,
        }

    report["cnn_vs_uv"] = compare("pretrained_cnn", "reconstruction_only")
    report["cnn_vs_topology_inspired"] = compare("pretrained_cnn", "topology_inspired")

    out_json = out / "phase5a_step2b_gt_provenance_diagnostic.json"
    out_json.write_text(json.dumps(report, indent=2))

    print("===== PHASE 5A STEP 2B — GT PROVENANCE EXCEPTION DIAGNOSTIC =====")
    print()

    print("PD display-geometry failures from Step 2:")
    for m, ff in report["pd_geometry_failures"].items():
        print(f"  {m}: {len(ff)}")
        for x in ff:
            print("   ", x)

    print()
    print("Finite D0/D1 counts + exact scalar-key duplicate audit:")
    for m in finite:
        d = report["exact_scalar_key_duplicates"][m]
        print(
            f"  {m:22s} finite={len(finite[m]):4d} "
            f"duplicate_keys={d['duplicate_key_count']:3d} "
            f"duplicate_row_excess={d['duplicate_row_excess']:3d}"
        )

    print()
    p = report["pair_identifier_cross_track"]
    print(
        "Common finite PairIdentifiers across all 3 GT tracks:",
        p["common_finite_pair_identifiers"],
    )
    print("PairIdentifier scalar/coordinate disagreements:", p["disagreement_count"])
    for x in p["disagreements"][:20]:
        print(json.dumps(x, indent=2))

    for key in ("cnn_vs_uv", "cnn_vs_topology_inspired"):
        c = report[key]
        print()
        print(f"--- {c['a']} vs {c['b']} ---")
        print("finite rows:", c["finite_rows_a"], c["finite_rows_b"])
        print("exact scalar keys:", c["exact_scalar_keys_a"], c["exact_scalar_keys_b"])
        print("common exact scalar keys:", c["common_exact_scalar_keys"])
        print("a-only key count:", len(c["a_only_keys"]))
        print("b-only key count:", len(c["b_only_keys"]))
        print("exact coordinate matches:", c["exact_coordinate_matches_count"])
        print("unique coordinate mismatches:", c["unique_coordinate_mismatch_count"])
        print(
            "multiplicity/duplicate mismatches:",
            c["multiplicity_or_duplicate_mismatch_count"],
        )

        if c["a_only_keys"]:
            print("First A-only keys:")
            print(json.dumps(c["a_only_keys"][:10], indent=2))
        if c["b_only_keys"]:
            print("First B-only keys:")
            print(json.dumps(c["b_only_keys"][:10], indent=2))
        if c["unique_coordinate_mismatches"]:
            print("Coordinate mismatches:")
            print(json.dumps(c["unique_coordinate_mismatches"][:20], indent=2))
        if c["multiplicity_or_duplicate_mismatches"]:
            print("Multiplicity/duplicate mismatches:")
            print(json.dumps(c["multiplicity_or_duplicate_mismatches"][:20], indent=2))

    print()
    print("Wrote:", out_json)


if __name__ == "__main__":
    main()
