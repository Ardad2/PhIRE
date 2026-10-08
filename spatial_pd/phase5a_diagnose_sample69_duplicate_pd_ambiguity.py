#!/usr/bin/env python3
"""
Phase 5A Step 4D — exact duplicate-persistence ambiguity diagnostic.

Purpose:
- enumerate exact duplicate (birth, death) points in canonical sample-69 GT;
- record their spatial critical-point coordinates and vertex ids;
- reproduce the zero-cost self-match under W2,2 and W2,infinity;
- identify exactly which features are non-identity matched;
- quantify within-duplicate spatial separation.

Run inside gudhi-audit.
"""

from pathlib import Path
from collections import defaultdict
import argparse, csv, json, math, hashlib
import numpy as np
import vtk
import gudhi
from gudhi.wasserstein import wasserstein_distance

TOL = 1e-10

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def read_gt(path):
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(path))
    r.Update()
    g = r.GetOutput()

    cd = g.GetCellData()
    pd = g.GetPointData()

    ptype = cd.GetArray("PairType")
    pers = cd.GetArray("Persistence")
    birth = cd.GetArray("Birth")
    finite = cd.GetArray("IsFinite")
    pid = cd.GetArray("PairIdentifier")
    vid = pd.GetArray("ttkVertexScalarField")
    coord = pd.GetArray("Coordinates")
    crit = pd.GetArray("CriticalType")

    out = {0: [], 1: []}

    for ci in range(g.GetNumberOfCells()):
        typ = int(round(ptype.GetTuple1(ci)))
        fin = int(round(finite.GetTuple1(ci)))
        p = float(pers.GetTuple1(ci))
        if fin != 1 or typ not in (0, 1) or p <= TOL:
            continue

        c = g.GetCell(ci)
        p0 = int(c.GetPointId(0))
        p1 = int(c.GetPointId(1))

        b = float(birth.GetTuple1(ci))
        d = b + p
        c0 = coord.GetTuple(p0)
        c1 = coord.GetTuple(p1)

        out[typ].append({
            "index": len(out[typ]),
            "cell_index": ci,
            "pair_identifier": int(round(pid.GetTuple1(ci))),
            "pair_type": typ,
            "birth": b,
            "death": d,
            "persistence": p,
            "birth_hex": b.hex(),
            "death_hex": d.hex(),
            "birth_x": float(c0[0]),
            "birth_y": float(c0[1]),
            "death_x": float(c1[0]),
            "death_y": float(c1[1]),
            "birth_vertex_id": int(round(vid.GetTuple1(p0))),
            "death_vertex_id": int(round(vid.GetTuple1(p1))),
            "birth_critical_type": int(round(crit.GetTuple1(p0))),
            "death_critical_type": int(round(crit.GetTuple1(p1))),
        })

    return out

def points(fs):
    return np.asarray([[f["birth"], f["death"]] for f in fs], dtype=float).reshape(-1, 2)

def distance_xy(a, b, prefix):
    return math.hypot(
        a[f"{prefix}_x"] - b[f"{prefix}_x"],
        a[f"{prefix}_y"] - b[f"{prefix}_y"],
    )

def selfmatch(fs, internal_p):
    d, M = wasserstein_distance(
        points(fs),
        points(fs),
        matching=True,
        order=2.0,
        internal_p=internal_p,
        keep_essential_parts=False,
    )
    M = np.asarray(M, dtype=int).reshape(-1, 2)

    rows = []
    for i, j in M:
        if i < 0 or j < 0:
            rows.append({
                "i": int(i), "j": int(j), "match_type": "diagonal"
            })
            continue
        a = fs[i]
        b = fs[j]
        rows.append({
            "i": int(i),
            "j": int(j),
            "match_type": "real_real",
            "same_index": bool(i == j),
            "same_exact_scalar_point": bool(
                a["birth_hex"] == b["birth_hex"]
                and a["death_hex"] == b["death_hex"]
            ),
            "birth_displacement_px": distance_xy(a, b, "birth"),
            "death_displacement_px": distance_xy(a, b, "death"),
            "i_pair_identifier": a["pair_identifier"],
            "j_pair_identifier": b["pair_identifier"],
            "i_birth_x": a["birth_x"],
            "i_birth_y": a["birth_y"],
            "j_birth_x": b["birth_x"],
            "j_birth_y": b["birth_y"],
            "i_death_x": a["death_x"],
            "i_death_y": a["death_y"],
            "j_death_x": b["death_x"],
            "j_death_y": b["death_y"],
            "birth": a["birth"],
            "death": a["death"],
            "persistence": a["persistence"],
        })
    return float(d), rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gt",
        default=str(
            Path.home()
            / "PhIRE/spatial_pd/phase5a_sample69_canonical_pd/pd/"
              "phase5_GT_s69_speed_p160_x0_y0_pd_port_0.vtu"
        ),
    )
    ap.add_argument(
        "--out",
        default=str(
            Path.home()
            / "PhIRE/spatial_pd/phase5a_sample69_duplicate_pd_ambiguity"
        ),
    )
    args = ap.parse_args()

    gt_path = Path(args.gt).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    F = read_gt(gt_path)
    report = {
        "sample": 69,
        "gudhi_version": gudhi.__version__,
        "gt_path": str(gt_path),
        "gt_sha256": sha256(gt_path),
        "dimensions": {},
        "self_match": {},
    }

    duplicate_rows = []

    print("===== EXACT DUPLICATE PD POINTS =====")
    for dim in (0, 1):
        groups = defaultdict(list)
        for f in F[dim]:
            groups[(f["birth_hex"], f["death_hex"])].append(f)

        dups = [v for v in groups.values() if len(v) > 1]
        print(f"D{dim}: positive features={len(F[dim])} duplicate scalar groups={len(dups)}")

        dim_report = []
        for gi, members in enumerate(dups):
            pairwise = []
            for a_idx in range(len(members)):
                for b_idx in range(a_idx + 1, len(members)):
                    a, b = members[a_idx], members[b_idx]
                    pairwise.append({
                        "member_a_index": a["index"],
                        "member_b_index": b["index"],
                        "birth_endpoint_separation_px": distance_xy(a, b, "birth"),
                        "death_endpoint_separation_px": distance_xy(a, b, "death"),
                    })

            item = {
                "group_id": gi,
                "multiplicity": len(members),
                "birth": members[0]["birth"],
                "death": members[0]["death"],
                "persistence": members[0]["persistence"],
                "members": members,
                "pairwise_spatial_separation": pairwise,
            }
            dim_report.append(item)

            print()
            print(
                f"  D{dim} group {gi}: multiplicity={len(members)} "
                f"birth={members[0]['birth']:.15g} "
                f"death={members[0]['death']:.15g} "
                f"persistence={members[0]['persistence']:.15g}"
            )
            for m in members:
                print(
                    "    index=", m["index"],
                    "pair_id=", m["pair_identifier"],
                    "birth_xy=", (m["birth_x"], m["birth_y"]),
                    "death_xy=", (m["death_x"], m["death_y"]),
                    "birth_vid=", m["birth_vertex_id"],
                    "death_vid=", m["death_vertex_id"],
                )
            for p in pairwise:
                print("    separation:", p)

            for m in members:
                duplicate_rows.append({
                    "dimension": dim,
                    "group_id": gi,
                    "multiplicity": len(members),
                    **m,
                })

        report["dimensions"][f"D{dim}"] = {
            "positive_feature_count": len(F[dim]),
            "duplicate_group_count": len(dups),
            "groups": dim_report,
        }

    print()
    print("===== ZERO-COST SELF-MATCH NONIDENTITY ROWS =====")
    for label, p in (("W22", 2.0), ("W2inf", np.inf)):
        report["self_match"][label] = {}
        for dim in (0, 1):
            d, rows = selfmatch(F[dim], p)
            nonid = [
                r for r in rows
                if r.get("match_type") == "real_real"
                and not r.get("same_index", True)
            ]
            report["self_match"][label][f"D{dim}"] = {
                "distance": d,
                "nonidentity_count": len(nonid),
                "nonidentity_rows": nonid,
            }
            print(
                f"{label} D{dim}: distance={d:.15g} "
                f"nonidentity={len(nonid)}"
            )
            for r in nonid:
                print(" ", r)

    json_path = out / "sample69_duplicate_pd_ambiguity.json"
    json_path.write_text(json.dumps(report, indent=2))

    if duplicate_rows:
        csv_path = out / "sample69_duplicate_pd_groups.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(duplicate_rows[0].keys()))
            w.writeheader()
            w.writerows(duplicate_rows)

    print()
    print("Wrote:", out)

if __name__ == "__main__":
    main()
