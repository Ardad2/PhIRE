#!/usr/bin/env python3
"""
Phase 5B — controlled D0 translation benchmark.

Modes
-----
generate:
    Create canonical C-order VTI scalar fields for two synthetic scenarios.

analyze:
    Read TTK PD VTUs generated from those fields and compare:
      1. raw GUDHI W2,2 assignment
      2. raw GUDHI W2,infinity assignment
      3. exact-persistence-group spatial secondary tie-break

Scientific purpose
------------------
This is a methodological validation, not a wind-field result.

Scenario A ("unique"):
    finite D0 features have distinct persistence coordinates.

Scenario B ("duplicate"):
    two finite D0 features have exactly the same persistence coordinate but
    different physical locations.

All local minima are translated rigidly by dx in {0,1,2,4,8}.  Because only
location changes, the PD itself should remain identical and the true D0 birth
displacement is exactly dx pixels.

The benchmark deliberately evaluates the D0 birth/minimum location only.
Death/saddle representative locations are not used because the constructed
background is a flat merge plateau and therefore does not define a unique
physical saddle representative.
"""

from __future__ import annotations

from pathlib import Path
from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import json
import math

import numpy as np


SHIFTS = [0, 1, 2, 4, 8]
H = W = 160
BACKGROUND = 20.0


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def scenario_minima(name):
    if name == "unique":
        # deepest minimum is global / essential and is not part of finite D0.
        return [
            (30, 30, 0.0),
            (50, 50, 2.0),
            (90, 70, 4.0),
            (120, 110, 6.0),
        ]
    if name == "duplicate":
        return [
            (30, 30, 0.0),
            (50, 50, 4.0),
            (110, 90, 4.0),
            (90, 120, 6.0),
        ]
    raise ValueError(name)


def make_field(name, dx):
    a = np.full((H, W), BACKGROUND, dtype=np.float32)
    for x, y, value in scenario_minima(name):
        xx = x + dx
        if not (0 <= xx < W and 0 <= y < H):
            raise ValueError((name, dx, x, y))
        a[y, xx] = np.float32(value)
    return a


def generate(out):
    # Import the project's repaired writer directly from its file path.
    # This is deliberately path-based rather than `from scripts...` because
    # when this benchmark is launched as:
    #
    #   python spatial_pd/phase5b_controlled_translation_benchmark.py
    #
    # Python places /work/spatial_pd (the script directory), not necessarily
    # /work (the repository root), on sys.path inside the Docker container.
    # Loading the module explicitly keeps the benchmark independent of whether
    # `scripts/` is an importable package.
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    writer_path = repo_root / "scripts" / "convert_phire_to_vti.py"

    if not writer_path.exists():
        raise FileNotFoundError(
            f"Could not locate repaired VTI writer: {writer_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "phase5_repaired_vti_writer", writer_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load VTI writer: {writer_path}")

    writer_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(writer_module)
    make_vti_from_scalar = writer_module.make_vti_from_scalar

    vti_dir = out / "vti"
    fields_dir = out / "fields"
    vti_dir.mkdir(parents=True, exist_ok=True)
    fields_dir.mkdir(parents=True, exist_ok=True)

    manifest = []

    for scenario in ("unique", "duplicate"):
        gt = make_field(scenario, 0)
        np.save(fields_dir / f"{scenario}_GT.npy", gt)
        gt_vti = vti_dir / f"{scenario}_GT.vti"
        make_vti_from_scalar(gt, "wind_speed", str(gt_vti))

        manifest.append({
            "scenario": scenario,
            "kind": "GT",
            "dx": 0,
            "vti": str(gt_vti),
            "vti_sha256": sha256(gt_vti),
            "field_sha256": sha256(fields_dir / f"{scenario}_GT.npy"),
        })

        for dx in SHIFTS:
            sr = make_field(scenario, dx)
            np.save(fields_dir / f"{scenario}_SR_dx{dx}.npy", sr)
            sr_vti = vti_dir / f"{scenario}_SR_dx{dx}.vti"
            make_vti_from_scalar(sr, "wind_speed", str(sr_vti))
            manifest.append({
                "scenario": scenario,
                "kind": "SR",
                "dx": dx,
                "vti": str(sr_vti),
                "vti_sha256": sha256(sr_vti),
                "field_sha256": sha256(fields_dir / f"{scenario}_SR_dx{dx}.npy"),
            })

    (out / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    print("===== SYNTHETIC TRANSLATION FIELDS GENERATED =====")
    for x in manifest:
        print(
            x["scenario"], x["kind"], "dx=", x["dx"],
            Path(x["vti"]).name
        )


def read_positive_d0(path):
    import vtk

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
    coord = pd.GetArray("Coordinates")
    vid = pd.GetArray("ttkVertexScalarField")

    out = []

    for ci in range(g.GetNumberOfCells()):
        typ = int(round(ptype.GetTuple1(ci)))
        fin = int(round(finite.GetTuple1(ci)))
        p = float(pers.GetTuple1(ci))

        if typ != 0 or fin != 1 or p <= 1e-10:
            continue

        cell = g.GetCell(ci)
        p0 = int(cell.GetPointId(0))
        b = float(birth.GetTuple1(ci))
        d = b + p
        c0 = coord.GetTuple(p0)

        out.append({
            "index": len(out),
            "cell_index": ci,
            "pair_identifier": int(round(pid.GetTuple1(ci))),
            "birth": b,
            "death": d,
            "persistence": p,
            "key": (b.hex(), d.hex()),
            "birth_x": float(c0[0]),
            "birth_y": float(c0[1]),
            "birth_vertex_id": int(round(vid.GetTuple1(p0))),
        })

    return out


def points(fs):
    return np.asarray(
        [[f["birth"], f["death"]] for f in fs], dtype=float
    ).reshape(-1, 2)


def birth_dist(a, b):
    return math.hypot(
        a["birth_x"] - b["birth_x"],
        a["birth_y"] - b["birth_y"],
    )


def gudhi_match(gt, sr, internal_p):
    import gudhi
    from gudhi.wasserstein import wasserstein_distance

    d, M = wasserstein_distance(
        points(gt),
        points(sr),
        matching=True,
        order=2.0,
        internal_p=internal_p,
        keep_essential_parts=False,
    )
    M = np.asarray(M, dtype=int).reshape(-1, 2)

    rows = []
    for i, j in M:
        if i >= 0 and j >= 0:
            rows.append({
                "gt_index": int(i),
                "sr_index": int(j),
                "type": "real_real",
                "birth_displacement": birth_dist(gt[i], sr[j]),
            })
        elif i >= 0:
            rows.append({
                "gt_index": int(i),
                "sr_index": -1,
                "type": "gt_to_diagonal",
                "birth_displacement": None,
            })
        else:
            rows.append({
                "gt_index": -1,
                "sr_index": int(j),
                "type": "diagonal_to_sr",
                "birth_displacement": None,
            })

    return float(d), rows


def secondary_exact_key_spatial(gt, sr):
    """
    Lexicographic rule for this controlled test only:

    Primary:
        preserve exact persistence coordinate equality.

    Secondary:
        within each exact (birth,death) group, minimize total birth-coordinate
        Euclidean distance.

    This does not alter primary PD cost because every assignment inside an
    exact-equal scalar group has identical zero persistence-plane cost.
    """
    from scipy.optimize import linear_sum_assignment

    G = defaultdict(list)
    S = defaultdict(list)

    for i, f in enumerate(gt):
        G[f["key"]].append(i)
    for j, f in enumerate(sr):
        S[f["key"]].append(j)

    if Counter({k: len(v) for k, v in G.items()}) != Counter(
        {k: len(v) for k, v in S.items()}
    ):
        raise RuntimeError("GT/SR exact persistence group multiplicities differ.")

    rows = []

    for key in sorted(G):
        gi = G[key]
        sj = S[key]

        C = np.zeros((len(gi), len(sj)), dtype=float)
        for a, i in enumerate(gi):
            for b, j in enumerate(sj):
                C[a, b] = birth_dist(gt[i], sr[j])

        rr, cc = linear_sum_assignment(C)

        for a, b in zip(rr, cc):
            i = gi[a]
            j = sj[b]
            rows.append({
                "gt_index": i,
                "sr_index": j,
                "birth_displacement": float(C[a, b]),
                "birth": gt[i]["birth"],
                "death": gt[i]["death"],
                "multiplicity": len(gi),
            })

    return rows


def summarize_displacements(rows):
    vals = np.asarray(
        [r["birth_displacement"] for r in rows
         if r.get("birth_displacement") is not None],
        dtype=float,
    )
    return {
        "count": int(len(vals)),
        "mean": float(vals.mean()) if len(vals) else None,
        "median": float(np.median(vals)) if len(vals) else None,
        "max": float(vals.max()) if len(vals) else None,
        "values": vals.tolist(),
    }


def analyze(out):
    import gudhi

    pd_dir = out / "pd"
    if not pd_dir.is_dir():
        raise FileNotFoundError(pd_dir)

    summary_rows = []
    detail_rows = []
    report = {
        "gudhi_version": gudhi.__version__,
        "shifts": SHIFTS,
        "scenarios": {},
    }

    for scenario in ("unique", "duplicate"):
        gt_path = pd_dir / f"{scenario}_GT_pd_port_0.vtu"
        gt = read_positive_d0(gt_path)

        report["scenarios"][scenario] = {
            "gt_path": str(gt_path),
            "gt_sha256": sha256(gt_path),
            "gt_positive_d0": gt,
            "shifts": {},
        }

        print()
        print("============================================================")
        print("SCENARIO:", scenario)
        print("GT positive D0:")
        for f in gt:
            print(
                "  birth=", f["birth"],
                "death=", f["death"],
                "xy=", (f["birth_x"], f["birth_y"]),
            )

        for dx in SHIFTS:
            sr_path = pd_dir / f"{scenario}_SR_dx{dx}_pd_port_0.vtu"
            sr = read_positive_d0(sr_path)

            # Exact persistence multiset gate.
            gt_keys = Counter(f["key"] for f in gt)
            sr_keys = Counter(f["key"] for f in sr)
            scalar_equal = gt_keys == sr_keys
            if not scalar_equal:
                raise RuntimeError(
                    f"{scenario} dx={dx}: scalar PD multiset changed"
                )

            d22, m22 = gudhi_match(gt, sr, 2.0)
            dinf, minf = gudhi_match(gt, sr, np.inf)
            sec = secondary_exact_key_spatial(gt, sr)

            s22 = summarize_displacements(m22)
            sinf = summarize_displacements(minf)
            ssec = summarize_displacements(sec)

            # The secondary exact-key spatial rule should recover the known
            # rigid translation in this benchmark.
            sec_ok = all(
                math.isclose(v, float(dx), rel_tol=0.0, abs_tol=1e-12)
                for v in ssec["values"]
            )
            if not sec_ok:
                raise RuntimeError(
                    f"{scenario} dx={dx}: secondary spatial rule did not "
                    f"recover known shift; values={ssec['values']}"
                )

            print()
            print(f"dx={dx}")
            print("  PD scalar multiset equal:", scalar_equal)
            print("  W22 distance:", d22)
            print("  W2inf distance:", dinf)
            print(
                "  raw W22 birth displacement:",
                s22["values"],
                "mean=", s22["mean"],
            )
            print(
                "  raw W2inf birth displacement:",
                sinf["values"],
                "mean=", sinf["mean"],
            )
            print(
                "  exact-key spatial tie-break:",
                ssec["values"],
                "mean=", ssec["mean"],
                "known-shift-pass=", sec_ok,
            )

            report["scenarios"][scenario]["shifts"][str(dx)] = {
                "sr_path": str(sr_path),
                "sr_sha256": sha256(sr_path),
                "pd_scalar_multiset_equal": scalar_equal,
                "W22_distance": d22,
                "W2inf_distance": dinf,
                "raw_W22_birth_displacement": s22,
                "raw_W2inf_birth_displacement": sinf,
                "secondary_exact_key_spatial": ssec,
                "secondary_recovers_known_shift": sec_ok,
            }

            summary_rows.append({
                "scenario": scenario,
                "dx": dx,
                "pd_scalar_multiset_equal": scalar_equal,
                "W22_distance": d22,
                "W2inf_distance": dinf,
                "raw_W22_mean_birth_disp": s22["mean"],
                "raw_W2inf_mean_birth_disp": sinf["mean"],
                "secondary_mean_birth_disp": ssec["mean"],
                "secondary_recovers_known_shift": sec_ok,
            })

            for label, rows in (
                ("W22", m22),
                ("W2inf", minf),
                ("secondary", sec),
            ):
                for r in rows:
                    detail_rows.append({
                        "scenario": scenario,
                        "dx": dx,
                        "rule": label,
                        **r,
                    })

    json_path = out / "translation_benchmark_summary.json"
    json_path.write_text(json.dumps(report, indent=2))

    with (out / "translation_benchmark_summary.csv").open(
        "w", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    # union all detail keys
    fieldnames = []
    for r in detail_rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)

    with (out / "translation_benchmark_detail.csv").open(
        "w", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(detail_rows)

    print()
    print("TRANSLATION BENCHMARK ANALYSIS: PASS")
    print("Wrote:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["generate", "analyze"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == "generate":
        generate(out)
    else:
        analyze(out)


if __name__ == "__main__":
    main()
