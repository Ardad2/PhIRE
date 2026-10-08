#!/usr/bin/env python3
"""
Phase 5B2 — near-degenerate persistence crossover benchmark.

Purpose
-------
Test the boundary between:
  (a) persistence-optimal correspondence, and
  (b) known physical feature identity,

when two spatially separated D0 features have similar but not always identical
persistence coordinates.

Construction
------------
Background = 20.
One global minimum at (30,30), value 0 (essential; excluded from finite D0).

Two tracked finite D0 features:
  A_GT: birth=4.0 at (50,50)
  B_GT: birth=4.4 at (110,90)

One anchor:
  C_GT: birth=7.0 at (90,120)

SR:
  all tracked minima translated +4 pixels in x.
  A birth = 4.0 + delta
  B birth = 4.4 - delta
  C birth = 7.0

delta grid:
  0.00, 0.10, 0.19, 0.20, 0.21, 0.30, 0.40

At delta=0.20:
  A_SR and B_SR have identical persistence coordinates.

For delta>0.20:
  their scalar ordering crosses.  Persistence-only assignment can prefer a
  crossed correspondence even though physical identity remains known by
  construction.

Modes
-----
generate:
  write canonical C-order VTIs.

analyze:
  read TTK PDs and compare:
    - W2,2 optimal assignment
    - W2,infinity optimal assignment
    - known physical-identity assignment
    - exact-degeneracy spatial tie-break at delta=0.20

This benchmark deliberately evaluates D0 birth/minimum locations only.
"""

from pathlib import Path
from collections import defaultdict
import argparse, csv, hashlib, json, math
import numpy as np

H = W = 160
BACKGROUND = 20.0
DX = 4
DELTAS = [0.00, 0.10, 0.19, 0.20, 0.21, 0.30, 0.40]
TOL = 1e-10

GT_FEATURES = {
    "A": (50, 50, 4.0),
    "B": (110, 90, 4.4),
    "C": (90, 120, 7.0),
}
GLOBAL = (30, 30, 0.0)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_writer():
    import importlib.util
    repo_root = Path(__file__).resolve().parents[1]
    p = repo_root / "scripts" / "convert_phire_to_vti.py"
    spec = importlib.util.spec_from_file_location("phase5_writer", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.make_vti_from_scalar


def field(delta=None):
    a = np.full((H, W), BACKGROUND, dtype=np.float32)
    gx, gy, gv = GLOBAL
    a[gy, gx] = np.float32(gv)

    for label, (x, y, v) in GT_FEATURES.items():
        if delta is None:
            xx = x
            vv = v
        else:
            xx = x + DX
            if label == "A":
                vv = 4.0 + delta
            elif label == "B":
                vv = 4.4 - delta
            else:
                vv = 7.0
        a[y, xx] = np.float32(vv)

    return a


def generate(out):
    writer = load_writer()
    vti = out / "vti"
    fields = out / "fields"
    vti.mkdir(parents=True, exist_ok=True)
    fields.mkdir(parents=True, exist_ok=True)

    gt = field(None)
    np.save(fields / "GT.npy", gt)
    writer(gt, "wind_speed", str(vti / "GT.vti"))

    manifest = [{
        "kind": "GT",
        "delta": None,
        "vti": str(vti / "GT.vti"),
        "vti_sha256": sha256(vti / "GT.vti"),
    }]

    for d in DELTAS:
        sr = field(d)
        tag = f"{d:.2f}".replace(".", "p")
        np.save(fields / f"SR_delta_{tag}.npy", sr)
        p = vti / f"SR_delta_{tag}.vti"
        writer(sr, "wind_speed", str(p))
        manifest.append({
            "kind": "SR",
            "delta": d,
            "vti": str(p),
            "vti_sha256": sha256(p),
        })

    (out / "generation_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Generated", len(manifest), "VTIs")
    for x in manifest:
        print(x["kind"], x["delta"], Path(x["vti"]).name)


def read_d0(path):
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
    coord = pd.GetArray("Coordinates")

    out = []
    for ci in range(g.GetNumberOfCells()):
        typ = int(round(ptype.GetTuple1(ci)))
        fin = int(round(finite.GetTuple1(ci)))
        p = float(pers.GetTuple1(ci))
        if typ != 0 or fin != 1 or p <= TOL:
            continue
        c = g.GetCell(ci)
        p0 = int(c.GetPointId(0))
        b = float(birth.GetTuple1(ci))
        d = b + p
        xy = coord.GetTuple(p0)
        out.append({
            "index": len(out),
            "birth": b,
            "death": d,
            "persistence": p,
            "x": float(xy[0]),
            "y": float(xy[1]),
        })
    return out


def pts(fs):
    return np.asarray([[f["birth"], f["death"]] for f in fs], dtype=float)


def pd_cost(a, b, p):
    dx = abs(a["birth"] - b["birth"])
    dy = abs(a["death"] - b["death"])
    if np.isinf(p):
        return max(dx, dy)
    return math.hypot(dx, dy)


def spatial(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def gmatch(gt, sr, p):
    from gudhi.wasserstein import wasserstein_distance
    dist, M = wasserstein_distance(
        pts(gt), pts(sr), matching=True, order=2.0, internal_p=p,
        keep_essential_parts=False
    )
    M = np.asarray(M, dtype=int).reshape(-1, 2)
    return float(dist), M


def label_features(fs, expected_positions):
    """
    Assign feature labels by nearest expected birth location.
    Controlled benchmark only.
    """
    remaining = set(range(len(fs)))
    labeled = {}

    for label, (x, y) in expected_positions.items():
        j = min(
            remaining,
            key=lambda k: math.hypot(fs[k]["x"] - x, fs[k]["y"] - y)
        )
        labeled[label] = j
        remaining.remove(j)

    return labeled


def total_assignment_pd_q2(gt, sr, pairs, internal_p):
    costs = [pd_cost(gt[i], sr[j], internal_p) for i, j in pairs]
    return math.sqrt(sum(c*c for c in costs))


def analyze(out):
    import gudhi
    pdroot = out / "pd"
    gt = read_d0(pdroot / "GT_pd_port_0.vtu")

    gt_idx = label_features(
        gt,
        {k: (v[0], v[1]) for k, v in GT_FEATURES.items()}
    )

    report = {
        "gudhi": gudhi.__version__,
        "dx": DX,
        "deltas": DELTAS,
        "gt": gt,
        "rows": [],
    }

    print("===== GT D0 FEATURES =====")
    for label, i in gt_idx.items():
        print(label, gt[i])

    for delta in DELTAS:
        tag = f"{delta:.2f}".replace(".", "p")
        sr = read_d0(pdroot / f"SR_delta_{tag}_pd_port_0.vtu")
        sr_idx = label_features(
            sr,
            {k: (v[0] + DX, v[1]) for k, v in GT_FEATURES.items()}
        )

        true_pairs = [(gt_idx[k], sr_idx[k]) for k in ("A", "B", "C")]

        print()
        print("====================================================")
        print("delta =", delta)

        row = {"delta": delta}

        for name, p in (("W22", 2.0), ("W2inf", np.inf)):
            d, M = gmatch(gt, sr, p)
            mapping = {int(i): int(j) for i, j in M if i >= 0}
            ab_truth = {
                "A": mapping.get(gt_idx["A"]) == sr_idx["A"],
                "B": mapping.get(gt_idx["B"]) == sr_idx["B"],
            }

            true_pd = total_assignment_pd_q2(gt, sr, true_pairs, p)

            raw_spatial = {}
            for label in ("A", "B", "C"):
                i = gt_idx[label]
                j = mapping.get(i, -1)
                raw_spatial[label] = (
                    spatial(gt[i], sr[j]) if j >= 0 else None
                )

            row[f"{name}_optimal_distance"] = d
            row[f"{name}_true_identity_pd_cost"] = true_pd
            row[f"{name}_true_identity_extra_cost"] = true_pd - d
            row[f"{name}_A_correct"] = ab_truth["A"]
            row[f"{name}_B_correct"] = ab_truth["B"]
            row[f"{name}_raw_A_spatial"] = raw_spatial["A"]
            row[f"{name}_raw_B_spatial"] = raw_spatial["B"]

            print(name)
            print("  optimal PD distance:", d)
            print("  true-identity PD cost:", true_pd)
            print("  extra persistence cost for true identity:", true_pd - d)
            print("  A correct:", ab_truth["A"])
            print("  B correct:", ab_truth["B"])
            print("  raw spatial A/B:", raw_spatial["A"], raw_spatial["B"])

        # At exact scalar degeneracy, perform secondary spatial assignment
        # inside the A/B equal-persistence group.
        if math.isclose(delta, 0.20, abs_tol=1e-12):
            from scipy.optimize import linear_sum_assignment
            gi = [gt_idx["A"], gt_idx["B"]]
            sj = [sr_idx["A"], sr_idx["B"]]
            C = np.asarray(
                [[spatial(gt[i], sr[j]) for j in sj] for i in gi],
                dtype=float,
            )
            rr, cc = linear_sum_assignment(C)
            vals = [float(C[a, b]) for a, b in zip(rr, cc)]
            row["degenerate_spatial_tiebreak_values"] = vals
            print("  exact-degenerate spatial tie-break:", vals)

        report["rows"].append(row)

    (out / "near_degenerate_crossover_summary.json").write_text(
        json.dumps(report, indent=2)
    )

    fieldnames = []
    for r in report["rows"]:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)

    with (out / "near_degenerate_crossover_summary.csv").open(
        "w", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(report["rows"])

    print()
    print("NEAR-DEGENERATE CROSSOVER ANALYSIS COMPLETE")
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
