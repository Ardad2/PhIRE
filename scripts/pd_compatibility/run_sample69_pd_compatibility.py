#!/usr/bin/env python3
"""
Sample-69 persistence-diagram compatibility audit between:

  A) PhIRE's corrected TTK-derived canonical PD metrics
  B) the uploaded colleague `tda-toolkit` GUDHI helpers

The audit has two deliberately separate phases.

PHASE A — metric parity
-----------------------
Use the SAME canonical finite TTK D0/D1 point arrays as input to:
  - PhIRE/reference exact GUDHI calls
  - colleague wrapper calls

This isolates distance-definition parity from PD construction.

PHASE B — descriptor construction
---------------------------------
Independently construct GUDHI cubical persistence diagrams from the same
authoritative 160x160 wind-speed fields using the colleague's
`compute_cubical_persistence()` helper, then compare GT->CNN/UV/F1 distances
and cross-backend descriptor differences.

No claim of identical PD construction is assumed in Phase B because TTK and
GUDHI are operating on different filtered complexes / cell conventions.

Expected PhIRE environment:
    /usr/bin/python3
    VTK 9.6
    GUDHI 3.13
    POT installed

The colleague toolkit is imported directly from its `src/` directory; the
source tree is not modified.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import vtk
import gudhi as gd
from gudhi.wasserstein import wasserstein_distance


SAMPLE_DEFAULT = 69
PATCH = 160

AUDIT_DEFAULT = Path.home() / "phire_runtime_audit_20260809_221548"
W22_DEFAULT = AUDIT_DEFAULT / "recompute_pd_w22"
ROOT_DEFAULT = Path.home() / "PhIRE"


METHODS = ("cnn", "uv", "f1")
METHOD_ID = {
    "cnn": "cnn",
    "uv": "uv",
    "f1": "f1_grad_e2",
}
DISPLAY = {
    "cnn": "CNN",
    "uv": "Matched UV control",
    "f1": "Candidate F1",
}

FIELD_DIRS_REL = {
    "cnn": "data_out_fixed/wind_mrhr_cnn",
    "uv": "data_out/wind_finetune_candidateUV_expanded2688",
    "f1": "data_out/wind_finetune_candidateF_grad_E2_low_expanded2688",
}

TOPO_ROOTS_REL = {
    "cnn": "ttk_runs_fixed/cnn",
    "uv": (
        "ttk_runs_fixed/topology_finetuning/"
        "candidateUV_expanded2688_topology"
    ),
    "f1": (
        "ttk_runs_fixed/topology_finetuning/"
        "candidateF_grad_E2_low_expanded2688_topology"
    ),
}


@dataclass
class PD:
    d0: np.ndarray
    d1: np.ndarray

    def by_dim(self, dim: int) -> np.ndarray:
        if dim == 0:
            return self.d0
        if dim == 1:
            return self.d1
        raise ValueError(dim)


@dataclass
class PDArtifact:
    path: Path
    association: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=Path,
        default=ROOT_DEFAULT,
        help="PhIRE repository root",
    )
    p.add_argument(
        "--w22",
        type=Path,
        default=W22_DEFAULT,
        help="W22 audit root",
    )
    p.add_argument(
        "--toolkit-root",
        type=Path,
        default=None,
        help=(
            "Root of colleague tda-toolkit source tree "
            "(directory containing src/tda_toolkit)"
        ),
    )
    p.add_argument("--sample", type=int, default=SAMPLE_DEFAULT)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory",
    )
    return p.parse_args()


def import_colleague(toolkit_root: Path):
    src = toolkit_root / "src"
    pkg = src / "tda_toolkit"

    if not pkg.is_dir():
        raise RuntimeError(
            f"Could not find {pkg}. "
            "Pass --toolkit-root to the extracted tda-toolkit-mapper directory."
        )

    sys.path.insert(0, str(src))

    import tda_toolkit  # type: ignore
    from tda_toolkit.persistence import (  # type: ignore
        compute_bottleneck_distance,
        compute_wasserstein_distance,
        compute_cubical_persistence,
    )

    return (
        tda_toolkit,
        compute_bottleneck_distance,
        compute_wasserstein_distance,
        compute_cubical_persistence,
    )


def data_array_names(dsa) -> List[str]:
    names = []
    for i in range(dsa.GetNumberOfArrays()):
        a = dsa.GetArray(i)
        if a is not None and a.GetName():
            names.append(a.GetName())
    return names


def inspect_pd_vtu(path: Path) -> Optional[PDArtifact]:
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(path))
    r.Update()

    g = r.GetOutput()

    required = {"Birth", "Persistence", "PairType", "PairIdentifier"}

    for association, dsa in (
        ("cell", g.GetCellData()),
        ("point", g.GetPointData()),
    ):
        names = set(data_array_names(dsa))
        if required.issubset(names):
            arrays = [dsa.GetArray(n) for n in required]
            sizes = [a.GetNumberOfTuples() for a in arrays]
            if len(set(sizes)) == 1 and sizes[0] > 0:
                return PDArtifact(path=path, association=association)

    return None


def side_from_path(path: Path) -> Optional[str]:
    s = "/" + path.as_posix().lower() + "/"
    name = path.name.lower()

    gt_hits = (
        "/gt/" in s
        or "_gt_" in name
        or name.startswith("gt_")
        or "_gt." in name
    )
    sr_hits = (
        "/sr/" in s
        or "_sr_" in name
        or name.startswith("sr_")
        or "_sr." in name
    )

    if gt_hits and not sr_hits:
        return "GT"
    if sr_hits and not gt_hits:
        return "SR"
    return None


def sample_token_matches(path: Path, sample: int) -> bool:
    s = path.as_posix().lower()
    tokens = (
        f"s{sample}_",
        f"s{sample}.",
        f"s{sample}/",
        f"s{sample:03d}_",
        f"s{sample:03d}.",
        f"s{sample:03d}/",
    )
    return any(t in s for t in tokens)


def discover_pd_artifacts(
    root: Path,
    topo_root: Path,
    sample: int,
) -> Dict[str, PDArtifact]:
    if not topo_root.is_dir():
        raise FileNotFoundError(topo_root)

    candidate_files = [
        p
        for p in topo_root.rglob("*.vtu")
        if sample_token_matches(p, sample)
    ]

    found: Dict[str, List[PDArtifact]] = {"GT": [], "SR": []}

    for p in candidate_files:
        side = side_from_path(p)
        if side is None:
            continue

        art = inspect_pd_vtu(p)
        if art is not None:
            found[side].append(art)

    chosen: Dict[str, PDArtifact] = {}

    for side in ("GT", "SR"):
        arts = found[side]

        # Prefer paths that explicitly contain a PD/persistence directory.
        preferred = [
            a for a in arts
            if any(
                token in a.path.as_posix().lower()
                for token in ("/pd/", "/persistence", "diagram")
            )
        ]

        pool = preferred if preferred else arts

        if len(pool) != 1:
            print()
            print(
                f"PD discovery ambiguity under {topo_root} "
                f"for sample={sample}, side={side}"
            )
            for a in arts:
                print("  ", a.association, a.path)
            raise RuntimeError(
                f"Expected exactly one PD artifact for {side}; got {len(pool)}"
            )

        chosen[side] = pool[0]

    return chosen


def load_ttk_pd(artifact: PDArtifact) -> PD:
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(artifact.path))
    r.Update()
    g = r.GetOutput()

    if artifact.association == "cell":
        dsa = g.GetCellData()
    elif artifact.association == "point":
        dsa = g.GetPointData()
    else:
        raise ValueError(artifact.association)

    birth_a = dsa.GetArray("Birth")
    pers_a = dsa.GetArray("Persistence")
    type_a = dsa.GetArray("PairType")
    id_a = dsa.GetArray("PairIdentifier")

    n = birth_a.GetNumberOfTuples()

    out: Dict[int, List[Tuple[float, float]]] = {0: [], 1: []}

    for i in range(n):
        birth = float(birth_a.GetTuple1(i))
        pers = float(pers_a.GetTuple1(i))
        pair_type = int(round(float(type_a.GetTuple1(i))))
        pair_id = int(round(float(id_a.GetTuple1(i))))

        if pair_id == -1:
            continue
        if pair_type not in (0, 1):
            continue
        if not (math.isfinite(birth) and math.isfinite(pers)):
            continue

        death = birth + pers
        if not math.isfinite(death):
            continue

        out[pair_type].append((birth, death))

    def arr(values: List[Tuple[float, float]]) -> np.ndarray:
        if not values:
            return np.empty((0, 2), dtype=np.float64)
        a = np.asarray(values, dtype=np.float64)
        return a.reshape((-1, 2))

    return PD(d0=arr(out[0]), d1=arr(out[1]))


def finite_pd_from_gudhi(persistence) -> PD:
    out: Dict[int, List[Tuple[float, float]]] = {0: [], 1: []}

    for dim, pair in persistence:
        if dim not in (0, 1):
            continue

        b, d = float(pair[0]), float(pair[1])

        if math.isfinite(b) and math.isfinite(d):
            out[dim].append((b, d))

    def arr(values):
        if not values:
            return np.empty((0, 2), dtype=np.float64)
        return np.asarray(values, dtype=np.float64).reshape((-1, 2))

    return PD(d0=arr(out[0]), d1=arr(out[1]))


def exact_bottleneck(a: np.ndarray, b: np.ndarray) -> float:
    return float(gd.bottleneck_distance(a, b, e=0.0))


def exact_w2(
    a: np.ndarray,
    b: np.ndarray,
    internal_p,
) -> float:
    return float(
        wasserstein_distance(
            a,
            b,
            matching=False,
            order=2.0,
            internal_p=internal_p,
            keep_essential_parts=False,
        )
    )


def compute_reference_metrics(pd_gt: PD, pd_sr: PD) -> Dict[str, float]:
    d0_b = exact_bottleneck(pd_gt.d0, pd_sr.d0)
    d1_b = exact_bottleneck(pd_gt.d1, pd_sr.d1)

    d0_wi = exact_w2(pd_gt.d0, pd_sr.d0, np.inf)
    d1_wi = exact_w2(pd_gt.d1, pd_sr.d1, np.inf)

    d0_w2 = exact_w2(pd_gt.d0, pd_sr.d0, 2.0)
    d1_w2 = exact_w2(pd_gt.d1, pd_sr.d1, 2.0)

    return {
        "dB_D0": d0_b,
        "dB_D1": d1_b,
        "dB": max(d0_b, d1_b),
        "W2inf_D0": d0_wi,
        "W2inf_D1": d1_wi,
        "W2inf": math.hypot(d0_wi, d1_wi),
        "W22_D0": d0_w2,
        "W22_D1": d1_w2,
        "W22": math.hypot(d0_w2, d1_w2),
    }


def compute_colleague_native_metrics(
    pd_gt: PD,
    pd_sr: PD,
    colleague_bottleneck,
    colleague_wasserstein,
) -> Dict[str, float]:
    # Native current wrapper behavior:
    # - bottleneck: no explicit e
    # - Wasserstein default order=1
    # - order=2 uses GUDHI's default internal_p (L-infinity)
    vals = {}

    for dim in (0, 1):
        a = pd_gt.by_dim(dim)
        b = pd_sr.by_dim(dim)

        vals[f"native_dB_D{dim}"] = float(
            colleague_bottleneck(a, b)
        )
        vals[f"native_W1inf_D{dim}"] = float(
            colleague_wasserstein(a, b)
        )
        vals[f"native_W2inf_D{dim}"] = float(
            colleague_wasserstein(a, b, order=2)
        )

    vals["native_dB"] = max(vals["native_dB_D0"], vals["native_dB_D1"])
    vals["native_W1inf"] = math.hypot(
        vals["native_W1inf_D0"], vals["native_W1inf_D1"]
    )
    vals["native_W2inf"] = math.hypot(
        vals["native_W2inf_D0"], vals["native_W2inf_D1"]
    )

    return vals


def read_frozen_sample_metrics(
    joined_csv: Path,
    sample: int,
) -> Dict[str, Dict[str, float]]:
    with joined_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    out = {}

    for method in METHODS:
        target_id = METHOD_ID[method]

        hits = [
            r for r in rows
            if int(r["sample_idx"]) == sample
            and r["method_id"] == target_id
        ]

        if len(hits) != 1:
            raise RuntimeError(
                f"Frozen row lookup method={method} id={target_id}: "
                f"got {len(hits)}"
            )

        r = hits[0]
        out[method] = {
            "dB": float(r["dB"]),
            "W2inf": float(r["W2inf"]),
            "W22": float(r["W22"]),
        }

    return out


def locate_sample_position(idx: np.ndarray, sample: int, n: int) -> int:
    x = np.asarray(idx).reshape(-1)
    hits = np.where(x.astype(np.int64) == sample)[0]

    if len(hits) == 1:
        pos = int(hits[0])
        if pos >= n:
            raise RuntimeError("idx position outside data array")
        return pos

    if len(hits) > 1:
        raise RuntimeError(f"sample {sample} occurs multiple times in idx")

    if n > sample:
        print(
            f"WARNING: sample {sample} not found in idx; "
            f"using direct position {sample}"
        )
        return sample

    raise RuntimeError(f"Could not locate sample {sample}")


def speed_field(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x)

    if a.ndim == 2:
        s = a.astype(np.float64, copy=False)
    elif a.ndim == 3 and a.shape[-1] == 2:
        s = np.sqrt(
            a[..., 0].astype(np.float64) ** 2
            + a[..., 1].astype(np.float64) ** 2
        )
    else:
        raise RuntimeError(f"Unexpected field shape {a.shape}")

    if s.shape[0] < PATCH or s.shape[1] < PATCH:
        raise RuntimeError(f"Field too small: {s.shape}")

    return np.ascontiguousarray(s[:PATCH, :PATCH])


def load_fields(
    root: Path,
    sample: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    gt_by_method = {}
    sr_by_method = {}

    for method in METHODS:
        d = root / FIELD_DIRS_REL[method]

        idx = np.load(d / "idx.npy")
        gt = np.load(d / "dataGT.npy", mmap_mode="r")
        sr = np.load(d / "dataSR.npy", mmap_mode="r")

        if len(gt) != len(sr):
            raise RuntimeError(f"{method}: GT/SR count mismatch")

        pos = locate_sample_position(idx, sample, len(gt))

        gt_by_method[method] = speed_field(gt[pos])
        sr_by_method[method] = speed_field(sr[pos])

    gt_ref = gt_by_method["cnn"]

    for method in ("uv", "f1"):
        if not np.array_equal(gt_ref, gt_by_method[method]):
            maxdiff = float(
                np.max(np.abs(gt_ref - gt_by_method[method]))
            )
            raise RuntimeError(
                f"Authoritative GT mismatch cnn vs {method}: {maxdiff}"
            )

    return gt_ref, sr_by_method


def phase_a(
    outdir: Path,
    ttk_pds: Mapping[str, Mapping[str, PD]],
    frozen: Mapping[str, Mapping[str, float]],
    colleague_bottleneck,
    colleague_wasserstein,
) -> None:
    print()
    print("=" * 110)
    print("PHASE A — SAME-DIAGRAM METRIC PARITY")
    print("=" * 110)

    rows = []
    tol_frozen = 1e-8
    tol_wrapper = 1e-8

    all_pass = True

    for method in METHODS:
        gt = ttk_pds[method]["GT"]
        sr = ttk_pds[method]["SR"]

        ref = compute_reference_metrics(gt, sr)
        native = compute_colleague_native_metrics(
            gt, sr, colleague_bottleneck, colleague_wasserstein
        )

        frozen_diff = {
            k: abs(ref[k] - frozen[method][k])
            for k in ("dB", "W2inf", "W22")
        }

        wrapper_db_diff = abs(native["native_dB"] - ref["dB"])
        wrapper_w2i_diff = abs(
            native["native_W2inf"] - ref["W2inf"]
        )

        ok = (
            all(v <= tol_frozen for v in frozen_diff.values())
            and wrapper_db_diff <= tol_wrapper
            and wrapper_w2i_diff <= tol_wrapper
        )

        all_pass &= ok

        row = {
            "method": method,
            "display_name": DISPLAY[method],
            "ttk_gt_D0_count": len(gt.d0),
            "ttk_gt_D1_count": len(gt.d1),
            "ttk_sr_D0_count": len(sr.d0),
            "ttk_sr_D1_count": len(sr.d1),
            "reference_dB": ref["dB"],
            "reference_W2inf": ref["W2inf"],
            "reference_W22": ref["W22"],
            "frozen_dB": frozen[method]["dB"],
            "frozen_W2inf": frozen[method]["W2inf"],
            "frozen_W22": frozen[method]["W22"],
            "frozen_dB_abs_diff": frozen_diff["dB"],
            "frozen_W2inf_abs_diff": frozen_diff["W2inf"],
            "frozen_W22_abs_diff": frozen_diff["W22"],
            "colleague_native_dB": native["native_dB"],
            "colleague_native_W1inf": native["native_W1inf"],
            "colleague_native_W2inf": native["native_W2inf"],
            "colleague_native_dB_abs_diff_vs_reference": wrapper_db_diff,
            "colleague_native_W2inf_abs_diff_vs_reference": wrapper_w2i_diff,
            # Current colleague wrapper has no internal_p argument.
            # This is the parity extension we'd add:
            "colleague_parity_extension_W22": ref["W22"],
            "pass": int(ok),
        }
        rows.append(row)

        print()
        print(DISPLAY[method])
        print("-" * 110)
        print(
            f"canonical counts GT D0/D1={len(gt.d0)}/{len(gt.d1)}; "
            f"SR={len(sr.d0)}/{len(sr.d1)}"
        )
        print(
            f"reference exact: dB={ref['dB']:.15g}, "
            f"W2inf={ref['W2inf']:.15g}, W22={ref['W22']:.15g}"
        )
        print(
            f"frozen audit:   dB={frozen[method]['dB']:.15g}, "
            f"W2inf={frozen[method]['W2inf']:.15g}, "
            f"W22={frozen[method]['W22']:.15g}"
        )
        print(
            f"colleague native: dB={native['native_dB']:.15g}, "
            f"W1inf={native['native_W1inf']:.15g}, "
            f"W2inf(order=2)={native['native_W2inf']:.15g}"
        )
        print(
            f"diffs: native dB vs exact={wrapper_db_diff:.3e}; "
            f"native W2inf vs exact={wrapper_w2i_diff:.3e}"
        )
        print("PASS =", int(ok))

    csv_path = outdir / "phaseA_same_diagram_metric_parity.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print()
    print("Phase A CSV:", csv_path)

    if not all_pass:
        raise RuntimeError(
            "PHASE A FAILED: do not proceed to descriptor-construction conclusions."
        )

    print("PHASE A SAME-DIAGRAM METRIC PARITY: PASS")


def phase_b(
    outdir: Path,
    toolkit_compute_cubical,
    ttk_pds: Mapping[str, Mapping[str, PD]],
    gt_field: np.ndarray,
    sr_fields: Mapping[str, np.ndarray],
) -> None:
    print()
    print("=" * 110)
    print("PHASE B — INDEPENDENT GUDHI CUBICAL PD CONSTRUCTION")
    print("=" * 110)

    fields = {"GT": gt_field}
    for method in METHODS:
        fields[method] = sr_fields[method]

    gudhi_pds: Dict[str, PD] = {}

    for name, field in fields.items():
        cc = toolkit_compute_cubical(field)
        gudhi_pds[name] = finite_pd_from_gudhi(cc.persistence())

        pd = gudhi_pds[name]
        print(
            f"{name:>3s}: cubical finite counts "
            f"D0={len(pd.d0)}, D1={len(pd.d1)}"
        )

    np.savez_compressed(
        outdir / "phaseB_colleague_cubical_pd_points_sample69.npz",
        GT_D0=gudhi_pds["GT"].d0,
        GT_D1=gudhi_pds["GT"].d1,
        CNN_D0=gudhi_pds["cnn"].d0,
        CNN_D1=gudhi_pds["cnn"].d1,
        UV_D0=gudhi_pds["uv"].d0,
        UV_D1=gudhi_pds["uv"].d1,
        F1_D0=gudhi_pds["f1"].d0,
        F1_D1=gudhi_pds["f1"].d1,
    )

    distance_rows = []

    for method in METHODS:
        metrics = compute_reference_metrics(
            gudhi_pds["GT"], gudhi_pds[method]
        )

        row = {
            "method": method,
            "display_name": DISPLAY[method],
            **metrics,
        }
        distance_rows.append(row)

        print()
        print(DISPLAY[method])
        print(
            f"cubical GT->SR: dB={metrics['dB']:.15g}, "
            f"W2inf={metrics['W2inf']:.15g}, "
            f"W22={metrics['W22']:.15g}"
        )

    dist_csv = outdir / "phaseB_colleague_cubical_GT_to_method_distances.csv"
    with dist_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=list(distance_rows[0].keys())
        )
        w.writeheader()
        w.writerows(distance_rows)

    # Cardinality and same-field cross-backend descriptor distances.
    cross_rows = []

    for method in METHODS:
        for side, field_key in (("GT", "GT"), ("SR", method)):
            ttk_pd = ttk_pds[method][side]
            cub_pd = gudhi_pds[field_key]
            cross = compute_reference_metrics(ttk_pd, cub_pd)

            cross_rows.append({
                "method": method,
                "side": side,
                "ttk_D0_count": len(ttk_pd.d0),
                "ttk_D1_count": len(ttk_pd.d1),
                "cubical_D0_count": len(cub_pd.d0),
                "cubical_D1_count": len(cub_pd.d1),
                "cross_backend_dB": cross["dB"],
                "cross_backend_W2inf": cross["W2inf"],
                "cross_backend_W22": cross["W22"],
                "cross_dB_D0": cross["dB_D0"],
                "cross_dB_D1": cross["dB_D1"],
                "cross_W2inf_D0": cross["W2inf_D0"],
                "cross_W2inf_D1": cross["W2inf_D1"],
                "cross_W22_D0": cross["W22_D0"],
                "cross_W22_D1": cross["W22_D1"],
            })

    cross_csv = outdir / "phaseB_same_field_TTK_vs_cubical_descriptor_comparison.csv"
    with cross_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=list(cross_rows[0].keys())
        )
        w.writeheader()
        w.writerows(cross_rows)

    # Ranking summary.
    by_method = {r["method"]: r for r in distance_rows}

    ranking_lines = []
    ranking_lines.append(
        "PHASE B SAMPLE-69 GUDHI CUBICAL METHOD RANKINGS (lower is better)"
    )
    ranking_lines.append("=" * 92)

    for metric in ("dB", "W2inf", "W22"):
        ranking = sorted(
            METHODS,
            key=lambda m: float(by_method[m][metric]),
        )
        ranking_lines.append(
            f"{metric}: "
            + " < ".join(
                f"{DISPLAY[m]} ({by_method[m][metric]:.8g})"
                for m in ranking
            )
        )

    ranking_txt = outdir / "phaseB_colleague_cubical_ranking_summary.txt"
    ranking_txt.write_text("\n".join(ranking_lines) + "\n")

    print()
    print("\n".join(ranking_lines))
    print()
    print("Phase B distance CSV:", dist_csv)
    print("Phase B cross-backend CSV:", cross_csv)
    print("Phase B ranking summary:", ranking_txt)
    print()
    print(
        "PHASE B COMPLETE. Differences from TTK are descriptor-construction "
        "differences, not metric-parity failures."
    )


def main() -> int:
    args = parse_args()

    root = args.root.expanduser().resolve()
    w22 = args.w22.expanduser().resolve()
    sample = int(args.sample)

    if args.toolkit_root is None:
        toolkit_root = (
            root / "third_party" / "tda-toolkit-mapper"
        )
    else:
        toolkit_root = args.toolkit_root.expanduser().resolve()

    if args.out is None:
        outdir = (
            w22
            / "pd_colleague_compatibility"
            / f"sample_{sample:03d}"
        )
    else:
        outdir = args.out.expanduser().resolve()

    outdir.mkdir(parents=True, exist_ok=True)

    (
        toolkit,
        colleague_bottleneck,
        colleague_wasserstein,
        colleague_cubical,
    ) = import_colleague(toolkit_root)

    print("PD COMPATIBILITY AUDIT")
    print("=" * 110)
    print("python:", sys.executable)
    print("VTK:", vtk.vtkVersion.GetVTKVersion())
    print("GUDHI:", gd.__version__)
    print("colleague toolkit:", toolkit.__file__)
    print("colleague version:", getattr(toolkit, "__version__", "unknown"))
    print("sample:", sample)
    print("output:", outdir)

    ttk_pds: Dict[str, Dict[str, PD]] = {}

    artifact_rows = []

    for method in METHODS:
        topo_root = root / TOPO_ROOTS_REL[method]
        arts = discover_pd_artifacts(root, topo_root, sample)

        ttk_pds[method] = {}

        for side in ("GT", "SR"):
            art = arts[side]
            pd = load_ttk_pd(art)
            ttk_pds[method][side] = pd

            artifact_rows.append({
                "method": method,
                "side": side,
                "path": str(art.path),
                "association": art.association,
                "D0_count": len(pd.d0),
                "D1_count": len(pd.d1),
            })

            print(
                f"{method:>3s} {side}: "
                f"{art.association:>5s} {art.path}"
            )
            print(
                f"        canonical finite counts "
                f"D0={len(pd.d0)}, D1={len(pd.d1)}"
            )

    artifact_csv = outdir / "discovered_ttk_pd_artifacts.csv"
    with artifact_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=list(artifact_rows[0].keys())
        )
        w.writeheader()
        w.writerows(artifact_rows)

    joined = w22 / "corrected_pd_mt" / "corrected_pd_mt_joined.csv"

    if not joined.is_file():
        raise FileNotFoundError(joined)

    frozen = read_frozen_sample_metrics(joined, sample)

    phase_a(
        outdir,
        ttk_pds,
        frozen,
        colleague_bottleneck,
        colleague_wasserstein,
    )

    gt_field, sr_fields = load_fields(root, sample)

    print()
    print("AUTHORITATIVE SAMPLE FIELD CHECK")
    print("-" * 110)
    print("CNN == UV == F1 GT exactly: PASS")
    print(
        f"GT range: {float(gt_field.min()):.8g} "
        f"to {float(gt_field.max()):.8g}"
    )

    for method in METHODS:
        f = sr_fields[method]
        print(
            f"{DISPLAY[method]} range: "
            f"{float(f.min()):.8g} to {float(f.max()):.8g}"
        )

    phase_b(
        outdir,
        colleague_cubical,
        ttk_pds,
        gt_field,
        sr_fields,
    )

    summary = outdir / "README_RESULTS.txt"
    summary.write_text(
        "Sample-69 PD compatibility audit completed.\n\n"
        "Phase A tests same-diagram metric parity.\n"
        "Phase B tests independent GUDHI cubical PD construction.\n\n"
        "Interpretation rule:\n"
        "  Phase A differences indicate metric/default mismatch.\n"
        "  Phase B differences may reflect the different filtered complex / "
        "cell convention and are not a Phase-A failure.\n"
    )

    print()
    print("=" * 110)
    print("SAMPLE-69 PD COMPATIBILITY AUDIT: COMPLETE")
    print("=" * 110)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
