#!/usr/bin/env python3
"""Export 160x160 wind-speed fields (GT, CNN, UV, F1) as .npy files for the TDA toolkit UI.

Uses the same conventions as the topology evaluation:
  - top-left crop [:patch, :patch]  (x0 = 0, y0 = 0)
  - wind speed = hypot(u, v)
  - C-order arrays, field[y, x]
  - float32, matching the values TTK reads from the VTIs

Run on Spark from ~/PhIRE:
  PYTHONNOUSERSITE=1 /usr/bin/python3 export_example_fields.py --samples 69
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOTS = {
    "CNN": "data_out_fixed/wind_mrhr_cnn",
    "UV": "data_out/wind_finetune_candidateUV_expanded2688",
    "F1": "data_out/wind_finetune_candidateF_grad_E2_low_expanded2688",
}


def load(root: Path, name: str) -> np.ndarray:
    path = root / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return np.load(path, mmap_mode="r")


def position_of(idx: np.ndarray, sample: int) -> int:
    hits = np.flatnonzero(np.asarray(idx) == sample)
    if hits.size != 1:
        raise ValueError(f"sample {sample} found {hits.size} times in idx.npy")
    return int(hits[0])


def speed_crop(arr: np.ndarray, pos: int, patch: int) -> np.ndarray:
    if arr.ndim != 4 or arr.shape[-1] != 2:
        raise ValueError(f"expected (N, H, W, 2) u/v array, got {arr.shape}")
    uv = np.asarray(arr[pos, :patch, :patch, :], dtype=np.float64)
    return np.ascontiguousarray(np.hypot(uv[..., 0], uv[..., 1]).astype(np.float32))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phire", default="~/PhIRE", help="PhIRE repo root (default ~/PhIRE)")
    ap.add_argument("--samples", type=int, nargs="+", default=[69], help="benchmark sample ids (default 69)")
    ap.add_argument("--patch", type=int, default=160, help="crop size (default 160)")
    ap.add_argument("--out", default="~/demi_examples", help="output folder (default ~/demi_examples)")
    args = ap.parse_args()

    phire = Path(args.phire).expanduser()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    arrays = {}
    for name, rel in ROOTS.items():
        root = phire / rel
        arrays[name] = {
            "idx": np.asarray(load(root, "idx.npy")),
            "GT": load(root, "dataGT.npy"),
            "SR": load(root, "dataSR.npy"),
            "root": str(root),
        }
        print(f"{name:3s} {root}  GT {arrays[name]['GT'].shape}  SR {arrays[name]['SR'].shape}")

    manifest = {"crop": f"[:{args.patch}, :{args.patch}] (x0=0, y0=0)", "scalar": "hypot(u, v)",
                "dtype": "float32", "layout": "C-order field[y, x]", "roots": {k: v["root"] for k, v in arrays.items()},
                "files": []}

    for s in args.samples:
        gts = {}
        for name, a in arrays.items():
            pos = position_of(a["idx"], s)
            gts[name] = speed_crop(a["GT"], pos, args.patch)
            sr = speed_crop(a["SR"], pos, args.patch)
            p = out / f"s{s:03d}_{name}_SR_speed{args.patch}.npy"
            np.save(p, sr)
            manifest["files"].append({"file": p.name, "sample": s, "field": f"{name} SR",
                                      "shape": list(sr.shape), "min": float(sr.min()), "max": float(sr.max()),
                                      "sha256": sha256(p)})

        ref = gts["CNN"]
        for name, g in gts.items():
            if not np.array_equal(g, ref):
                raise RuntimeError(f"sample {s}: GT from {name} root differs from CNN root")

        p = out / f"s{s:03d}_GT_speed{args.patch}.npy"
        np.save(p, ref)
        manifest["files"].append({"file": p.name, "sample": s, "field": "GT (identical across roots)",
                                  "shape": list(ref.shape), "min": float(ref.min()), "max": float(ref.max()),
                                  "sha256": sha256(p)})

        # One middle row of the GT field, for the toolkit's 1D merge-tree view
        row = args.patch // 2
        p = out / f"s{s:03d}_GT_row{row}_1d.npy"
        np.save(p, np.ascontiguousarray(ref[row, :]))
        manifest["files"].append({"file": p.name, "sample": s, "field": f"GT row y={row} (1D)",
                                  "shape": [args.patch], "sha256": sha256(p)})
        print(f"sample {s}: GT identical across CNN/UV/F1 roots; wrote GT, 3 SR fields and a 1D row")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
