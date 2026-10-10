#!/usr/bin/env python3

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


# ============================================================================
# Configuration
# ============================================================================

HOME = Path.home()
REPO = HOME / "PhIRE"
W22 = Path(os.environ["W22"])

SAMPLE = 69
PATCH = 160

OUT = (
    W22
    / "corrected_pd_mt"
    / "discordance_visuals"
    / f"sample_{SAMPLE:03d}"
)

INPUT_DIR = OUT / "inputs"
INPUT_DIR.mkdir(parents=True, exist_ok=True)


METHODS = {
    "cnn": {
        "display_name": "CNN",
        "idx": REPO / "data_out_fixed/wind_mrhr_cnn/idx.npy",
        "gt": REPO / "data_out_fixed/wind_mrhr_cnn/dataGT.npy",
        "sr": REPO / "data_out_fixed/wind_mrhr_cnn/dataSR.npy",
    },

    "uv": {
        "display_name": "Ablation (Luv only)",
        "idx": (
            REPO
            / "data_out/wind_finetune_candidateUV_expanded2688/idx.npy"
        ),
        "gt": (
            REPO
            / "data_out/wind_finetune_candidateUV_expanded2688/dataGT.npy"
        ),
        "sr": (
            REPO
            / "data_out/wind_finetune_candidateUV_expanded2688/dataSR.npy"
        ),
    },

    "f1": {
        "display_name": "Candidate F (grad + repaired E2-low)",
        "idx": (
            REPO
            / "data_out/wind_finetune_candidateF_grad_E2_low_expanded2688/"
              "idx.npy"
        ),
        "gt": (
            REPO
            / "data_out/wind_finetune_candidateF_grad_E2_low_expanded2688/"
              "dataGT.npy"
        ),
        "sr": (
            REPO
            / "data_out/wind_finetune_candidateF_grad_E2_low_expanded2688/"
              "dataSR.npy"
        ),
    },
}


# ============================================================================
# Helpers
# ============================================================================

def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def load_sample(path: Path, idx_path: Path, sample_idx: int):
    if not path.is_file():
        raise FileNotFoundError(path)

    if not idx_path.is_file():
        raise FileNotFoundError(idx_path)

    idx = np.load(idx_path)

    where = np.where(
        np.asarray(idx).reshape(-1) == sample_idx
    )[0]

    if len(where) != 1:
        raise RuntimeError(
            f"{idx_path}: sample {sample_idx} occurs {len(where)} times"
        )

    arr = np.load(path, mmap_mode="r")

    if len(arr) != len(idx):
        raise RuntimeError(
            f"array/index length mismatch: {path}"
        )

    sample = np.asarray(arr[int(where[0])])

    if (
        sample.ndim != 3
        or sample.shape[-1] != 2
    ):
        raise RuntimeError(
            f"expected HxWx2 array; got {sample.shape} from {path}"
        )

    if sample.shape[0] < PATCH or sample.shape[1] < PATCH:
        raise RuntimeError(
            f"sample too small for {PATCH}x{PATCH}: {sample.shape}"
        )

    if not np.all(np.isfinite(sample)):
        raise RuntimeError(
            f"non-finite values in {path}"
        )

    return sample


def speed_crop(uv):
    # Match the current corrected writer's float32 scalarization.
    u = uv[..., 0].astype(np.float32)
    v = uv[..., 1].astype(np.float32)

    speed = np.sqrt(
        u ** 2 + v ** 2
    ).astype(np.float32)

    return np.ascontiguousarray(
        speed[:PATCH, :PATCH]
    )


def write_speed_vti(speed_2d: np.ndarray, path: Path):
    if speed_2d.shape != (PATCH, PATCH):
        raise RuntimeError(
            f"unexpected speed shape: {speed_2d.shape}"
        )

    H, W = speed_2d.shape

    img = vtk.vtkImageData()

    # Corrected spatial convention:
    # VTK x dimension = W, y dimension = H.
    img.SetDimensions(W, H, 1)
    img.SetOrigin(0.0, 0.0, 0.0)
    img.SetSpacing(1.0, 1.0, 1.0)

    # Current authoritative convention:
    # scalar[y, x] -> flat[x + y*W].
    flat = (
        np.ascontiguousarray(speed_2d)
        .ravel(order="C")
    )

    vtk_arr = numpy_to_vtk(
        flat,
        deep=True,
        array_type=vtk.VTK_FLOAT,
    )

    vtk_arr.SetName("wind_speed")

    img.GetPointData().SetScalars(vtk_arr)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(img)
    writer.SetDataModeToBinary()

    status = writer.Write()

    if status != 1:
        raise RuntimeError(
            f"VTK writer failed: {path}"
        )


def read_speed_vti(path: Path):
    r = vtk.vtkXMLImageDataReader()
    r.SetFileName(str(path))
    r.Update()

    img = r.GetOutput()

    if img is None:
        raise RuntimeError(
            f"failed to read {path}"
        )

    arr = img.GetPointData().GetArray("wind_speed")

    if arr is None:
        raise RuntimeError(
            f"wind_speed missing from {path}"
        )

    return vtk_to_numpy(arr).astype(
        np.float32,
        copy=False,
    )


# ============================================================================
# Load all authoritative fields
# ============================================================================

print("=" * 100)
print("SAMPLE 69 AUTHORITATIVE MT VISUALIZATION INPUT PREPARATION")
print("=" * 100)

loaded = {}

for method_id, info in METHODS.items():

    loaded[method_id] = {
        "gt": load_sample(
            info["gt"],
            info["idx"],
            SAMPLE,
        ),

        "sr": load_sample(
            info["sr"],
            info["idx"],
            SAMPLE,
        ),
    }

    print(
        f"{method_id:4s}: "
        f"GT={loaded[method_id]['gt'].shape}, "
        f"SR={loaded[method_id]['sr'].shape}"
    )


# ============================================================================
# GT identity check
# ============================================================================

print()
print("=" * 100)
print("GT IDENTITY CHECK")
print("=" * 100)

canonical_gt = loaded["cnn"]["gt"]

for method_id in ["uv", "f1"]:

    other = loaded[method_id]["gt"]

    exact = np.array_equal(
        canonical_gt,
        other,
    )

    diff = np.abs(
        canonical_gt.astype(np.float64)
        - other.astype(np.float64)
    )

    max_diff = float(
        np.max(diff)
    )

    print(
        f"CNN GT vs {method_id:4s}: "
        f"exact={exact}, "
        f"max_abs_diff={max_diff:.3e}"
    )

    if not exact:
        raise RuntimeError(
            f"GT mismatch CNN vs {method_id}"
        )

print("GT identity across CNN / UV / Candidate F: PASS")


# ============================================================================
# Construct scalar inputs
# ============================================================================

fields = {
    "gt": speed_crop(
        canonical_gt
    ),

    "cnn": speed_crop(
        loaded["cnn"]["sr"]
    ),

    "uv": speed_crop(
        loaded["uv"]["sr"]
    ),

    "f1": speed_crop(
        loaded["f1"]["sr"]
    ),
}


# ============================================================================
# Write C-order VTI files and verify round-trip
# ============================================================================

print()
print("=" * 100)
print("C-ORDER VTI GENERATION")
print("=" * 100)

manifest = {
    "sample_idx": SAMPLE,
    "patch": PATCH,
    "scalar": "wind_speed = sqrt(u^2 + v^2)",
    "crop": "[0:160, 0:160]",
    "vtk_dimensions": [PATCH, PATCH, 1],
    "flatten_order": "C",
    "methods": {},
}

for method_id, speed in fields.items():

    path = (
        INPUT_DIR
        / f"{method_id}_s{SAMPLE:03d}.vti"
    )

    write_speed_vti(
        speed,
        path,
    )

    actual = read_speed_vti(path)

    ref_c = (
        np.ascontiguousarray(speed)
        .ravel(order="C")
    )

    ref_f = (
        np.ascontiguousarray(speed)
        .ravel(order="F")
    )

    c_exact = np.array_equal(
        actual,
        ref_c,
    )

    f_exact = np.array_equal(
        actual,
        ref_f,
    )

    c_max = float(
        np.max(
            np.abs(
                actual.astype(np.float64)
                - ref_c.astype(np.float64)
            )
        )
    )

    f_mean = float(
        np.mean(
            np.abs(
                actual.astype(np.float64)
                - ref_f.astype(np.float64)
            )
        )
    )

    if not c_exact:
        raise RuntimeError(
            f"C-order round-trip failed: {path}"
        )

    print()
    print(method_id)
    print("-" * 60)
    print("path:", path)
    print("shape:", speed.shape)
    print(
        "speed range:",
        float(np.min(speed)),
        float(np.max(speed)),
    )
    print("C-order exact:", c_exact)
    print("C-order max diff:", c_max)
    print("F-order exact:", f_exact)
    print("F-order mean diff:", f_mean)
    print("SHA256:", sha256(path))

    manifest["methods"][method_id] = {
        "vti": str(path),
        "sha256": sha256(path),
        "speed_min": float(np.min(speed)),
        "speed_max": float(np.max(speed)),
        "c_order_exact_roundtrip": bool(c_exact),
        "f_order_exact_roundtrip": bool(f_exact),
        "f_order_mean_difference": f_mean,
    }


# ============================================================================
# Source provenance
# ============================================================================

manifest["source_paths"] = {
    method_id: {
        key: str(value)
        for key, value in info.items()
        if key in {"idx", "gt", "sr"}
    }
    for method_id, info in METHODS.items()
}


manifest_path = (
    OUT
    / "authoritative_input_manifest.json"
)

manifest_path.write_text(
    json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
    )
    + "\n"
)


sha_path = (
    OUT
    / "authoritative_inputs_sha256.txt"
)

with sha_path.open("w") as f:

    for method_id in [
        "gt",
        "cnn",
        "uv",
        "f1",
    ]:

        path = (
            INPUT_DIR
            / f"{method_id}_s{SAMPLE:03d}.vti"
        )

        f.write(
            f"{sha256(path)}  {path}\n"
        )

    f.write(
        f"{sha256(manifest_path)}  "
        f"{manifest_path}\n"
    )


print()
print("=" * 100)
print("AUTHORITATIVE INPUT PREPARATION: PASS")
print("=" * 100)

print("output:", OUT)
print("manifest:", manifest_path)
print("SHA manifest:", sha_path)
