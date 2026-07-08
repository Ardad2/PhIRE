#!/usr/bin/env python3
"""
Generate pretrained-CNN SR/GT/IN/idx arrays for an expanded seasonal dataset,
for use as training input to the PyTorch RefinerNet candidates (D/Dpd/E/E2).

This is a thin, parameterized wrapper around the exact procedure already used
(as an inline heredoc, not previously a committed script) to produce
data_out/wind_mrhr_cnn_expanded672/ -- see docs/candidateE2_expanded672_notes.md
and docs/candidateDpd_expanded672_notes.md, "Step 0".

Method choice (verified, not guessed):
    PhIREGANs.pretrain()     trains/updates the generator's weights
                              (calls optimizer.minimize(model.g_loss, ...)).
                              WRONG for this purpose -- would corrupt the
                              pretrained checkpoint.
    PhIREGANs.test_paired()  restores a saved checkpoint via
                              tf.train.Saver.restore() and only runs forward
                              passes (sess.run(model.x_SR, ...)). No optimizer,
                              no gradient updates. This is what the existing
                              672 recipe uses, and what this script uses.

Must be run under TF1 (tensorflow.compat.v1), the same native environment used
for the existing wind_mrhr_cnn_expanded672 generation -- NOT the PyTorch
.mamba_candidateD_pd / .venv_candidateD_pd environment used by the RefinerNet
scripts.

Usage (1344):
    python3 scripts/generate_expanded_cnn_sr.py \\
        --data-type wind_mrhr_cnn_expanded1344 \\
        --tfrecord example_data_topology_expanded_1344/wind_MR-HR.tfrecord \\
        --n-expected 1344 \\
        --out-dir data_out/wind_mrhr_cnn_expanded1344

Usage (2688):
    python3 scripts/generate_expanded_cnn_sr.py \\
        --data-type wind_mrhr_cnn_expanded2688 \\
        --tfrecord example_data_topology_expanded_2688/wind_MR-HR.tfrecord \\
        --n-expected 2688 \\
        --out-dir data_out/wind_mrhr_cnn_expanded2688

Output (matches the existing 672 convention exactly):
    <out-dir>/idx.npy     (N,)
    <out-dir>/dataIN.npy  (N, 100, 100, 2)  -- physical [u, v], LR input
    <out-dir>/dataGT.npy  (N, 500, 500, 2)  -- physical [u, v], HR ground truth
    <out-dir>/dataSR.npy  (N, 500, 500, 2)  -- physical [u, v], pretrained CNN SR
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Established convention (matches the 672 recipe exactly; do not change without
# re-deriving mu/sig, since a mismatch here silently mis-normalizes every
# downstream RefinerNet loss).
DEFAULT_MU  = [0.7684, -0.4575]
DEFAULT_SIG = [5.02455, 5.9017]
DEFAULT_MODEL_PATH = "models/wind_mr-hr/trained_cnn/cnn"
DEFAULT_R = [5]
DEFAULT_BATCH_SIZE = 1

# Rough per-sample disk cost at (500,500,2) HR + (100,100,2) LR/IN, float32,
# for dataGT + dataSR + dataIN combined. Used only for the preflight
# disk-space estimate below; not a hard physical constant.
_BYTES_PER_SAMPLE_HR = 500 * 500 * 2 * 4      # dataGT.npy or dataSR.npy, one sample
_BYTES_PER_SAMPLE_LR = 100 * 100 * 2 * 4      # dataIN.npy, one sample
_BYTES_PER_SAMPLE_TOTAL = 2 * _BYTES_PER_SAMPLE_HR + _BYTES_PER_SAMPLE_LR
_DISK_SAFETY_MARGIN = 1.5  # require 1.5x the estimated size free, not just 1.0x


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-type", required=True,
                   help="PhIREGANs data_type label, e.g. wind_mrhr_cnn_expanded1344")
    p.add_argument("--tfrecord", required=True, type=Path,
                   help="Path to the wind_MR-HR.tfrecord to run inference on")
    p.add_argument("--n-expected", required=True, type=int,
                   help="Expected sample count (e.g. 1344, 2688) -- hard-checked "
                        "against the output shape before writing")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Output directory for idx.npy/dataIN.npy/dataGT.npy/dataSR.npy")
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                   help=f"Pretrained CNN checkpoint path (default: {DEFAULT_MODEL_PATH})")
    p.add_argument("--mu", type=float, nargs=2, default=DEFAULT_MU, metavar=("MU_U", "MU_V"),
                   help=f"Normalization mean [u, v] (default: {DEFAULT_MU})")
    p.add_argument("--sig", type=float, nargs=2, default=DEFAULT_SIG, metavar=("SIG_U", "SIG_V"),
                   help=f"Normalization std [u, v] (default: {DEFAULT_SIG})")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--skip-disk-check", action="store_true",
                   help="Skip the preflight free-disk-space check (not recommended)")
    return p


def _check_disk_space(out_dir: Path, n_expected: int, skip: bool) -> None:
    estimated_bytes = n_expected * _BYTES_PER_SAMPLE_TOTAL
    required_bytes = int(estimated_bytes * _DISK_SAFETY_MARGIN)

    check_dir = out_dir if out_dir.exists() else out_dir.parent
    while not check_dir.exists():
        check_dir = check_dir.parent
    free_bytes = shutil.disk_usage(check_dir).free

    est_gib = estimated_bytes / (1024 ** 3)
    req_gib = required_bytes / (1024 ** 3)
    free_gib = free_bytes / (1024 ** 3)
    print(f"[preflight] Estimated output size : {est_gib:.2f} GiB "
          f"({n_expected} samples x (dataGT+dataSR+dataIN))")
    print(f"[preflight] Required free space    : {req_gib:.2f} GiB "
          f"({_DISK_SAFETY_MARGIN}x safety margin)")
    print(f"[preflight] Free space at {check_dir} : {free_gib:.2f} GiB")

    if skip:
        print("[preflight] --skip-disk-check set; not enforcing.")
        return

    if free_bytes < required_bytes:
        sys.exit(
            f"[error] Insufficient free disk space at {check_dir}: "
            f"{free_gib:.2f} GiB free, need >= {req_gib:.2f} GiB "
            f"(estimated output {est_gib:.2f} GiB x {_DISK_SAFETY_MARGIN} margin).\n"
            "  Free up space, point --out-dir at a larger volume, or pass "
            "--skip-disk-check to override (not recommended)."
        )
    print("[preflight] Disk space OK.")


def _check_protected(out_dir: Path) -> None:
    """Refuse to write into any of the fixed/authoritative dataset paths."""
    protected = [
        REPO_ROOT / "data_out_fixed",
        REPO_ROOT / "example_data_fixed",
        REPO_ROOT / "data_out" / "wind_mrhr_cnn_expanded672",
    ]
    for p in protected:
        try:
            out_dir.resolve().relative_to(p.resolve())
            sys.exit(
                f"[error] --out-dir ({out_dir}) would write inside protected "
                f"directory {p}. Aborting."
            )
        except ValueError:
            pass


def main() -> None:
    args = build_arg_parser().parse_args()

    if not args.tfrecord.exists():
        sys.exit(f"[error] TFRecord not found: {args.tfrecord}")

    _check_protected(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _check_disk_space(args.out_dir, args.n_expected, args.skip_disk_check)

    print("=" * 64)
    print("Generate expanded pretrained-CNN SR arrays (test_paired, inference-only)")
    print("=" * 64)
    print(f"  data_type    : {args.data_type}")
    print(f"  tfrecord     : {args.tfrecord}")
    print(f"  n_expected   : {args.n_expected}")
    print(f"  out_dir      : {args.out_dir}")
    print(f"  model_path   : {args.model_path}")
    print(f"  mu, sig      : {args.mu}, {args.sig}")
    print(f"  batch_size   : {args.batch_size}")
    print()

    sys.path.insert(0, str(REPO_ROOT))
    import tensorflow.compat.v1 as tf  # noqa: E402
    tf.disable_v2_behavior()
    from PhIREGANs import PhIREGANs  # noqa: E402
    import numpy as np  # noqa: E402

    phire = PhIREGANs(
        data_type=args.data_type,
        mu_sig=[list(args.mu), list(args.sig)],
    )
    phire.set_data_out_path(str(args.out_dir))
    phire.test_paired(
        r=DEFAULT_R,
        data_path=str(args.tfrecord),
        model_path=args.model_path,
        batch_size=args.batch_size,
        save_inputs=True,
    )

    print("\n[validate] Checking output shapes …")
    expected = {
        "idx.npy":    (args.n_expected,),
        "dataIN.npy": (args.n_expected, 100, 100, 2),
        "dataGT.npy": (args.n_expected, 500, 500, 2),
        "dataSR.npy": (args.n_expected, 500, 500, 2),
    }
    all_ok = True
    for fname, exp_shape in expected.items():
        p = args.out_dir / fname
        if not p.exists():
            print(f"  [FAIL] {fname}: NOT FOUND")
            all_ok = False
            continue
        arr = np.load(p, mmap_mode="r")
        if arr.shape != exp_shape:
            print(f"  [FAIL] {fname}: shape {arr.shape} != {exp_shape}")
            all_ok = False
        else:
            print(f"  [OK]   {fname}: {arr.shape}")

    idx = np.load(args.out_dir / "idx.npy")
    if len(set(idx.tolist())) != len(idx):
        print(f"  [FAIL] idx.npy: contains duplicate sample indices")
        all_ok = False
    else:
        print(f"  [OK]   idx.npy: {len(idx)} unique indices, "
              f"range [{int(idx.min())}, {int(idx.max())}]")

    if not all_ok:
        sys.exit("[error] Output validation failed.")

    print("\n" + "=" * 64)
    print("Done.")
    print(f"  Output: {args.out_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
