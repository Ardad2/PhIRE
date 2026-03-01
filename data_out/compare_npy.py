#!/usr/bin/env python3
import os
import numpy as np

a_path = os.path.expanduser("~/PhIRE/data_out/wind_mrhr_gan/dataGT.npy")
b_path = os.path.expanduser("~/PhIRE/data_out/wind_mrhr_cnn/dataGT.npy")

A = np.load(a_path, allow_pickle=False)
B = np.load(b_path, allow_pickle=False)

print("A:", a_path)
print("B:", b_path)
print("dtype:", A.dtype, B.dtype)
print("shape:", A.shape, B.shape)

if A.shape != B.shape:
    print("NOT equivalent: shapes differ")
    raise SystemExit(1)

if A.dtype != B.dtype:
    print("Note: dtypes differ (still may be numerically close)")

# Strict equality (handles NaNs as equal)
exact = np.array_equal(A, B, equal_nan=True)
print("exact_equal:", exact)

if not exact:
    # Numeric closeness (good for floats)
    if np.issubdtype(A.dtype, np.number) and np.issubdtype(B.dtype, np.number):
        close = np.allclose(A, B, rtol=1e-5, atol=1e-8, equal_nan=True)
        print("allclose(rtol=1e-5, atol=1e-8):", close)

        diff = np.abs(A - B)
        # guard for empty arrays
        if diff.size:
            max_diff = np.nanmax(diff)
            mean_diff = np.nanmean(diff)
            idx = np.unravel_index(np.nanargmax(diff), diff.shape)
            print("max_abs_diff:", max_diff)
            print("mean_abs_diff:", mean_diff)
            print("argmax_diff_index:", idx)
            print("A[idx], B[idx]:", A[idx], B[idx])
    else:
        print("Non-numeric dtype; only exact comparison is meaningful here.")
