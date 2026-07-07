# Candidate D/E Infrastructure Fixes — Summary

**Scope:** Implements the infrastructure-fix phase of
`docs/candidateD_E_topology_audit.md`, Findings 2.1 and 2.3 only. Finding
2.2 (Dpd's superlevel training loss vs. sublevel evaluation convention) is
**not** addressed here — it was out of scope for this pass.

No training was launched. No existing experiment results, checkpoints, CSVs,
or constraint files were modified, deleted, or regenerated.

---

## 1. Finding 2.3 — VTI writer coordinate transpose (fixed)

**File changed:** `scripts/convert_phire_to_vti.py`

`make_vti_from_scalar()` used `scalar_2d.ravel(order="F")` together with
`img.SetDimensions(W, H, 1)`. VTK's point-id convention (`pointId = ix +
iy*dimX`, x fastest) requires the flat buffer to vary the array's column
axis (`W`, x) fastest; Fortran-order raveling instead varies the row axis
(`H`, y) fastest. For the square 160×160 patches used throughout this
pipeline this reduced to a pure transpose (`img(x,y) == scalar_2d[x,y]`
instead of the intended `scalar_2d[y,x]`); for non-square patches it would
have produced fully scrambled data.

**Fix:** changed `ravel(order="F")` to `ravel(order="C")` (ordinary
row-major order, numpy's default). Updated the function's docstring and the
inline comment to state the intended convention explicitly
(`scalar_2d[y, x] -> VTK point (x, y)`) and to record why the old order was
wrong, for future readers.

**Verified:**
- Added `scripts/verify_vti_coordinate_mapping.py`: an end-to-end regression
  test that calls the real `make_vti_from_scalar()` with a **non-square**
  (5×8) synthetic field, writes an actual `.vti` file, reads it back with
  VTK's own `vtkXMLImageDataReader`, and asserts `grid[x, y] ==
  scalar_2d[y, x]`. Non-square dimensions are deliberate: a square field
  cannot distinguish "correctly fixed" from "still transposed," since a
  transpose of a square array can look like a coincidental pass.
- Ran the test against the fixed writer: **PASS**.
- Separately confirmed the test has discriminating power by running the
  same assertion against a reimplementation of the *old* buggy writer: it
  correctly **fails** (`grid[x,y] == scalar_2d[y,x]` is `False`).
- The original `scripts/verify_vti_transpose_bug.py` (pure-Python,
  no-vtk-needed simulation) is left untouched as a historical record of
  what the bug was; it is superseded going forward by
  `verify_vti_coordinate_mapping.py`, which exercises the real production
  code path instead of a reimplementation.

**Important — what must be regenerated:**
Every existing VTI file, and everything downstream of it, was written by the
old (transposed) writer. Persistence-diagram and merge-tree **distances**
computed from those files remain valid (bottleneck/MT distances only depend
on the multiset of scalar values, not on axis labeling — a consistent
transpose of both GT and SR fields does not change these distances; see
audit §2.3.2). However, anything that uses **spatial (y, x) vertex
coordinates** read from those old files is now stale and must be rebuilt
from scratch against the fixed writer, specifically:

- `ttk_runs_fixed/topology_finetuning/candidateE_constraints/ttk_pd_critical_pairs.csv`
  (existing, pre-fix — already flagged in the audit as a non-representative
  5-sample dev artifact, but in any case built from transposed VTIs)
- Any future run of `scripts/extract_ttk_pd_critical_pairs.py`
- Any future run of `scripts/build_candidateE2_expanded672_ttk_constraints.py`,
  and the NPZ/CSV constraint files it produces
  (`ttk_pd_critical_pairs_gtvalues.npz` and friends)
- Any Candidate E or E2 training run and its resulting checkpoints/outputs,
  since they were (or would be) trained against constraints built from the
  old, transposed vertex coordinates

None of these were regenerated as part of this change — regenerating them
requires Docker + the TTK image and was explicitly out of scope for this
pass ("do not launch long training runs").

---

## 2. Finding 2.1 — `L_uv` normalization (fixed)

**Files changed** (identical pattern applied to all five PyTorch RefinerNet
scripts):
- `scripts/run_candidateD_pd_refiner.py`
- `scripts/run_candidateD_expanded672_pd_refiner.py`
- `scripts/run_candidateDpd_expanded672_pd_refiner.py`
- `scripts/run_candidateE_ttkcrit_refiner.py`
- `scripts/run_candidateE2_expanded672_ttkcrit_refiner.py`

Each script's `l_uv(sr, gt)` previously computed `F.mse_loss(sr, gt)`
directly on `sr`/`gt` as loaded from `dataSR.npy`/`dataGT.npy`. Those arrays
are physical (denormalized) `[u, v]` — `PhIREGANs.test_paired()` applies
`mu_sig[1]*batch + mu_sig[0]` before saving them — so `L_uv` was being
computed in physical units, unlike the TF1 Candidate B/C convention
(`sr_network.py`'s `content_loss`, computed on normalized `x_HR`/`x_SR`).

**Fix:** added, in each file:

```python
_MU_UV    = (0.7684, -0.4575)
_SIGMA_UV = (5.02455, 5.9017)

def _normalize_uv(uv: torch.Tensor) -> torch.Tensor:
    """(B, 2, H, W) physical [u, v] -> normalized [u, v]."""
    mu    = torch.tensor(_MU_UV,    dtype=uv.dtype, device=uv.device).view(1, 2, 1, 1)
    sigma = torch.tensor(_SIGMA_UV, dtype=uv.dtype, device=uv.device).view(1, 2, 1, 1)
    return (uv - mu) / sigma

def l_uv(sr: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(_normalize_uv(sr), _normalize_uv(gt))
```

Deliberately **not** done: no global renormalization of the loaded
`sr`/`gt` arrays. `_speed_t`, `l_speed`, `l_grad`, `l_crit`, and (in Dpd)
`l_pd` all continue to receive and operate on the original physical-unit
tensors unchanged — normalization is applied locally, only inside `l_uv`,
exactly as requested, so the scalar-speed/topology losses' physical-unit
behavior is untouched.

**Verified:**
- `python3 -m py_compile` on all five modified files: clean.
- The `_normalize_uv` broadcasting logic (`.view(1, 2, 1, 1)` against an
  `(B, 2, H, W)` NCHW tensor) was verified with an equivalent numpy
  reproduction: per-channel `mu`/`sigma` broadcast correctly across
  `batch`/`H`/`W` without cross-channel mixing, and normalizing synthetic
  data drawn with the same `mu`/`sigma` per channel recovers ~zero mean /
  unit std per channel, as expected. Real `torch` was not available in this
  sandbox to execute the actual PyTorch code path; the broadcasting
  semantics of `.view()` vs. numpy's `.reshape()` are identical for this
  case, so this is a faithful proxy, not a substitute for running the real
  script once on the target machine before a full training run.

**What this changes for future runs:** the effective relative weighting
between `L_uv` and the auxiliary topology terms (`L_speed`, `L_grad`,
`L_crit`, `L_PD`, `L_ttkcv`, `L_ttkpers`) will shift once `L_uv` returns to
the smaller, normalized-space scale — the existing lambda weights
(`lambda_speed=0.01`, `lambda_grad=0.05`, `lambda_crit=0.001`,
`lambda_pd=0.001904`, `lambda_ttkcv=0.04`, `lambda_ttkpers=0.02`) were not
changed in this pass and may be worth re-diagnosing (per the audit's Phase
4 recommendations) now that `L_uv`'s scale matches the regime they were
originally calibrated for.

---

## 3. What still needs to happen before drawing new conclusions

In order:

1. Rebuild the Docker-based TTK constraint pipeline
   (`scripts/build_candidateE2_expanded672_ttk_constraints.py` and
   `scripts/extract_ttk_pd_critical_pairs.py`) from scratch against the
   fixed VTI writer. The existing constraint NPZ/CSVs are stale.
2. Re-run Candidate D/Dpd/E/E2 training from scratch with the fixed `L_uv`
   and the rebuilt (correct-coordinate) constraints — not launched here per
   instructions.
3. Only after both of the above: re-evaluate whether E2's `L_ttkcv`/
   `L_ttkpers` losses correlate with final TTK PD/MT improvement, using
   `scripts/diagnose_candidate_topology_alignment.py` from the prior audit
   pass.
4. Finding 2.2 (Dpd's `CubicalComplex(superlevel=True)` vs. the sublevel
   evaluation convention) remains open and unaddressed by this pass.

---

## 4. Files touched in this pass

Modified:
- `scripts/convert_phire_to_vti.py`
- `scripts/run_candidateD_pd_refiner.py`
- `scripts/run_candidateD_expanded672_pd_refiner.py`
- `scripts/run_candidateDpd_expanded672_pd_refiner.py`
- `scripts/run_candidateE_ttkcrit_refiner.py`
- `scripts/run_candidateE2_expanded672_ttkcrit_refiner.py`

Added:
- `scripts/verify_vti_coordinate_mapping.py`
- `docs/candidateD_E_infra_fix_notes.md` (this file)

Not touched: `scripts/verify_vti_transpose_bug.py` (kept as historical
record), anything under `data_out*/`, `ttk_runs*/`, `models_fixed/`, or any
other existing experiment artifact.

Not committed or pushed, per instructions.

---

## 5. Update — Step 3: E2 constraint regeneration (Section 2.3 continued)

**Important discovery:** `build_candidateE2_expanded672_ttk_constraints.py`
does **not** call `make_vti_from_scalar()` from `convert_phire_to_vti.py` —
it has its own independent ASCII VTI writer, `_write_vti_ascii()`, with the
exact same `ravel(order="F")` bug, plus an incorrect comment claiming
"F-order == C-order" for square patches (false in general — see Section
2.3 of `candidateD_E_topology_audit.md`; it only holds for a
diagonally-symmetric field, which wind fields are not). **Fixing
`convert_phire_to_vti.py` alone did not fix the E2 constraint pipeline.**
This has now also been fixed (`ravel(order="C")`, corrected docstring).

**CLI changes** (minimal, old defaults unchanged): added `--out-dir`,
`--vti-dir`, `--pd-dir`, `--vti-label` to
`build_candidateE2_expanded672_ttk_constraints.py`. Running the script with
no arguments behaves exactly as before (writes to
`ttk_runs_fixed/topology_finetuning/candidateE2_expanded672_constraints/`).
`extract_ttk_pd_critical_pairs.py` needed no changes — `--out-dir` was
already a required (non-hardcoded) argument.

**Real regeneration status: not run.** This environment has no running
Docker daemon (attempting to start it fails with a permission error —
sandboxed, cannot be worked around), no `phire-ttk:latest` image, and no
real `data_out/wind_mrhr_cnn_expanded672/{dataGT,idx}.npy` (the 672-sample
expanded dataset does not exist in this checkout). Consequently:

- The fixed VTI writer and the new CLI plumbing were verified with real VTK
  write/read round-trips and a Docker-free smoke test
  (`scripts/smoke_test_candidateE2_fixed_constraints.py`, new file) against
  **fabricated synthetic data only**, written to
  `diagnostics/candidateE2_fixed_constraints/smoke_test/` — never to any
  `ttk_runs_fixed/...` path. All checks passed (VTI coordinate mapping
  correct; `_sanity_check()` correctly passes on correct data and correctly
  fails on deliberately-transposed fabricated data).
- No files under `ttk_runs_fixed/topology_finetuning/candidateE2_fixed_constraints/`
  or `.../candidateE2_fixed_vti/` were created — the real, Docker-dependent
  regeneration against real data has not happened yet and must be run where
  Docker + the real dataset exist (e.g. Spark), using:

  ```bash
  python3 scripts/build_candidateE2_expanded672_ttk_constraints.py \
    --out-dir ttk_runs_fixed/topology_finetuning/candidateE2_fixed_constraints \
    --vti-dir ttk_runs_fixed/topology_finetuning/candidateE2_fixed_vti
  ```

  This does not touch the old `candidateE2_expanded672_constraints/` output.

Files added this round: `scripts/smoke_test_candidateE2_fixed_constraints.py`,
`diagnostics/candidateE2_fixed_constraints/smoke_test/` (synthetic-only).
Files modified: `scripts/build_candidateE2_expanded672_ttk_constraints.py`.
