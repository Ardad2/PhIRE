# Candidate E TTK Critical-Pair Refiner Notes

**Updated:** 2026-05-20 (E1-fix)

Candidate E = Candidate C losses + Candidate B level-set loss + TTK critical-pair losses (Kissi-style).

---

## Status

Real data not yet available in this environment. Both scripts abort cleanly
with actionable error messages:

- `extract_ttk_pd_critical_pairs.py` — aborts (exit 1) when `candidateD_topology/pd/GT/` is missing/empty.
- `run_candidateE_ttkcrit_refiner.py` — aborts (exit 1) when:
  - NPZ is missing required key `sample_idx` (old format — regenerate NPZ).
  - NPZ has fewer than 168 samples (without `--allow-partial-constraints`).
  - `data_out_fixed/wind_mrhr_cnn/dataSR.npy` is missing.

**Pre-conditions before training:**

1. Run TTK pipeline on CNN-baseline or Candidate C output to produce 168 GT PD VTU files.
2. Run extraction (requires 168 samples):
   ```
   python3 scripts/extract_ttk_pd_critical_pairs.py \
     --pd-dir ttk_runs_fixed/topology_finetuning/candidateD_topology/pd/GT \
     --out-dir ttk_runs_fixed/topology_finetuning/candidateE_constraints \
     --patch 160 --persistence-frac 0.01 --top-k 64 --expected-samples 168
   ```
3. Ensure `data_out_fixed/wind_mrhr_cnn/` has `dataSR.npy` and `dataGT.npy`.
4. Run diagnostic (no training):
   ```
   python3 scripts/run_candidateE_ttkcrit_refiner.py \
     --diagnostic-only \
     --data-dir data_out_fixed/wind_mrhr_cnn \
     --constraints ttk_runs_fixed/topology_finetuning/candidateE_constraints/ttk_pd_critical_pairs.npz \
     --lambda-ttkcv 0.0 --lambda-ttkpers 0.0
   ```

---

## Fixes Applied (E1-fix)

### A. extract_ttk_pd_critical_pairs.py

- Removed silent archive fallback; added `--allow-archive-fallback` (CI only, default off).
- Aborts with clear message when `--pd-dir` has zero VTU files.
- Parses `sample_idx` from filenames using regex `_s(\d+)_` (e.g. `candidateD_GT_s113_...`).
- Stores `sample_idx` array in NPZ; sorts results by numeric sample index.
- `--expected-samples 168` guard; `--allow-partial` to override.
- Writes human-readable CSV with columns: `sample_idx`, `sample_name`, `pair_id`, `pair_type`, `birth_vid`, `death_vid`, `birth_y`, `birth_x`, `death_y`, `death_x`, `birth_val`, `death_val`, `persistence`.
- Full output validation: finite values, vertex IDs in `[0, patch*patch)`, `y`/`x` in `[0, patch)`, `pair_type != -1`.
- Prints per-sample stats and examples for samples 6, 18, 25, 80, 162 if present.

### B. run_candidateE_ttkcrit_refiner.py

- `--constraints` is now required; aborts if file missing or invalid.
- `TTKConstraints.__init__` validates required NPZ keys; aborts with clear message if missing.
- `constraints.n_samples < 168` guard with `--allow-partial-constraints` override.
- `TTKConstraints.get(sample_idx)` uses dict lookup (sample_idx → row); no modulo fallback.
- Uses `idx.npy` for actual sample id during training when available.
- `lambda_ttkcv` and `lambda_ttkpers` default to `0.0` (calibrate from diagnostic).
- Added `L_levelset`: sigmoid soft exceedance at 5/10/15 m/s with `k=10`.
- NaN/inf abort on every loss term at every training step.
- Added `--dry-run` (5-step epoch, no output save) and `--max-pairs-per-sample`.

---

## Loss Structure

```
L_total = L_uv
        + lambda_speed    * L_speed         (default 0.01)
        + lambda_grad     * L_grad          (default 0.05)
        + lambda_crit     * L_crit          (default 0.001; Candidate C pool-3 maxima proxy)
        + lambda_levelset * L_levelset      (default 0.25; sigmoid 5/10/15 m/s)
        + lambda_ttkcv    * L_ttkcv         (default 0.0; calibrate from diagnostic)
        + lambda_ttkpers  * L_ttkpers       (default 0.0; calibrate from diagnostic)
```

---

## Persistence Sign Convention

TTK stores `persistence = |death_scalar - birth_scalar|` (unsigned).

The extraction script stores:
- `birth_val` = scalar at birth vertex
- `death_val` = `birth_val + persistence_raw`
- `persistence` = `|death_val - birth_val|`

Both TTK loss functions are direction-agnostic:

- **L_ttkcv**: `0.5 * (MSE(sr[birth_yx], birth_val) + MSE(sr[death_yx], death_val))`
- **L_ttkpers**: `MSE(|sr[death] - sr[birth]|, gt_persistence)`

The ordering of birth and death vertices depends on the TTK filtration direction
and is not assumed. The unsigned convention matches how TTK stores data in VTU files.

---

## Abort Behavior Verified

| Scenario | Script | Exit |
|---|---|---|
| Real 168 GT VTU missing | extraction | 1 (clear error) |
| Archive fallback without flag | extraction | 1 (clear error) |
| NPZ missing `sample_idx` key | refiner | 1 (ValueError) |
| NPZ has 8 < 168 samples | refiner | 1 (without --allow-partial-constraints) |
| dataSR.npy missing | refiner | 1 (clear error) |
