# Phase 8 — Temporally separated, out-of-pilot development evaluation

**Status:** Source/protocol and local mathematical/metadata tests prepared on 2026-11-17; **NOT run on Spark**. Do not record its results as achieved until full verification.

## Why this evaluation is necessary

The 2026-11-16 four-field/eight-update experiment improved true TTK/GUDHI W2,2 on every evaluation immediately after a training step, but those evaluations were on the same four examples that received optimizer updates. The mean normalized L_uv rose by 0.000044107321 relative to the control, while mean L_grad fell by 0.00039665774. This warrants a separate development evaluation before expanding training.

The builder `scripts/build_wind_mrhr_expanded_dataset_2688.py` constructs 2,688 hourly WTK fields from **16 distinct seven-day seasonal windows**, each with 168 successive hours, using a single consistent spatial crop. The `data_out/wind_mrhr_cnn_expanded2688/idx.npy` array confirms row identity only, not independence; consult the CSV source time mapping.

The first four training examples have WTK time indices 336–339 in `winter_1`. Indices 4–7 are adjacent hours and unsuitable as an independent-style evaluation. This gate selects four other, predetermined indices from distinct seasonal weeks:

| Training-pool sample | WTK time index | Source window | Role |
|---:|---:|---|---|
| 0–3 | 336–339 | `winter_1` | Eight-step optimizer updates (repeated twice) |
| 672 | 2160 | `spring_1` | Out-of-pilot evaluation only |
| 1344 | 4344 | `summer_1` | Out-of-pilot evaluation only |
| 2016 | 6552 | `fall_1` | Out-of-pilot evaluation only |
| 2520 | 7056 | `fall_4` | Out-of-pilot evaluation only |

**Scientific boundary:** These four fields remain **in the existing 2,688-field training pool**, and share geographic location with the optimizer fields. They are separate from this eight-step pilot's optimizer examples, but are NOT an independent benchmark, new geography, or evidence of statistical generalization. Do not use the sealed 168-field benchmark for this gate. Do not call these “test” fields.

## Implementation strategy

Source: `phase8_temporal_out_of_pilot_eval_20261117.py` placed directly in `~/PhIRE`.

- `selftest`: read-only math and ID mapping checks, no external imports.
- `preflight`: verifies the historical eight-step artifacts and all their recorded source/checkpoint hashes, then verifies **all 2,688 CSV rows** match the builder's 16 frozen WTK windows. Stops if metadata or upstream source drift is detected.
- `export`: streams the original expanded TFRecord through `PhIREGANs._parse_train_` in native TensorFlow, saves normalized inputs and physical-speed GT crops for indices 672, 1344, 2016, 2520. Nothing is written to the source dataset.
- `replay`: reconstructs two independent optimizer sessions from the original frozen CNN checkpoint. Replays the exact eight-example order `(0,1,2,3,0,1,2,3)` with `L_uv + 0.05 L_grad` versus that objective plus `1e-5 J_dynamic`. Uses the **already recorded fresh per-step TTK/GUDHI matchings**; it does NOT repeat matching. Every replayed prediction *before* and *after* every step must agree with the verified historical field arrays to within a predeclared `0.005 m/s` maximum absolute tolerance. It then evaluates the original CNN, control, and dynamic arms on the four out-of-pilot fields **without further optimizer updates**. Saved artifacts are only 160x160 crop arrays and scalar metrics—not model weights or checkpoints.
- `score`: runs in `gudhi-audit`; extracts fresh TTK diagrams for each evaluation GT and the three model predictions (16 diagrams) and computes corrected canonical Phase 6 `dB`, `W2inf`, and `W22` with separate D0/D1 handling. Reports normalized vector `L_uv`, `L_grad`, and physical-speed crop MAE alongside the diagram metrics. Reports paired differences and individual cases, without claiming significance.
- `verify`: independently reloads and recomputes the twelve GT-vs-model diagram comparisons from the frozen VTUs, checks speed crop MAE, summaries, original source lineage, archived replay fidelity, and output hashes.

All writes go to a previously nonexistent isolated directory: `ttk_runs_fixed/topology_finetuning/phase8_cross_season_out_of_pilot_20261117_r01`. Each stage is append-only; it does not overwrite completed prior stages. Failed stages create `FAILED.json` under an explicitly incomplete staging directory. No pretrained checkpoint or 168-field held-out benchmark is touched.

The earlier result’s original fresh-match behavior is preserved and validated in its archived files; this follow-up specifically tests **how those final eight-step model states behave on fields that received no step**, rather than conducting a new eight-step optimization experiment. A replay failure must not be bypassed by widening the 0.005 m/s tolerance without a documented numerical investigation.

## Commands — from Spark, after transferring the new source into repo root

```bash
cd ~/PhIRE

# 1. Local math and fixed-lineage mapping
python3 phase8_temporal_out_of_pilot_eval_20261117.py --mode selftest

# 2. Read-only checks of 2688-row source manifest and verified prior training
python3 phase8_temporal_out_of_pilot_eval_20261117.py --mode preflight

# 3. Export 4 temporally separated TFRecord examples, not used to optimize
python3 phase8_temporal_out_of_pilot_eval_20261117.py --mode export

# 4. Reproduce the historical eight-step run in memory, validate replay
#    against the original artifact at EVERY step; evaluate four other fields
python3 phase8_temporal_out_of_pilot_eval_20261117.py --mode replay

# 5. Fresh TTK/GUDHI scoring of all 4×3 predictions
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit \
  python phase8_temporal_out_of_pilot_eval_20261117.py --mode score --ttk-backend docker

# 6. Independent numerical/hash verification
PYTHONNOUSERSITE=1 micromamba run -n gudhi-audit \
  python phase8_temporal_out_of_pilot_eval_20261117.py --mode verify --ttk-backend docker
```

Stop at the first error. No pip installs and no environment changes should be necessary; the Python 3.12 NumPy binary ABI warnings seen in previous successful native TF1 runs may recur. An actual final `PASS` message, not absence of warnings, establishes completion.

The results after success are under:

```
ttk_runs_fixed/topology_finetuning/phase8_cross_season_out_of_pilot_20261117_r01/
  eval_inputs/manifest.json
  replay/manifest.json
  score/report.json
  score/report.md
  score/manifest.json
```

After successful verification, inspect the **four individual cases**, not just mean values. A good result requires considering true W22 together with `L_uv`, `L_grad`, and speed MAE; improvement on topology alone is insufficient to select a final model.

## Audit convention

The current master notes end at Part CXVII, committed `aa518273`. The current document and new script are **planned, not yet executed**, and must not be inserted as a successful result. After Spark validation, record a new **Part CXVIII** containing: actual source SHA256, manifest CSV lineage, selected WTK indices/timestamps, original eight-step source hashes, replay max differences, per-case and mean metrics for all three arms, TTK/GUDHI verification status, software/runtime issues, interpretation and limitations. Keep prior sections and checkpoints unchanged; commit only the notes after review.

If the replay fails, retain the incomplete staging artifact and investigate before further training. If evaluation shows a reconstruction tradeoff or inconsistent topology benefits, report that transparently and adjust only through a new versioned experiment with predeclared hyperparameters.
