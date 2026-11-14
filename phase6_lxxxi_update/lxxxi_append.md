
---

# Part LXXXI — Phase 6: verified all-168 paired persistence audit and Phase 5E locality-port readiness (October 14, 2026)

## LXXXI.1 Observed completion of Part LXXX on Spark

The researcher executed the LXXX source/paired-audit preflight and report command; both exited `0`. Exact observed markers included:

```text
[SOURCE] PASS: 840 matched rows, 168 indices, all source hashes and LXXIX means/effects verified
[PAIRED] dB best_counts={'cnn': 2, 'grad': 20, 'cv': 34, 'pers': 53, 'f1': 59, 'tie': 0}
[PAIRED] W2inf best_counts={'cnn': 1, 'grad': 41, 'cv': 31, 'pers': 68, 'f1': 27, 'tie': 0}
[PAIRED] W22 best_counts={'cnn': 1, 'grad': 53, 'cv': 24, 'pers': 66, 'f1': 24, 'tie': 0}
[LXXX] PREFLIGHT PASS
[LXXX] REPORT PASS
```

The exact frozen input CSV digest displayed in both executions is:

```text
3d53189ec69cc2f89c5561159d0768c973351d7b9998deceeaef42ebb6b379ea
```

The independently run `sha256sum -c SHA256SUMS` in the new LXXX output folder reported `OK` for `source_csv_sha256.txt`, `paired_audit.json`, and `paired_audit.md`. The report lives under `ttk_runs_fixed/topology_finetuning/phase6_lxxx_paired_effect_audit_20261014/` and is purely derived from LXXIX. **No new TTK/GUDHI computation** was performed by LXXX.

## LXXXI.2 Paired-field scientific findings

All 168 same-hour fields are paired across the five methods; lower is better.

| Metric | Best-of-five by sample: CNN / grad / CV / PERS / F1 | Paired F1 wins vs PERS | Mean winner |
|---|---|---:|---|
| Corrected bottleneck `dB` | 2 / 20 / 34 / 53 / 59 | 101 / 168 | F1 |
| Corrected `W2inf` | 1 / 41 / 31 / 68 / 27 | 49 / 168 | PERS-only |
| Corrected `W22` | 1 / 53 / 24 / 66 / 24 | 34 / 168 | PERS-only |

The aggregate corrected `W22` means were 17.484939 for PERS-only and 17.831897 for F1. PERS-only outperformed F1 on 134 / 168 individual fields for `W22`, and 119 / 168 for `W2inf`, strengthening the descriptive observation that this is not merely a mean driven by a few unusually large outliers. F1 outperformed PERS-only on 101 / 168 fields for `dB`. Reported F1-minus-PERS paired means and medians were:

| Metric | Mean difference | Paired median | Better/worse/tied F1 |
|---|---:|---:|---|
| dB | -0.046375 | -0.075414 | 101 / 67 / 0 |
| W2inf | +0.172446 | +0.151641 | 49 / 119 / 0 |
| W22 | +0.346958 | +0.330417 | 34 / 134 / 0 |

In the seven contiguous 24-hour blocks, both individual CV and PERS main effects for bottleneck `dB` were negative in every block; for Wasserstein, some block effects changed sign. CV×PERS interaction means were positive across all seven blocks for all three metrics. This is less-than-additive *descriptive* performance at the current weights, not evidence of statistically significant antagonism, replicated training variability, or a universal property of the loss combination. Do not treat 168 consecutive hours as independent realizations.

The pretrained CNN is a comparator and not cell 00; 00 is the gradient-only model. The archived F1 checkpoint/data provenance remains subject to a separate documentation verification before manuscript attribution.

## LXXXI.3 Planned next gate: Phase 5E source compatibility, not yet an execution

The new `Phase6_LXXXI_Locality_Readiness_20261014.zip` workpack is **prepared locally, not executed on Spark**. Its Python script performs a read-only source CSV hash/paired audit manifest check, counts all canonical GT/CNN/F1 and new-arm PD outputs, checks new-arm VTIs, locates plausible existing Phase 5E locality/sparse-matching source scripts and prints their hashes/possible CLI flags, and inventories F1 source/checkpoint paths where available. No locality objective is evaluated and source candidates are **not** declared validated merely because they were found. If an expected F1 source directory is absent, the script reports unverified provenance rather than deleting any artifact.

The mathematical and procedural target remains the existing Phase 5E *sparse-savings matching* objective, evaluated for both D0/D1 and `r={4,8,16,32,64,96,128,infinity}`, with GT-side and SR-side persistence-mass coverage separately measured. Phase 5E's 8-sample dense-vs-sparse parity and its existing 168-field CNN/F1 locality outputs are the appropriate regression references. **At r=infinity, the new calculation must match LXXIX's canonical per-sample W22 to numerical tolerance, for every model/sample, not merely the mean.**

Do not replace the spatial matching with unconstrained birth/death Wasserstein, do not substitute a different radius grid, do not infer kilometer units without grid calibration, and do not overwrite Phase 5E or LXXIX archives. A matched new-arm locality pilot should precede all-168 locality execution; merge-tree evaluation remains a distinct outstanding task.
