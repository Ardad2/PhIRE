# Phase 6 quantitative documentation update (Part LXVIII)

## Important: select the patch matching your repository's current notes

Your last CONFIRMED applied notes update was **Part LXVI**, not LXVII. Accordingly the recommended patch for your checkout is the **cumulative LXVI -> LXVIII patch**, which adds BOTH Parts LXVII and LXVIII in one operation. Do NOT apply the LXVII patch separately if you choose this route.

From `~/PhIRE` after unzipping into `phase6_lxviii_update`:

```
PATCH=phase6_lxviii_update/patches/phase6_grad_pairzero_LXVI_to_LXVIII_cumulative.patch
git apply --check "$PATCH"
git apply "$PATCH"
git diff --stat -- dataset_generation_and_repair_notes.md
```

Only if you had ALREADY successfully applied Part LXVII, instead use the incremental patch:

```
PATCH=phase6_lxviii_update/patches/phase6_grad_pairzero_quantitative_LXVIII_incremental.patch
git apply --check "$PATCH"
git apply "$PATCH"
```

Either route preserves your earlier notes. Do not use both patches. Both paths were validated to produce a byte-identical Part LXVIII master from their respective bases.

Optional reproducibility script:

```
cp phase6_lxviii_update/scripts/compare_phase6_scalar_reports.py scripts/
/usr/bin/python3 scripts/compare_phase6_scalar_reports.py \
 --control docs/topology_finetuning_candidateF_grad_pairzero_expanded2688_eval.md \
 --f1 docs/topology_finetuning_candidateF_grad_E2_low_expanded2688_eval.md
```

The script uses only the standard library. It compares aggregate report means, not raw per-field results, causal impacts, or corrected topological distances.

Source evidence: October 13 Spark logs from user, current grad-pairzero evaluation report in `sources/`, and historical F1 report in frozen tag `topoinvis-2026-arxiv-v1.1`. No files in Spark are modified by this ZIP by itself; patches must be applied explicitly. No original experiments or frozen release tags should be overwritten. Avoid `git add .` with the user's other uncommitted files.
