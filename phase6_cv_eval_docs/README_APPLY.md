# Phase 6 CV-only quantified results — Part LXX (October 13, 2026)

## What changed

Adds verified CV-only training/inference and cheap-evaluation numerical results, documents the CV-only versus gradient-only/CNN/archived F1 trade-offs, flags the spatial-wind-speed maximum overshoot and missing SSIM, and records the incorrect patch-selection failure. **Does not alter** training scripts, benchmark data, weights or earlier notes. The archived F1 numbers refer to `docs/topology_finetuning_candidateF_grad_E2_low_expanded2688_eval.md` at frozen release `topoinvis-2026-arxiv-v1.1`.

## Choose EXACTLY ONE patch based on your latest existing heading

The user's last shell output showed Part **LXVII** as last applied. Therefore use:

    patches/phase6_notes_LXVII_to_LXX.patch

If instead LXVIII or LXIX is already applied, use:

    patches/phase6_notes_LXVIII_to_LXX.patch
    patches/phase6_notes_LXIX_to_LXX.patch

All three transitions have been checked with `git apply --check` and `git apply` against the matching released update files and produce the LXX master document byte-for-byte. Never apply more than one. Do NOT replace the original document with the full one unless deliberately resolving a local-content conflict.

## On Spark (after copying ZIP into `~/PhIRE/`)

```bash
cd ~/PhIRE
unzip Phase6_CV_Only_Quantitative_Notes_LXX_20261013.zip -d phase6_cv_eval_docs

grep '^# Part LX' dataset_generation_and_repair_notes.md | tail -6
PATCH=phase6_cv_eval_docs/patches/phase6_notes_LXVII_to_LXX.patch

git apply --check "$PATCH" && git apply "$PATCH"
grep '^# Part LX' dataset_generation_and_repair_notes.md | tail -6
git diff --stat -- dataset_generation_and_repair_notes.md
```

Optional pure-Python report comparison script (must be copied deliberately):

```bash
cp phase6_cv_eval_docs/scripts/compare_phase6_cheap_reports.py scripts/
/usr/bin/python3 scripts/compare_phase6_cheap_reports.py
```

Caution: the user's `master` worktree includes unrelated modified figures and untracked ZIPs. Stage only intended files; do not `git add .`, hard-reset, or alter the frozen v1.1 tag. The full updated master notes and standalone record are provided for auditing; the patch is the safer way to update the tracked file.
