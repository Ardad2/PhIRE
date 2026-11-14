# Phase 6 PERS-only training completion — documentation update (14 October 2026)

This bundle extends the original `dataset_generation_and_repair_notes.md` to Part LXXI. It records the successful **PERS-only training pipeline exit 0** and the next read-only verification / cheap-evaluation commands; it does not claim PERS-only numerical results or corrected PD/MT outcomes.

**This ZIP is intended for Spark. Download/transfer it to `~/PhIRE/` before running the commands below.** No operation here changes published checkpoints, other scripts, or the frozen TopoInVis release.

1. `cd ~/PhIRE`
2. `unzip Phase6_PERS_Only_Training_Notes_LXXI_20261014.zip -d phase6_pers_notes_update`
3. `grep '^# Part LX' dataset_generation_and_repair_notes.md | tail -7`
4. Select the patch matching your **current latest section** (do not assume LXX is already installed):

| Latest current section | Choose this patch |
|---|---|
| LXVII | `phase6_notes_LXVII_to_LXXI.patch` |
| LXVIII | `phase6_notes_LXVIII_to_LXXI.patch` |
| LXIX | `phase6_notes_LXIX_to_LXXI.patch` |
| LXX | `phase6_notes_LXX_to_LXXI.patch` |

5. For example, when current notes **end at LXVII**:

```bash
PATCH=phase6_pers_notes_update/patches/phase6_notes_LXVII_to_LXXI.patch
git apply --check "$PATCH" && git apply "$PATCH"
grep '^# Part LX' dataset_generation_and_repair_notes.md | tail -5
git diff --stat -- dataset_generation_and_repair_notes.md
```

If `git apply --check` fails, **do not force**. Check the latest heading and for manual edits to the document; send the output to the assistant. If the latest heading is already LXXI, do not reapply. `git apply` only modifies the master `.md` file, not the unrelated working-tree changes.

Other bundle components:
- `dataset_generation_and_repair_notes.md`: complete Part LXXI document (reference copy; don't blindly overwrite your live notes)
- `docs/phase6_grad_pers_only_training_exit0_20261014.md`: standalone new phase handoff section
- `patches/*.patch`: four incremental/cumulative patches, **all independently applied and SHA256-checked** against the specified base document

After verifying the PERS-only files and running the cheap evaluator, share `docs/topology_finetuning_candidateF_grad_PERS_only_expanded2688_eval.md` and the console verification output. The next documentation addition will include its **actual** metrics and 2×2 factorial contrasts.
