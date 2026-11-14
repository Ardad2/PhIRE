# Phase 6 first-arm run status

This is Part LXVI from the master document.


---

# Part LXVI — Phase 6 First Real Training Run: `grad_pair_zero` (2026-10-13)

**Evidence status (2026-10-13, approximately 18:05 CDT): End-of-run message observed; detailed training log, checkpoint inventory, and quantitative evaluation pending inspection.** This section updates (and supersedes the earlier as-of status in) Part LXV. It does not revise any frozen paper or Phase 5E results.

## LXVI.1 Repository documentation update and local worktree state

The author extracted `Phase6_Notes_Documentation_Update_20261013.zip` on Spark and applied its incremental patch to the repository-root `dataset_generation_and_repair_notes.md`:

```bash
cd ~/PhIRE
unzip Phase6_Notes_Documentation_Update_20261013.zip -d phase6_docs_update
git status --short
git branch --show-current
git apply --check phase6_docs_update/patches/phase6_notes_update_20261013.patch
git apply --whitespace=nowarn phase6_docs_update/patches/phase6_notes_update_20261013.patch
git diff --stat
```

Observed current branch: `master`. `git apply --check` and the actual `git apply` returned without error. The ensuing diff showed `dataset_generation_and_repair_notes.md | 315 +++++++++++-` and existing, unrelated local edits to `figures/paper_sample78/sample078_paper.pdf`, `.png`, and `scripts/paper_sample78_figure.py`. Several workpack ZIPs, workpack folders, supporting scripts, and patches are untracked. **Do not indiscriminately `git add .` or reset/overwrite those unrelated changes.** A patch applied in a working tree is not yet a Git commit or push. The frozen tag `topoinvis-2026-arxiv-v1.1` remains the reproducibility baseline.

## LXVI.2 Actual `grad_pair_zero` training invocation and observed end

The new first-arm run was initiated from a `tmux` session using the previously supplied command:

```bash
cd ~/PhIRE
mkdir -p logs
set -o pipefail
env -u PYTHONNOUSERSITE /usr/bin/python3 \
    scripts/run_candidateF_cv_pers_ablation.py \
    --variant grad_pair_zero \
    2>&1 | tee logs/phase6_grad_pairzero_console.log
```

The user-provided **final excerpt** of the Spark terminal showed the printed `scripts/evaluate_finetune_candidate.py` command for `candidateF_grad_pairzero_expanded2688`, followed by the reminder to run `scripts/run_candidate_topology_pipeline.sh` *only after* cheap evaluation, and then the shell prompt. No Python traceback appeared in the supplied excerpt. The tmux status at the end included `phase6_pa0:bash*` (the actual session name used on Spark may differ from the originally suggested one).

**Source-code control-flow finding:** In the frozen runner `scripts/run_candidateF_expanded2688_finetune.py`, lines 1016–1052 invoke real training, paired inference, and `_validate_post_inference()`. Lines 1057–1091 print the `Run complete` report and the exact cheap-evaluation and TTK commands, then `return 0`. The presence of these *final commands* in the user transcript is therefore strong evidence that those earlier operations completed and the validation function returned normally. However, the provided excerpt does **not** include epoch-by-epoch losses, saved-checkpoint listing, full validation report, process exit status, or computed domain metrics; retain them as **not yet independently reviewed**.

### Expected new output locations (verify actual existence, not yet inventoried here)

```text
Training and checkpoint:
  models_fixed/topology_finetuning/wind_finetune_candidateF_grad_pairzero_expanded2688/
Canonical runner log:
  logs/wind_finetune_candidateF_grad_pairzero_expanded2688.log
Full tee transcript:
  logs/phase6_grad_pairzero_console.log
Paired inference (168 fixed benchmark fields):
  data_out/wind_finetune_candidateF_grad_pairzero_expanded2688/idx.npy
  data_out/wind_finetune_candidateF_grad_pairzero_expanded2688/dataIN.npy
  data_out/wind_finetune_candidateF_grad_pairzero_expanded2688/dataGT.npy
  data_out/wind_finetune_candidateF_grad_pairzero_expanded2688/dataSR.npy
```

### Verify and preserve the final training validation

```bash
cd ~/PhIRE
METHOD=candidateF_grad_pairzero_expanded2688
LOG=logs/wind_finetune_${METHOD}.log
OUT=data_out/wind_finetune_${METHOD}
MODEL=models_fixed/topology_finetuning/wind_finetune_${METHOD}

grep -E 'Fine-tuning complete|Run complete|Post-inference validation complete|^\[OK\]|^\[WARNING\]|^\[report\]|Loss-magnitude diagnostic|Checkpoint:' "$LOG" | tail -45
ls -lh "$OUT"/{idx,dataIN,dataGT,dataSR}.npy
ls -lh "$MODEL"/cnn*
```

The runner's mandatory validation checks each output shape, finiteness, ordered indices `0..167`, and exact equality of GT and input arrays to the fixed CNN benchmark. Capture the full `[validate]` and `[report]` output before assigning numerical scientific claims to this run. Synthetic-smoke loss `3.05885259876959e-05` from LXV was not a training or evaluation metric and must not be conflated with this actual run.

## LXVI.3 Next step: cheap scalar/domain evaluation (not observed executed yet)

After the preceding verification, use the first-arm method name exactly:

```bash
cd ~/PhIRE
mkdir -p logs
set -o pipefail
METHOD=candidateF_grad_pairzero_expanded2688
env -u PYTHONNOUSERSITE /usr/bin/python3 \
  scripts/evaluate_finetune_candidate.py \
  --candidate-name "$METHOD" \
  --candidate-dir "data_out/wind_finetune_${METHOD}" \
  --cnn-dir data_out_fixed/wind_mrhr_cnn \
  --gan-dir data_out_fixed/wind_mrhr_gan \
  --merged-csv ttk_runs_fixed/combined/psnr_topology_physics_merged.csv \
  --out-dir "ttk_runs_fixed/topology_finetuning/${METHOD}_eval" \
  2>&1 | tee "logs/phase6_${METHOD}_cheap_eval.log"
echo "Evaluation exit status: $?"
```

The evaluation script generates a Markdown report (`docs/topology_finetuning_<method>_eval.md`) and per-sample and summary CSVs under the specified `_eval/` folder. Consult their actual numerical outputs before judging physical fidelity or deciding whether to proceed to costly topology extraction. **Do not interpret `pd_distance` inherited from the historical merged CSV as the newly computed corrected PD metric for this candidate.** Its corrected PD bottleneck/W₂ metrics and MT distance require separate evaluation; preserve the distinction from all previous audited manuscript scores.

### Updated Phase 6 status

| Phase | Status at time of this addendum |
|---|---|
| New wrapper, static assertions, three dry-runs | PASS (see Part LXV) |
| TensorFlow import / generator checkpoint+loss+optimizer smoke | PASS (with NumPy ABI warning; see Part LXV) |
| Actual `grad_pair_zero` training + paired inference | Final runner next-step block observed; very likely completed successfully; **full numeric log review pending** |
| `grad_cv_only`, `grad_pers_only` training | Not observed executed |
| New first-arm cheap scalar/domain evaluation | Not observed executed |
| New corrected PD/MT evaluation | Not observed executed |

**Research integrity:** New work belongs on the development branch; do not overwrite frozen models, benchmark data, release tags, paper tables, or unrelated local edits. Once logs and cheap evaluation have been reviewed, add a new status entry with the precise numerical outputs, executed command, saved-checkpoint path, TF/NumPy version manifest and flags, and any scientific interpretation.
