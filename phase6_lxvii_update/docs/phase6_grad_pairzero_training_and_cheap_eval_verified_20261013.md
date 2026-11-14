---

# Part LXVII — Phase 6 `grad_pair_zero`: Verified Training/Inference and Cheap-Evaluation Completion (2026-10-13)

**Evidence status:** Direct Spark terminal outputs, provided on October 13, 2026. This entry supersedes LXVI's "pending training verification" and "cheap evaluation not yet executed" statuses for **only** the `grad_pair_zero` variant; Parts LXV–LXVI are retained intact as chronological historical records. Other arm runs, corrected PD/MT analysis, and new spatial-locality analysis have **not** been established by the supplied logs.

## LXVII.1 Trained arm and provenance

- Variant: `grad_pair_zero`; method: `candidateF_grad_pairzero_expanded2688`.
- Implementing wrapper: `scripts/run_candidateF_cv_pers_ablation.py`.
- Shared trainer: `scripts/run_candidateF_expanded2688_finetune.py` (repaired E2 custom-loop training pathway).
- Loss configuration confirmed in preflight (LXV): `lambda_uv=1`, `lambda_grad=0.05`, `lambda_TTKCV=0`, `lambda_TTKpers=0`; all other auxiliary losses 0; `requires_e2=True` to keep a matched runner and GT-pair data path. **This is a gradient-only *weighted-loss arm within the shared repaired-E2 pathway***, not proof that the network, optimizer graph or RNG is identical to every historical standalone gradient-only trainer.
- Epoch count: 3; learning rate: `1e-5`; batch size: 4; 2,688 training fields; fixed separate 168-field evaluation benchmark; unchanged pretrained PhIRE CNN initialization (see LXV configuration and preflight).
- Native canonical training log: `logs/wind_finetune_candidateF_grad_pairzero_expanded2688.log`.
- Optional full `tee` console transcript, if retained from initial invocation: `logs/phase6_grad_pairzero_console.log`.

## LXVII.2 Verified checkpoint restoration, epoch snapshots, and final inference

The author executed the exact read-only verification command from LXVI.2:

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

**Observed decisive log lines (verbatim values, condensed):**

```text
Loss-magnitude diagnostic (first batch, no training step)
Checkpoint: models_fixed/topology_finetuning/wind_finetune_candidateF_grad_pairzero_expanded2688/cnn00001/cnn
Checkpoint: models_fixed/topology_finetuning/wind_finetune_candidateF_grad_pairzero_expanded2688/cnn00002/cnn
Checkpoint: models_fixed/topology_finetuning/wind_finetune_candidateF_grad_pairzero_expanded2688/cnn00003/cnn
Fine-tuning complete (repaired E2 path).
Fine-tuning complete. Final checkpoint: models_fixed/topology_finetuning/wind_finetune_candidateF_grad_pairzero_expanded2688/cnn/cnn
[OK] idx.npy  shape=(168,)
[OK] dataIN.npy  shape=(168, 100, 100, 2)
[OK] dataGT.npy  shape=(168, 500, 500, 2)
[OK] dataSR.npy  shape=(168, 500, 500, 2)
[OK] idx.npy values: exactly ordered 0..167 (no duplicate/missing samples).
[OK] fixed CNN baseline idx.npy: exactly ordered 0..167.
[OK] dataGT.npy exactly aligned with fixed CNN/GAN benchmark GT (np.array_equal=True, max_abs_diff=0.0000e+00 for diagnostics).
[OK] dataIN.npy exactly aligned with fixed benchmark input (np.array_equal=True, max_abs_diff=0.0000e+00 for diagnostics).
[report] dataGT.npy [u,v] range : u=[-16.1737, 30.8122] v=[-30.4676, 28.1310]
[report] dataSR.npy [u,v] range : u=[-17.8825, 27.6427] v=[-26.8877, 28.6427]
[report] GT scalar-speed range : [0.0122, 33.9885] m/s
[report] SR scalar-speed range : [0.0018, 33.4720] m/s
[report] Mean absolute scalar-speed error (SR vs GT): 0.5275 m/s
[validate] Post-inference validation complete: shapes, exact idx order, finiteness, and (where the baseline was available) GT/IN alignment all passed.
Run complete.
```

**Observed output artifact sizes/times (`ls -lh`, human-rounded):**

| Artifact | Reported size | Modification date |
|---|---:|---|
| `data_out/wind_finetune_candidateF_grad_pairzero_expanded2688/dataGT.npy` | 641M | Oct 13 01:28 |
| `data_out/wind_finetune_candidateF_grad_pairzero_expanded2688/dataIN.npy` | 26M | Oct 13 01:28 |
| `data_out/wind_finetune_candidateF_grad_pairzero_expanded2688/dataSR.npy` | 641M | Oct 13 01:28 |
| `data_out/wind_finetune_candidateF_grad_pairzero_expanded2688/idx.npy` | 1.5K | Oct 13 01:28 |
| `models_fixed/topology_finetuning/wind_finetune_candidateF_grad_pairzero_expanded2688/cnn/cnn.{data-00000-of-00001,index,meta}` | 8.2M, 2.8K, 956K | Oct 13 01:27 |
| `.../cnn00001/cnn.*`, `.../cnn00002/cnn.*`, `.../cnn00003/cnn.*` | ~9.2M each snapshot directory | Oct 13 00:33, 01:00, 01:27 |

**Conclusion:** The actual custom E2-path training completed; epoch snapshots and final CNN weights exist; paired inference produced 168 outputs with finite values, correct shapes and order; GT and input tensors match the authoritative fixed benchmark **exactly**, not just within tolerance. The output speed range has no obvious catastrophic excursion relative to GT; the aggregate physical speed MAE is `0.5275 m/s`. No comparative claim against CNN/GAN/F1 or topology improvement follows from this single absolute metric. The first-batch *loss-magnitude diagnostic was printed*, but its individual loss values and epoch-by-epoch loss values were not included in the provided excerpt and remain to be extracted if needed.

## LXVII.3 First-arm scalar/domain cheap evaluation: Completed, with SSIM unavailable

Executed:

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

**Observed evaluation summary:**

```text
[warning] skimage SSIM unavailable — SSIM will be NaN.
Reason: ValueError('numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject')
Loading bicubic from .../data_out_fixed/wind_mrhr_bicubic
Common samples: 168
Checking GT alignment … OK (max_abs_diff=0.00e+00)
Topology lookup loaded: 336 entries (from psnr_topology_physics_merged.csv)
Computing metrics for [bicubic] … physics … done; components … done
Computing metrics for [cnn] … physics … done; components … done
Computing metrics for [gan] … physics … done; components … done
Computing metrics for [candidateF_grad_pairzero_expanded2688] … physics … done; components … done
Total metric rows: 672
Evaluation complete.
Evaluation exit status: 0
```

**Files written:**

```text
logs/phase6_candidateF_grad_pairzero_expanded2688_cheap_eval.log
ttk_runs_fixed/topology_finetuning/candidateF_grad_pairzero_expanded2688_eval/
  all_sample_metrics_candidateF_grad_pairzero_expanded2688.csv
  winner_counts_candidateF_grad_pairzero_expanded2688.csv
  pairwise_cnn_vs_candidateF_grad_pairzero_expanded2688.csv
  adjacent_cluster_table_candidateF_grad_pairzero_expanded2688.csv
docs/topology_finetuning_candidateF_grad_pairzero_expanded2688_eval.md
```

The `672` rows equal `4 methods × 168 fields`. A successful exit confirms those **non-SSIM** calculations and output writes completed; it **does not** mean all expected metrics were numerically available. Specifically, scikit-image was binary-incompatible with the NumPy environment: the script warned that **SSIM will be NaN**. Do not rank the models by SSIM from these CSVs, silently replace missing values by zeros, or represent this as a successful SSIM calculation. Prefer the existing independently validated clean-environment SSIM procedure with consistent conventions, or repair the ABI issue in a **separate** environment after capturing a dependency manifest. Avoid system-wide `pip install --upgrade` while these trained outputs are being audited.

**Topology caveat:** The `336` lookup entries came from the existing `psnr_topology_physics_merged.csv` (CNN/GAN historical topology references). This is **not** a new corrected PD or MT evaluation for `grad_pair_zero`; those fields require the dedicated topology/distance pipeline, using the paper's **corrected/audited** distance definitions.

## LXVII.4 Reproducible numerical comparison extraction — pending execution

Only output *file paths*, not the report's actual model-vs-model numerical table, were shown in the shared terminal logs. To inspect the new arm fairly, extract **the candidate versus CNN** from its generated pairwise CSV; the following command uses only Python's standard-library `csv`, avoiding NumPy/pandas ABI issues:

```bash
cd ~/PhIRE
METHOD=candidateF_grad_pairzero_expanded2688
python3 - <<'PY'
import csv
from pathlib import Path
method = 'candidateF_grad_pairzero_expanded2688'
p = Path(f'ttk_runs_fixed/topology_finetuning/{method}_eval/pairwise_cnn_vs_{method}.csv')
with p.open(newline='') as f:
    table = {row['metric']: row for row in csv.DictReader(f)}
for key in ('psnruv','speed_mae','speed_rmse','grad_mae','grad_w1',
            'wpd_mae','wpd_w1','exceed_abs_t5','exceed_abs_t10',
            'comp_curve_l1','ssim'):
    row = table.get(key)
    if not row:
        print(f'{key:20s} [not in CSV]')
        continue
    print(f'{key:20s} CNN={row["mean_baseline_cnn"]}  '
          f'pair-zero={row[f"mean_{method}"]}  delta={row["delta"]}  '
          f'better={row["n_improved"]}  worse={row["n_worsened"]}  '
          f'valid={row["n_valid"]}')
PY
sed -n '1,130p' "docs/topology_finetuning_${METHOD}_eval.md"
```

For reproducibility/provenance, also capture a small **post-run** checksum/version manifest (do not invent missing SHA-256 values):

```bash
cd ~/PhIRE
METHOD=candidateF_grad_pairzero_expanded2688
sha256sum \
  scripts/run_candidateF_cv_pers_ablation.py \
  scripts/run_candidateF_expanded2688_finetune.py \
  scripts/evaluate_finetune_candidate.py \
  models_fixed/topology_finetuning/wind_finetune_${METHOD}/cnn/cnn.index \
  models_fixed/topology_finetuning/wind_finetune_${METHOD}/cnn/cnn.data-00000-of-00001 \
  ttk_runs_fixed/topology_finetuning/${METHOD}_eval/pairwise_cnn_vs_${METHOD}.csv
/usr/bin/python3 -m pip show tensorflow numpy scikit-image pandas
```

A clean comparison with the archived F1 model must use the **same** 168 fields and metric definitions. Existing Phase 5E locality and published corrected PD numbers are a **separate evaluation dimension**, not computable from this cheap evaluation alone. After inspecting the report and extracting comparable aggregate metrics, assess whether to launch `grad_cv_only` and `grad_pers_only`, then run the corrected PD/MT and spatial-locality pipelines in a controlled, matched comparison. The TTK pipeline has **not** been run for this arm as of the evidence provided here.

## LXVII.5 Git documentation status and explicit scientific boundaries

The author then extracted `Phase6_GradPairZero_Notes_Update_20261013.zip` and successfully applied `phase6_grad_pairzero_notes_LXVI_incremental.patch` (`git apply --check` followed by `git apply`) to the repository-root `dataset_generation_and_repair_notes.md`. The author was on the `master` worktree in the earlier status report, with unrelated paper figure edits and untracked workpack files. These local changes should not be overwritten or accidentally committed together. **Applying the patch is not a commit or a GitHub push.** Do not alter `topoinvis-2026-arxiv-v1.1` or the frozen release artifacts.

**Updated status (2026-10-13):**

| Phase 6 component | Verified status |
|---|---|
| Static 2×2 arm checks; all three dry-runs; TF compatibility smoke | PASS (LXV) |
| `grad_pair_zero` training, 3 epoch checkpoint series, final weights | **PASS** |
| `grad_pair_zero` inference, 168 fields, shape/finiteness/ordered-index and exact GT/IN alignment | **PASS** |
| `grad_pair_zero` cheap evaluation, 4 methods × 168 fields, CSV/report output | **PASS, SSIM MISSING (NaN)** |
| Mean scalar-speed MAE for `grad_pair_zero` | **0.5275 m/s** (absolute, not comparative) |
| Model-vs-CNN/F1 numerical comparisons from new CSV | **Pending inspection** |
| New corrected PD / merge-tree distances, Phase 6 spatial matching | **NOT EXECUTED / NOT REPORTED** |
| `grad_cv_only` and `grad_pers_only` real training | **NOT REPORTED** |

Retain the original run logs and generated CSVs. The next record should quote actual candidate-vs-CNN, candidate-vs-GAN/F1 metrics, exact CSV filenames, any relevant constraints/checkpoint hashes, and separately tracked topology evaluations. Do not infer improved scientific fidelity from checkpoint existence or a successful exit code alone.
