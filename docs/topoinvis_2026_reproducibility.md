# TopoInVis 2026 / arXiv v1 — Reproducibility Guide

**Paper:** *Topology-Inspired Fine-Tuning for Wind-Field Super-Resolution* (A. Dadhwal, B. Summa)
**Release tag:** [`topoinvis-2026-arxiv-v1`](https://github.com/Ardad2/PhIRE/releases/tag/topoinvis-2026-arxiv-v1)

This repository is a research fork of [PhIRE](https://github.com/NREL/PhIRE) that deliberately keeps its full development history: earlier loss designs, smaller-scale runs, PyTorch refiner experiments, and a superseded persistence-diagram metric. This guide identifies exactly which code, checkpoints and tables produce the results in the paper, and how to check or regenerate them.

Status labels used below:

| Label | Meaning |
|---|---|
| **in repo** | committed at the release tag |
| **asset** | attached to the GitHub release (too large for Git) |
| **regenerate** | git-ignored output; the listed script recreates it |

---

## 1. Check the paper's numbers in one step

The quantitative topology tables and loss-study figures are derived from one per-field table: `corrected_pd_mt_joined.csv`, with 18 methods × 168 benchmark fields, giving d_B, W₂,∞, W₂,₂ and MT distance for each pair. To regenerate the tables and check each value against the manuscript, run:

```bash
git clone https://github.com/Ardad2/PhIRE.git && cd PhIRE
git checkout topoinvis-2026-arxiv-v1

J=reproducibility/topoinvis2026/pd_audit/recompute_pd_w22/corrected_pd_mt/corrected_pd_mt_joined.csv
python3 reproducibility/topoinvis2026/manuscript_tools/make_tables.py  --joined $J --out /tmp/tables
python3 reproducibility/topoinvis2026/manuscript_tools/make_figures.py --joined $J --out /tmp/figures
```

The first command recomputes every mean, median, rank, per-field win count, factorial effect, targeted contrast and Pareto front. It ends with `All derived quantities match the values reported in the manuscript`, or else lists each mismatch and exits with status 1. The only requirements are Python 3, NumPy and Matplotlib. The outputs of this check, made when the release was prepared, are in `reproducibility/topoinvis2026/manuscript_outputs/`.

---

## 2. Where everything is

| Paper element | Location | Status |
|---|---|---|
| Training set (2688 fields, 16 seasonal 168-h windows) | built by `scripts/build_wind_mrhr_expanded_dataset_2688.py` → `example_data_topology_expanded_2688/wind_MR-HR.tfrecord` (`manifest.csv`, `stats.json` committed) | regenerate (needs NREL HSDS access) |
| Evaluation benchmark (168 fields) | `example_data_fixed/wind_MR-HR.tfrecord`; provenance in `repair_wind_mr_hr_from_hr.py`, `build_phire_native_tfrecords.py`, `scripts/audit_wind_mr_hr_pairing.py` | asset (`topoinvis2026_benchmark_wind_MR-HR.tfrecord`) |
| Pretrained PhIRE CNN / GAN | `models/wind_mr-hr/trained_cnn`, `models/wind_mr-hr/trained_gan` | in repo |
| 16 fine-tuned checkpoints | `models_fixed/topology_finetuning/wind_finetune_<run>/` (see §3) | in repo |
| Fixed-pair constraints (64 GT pairs per training field) | `ttk_runs_fixed/topology_finetuning/candidateE2_fixed_lowlambda_expanded2688_constraints/ttk_pd_critical_pairs_gtvalues.npz` | in repo |
| Reconstructions of the 168 benchmark fields | `data_out_fixed/wind_mrhr_{cnn,gan}/`, `data_out/wind_finetune_<run>/` (`idx.npy`, `dataGT.npy`, `dataSR.npy`) | regenerate (§4.3) |
| TTK persistence diagrams and merge trees (VTU) | written by `scripts/run_candidate_topology_pipeline.sh` | regenerate (§4.4) |
| Conventional / domain metrics per field | `ttk_runs_fixed/topology_finetuning/<run>_eval/all_sample_metrics_<run>.csv`; unified: `ttk_runs_fixed/unified_candidate_evaluation/unified_primary_per_sample_long.csv` | in repo |
| Recomputed speed SSIM | `ttk_runs_fixed/ssim_recomputed_topology_inspired_historical_convention/ssim_per_sample.csv`; `ttk_runs_fixed/ssim_recomputed_scale_sweep/ssim_per_sample_scale_sweep.csv` | in repo |
| MT distance per field | `ttk_runs_fixed/unified_candidate_evaluation/unified_primary_per_sample_long.csv`, column `mt_distance` | in repo |
| Corrected PD distances (d_B, W₂,∞, W₂,₂) per run and field | `reproducibility/topoinvis2026/pd_audit/recompute_pd_w22/w22_full_sweep.csv` (51 archived runs × 168 fields) | in repo |
| PD distance implementation | `pd_audit/recompute_pd/canonical_pd_pilot.py` (diagram parsing, D₀/D₁ split), `pd_audit/recompute_pd_w22/w22_full_sweep.py` | in repo |
| Joined PD + MT table and analyses | `pd_audit/recompute_pd_w22/corrected_pd_mt/` (from `scripts/analyze_corrected_pd_mt_tradeoff.py`) | in repo |
| PD validation against GUDHI | `pd_audit/recompute_pd/gudhi_crosscheck_full.py`, `pd_audit/recompute_pd_w22/gudhi_w22_crosscheck_full.py` with CSV and summary outputs | in repo |
| Cubical-complex diagram-construction check | `scripts/run_all168_pd_descriptor_compatibility.py`, `cross_toolkit_closeout_bundle.zip` | in repo |
| Join-tree construction check | `join_tree_harness/` | in repo |
| Complete frozen audit directories, including logs | `gudhi_distance_audit_20260907.tar.gz`, `w22_distance_audit_20260908.tar.gz` | asset |
| Near-tie cohort and predeclared samples | `pd_audit/recompute_pd_w22/select_near_tie_visual_cases.py`, `near_tie_candidateF_grad_E2_vs_cnn_*.csv` | in repo |
| Paper tables and loss-study figures | `reproducibility/topoinvis2026/manuscript_tools/` | in repo |
| Figure 6 (Sample 78) | `scripts/paper_sample78_figure.py` → `figures/paper_sample78/sample078_paper.pdf` | in repo (rerunning needs the reconstructions and diagrams) |
| Loss-scale calibration (Appendix) | `scripts/run_physics_loss_diagnostic.py` → `ttk_runs_fixed/topology_finetuning/loss_magnitude_diagnostic.csv`, `docs/topology_finetuning_loss_diagnostic.md` | in repo |

**SSIM convention.** The manuscript's recomputed speed-SSIM
means use `compute_sample_ssim` from
`scripts/recompute_scale_sweep_ssim.py`. The data range for
each field includes both ground-truth and reconstructed values.

Under this convention, CNN wins 165 of the 168 benchmark
fields against GAN; GAN wins samples 27, 33, and 34.

The older `ttk_runs_fixed/baseline_metrics/all_methods_per_sample.csv`
uses a ground-truth-only data range and yields 168/0.
These conventions must not be mixed.
`scripts/recompute_topology_inspired_ssim.py` also uses
the ground-truth-only convention.

Paths starting `pd_audit/` are under `reproducibility/topoinvis2026/`. Checksums for that directory are in `reproducibility/topoinvis2026/SHA256SUMS`, and for the release assets in the asset `SHA256SUMS`.

---

## 3. Configuration map

The paper names each configuration by the losses it adds to the reconstruction loss L_uv: speed (S), gradient (G), level-set (L), local maximum (M) and fixed-pair supervision (FP). The repository uses historical run names. All 16 configurations fine-tune the native PhIRE TensorFlow generator from the released CNN checkpoint, with 3 epochs, Adam, learning rate 1e-5, batch size 4 and the 2688-field training set.

| Paper label | `method_id` | Run name (`<run>`) | Training command (from repo root) |
|---|---|---|---|
| Pretrained CNN | `cnn` | — | released checkpoint |
| Pretrained GAN | `gan` | — | released checkpoint |
| Reconstruction-only control | `uv` | `candidateUV_expanded2688` | `python3 scripts/run_candidateUV_expanded2688_finetune.py` |
| + S | `speed_only` | `candidateB_factorial_speed_expanded2688` | `python3 scripts/run_candidateB_factorial_expanded2688_finetune.py --variant speed` |
| + L | `levelset_only` | `candidateB_factorial_levelset_expanded2688` | `… --variant levelset` |
| + S + L | `speed_levelset` | `candidateB_factorial_speed_levelset_expanded2688` | `… --variant speed_levelset` |
| + G | `grad_only` | `candidateB_factorial_grad_expanded2688` | `… --variant grad` |
| + S + G | `speed_grad` | `candidateB_factorial_speed_grad_expanded2688` | `… --variant speed_grad` |
| + G + L | `grad_levelset` | `candidateB_factorial_grad_levelset_expanded2688` | `… --variant grad_levelset` |
| + S + G + L | `candidate_b` | `candidateB_expanded2688` | `python3 scripts/run_candidateB_expanded2688_finetune.py` |
| + S + G + L + M | `candidate_c` | `candidateC_expanded2688` | `python3 scripts/run_candidateC_expanded2688_finetune.py` |
| + M | `uv_crit` | `candidateUV_plus_crit_expanded2688` | `python3 scripts/run_candidateUV_plus_crit_expanded2688_refiner.py` |
| + FP | `uv_e2` | `candidateUV_plus_E2_tf_lowlambda_expanded2688` | `python3 scripts/run_candidateUV_plus_E2_tf_lowlambda_expanded2688_ttkcrit_refiner.py` |
| + S + G + L + FP | `b_e2` | `candidateB_plus_E2_tf_lowlambda_expanded2688` | `python3 scripts/run_candidateB_plus_E2_tf_lowlambda_expanded2688_ttkcrit_refiner.py` |
| + S + G + L + M + FP | `c_e2` | `candidateE2_tf_lowlambda_expanded2688` | `python3 scripts/run_candidateE2_tf_lowlambda_expanded2688_ttkcrit_refiner.py` |
| **+ G + FP (final)** | `f1_grad_e2` | `candidateF_grad_E2_low_expanded2688` | `python3 scripts/run_candidateF_expanded2688_finetune.py --variant grad_e2_low` |
| + G + L + FP | `f2_grad_levelset_e2` | `candidateF_grad_levelset_E2_low_expanded2688` | `… --variant grad_levelset_e2_low` |
| + G + M | `f3_grad_crit` | `candidateF_grad_crit_expanded2688` | `… --variant grad_crit` |

Notes:

* Despite the `_refiner` / `_ttkcrit_refiner` suffix, the four scripts above fine-tune the PhIRE generator itself, not a separate residual head. The suffix is historical, as each script's docstring states.
* Loss weights are fixed across configurations: 0.01 (S), 0.05 (G), 0.25 (L), 0.001 (M), and 0.004 L_CV + 0.002 L_pers (FP). The final objective is `L_uv + 0.05 L_grad + 0.004 L_CV + 0.002 L_pers`.
* Each script has a usage block at the top. The F-family scripts support `--print-config` and `--dry-run`; the factorial script supports `--dry-run`.
* The artifact paths for every method (checkpoint, reconstruction, metric CSVs, reports) are listed in `docs/primary_candidate_artifact_reference.md`.

---

## 4. Re-running the pipeline

All commands below are run from the repository root. On the original machine they were run with `PYTHONNOUSERSITE=1 /usr/bin/python3`. TTK command-line tools were run inside Docker by the topology scripts.

### 4.1 Data

```bash
export HS_ENDPOINT="https://developer.nlr.gov/api/hsds"; export HSDS_ENDPOINT="$HS_ENDPOINT"
python3 scripts/build_wind_mrhr_expanded_dataset_2688.py --out-dir example_data_topology_expanded_2688
```

The evaluation benchmark is the release asset `topoinvis2026_benchmark_wind_MR-HR.tfrecord`. Place it at `example_data_fixed/wind_MR-HR.tfrecord`. How it was built and audited is described in Parts I–III of `dataset_generation_and_repair_notes.md`.

### 4.2 Fixed-pair constraints

The constraints file is committed. To rebuild it: compute 160×160 ground-truth persistence diagrams, keep finite pairs with persistence ≥ 1% of the field's maximum finite persistence, and retain the top 64.

```bash
python3 scripts/generate_expanded_cnn_sr.py --data-type wind_mrhr_cnn_expanded2688 \
  --tfrecord example_data_topology_expanded_2688/wind_MR-HR.tfrecord --n-expected 2688 \
  --out-dir data_out/wind_mrhr_cnn_expanded2688
python3 scripts/build_candidateE2_expanded672_ttk_constraints.py \
  --gt-path data_out/wind_mrhr_cnn_expanded2688/dataGT.npy \
  --idx-path data_out/wind_mrhr_cnn_expanded2688/idx.npy --n-expected 2688 \
  --out-dir ttk_runs_fixed/topology_finetuning/candidateE2_fixed_lowlambda_expanded2688_constraints \
  --vti-dir ttk_runs_fixed/topology_finetuning/candidateE2_fixed_lowlambda_expanded2688_vti \
  --vti-label candidateE2fixedlowlambda2688_GT
```

### 4.3 Fine-tuning and reconstructions

Run the commands in §3. Each training script writes checkpoints to `models_fixed/topology_finetuning/wind_finetune_<run>/` and benchmark reconstructions to `data_out/wind_finetune_<run>/`. The baseline reconstructions in `data_out_fixed/wind_mrhr_{cnn,gan}/` come from the original PhIRE test workflow on the benchmark; see `dataset_generation_and_repair_notes.md`.

For the repaired CNN/GAN baseline,
`scripts/run_full_experiment_fixed.sh` provides the historical
inference and topology workflow. For fine-tuned candidates,
`scripts/run_candidate_topology_pipeline.sh` processes
existing reconstructed fields. These are topology-generation
workflows; their historical `pd_distance` outputs are not
the corrected PD distances reported in the manuscript.
The exact historical wrapper invocation has not been verified.


### 4.4 Evaluation

Conventional and domain metrics:

```bash
python3 scripts/evaluate_finetune_candidate.py --candidate-name <run> \
  --candidate-dir data_out/wind_finetune_<run> \
  --cnn-dir data_out_fixed/wind_mrhr_cnn --gan-dir data_out_fixed/wind_mrhr_gan
```

TTK diagrams, merge trees and MT distance, on the fixed `y=0:160, x=0:160` crop:

```bash
bash scripts/run_candidate_topology_pipeline.sh --method <run> --data-dir data_out/wind_finetune_<run> \
  --vti-dir ttk_runs_fixed/topology_finetuning/<run>_topology_vti \
  --out-base ttk_runs_fixed/topology_finetuning/<run>_topology --n-samples 168 --threads 1 --skip-viz
```

The unified per-field table (conventional metrics and MT) is built by `scripts/build_unified_candidate_evaluation.py`.

Corrected PD distances. The canonical audit computes
bottleneck distance and W₂,∞. The separate
`pd_audit/recompute_pd_w22/w22_full_sweep.py` computes
W₂,₂ and carries forward the frozen canonical bottleneck
and W₂,∞ values. D₀ and D₁ are evaluated separately;
their distances are combined using the maximum for
bottleneck distance and root-sum-of-squares for
Wasserstein distances. The non-finite global pair is excluded.

The historical canonical sweep script is
`pd_audit/recompute_pd/run_canonical_pd_full_sweep.py`.
Its original manifest is preserved at
`pd_audit/manifests/pd_result_run_manifest_v2.csv`.
The script still contains Spark-specific path assumptions
and requires historical per-run persistence diagrams and
distance files. Including the manifest does not by itself
make this runner portable.

For independent GUDHI verification, use
`pd_audit/recompute_pd/gudhi_crosscheck_full.py`.
It accepts an explicit repository root and a canonical
reference CSV. The full historical cross-check requires
8,568 comparisons across 51 archived runs, plus their
persistence diagrams. The released numerical results
can be inspected without regenerating those diagrams.

Joined table and analyses. This script writes into `$W22/corrected_pd_mt/`. Point `W22` at a scratch copy so the released files are not overwritten:

```bash
export W22=$(mktemp -d) && cp -r reproducibility/topoinvis2026/pd_audit/recompute_pd_w22/. "$W22"
python3 scripts/analyze_corrected_pd_mt_tradeoff.py      # expects the repository at ~/PhIRE
diff -r "$W22/corrected_pd_mt" reproducibility/topoinvis2026/pd_audit/recompute_pd_w22/corrected_pd_mt
```

**Historical PD warning:** Any historical column named
`pd_distance` represents a superseded TTK quantity, not the
corrected bottleneck, W₂,∞ or W₂,₂ distance.

This includes columns in candidate `*_eval/*.csv`,
historical `*_pd_mt_distances.csv`, unified evaluation
tables, and Phase-2 analysis files.

Use the corrected PD audit metrics instead. The final
analysis script explicitly rejects historical PD columns.

### 4.5 Loss-scale calibration

```bash
python3 scripts/run_physics_loss_diagnostic.py --data-path example_data_fixed/wind_MR-HR.tfrecord \
  --model-path models/wind_mr-hr/trained_cnn/cnn --batch-size 4 --max-batches 0
```

This runs the pretrained CNN on all 168 benchmark fields in 42 batches of four, without updating weights.

### 4.6 Paper tables and figures

The quantitative topology tables and loss-study Figures 4–5 use the frozen joined per-field table (see §1). Figure 6 is a separate qualitative visualization and needs the CNN, control and final reconstructed fields, their audited persistence diagrams, and the same corrected sweep. Assuming the required reconstructions and diagrams have been regenerated or restored, run:

```bash
export AUDIT="$PWD/reproducibility/topoinvis2026/pd_audit"
export W22="$AUDIT/recompute_pd_w22"
python3 scripts/paper_sample78_figure.py
```

The conceptual diagrams and data-pipeline schematic are authored separately in the manuscript LaTeX source. They cannot be regenerated from the per-field distances alone.

---

## 5. Historical names

| Repository term | Paper term |
|---|---|
| `UV`, `candidateUV` | reconstruction-only control |
| `E2`, `repaired E2`, `TTKCV` / `TTKpers` | fixed-pair supervision, L_CV / L_pers |
| `crit`, critical / local-maxima proxy | local-maximum loss (M) |
| `F1`, `candidateF_grad_E2_low` | final objective (G + FP) |
| `Candidate B` / `Candidate C` | S + G + L / S + G + L + M |
| `pd_distance`, TTK "2" | superseded PD quantity; not reported |

---

## 6. Not part of the paper's results

The following are kept for provenance but are not interchangeable with the paper's experiments:

* 168-, 672- and 1344-sample fine-tuning runs (`*_expanded672`, `*_expanded1344`, unsuffixed pilots);
* PyTorch residual-refiner and differentiable-PD experiments (`candidateD*`, `candidateE2_fixed*`, `.mamba_candidateD_pd/`);
* unrepaired early fixed-pair runs and pre-repair VTI outputs;
* the historical PD columns described in §4.4, and the Phase-2 analyses built on them;
* `spatial_pd/`, which is follow-up spatial-persistence work outside this paper;
* `archive/`, `poster_figures/`, `figures/poster_sample78_final/` (poster-era assets), and other development folders not referenced above.

`docs/unified_candidate_evaluation_inventory.md` gives the primary / secondary / deprecated classification used during the experiment audit, and `dataset_generation_and_repair_notes.md` is the complete chronological lab record.

---

## 7. Environment

* PhIRE / TensorFlow: `tf_env.yml`
* Topology analysis: `topo_env_freeze.txt`
* Spark TTK Docker image: `phire-ttk:latest`. Recorded image ID:
  `sha256:858d3154edb75c098ccda15f9f812eecbd0d102af77055906d42a67a390d5c86`.
  The `latest` tag is mutable. The image ID documents the
  original environment but does not guarantee image availability.
* Historical manual-install record:
  `docs/phase2db_manual_topology_qa/authoritative_install_record.txt`
  records TTK 1.3.0 / VTK 9.1.0 for that installation.
  These versions should not be assumed to describe the
  Docker image. Historical Docker build files remain in `archive/`.
* PD audit and GUDHI cross-checks: `pd_audit/*/gudhi-audit-environment.yml`, `pd_audit/*/gudhi_versions.txt`

## 8. Scope of this release

The release contains:
* the code for every training, evaluation and analysis step above;
* all 16 fine-tuned checkpoints and the fixed-pair constraints;
* the per-field sources supporting the quantitative topology
  analyses, together with conventional metric, SSIM and
  loss-calibration data;
* scripts that regenerate the quantitative topology tables
  and loss-study figures, with numerical consistency checks.

Three reproducibility levels should be distinguished:

1. **Directly checkable from released files:** corrected PD/MT
   distances, quantitative topology analyses, GUDHI cross-check
   summaries, conventional metrics, SSIM and loss calibration.
2. **Requires regenerated or restored intermediate data:**
   model reconstructions, persistence diagrams, merge trees
   and the qualitative Sample 78 figure.
3. **Requires further computation or external data access:**
   constructing the 2,688-field training dataset, retraining,
   and rerunning the cubical-complex descriptor comparison.

The cubical-complex comparison is implemented in
`scripts/run_all168_pd_descriptor_compatibility.py`, with
third-party source material under `third_party/tda-toolkit-mapper/`.
Its summary is retained in the research notes, but complete
per-field outputs are not included in this Git snapshot.

Large intermediate data are regenerated by the listed scripts and are not stored in Git: the training TFRecord, the reconstructions, and the TTK diagram files. The full pipeline has not been re-run end to end from a clean machine with a single command.

## 9. Citation

Please cite the paper and the original PhIRE work: K. Stengel, A. Glaws, D. Hettinger, and R. N. King, "Adversarial super-resolution of climatological wind and solar data," *PNAS* 117(29):16805–16815, 2020.
