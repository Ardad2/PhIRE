# PhIRE Wind-Field Super-Resolution Project Record

**Project:** Topology-aware evaluation and fine-tuning of PhIRE wind-field super-resolution  
**Researcher:** Arjun Dadhwal  
**Status date:** July 21, 2026  
**Scope:** Dataset provenance and repair, topology-aware candidate development, expanded ablations, unified Phase-1 evaluation, authoritative Spark validation, preservation of Phase-1 outputs, and Phase-2A kickoff

---

## 1. Purpose of this document

This document consolidates the work completed so far on the PhIRE wind-field super-resolution project into a single project record.

It is intended to serve as:

1. a handoff document for continuing the work in a new chat or coding session;
2. a reproducibility and audit record;
3. a source for later paper writing;
4. a map connecting the many candidate families, datasets, scripts, result files, and validation steps;
5. a clear statement of what is complete, what is authoritative, and what remains pending.

This master record summarizes the main scientific and engineering history. The two large source notes remain the command-level and experiment-level references:

- `loss_candidates.MD`
- `dataset_generation_and_repair_notes.md`

The Phase-1 generated reports are also authoritative references:

- `docs/unified_candidate_evaluation_phase1.md`
- `docs/unified_candidate_evaluation_inventory.md`
- `docs/primary_candidate_artifact_reference.md`

---

# Part I — Scientific objective and representation

## 2. Core scientific question

The project studies whether topology-inspired and topology-derived losses can improve scientific super-resolution of wind fields beyond what is captured by conventional pointwise metrics.

The central observation motivating the work is that:

- a fidelity-oriented CNN can produce strong pointwise accuracy while smoothing or altering important structural features;
- an adversarial GAN can produce poorer PSNR/SSIM while sometimes preserving persistence-diagram structure better;
- persistence diagrams and merge trees do not necessarily respond to the same training signals;
- therefore, scientific SR should be evaluated across fidelity, physical, distributional, threshold-geometry, and topological metrics rather than with a single metric.

The current paper-level scientific direction is:

> Differentiable scalar-field losses influence persistence-diagram and merge-tree agreement in descriptor-specific ways. Gradient supervision is the dominant driver of persistence-diagram improvement, while repaired TTK critical-pair supervision is the dominant driver of merge-tree improvement.

## 3. Model representation

All learned candidates start from the pretrained PhIRE CNN.

The model predicts normalized vector components:

\[
\hat{u}_{\mathrm{norm}},\hat{v}_{\mathrm{norm}}.
\]

Physical components are recovered using the dataset normalization:

\[
u_{\mathrm{phys}}=\sigma_u u_{\mathrm{norm}}+\mu_u,
\qquad
v_{\mathrm{phys}}=\sigma_v v_{\mathrm{norm}}+\mu_v.
\]

Scalar wind speed is:

\[
s(x,y)=\sqrt{u(x,y)^2+v(x,y)^2}.
\]

The base reconstruction objective is:

\[
L_{uv}=
\operatorname{MSE}
\left(
[\hat u_{\mathrm{norm}},\hat v_{\mathrm{norm}}],
[u_{\mathrm{norm}},v_{\mathrm{norm}}]
\right).
\]

This preserves the full vector field. Scalar speed alone cannot recover direction, because different vectors can have the same magnitude.

## 4. Evaluation representation

The unified evaluation distinguishes between vector and scalar representations:

- **Vector-field fidelity:** `psnruv`
- **Scalar speed:** SSIM, speed MAE, speed RMSE
- **Wind-power distribution:** WPD MAE, WPD Wasserstein-1, absolute WPD bias
- **Gradient distribution:** gradient MAE, gradient Wasserstein-1, gradient-kurtosis deviation
- **Frequency domain:** PSD log-L2, spectral-slope deviation
- **Threshold geometry:** exceedance-fraction errors and connected-component errors
- **Topology:** persistence-diagram distance and merge-tree distance

Lower is better for all error and distance metrics. Higher is better for PSNR and SSIM.

The unified analysis defines an improvement-oriented paired difference so that positive always means better than CNN.

---

# Part II — Dataset provenance, repair, and authoritative benchmark

## 5. Original dataset provenance

The historical 168-sample wind dataset was generated from the NREL WIND Toolkit national gridded dataset:

- upstream path: `/nrel/wtk-us.h5`
- access: NREL HSDS
- client: `h5pyd`
- variables:
  - `windspeed_100m`
  - `winddirection_100m`
- spatial center: approximately \(39.5^\circ\mathrm{N},75.0^\circ\mathrm{W}\)
- native crop: \(500\times500\)
- temporal range: indices `0..167`
- corresponding period: January 1–7, 2007, hourly
- vector conversion:
  - \(u=-s\sin(\theta)\)
  - \(v=-s\cos(\theta)\)

The historical hierarchy was:

- HR: `(168, 500, 500, 2)`
- MR: `(168, 100, 100, 2)`
- LR: `(168, 10, 10, 2)`

The original pipeline was reconstructed using `build_example_data_extension_500.py`.

Although regenerated TFRecords were not byte-identical to archived TFRecords, reinstalling them and rerunning paired CNN/GAN inference reproduced all downstream arrays exactly. This established the historical generation path functionally.

## 6. Repair history

A later audit found that the original MR construction path was flawed. The project therefore separates:

1. **Historical provenance:** how the original 168-sample dataset was produced.
2. **Authoritative corrected pipeline:** the repaired dataset used for final scientific claims.

The repaired benchmark is the source of truth for the current work.

Primary corrected artifacts include:

```text
example_data_fixed/wind_MR-HR.tfrecord
data_out_fixed/wind_mrhr_cnn/
data_out_fixed/wind_mrhr_gan/
ttk_runs_fixed/combined/
ttk_runs_fixed/near_tie_study/
ttk_runs_fixed/selector_ablation_full/
ttk_runs_fixed/metric_trends/
ttk_runs_fixed/figure_sets/
```

The repaired pairing audit passed all 168 samples.

## 7. Fixed benchmark used in the unified evaluation

The authoritative evaluation benchmark contains 168 corrected samples.

For the unified Phase-1 analysis, expected shapes were:

```text
idx.npy     (168,)
dataIN.npy  (168, 100, 100, 2)
dataGT.npy  (168, 500, 500, 2)
dataSR.npy  (168, 500, 500, 2)
```

The CNN benchmark arrays under:

```text
data_out_fixed/wind_mrhr_cnn/
```

were used as canonical references for sample order, input alignment, and ground-truth alignment.

Every primary method was required to satisfy:

- sample indices exactly `0..167`;
- exact input equality with canonical CNN `dataIN.npy`;
- exact ground-truth equality with canonical CNN `dataGT.npy`;
- correct SR shape;
- all SR values finite.

Bicubic was exempt only from requiring a separately generated `idx.npy`.

---

# Part III — Candidate loss families

## 8. Candidate UV: vector-only fine-tuning control

\[
L_{\mathrm{UV}}=L_{uv}.
\]

Purpose:

- isolate the effect of fine-tuning itself;
- determine whether improvements from topology-inspired candidates arise merely from additional optimization.

At the expanded 2,688-sample scale:

- PD mean: `29.6121`
- MT mean: `6.0119`

UV did not produce the PD gains found in gradient-containing candidates.

## 9. Candidate B: scalar-field proxy scaffold

\[
L_B =
L_{uv}
+0.01L_{\mathrm{speed}}
+0.05L_{\mathrm{grad}}
+0.25L_{\mathrm{levelset}}.
\]

Definitions:

\[
L_{\mathrm{speed}}
=
\operatorname{MSE}(s_{\mathrm{SR}},s_{\mathrm{GT}})
\]

\[
L_{\mathrm{grad}}
=
\operatorname{MSE}
\left(
|\nabla s_{\mathrm{SR}}|,
|\nabla s_{\mathrm{GT}}|
\right)
\]

For thresholds \(T=\{5,10,15\}\) m/s:

\[
M_\tau(s)=\sigma(k(s-\tau)),\qquad k=10
\]

\[
L_{\mathrm{levelset}}
=
\frac{1}{|T|}
\sum_{\tau\in T}
\operatorname{MSE}
\left(
M_\tau(s_{\mathrm{SR}}),
M_\tau(s_{\mathrm{GT}})
\right).
\]

Expanded 2,688-sample result:

- PD mean: `22.7070`
- MT mean: `6.1612`

## 10. Candidate C: local-maxima / critical-value proxy

\[
L_C=L_B+0.001L_{\mathrm{crit}}.
\]

Prominent GT local maxima are selected using a \(3\times3\) neighborhood and an adaptive high-speed threshold:

\[
s_{\mathrm{GT}}(x,y)\geq\mu_s+\sigma_s.
\]

At these fixed GT-selected locations:

\[
L_{\mathrm{crit}}
=
\frac{
\sum M_{\max}(x,y)
(s_{\mathrm{SR}}(x,y)-s_{\mathrm{GT}}(x,y))^2
}{
\max(\sum M_{\max},1)
}.
\]

Expanded 2,688-sample result:

- PD mean: `22.4944`
- MT mean: `6.0803`

Candidate C became the main result in the submitted short paper because it produced strong PD improvement while remaining practical and differentiable. Its MT improvement was less robust across scale.

## 11. Candidate D: differentiable persistence-diagram loss

Candidate D used a PyTorch residual refiner on top of frozen CNN output:

\[
\hat u_D
=
\hat u_{\mathrm{CNN}}
+
\eta R_\theta(\hat u_{\mathrm{CNN}}),
\qquad \eta=0.1.
\]

Its objective included a differentiable Wasserstein-style cubical-persistence loss.

The approach was feasible but did not improve the final TTK PD/MT metrics. Likely reasons include mismatch between the training topology formulation and the final TTK evaluation metric, as well as architecture and optimization differences.

Candidate D remains a useful negative result rather than a primary method in the 19-method unified comparison.

## 12. Original Candidate E and repaired E2

Candidate E followed the topology-aware neural interpolation idea more directly:

1. run TTK offline on GT;
2. identify persistence-pair birth/death vertices;
3. supervise scalar values at those fixed vertices during training.

For pairs \(P=\{(b_i,d_i)\}\):

\[
L_{\mathrm{TTKCV}}
=
\frac{1}{2|P|}
\sum_{(b_i,d_i)\in P}
\left[
(\hat s(b_i)-s_{\mathrm{GT}}(b_i))^2+
(\hat s(d_i)-s_{\mathrm{GT}}(d_i))^2
\right]
\]

\[
L_{\mathrm{TTKpers}}
=
\frac{1}{|P|}
\sum_{(b_i,d_i)\in P}
\left(
|\hat s(b_i)-\hat s(d_i)|
-
|s_{\mathrm{GT}}(b_i)-s_{\mathrm{GT}}(d_i)|
\right)^2.
\]

The original E implementation was confounded by:

- inconsistent vector-loss normalization;
- incorrect VTI/VTK vertex mapping;
- incorrect target-value convention;
- overly strong topology weights;
- residual-refiner architecture differences;
- fragile topology extraction.

The repaired low-lambda E2 formulation uses:

\[
0.004L_{\mathrm{TTKCV}}+0.002L_{\mathrm{TTKpers}}.
\]

Repairs included:

- normalized \(L_{uv}\);
- C-order VTI flattening;
- direct GT scalar values at selected vertex IDs;
- low-lambda weighting;
- native TensorFlow/PhIRE implementation;
- resumable single-thread TTK evaluation.

## 13. Candidate F descriptor recombination study

Candidate F was designed after the factorial and repaired-E2 studies to isolate the strongest descriptor-specific terms.

### F1: gradient + repaired E2

\[
L_{F1}
=
L_{uv}
+0.05L_{\mathrm{grad}}
+0.004L_{\mathrm{TTKCV}}
+0.002L_{\mathrm{TTKpers}}.
\]

Result:

- PD mean: `23.8382`
- MT mean: `5.6566`

### F2: gradient + level-set + repaired E2

\[
L_{F2}
=
L_{uv}
+0.05L_{\mathrm{grad}}
+0.25L_{\mathrm{levelset}}
+0.004L_{\mathrm{TTKCV}}
+0.002L_{\mathrm{TTKpers}}.
\]

Result:

- PD mean: `23.7481`
- MT mean: `5.6742`

### F3: gradient + local-maxima proxy

\[
L_{F3}
=
L_{uv}
+0.05L_{\mathrm{grad}}
+0.001L_{\mathrm{crit}}.
\]

Result:

- PD mean: `22.0179`
- MT mean: `5.9840`

All three Candidate F variants completed training, cheap evaluation, and TTK evaluation.

---

# Part IV — Expanded factorial and descriptor-specific conclusions

## 14. Primary 19-method comparison set

The authoritative primary comparison contains 19 methods:

### Baselines

1. bicubic
2. CNN
3. GAN

### Vector/scalar factorial and full candidates

4. UV
5. speed only
6. level-set only
7. speed + level-set
8. gradient only
9. speed + gradient
10. gradient + level-set
11. Candidate B
12. Candidate C

### Targeted critical/E2 ablations

13. UV + critical proxy
14. UV + repaired E2
15. Candidate B + repaired E2
16. Candidate C + repaired E2

### Candidate F recombinations

17. gradient + repaired E2
18. gradient + level-set + repaired E2
19. gradient + critical proxy

## 15. Authoritative topology means

| Method ID | Description | PD mean | MT mean |
|---|---|---:|---:|
| `cnn` | pretrained CNN | 27.4063 | 5.8678 |
| `gan` | pretrained GAN | 20.8641 | 8.3481 |
| `uv` | vector-only control | 29.6121 | 6.0119 |
| `speed_only` | speed | 29.5783 | 5.9996 |
| `levelset_only` | level-set | 29.5953 | 6.0076 |
| `speed_levelset` | speed + level-set | 29.4363 | 5.9441 |
| `grad_only` | gradient | 22.9326 | 6.0560 |
| `speed_grad` | speed + gradient | 22.9706 | 6.2905 |
| `grad_levelset` | gradient + level-set | 22.6194 | 6.1996 |
| `candidate_b` | speed + gradient + level-set | 22.7070 | 6.1612 |
| `candidate_c` | Candidate B + critical proxy | 22.4944 | 6.0803 |
| `uv_crit` | UV + critical proxy | 29.1143 | 5.6899 |
| `uv_e2` | UV + repaired E2 | 25.0721 | 5.5940 |
| `b_e2` | Candidate B + repaired E2 | 23.9876 | 5.6774 |
| `c_e2` | Candidate C + repaired E2 | 24.2686 | 5.6628 |
| `f1_grad_e2` | gradient + repaired E2 | 23.8382 | 5.6566 |
| `f2_grad_levelset_e2` | gradient + level-set + repaired E2 | 23.7481 | 5.6742 |
| `f3_grad_crit` | gradient + critical proxy | 22.0179 | 5.9840 |

Bicubic was included in cheap metrics but had no required topology entry.

## 16. Main descriptor-specific interpretation

The expanded factorial and Candidate F experiments support the following interpretation.

### Gradient supervision

Gradient is the dominant driver of PD improvement.

Evidence:

- UV, speed-only, and level-set-only remain near PD `29.4–29.6`.
- Adding gradient reduces PD to approximately `22.6–23.0`.
- F3 achieves the strongest learned-candidate PD mean, `22.0179`.

### Repaired E2 supervision

Repaired TTK critical-pair supervision is the dominant driver of MT improvement.

Evidence:

- `uv_e2` reaches MT `5.5940`, the best mean MT among the primary candidates.
- F1 and F2 preserve strong MT improvement while retaining much of the gradient-driven PD gain.
- E2 improves MT even without Candidate B or Candidate C scaffolds.

### Local-maxima proxy

The critical/local-maxima proxy is a secondary PD enhancer and a context-dependent MT term.

Evidence:

- Candidate C improves PD slightly relative to Candidate B.
- F3 improves PD relative to gradient-only.
- `uv_crit` improves MT relative to UV, but not as strongly as repaired E2.
- The critical proxy is not equivalent to TTK critical-pair supervision.

### Level-set term

The level-set term is an interaction/modulation term rather than an independent topology driver.

Evidence:

- level-set only does not materially improve PD;
- level-set paired with gradient gives a modest PD improvement;
- F2 differs only slightly from F1, so the difference should be described cautiously.

### Speed term

The speed term is not a topology driver and can hurt MT when paired with gradient.

## 17. Current topology Pareto interpretation

Using mean PD and mean MT as two objectives, the main tradeoff sequence previously identified is:

```text
GAN → F3 → F2 → F1 → UV+E2
```

This reflects movement from stronger PD toward stronger MT.

It should not be treated as a single scalar ranking.

---

# Part V — Unified Phase-1 evaluation

## 18. Phase-1 objective

Phase 1 created a single authoritative per-sample table combining all metrics across the 19 primary methods.

The requirements were:

- one row per method × sample;
- 19 methods × 168 samples;
- no duplicate keys;
- preserve missing values;
- positive paired improvement always means better than CNN;
- validate all paths, indices, joins, raw arrays, topology values, and known means;
- do not run training, cheap evaluation, or TTK;
- do not manufacture missing values;
- keep primary and secondary experiments distinct.

## 19. Authoritative metric set

The Phase-1 schema contains 22 metrics:

1. `psnruv`
2. `ssim_speed`
3. `speed_mae`
4. `speed_rmse`
5. `wpd_mae`
6. `wpd_w1`
7. `wpd_bias_abs`
8. `grad_mae`
9. `grad_w1`
10. `grad_kurtosis_abs_delta`
11. `psd_log_l2`
12. `psd_slope_abs_delta`
13. `exceed_abs_t5`
14. `exceed_abs_t10`
15. `exceed_abs_t15`
16. `exceed_abs_p90`
17. `comp_curve_l1`
18. `comp_abs_t5`
19. `comp_abs_t10`
20. `comp_abs_t15`
21. `pd_distance`
22. `mt_distance`

SSIM was treated as optional-but-audited because of the known NumPy/scikit-image ABI problem. It was required to be either:

- complete: `168/168` finite;
- globally unavailable: `0/168` finite.

Partial coverage was prohibited.

## 20. Builder development chronology

The unified builder evolved through several audited commits.

### Commit `dfac9744`

Initial implementation of:

```text
scripts/build_unified_candidate_evaluation.py
```

It created:

- manifest and inventory;
- long and summary tables;
- strict and permissive modes;
- initial validation framework.

The lightweight checkout correctly reported that most large experiment artifacts were unavailable because they were gitignored.

### Commit `09a22402`

Added:

- harvesting of repeated bicubic/CNN/GAN rows from all candidate cheap-evaluation CSVs;
- cross-source consistency checks;
- comparison with the older combined baseline pipeline;
- strict mode as the default;
- human-readable primary artifact reference.

### Commit `7d7a6bd5`

Added:

- SSIM optional-but-audited handling;
- three missingness categories:
  - `no_source_artifact`
  - `unavailable_global_dependency`
  - `partial_source_coverage`
- raw-array validation;
- memory-mapped array loading;
- chunked input and GT comparison;
- exact sample-order checks;
- SR shape and finiteness checks.

### Commit `887ced2e`

Closed the baseline coverage blind spot:

- every required cheap metric must be complete in every baseline source;
- SSIM availability is resolved per source and per metric;
- finite SSIM is preserved when another source has all-NaN SSIM;
- availability mismatches are recorded rather than represented as numeric zero;
- noninteger `idx.npy` values can no longer pass through lossy integer truncation.

### Commit `749c8987`

Fixed the topology-column schema bug found on Spark:

Actual candidate topology columns were method-suffixed, for example:

```text
pd_distance_candidateF_grad_E2_low_expanded2688
mt_distance_candidateF_grad_E2_low_expanded2688
```

The original parser expected generic columns:

```text
pd_distance
mt_distance
```

The fix added strict resolution in this order:

1. generic exact column;
2. exact method-suffixed column;
3. unique prefix match.

It also added:

- ambiguous-schema hard failures;
- PD/MT suffix consistency checks;
- `pos_idx == sample_idx` validation;
- source-column reporting in inventory and column mapping;
- no topology row count until all values parse successfully.

## 21. Strict versus permissive modes

### Strict mode

```bash
python3 scripts/build_unified_candidate_evaluation.py --strict-primary
```

Strict mode refuses to write authoritative unified tables unless every primary method passes.

### Audit mode

```bash
python3 scripts/build_unified_candidate_evaluation.py --audit-allow-missing
```

Audit mode was used only in the lightweight checkout to test code honestly when large gitignored artifacts were absent.

The lightweight outputs were explicitly non-authoritative.

---

# Part VI — Spark validation and debugging chronology

## 22. Preliminary Spark checks

The script compiled successfully.

Spark contained:

```text
41
```

candidate `all_sample_metrics_*.csv` files.

Candidate F path checks confirmed the presence of:

- cheap-evaluation CSV;
- topology CSV;
- `idx.npy`;
- `dataIN.npy`;
- `dataGT.npy`;
- `dataSR.npy`.

Representative sizes were:

- `dataIN.npy`: about 26 MB
- `dataGT.npy`: about 641 MB
- `dataSR.npy`: about 641 MB
- `idx.npy`: about 1.5 KB

## 23. First strict-run failure

The first strict Spark run exited with status `1`.

All 16 learned candidates reported:

```text
pd_distance not finite/nonnegative for all 168 samples
mt_distance not finite/nonnegative for all 168 samples
```

Inventory inspection revealed:

- `row_count_topology = 168`
- topology path resolved exactly
- topology means were NaN
- cheap metrics and raw arrays passed

The actual topology headers were method-suffixed. The loader counted the rows but failed to read the distances.

No topology values were actually negative, and no TTK rerun was needed.

## 24. Topology schema survey

A repository-wide header survey showed that candidate topology outputs consistently used:

```text
pd_distance_<method>
mt_distance_<method>
```

This directly motivated commit `749c8987`.

## 25. Post-patch topology smoke test

After applying the topology resolver patch, all 16 learned candidates passed an isolated topology parser test.

Representative outputs:

```text
uv                       PD=29.612120 MT=6.011866 PASS
candidate_c              PD=22.494387 MT=6.080287 PASS
f1_grad_e2               PD=23.838249 MT=5.656570 PASS
f2_grad_levelset_e2      PD=23.748108 MT=5.674230 PASS
f3_grad_crit             PD=22.017884 MT=5.984007 PASS
```

All schemas resolved as:

```text
exact_method_suffixed_columns
```

## 26. Successful authoritative strict run

The final authoritative run exited with:

```text
STRICT AUDIT EXIT STATUS: 0
```

All strict checks passed:

```text
bicubic
cnn
gan
uv
speed_only
levelset_only
speed_levelset
grad_only
speed_grad
grad_levelset
candidate_b
candidate_c
uv_crit
uv_e2
b_e2
c_e2
f1_grad_e2
f2_grad_levelset_e2
f3_grad_crit
```

The conclusion was:

```text
RESULT: 19/19 primary methods have real per-sample data.
Strict criteria all-pass: True.
```

## 27. Validated raw-array results

For every primary method:

- index status: exact `0..167`;
- input alignment: exact;
- GT alignment: exact;
- SR shape: exact;
- SR finiteness: all finite.

The canonical CNN arrays were loaded with memory mapping.

## 28. Baseline consistency results

The script discovered 41 candidate evaluation CSVs.

For bicubic, CNN, and GAN:

- all 41 repeated sources passed required-metric consistency checks;
- the deterministic canonical baseline source was:
  - `candidateB_eval/all_sample_metrics_candidateB.csv`;
- cross-pipeline numeric differences were negligible;
- CNN/GAN SSIM showed an availability mismatch between harvested and legacy pipelines, which was explicitly recorded rather than treated as a numeric disagreement.

---

# Part VII — Authoritative Phase-1 outputs

## 29. Generated files and dimensions

Under:

```text
ttk_runs_fixed/unified_candidate_evaluation/
```

the strict run generated:

| File | Size in rows/shape |
|---|---:|
| `method_inventory.csv` | 43 rows |
| `column_mapping.csv` | 80 rows |
| `unified_primary_per_sample_long.csv` | 3,192 rows |
| `unified_primary_method_summary.csv` | 418 rows |
| `unified_primary_topology_validation.csv` | 18 rows |
| `unified_primary_pairwise_vs_cnn.csv` | 396 rows |
| `unified_primary_missingness.csv` | 418 rows |
| `unified_primary_wide.csv` | 168 rows × 419 columns |

The topology validation result was:

```text
PASS = 18
NO_DATA = 0
FAIL = 0
```

The long table is the primary source of truth for Phase 2.

## 30. Documentation generated by Phase 1

```text
docs/unified_candidate_evaluation_phase1.md
docs/unified_candidate_evaluation_inventory.md
docs/primary_candidate_artifact_reference.md
logs/build_unified_candidate_evaluation.log
```

---

# Part VIII — Preservation and checksum verification

## 31. Frozen archive

Phase 1 was archived as:

```text
unified_candidate_evaluation_phase1_2026-07-21.tar.gz
```

The archive contains:

```text
ttk_runs_fixed/unified_candidate_evaluation/
docs/unified_candidate_evaluation_phase1.md
docs/unified_candidate_evaluation_inventory.md
docs/primary_candidate_artifact_reference.md
logs/build_unified_candidate_evaluation.log
```

## 32. Phase-1 checksums

Checksums were written to:

```text
unified_candidate_evaluation_phase1_checksums.sha256
```

All listed Phase-1 CSVs and the Phase-1 report passed:

```text
OK
```

## 33. Archive checksum

The archive itself was checksummed into:

```text
unified_candidate_evaluation_phase1_archive.sha256
```

Phase 1 is therefore:

- validated;
- immutable for downstream analysis;
- archived;
- checksum-verifiable.

---

# Part IX — Phase-2 analysis plan

## 34. Phase-2 structure

Phase 2 is divided into four stages.

### Phase 2A — Descriptive and paired analysis

Questions:

- What is the distribution of each metric for each method?
- How does each method compare with CNN on the paired 168 samples?
- How often does each method win, tie, or lose?
- What benchmark-sample uncertainty is present?
- Can Phase-1 summary and pairwise tables be independently reproduced?

### Phase 2B — Loss-factor and targeted-contrast analysis

Planned analyses:

- speed × gradient × level-set factorial;
- main effects and interactions;
- targeted contrasts for critical proxy and repaired E2;
- descriptor-specific causal interpretation, with training-seed caveats.

### Phase 2C — Metric relationships and Pareto analysis

Planned analyses:

- pooled and within-method correlations;
- fidelity–physics–topology relationships;
- PD–MT disagreement;
- Pareto-optimal methods;
- no arbitrary composite score.

### Phase 2D — Sample archetypes and visualization

Planned analyses:

- algorithmic sample selection;
- fields and error maps;
- PD and MT views;
- critical points and connected components;
- representative failure and success archetypes.

## 35. Phase-2A prompt submitted to Codex

A detailed Phase-2A implementation request has been sent to Codex.

The requested script is:

```text
scripts/analyze_unified_candidate_metrics_phase2a.py
```

Requested output directory:

```text
ttk_runs_fixed/unified_candidate_analysis/phase2a/
```

Requested report:

```text
docs/unified_candidate_analysis_phase2a.md
```

Requested log:

```text
logs/unified_candidate_analysis_phase2a.log
```

## 36. Requested Phase-2A outputs

The Phase-2A request includes:

```text
phase2a_validation.csv
metric_coverage.csv
method_descriptive_summary.csv
paired_vs_cnn_detailed.csv
paired_vs_cnn_adjusted.csv
method_mean_improvement_matrix.csv
method_win_rate_matrix.csv
topology_tradeoff_summary.csv
topology_tradeoff_summary_sorted.csv
phase1_pairwise_reproduction.csv
phase1_immutability_check.csv
```

Methodological requirements include:

- standard-library CSV and NumPy where possible;
- no environment changes;
- deterministic bootstrap with seed `20260721`;
- 10,000 paired sample-axis bootstrap resamples;
- exact two-sided sign tests;
- Wilcoxon only if SciPy imports cleanly;
- Holm corrections globally and within metric;
- no aggregate weighted ranking;
- explicit caveat that the 168 samples are benchmark samples and each model was trained only once;
- pre/post SHA-256 verification that Phase-1 files remain unchanged;
- deterministic byte-identical outputs across reruns.

## 37. Current status

As of this document:

- Phase 1 is complete and frozen.
- The Phase-2A prompt has been sent to Codex.
- Phase 2A has not yet been reviewed or run authoritatively on Spark.
- Phase 2B has not begun.

---

# Part X — Current scientific conclusions

## 38. What the evidence currently supports

The completed experiments support these claims:

1. **Gradient supervision is the primary persistence-diagram driver.**
2. **Repaired TTK critical-pair supervision is the primary merge-tree driver.**
3. **PD and MT respond to different structural signals.**
4. **The local-maxima proxy is useful but is not equivalent to repaired E2.**
5. **Level-set supervision is mainly an interaction term.**
6. **Speed supervision is not a topology driver and can worsen MT in some combinations.**
7. **GAN remains strongest in mean PD but weak in mean MT.**
8. **UV+E2 produces the strongest primary-candidate mean MT.**
9. **F1/F2 provide balanced PD–MT compromises.**
10. **F3 gives the strongest learned-candidate mean PD but does not beat CNN in mean MT.**

## 39. What should not yet be claimed

The current results do not establish:

- robustness across independent training seeds;
- universal superiority of any single method;
- causal loss attribution outside the controlled factorial/contrast structure;
- a globally optimal scalar ranking across all metrics;
- that PD and MT are interchangeable topology measures;
- that benchmark-sample confidence intervals represent training-run uncertainty.

## 40. Current paper directions

### Submitted short paper

Main story:

- Candidate C;
- UV control;
- training-scale study;
- robust PD improvement;
- limited MT improvement;
- topology-inspired scalar-field losses.

### Follow-up ablation/descriptor paper

Strongest current direction:

> Persistence diagrams and merge trees respond differently to differentiable scalar-field supervision in scientific super-resolution.

Potential central evidence:

- Candidate B factorial;
- repaired E2 ablations;
- Candidate F recombinations;
- multi-metric Phase-2 analysis.

### Additional future directions

- merge-tree-aware GAN or hierarchy-aware SR;
- more explicit branch/hierarchy supervision;
- multi-seed training;
- broader datasets and regions;
- alternative PD distances and filtration conventions;
- benchmark/evaluation paper if broadened substantially.

---

# Part XI — Key artifact map

## 41. Primary candidate naming pattern

For internal method name `<METHOD>`:

```text
Model:
models_fixed/topology_finetuning/wind_finetune_<METHOD>/

Inference arrays:
data_out/wind_finetune_<METHOD>/

Cheap evaluation:
ttk_runs_fixed/topology_finetuning/<METHOD>_eval/
  all_sample_metrics_<METHOD>.csv
  pairwise_cnn_vs_<METHOD>.csv

Topology:
ttk_runs_fixed/topology_finetuning/<METHOD>_topology/
  <METHOD>_pd_mt_distances.csv
  <METHOD>_topology_comparison.csv

Reports:
docs/topology_finetuning_<METHOD>_eval.md
docs/topology_finetuning_<METHOD>_topology_eval.md
```

## 42. Exact internal names for primary learned methods

| Method ID | Internal artifact name |
|---|---|
| `uv` | `candidateUV_expanded2688` |
| `speed_only` | `candidateB_factorial_speed_expanded2688` |
| `levelset_only` | `candidateB_factorial_levelset_expanded2688` |
| `speed_levelset` | `candidateB_factorial_speed_levelset_expanded2688` |
| `grad_only` | `candidateB_factorial_grad_expanded2688` |
| `speed_grad` | `candidateB_factorial_speed_grad_expanded2688` |
| `grad_levelset` | `candidateB_factorial_grad_levelset_expanded2688` |
| `candidate_b` | `candidateB_expanded2688` |
| `candidate_c` | `candidateC_expanded2688` |
| `uv_crit` | `candidateUV_plus_crit_expanded2688` |
| `uv_e2` | `candidateUV_plus_E2_tf_lowlambda_expanded2688` |
| `b_e2` | `candidateB_plus_E2_tf_lowlambda_expanded2688` |
| `c_e2` | `candidateE2_tf_lowlambda_expanded2688` |
| `f1_grad_e2` | `candidateF_grad_E2_low_expanded2688` |
| `f2_grad_levelset_e2` | `candidateF_grad_levelset_E2_low_expanded2688` |
| `f3_grad_crit` | `candidateF_grad_crit_expanded2688` |

Historical aliases:

```text
uv_crit = C minus the B scaffold
uv_e2   = E minus C and B
b_e2    = E minus C
c_e2    = E
```

---

# Part XII — Reproducibility checkpoints

## 43. Phase-1 compile and run

```bash
cd ~/PhIRE

python3 -m py_compile \
  scripts/build_unified_candidate_evaluation.py

set -o pipefail

{ /usr/bin/time -v \
  python3 scripts/build_unified_candidate_evaluation.py \
    --strict-primary; \
} 2>&1 | tee logs/build_unified_candidate_evaluation.log

status=${PIPESTATUS[0]}
echo "STRICT AUDIT EXIT STATUS: ${status}"
```

Expected:

```text
STRICT AUDIT EXIT STATUS: 0
```

## 44. Strict-check verification

```bash
grep '^\[strict-check:' \
  logs/build_unified_candidate_evaluation.log
```

Expected: 19 `PASS` lines.

## 45. Archive verification

```bash
tar -tzf \
  unified_candidate_evaluation_phase1_2026-07-21.tar.gz \
  | head -n 30

sha256sum -c \
  unified_candidate_evaluation_phase1_checksums.sha256
```

Expected: every listed file reports `OK`.

---

# Part XIII — Important caveats and engineering notes

## 46. Python environment

Spark currently has NumPy 2.x with some optional packages compiled against NumPy 1.x.

Observed noisy imports include:

- `numexpr`
- `bottleneck`
- scikit-image-related ABI issues

The unified Phase-1 script avoids pandas and uses:

- standard-library CSV;
- NumPy;
- memory mapping.

Phase 2A was explicitly requested to avoid changing the environment.

## 47. TTK reproducibility

The repaired topology workflow includes:

- corrected VTI/vertex mapping;
- skip-completed behavior;
- retry only missing outputs;
- hard gates for incomplete extraction;
- single-thread execution for stability.

Small legacy reconstruction drift in merge-tree distance was previously observed and documented, but the fixed primary candidate topology means are now validated exactly against their expected values within `1e-4`.

## 48. Statistical interpretation

Every primary model was trained once.

Therefore:

- paired tests and bootstraps quantify variation across the 168 benchmark samples;
- they do not quantify variation across training seeds;
- claims of training robustness require future repeated training runs.

---

# Part XIV — Immediate next action

## 49. Await the Phase-2A Codex response

When Codex returns:

1. inspect the full script;
2. verify that Phase-1 files are read-only;
3. verify metric-direction handling;
4. verify optional SSIM behavior;
5. verify bootstrap pairing and deterministic seed;
6. verify sign test and Holm correction;
7. verify Phase-1 summary/pairwise reproduction;
8. verify pre/post checksums;
9. compile and run on Spark;
10. preserve Phase-2A outputs separately.

Do not begin Phase 2B until Phase 2A is validated.

---

# Final status

```text
Dataset provenance: reconstructed and documented
Corrected benchmark: authoritative
Candidate training/evaluation: complete for the 19-method primary set
Candidate B factorial: complete
Candidate F study: complete
Unified Phase 1: complete
Strict Spark validation: 19/19 PASS
Topology validation: 18/18 PASS
Raw-array validation: all primary methods PASS
Phase-1 archive: created
Phase-1 checksums: verified
Phase 2A prompt: submitted to Codex
Phase 2A authoritative run: pending
Phase 2B: not started
```

The project has now moved from experiment generation and data-integrity repair into structured statistical analysis.
