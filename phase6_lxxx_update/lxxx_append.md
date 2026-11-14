
---

# Part LXXX — Phase 6: verified all-168 corrected PD metrics and paired-effect interpretation (October 14, 2026)

## LXXX.1 New user-supplied Spark evidence

The Part LXXIX source preflight, full corrected GUDHI distance sweep, and post-run integrity verification each exited with code `0`. The observed gates were:

```text
[SOURCE] PASS: 1008 input PD VTUs (168 GT + 840 SR) hashed, including all 504 new-arm SR files
[PILOT] PASS: frozen GT and all three sample-0 SR hashes match independent GUDHI gate
[PREFLIGHT] PASS: no files written; corrected GUDHI s0 regression and full input inventory passed
[SWEEP] PASS 168/168 complete five-arm samples (840 rows)
[AUDIT] PASS: 840 unique (sample,arm) rows; 504 new-arm GUDHI comparisons; 336 Phase5E CNN/F1 controls
[SWEEP-GATE] PASS: CSV, summary and SHA256 generated
[VERIFY-GATE] PASS: all input hashes, rows, pilot regressions, invariants and CSV SHA256
```

Outputs were isolated under `ttk_runs_fixed/topology_finetuning/phase6_lxxix_corrected_pd_gudhi_20261014/`. The values below were read from the researcher-provided `summary.md`, not recomputed here on the Spark server.

## LXXX.2 Full five-arm corrected PD mean distances

All metrics are lower-is-better. The benchmark consists of 168 consecutive hourly fields; models include the pretrained CNN comparator and the fixed 2×2 topology-loss design with gradient-only (00), CV-only (10), PERS-only (01), and archived full F1 (11).

| Arm | Mean corrected bottleneck dB | Mean W2inf | Mean W22 |
|---|---:|---:|---:|
| cnn | 3.124108 | 19.152145 | 24.552151 |
| grad (00) | 2.562506 | 14.889700 | 18.306367 |
| cv (10) | 2.188473 | 14.407073 | 17.875645 |
| pers (01) | 2.196119 | **14.153230** | **17.484939** |
| f1 (11) | **2.149744** | 14.325676 | 17.831897 |

The full F1 model has the lowest *mean* bottleneck distance; PERS-only has the lowest mean Wasserstein distances. The new topology-only components provide better mean PD distances than the gradient-only 00 arm, while the previous scalar/domain comparison found gradient-only strongest for several pointwise measures. This pattern is descriptive and metric-specific; it does not establish spatial feature correctness, merge-tree fidelity, independent climate generalization, or robustness to random seeds.

## LXXX.3 Matched two-by-two effects

All contrasts use gradient-only as cell 00 and F1 as cell 11. The pretrained CNN is *not* the factorial control. A negative main effect improves the reported distance; a positive interaction means reduced additive benefit of CV+PERS relative to the 00 additive prediction.

| Metric | CV main | PERS main | Interaction |
|---|---:|---:|---:|
| dB | -0.210205 | -0.202558 | +0.327658 |
| W2inf | -0.155090 | -0.408934 | +0.655073 |
| W22 | -0.041882 | -0.432587 | +0.777680 |

The mathematical contrasts represent differences among these specific trained outputs; without replicated trainings and temporal dependence handling, they should not be interpreted as significance tests or causal estimates. In particular, positive interaction does not imply that F1 is always worse than individual losses: F1 still leads on mean bottleneck distance.

## LXXX.4 Verification scope and outstanding gates

The GUDHI calculations were completed on all 840 rows. The post-run `verify` command is a **hash, row-integrity, pilot-regression and invariant check of the already-computed output**, not a second independent recomputation of all 840 distances. The previous historical 8,568-comparison custom/GUDHI audit independently validated the distance implementation on different archived cases. Preserve these distinctions when documenting methods.

Prior to manuscript causal attribution, inspect the archived F1 checkpoint, training script/options and source provenance against the matched two-by-two 2688-sample protocol. This is a provenance requirement, not evidence that F1 is invalid. The F1 PDs are the frozen canonical Phase 5E diagrams and match the established Phase 5E mean W22 of approximately 17.8319, supporting metric continuity but not by itself proving identical model training conditions.

An independent read-only paired-field descriptive audit is prepared in `Phase6_LXXX_Paired_PD_Audit_20261014.zip`. It checks the exact LXXIX CSV SHA-256, sample pairing, row means, factorial means and per-input provenance hashes. It can produce an isolated report with field-wise win/loss/tie counts, median paired changes, worst-case examples, and seven contiguous 24-hour block means. It does not run GUDHI or TTK, alter prior outputs, or calculate statistical p-values. As of this Part LXXX note, **the actual Spark paired-field audit has not been run**.

After the paired audit and F1 lineage check, prioritize Phase 5E locality-constrained comparisons using the validated radii `4,8,16,32,64,96,128,infinity`, with GT coverage separated from constrained W22; independently plan matching merge-tree evaluation. Neither has been executed for the three new arms at this entry. Keep publication tag `topoinvis-2026-arxiv-v1.1`, historical checkpoints, original fixed arrays and frozen Phase 5E outputs unchanged.
