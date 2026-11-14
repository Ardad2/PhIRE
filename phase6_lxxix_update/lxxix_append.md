
---

# Part LXXIX — Phase 6 complete canonical PD extraction and preparation for all-168 corrected GUDHI sweep (October 14, 2026)

## LXXIX.1 New Spark evidence supplied by the researcher

The LXXVIII compatible PD XML reader hotfix was successfully used to complete the isolated LXXVII canonical SR persistence-diagram extraction for all three new Phase 6 ablation arms. The original extraction driver remains byte-for-byte unchanged; the compatibility launcher replaced only its VTU readback validator in memory. The final full-extraction log includes `[FULL] PASS pers: 168 SR VTIs and PD port-0 VTUs`, `[LXXVII] PASS mode=full arm=all`, `[COMPAT] PASS`, and `Full extraction exit: 0`.

A separate read-only verification across all three arms also completed with `Verification exit: 0`, reporting:

- `[VERIFY] PASS grad: 168 SR VTIs and PD port-0 VTUs`
- `[VERIFY] PASS cv: 168 SR VTIs and PD port-0 VTUs`
- `[VERIFY] PASS pers: 168 SR VTIs and PD port-0 VTUs`
- `[LXXVII] PASS mode=verify arm=all`
- `[COMPAT] PASS: XML-level validation complete`

Thus **504/504 new SR VTIs and 504/504 new SR PD port-0 VTUs** have completed the configured file-inventory, VTI value/orientation and XML/schema validations. They occupy the isolated output root `ttk_runs_fixed/topology_finetuning/phase6_lxxvii_canonical_pd_newarms_20261014/`. No superseded historical `pd_distance` calculation was run. The separate Phase5E 168 GT diagrams, 168 CNN diagrams, and 168 F1 diagrams remain reference candidates without regeneration.

**Scientific limitation:** XML/schema completion is NOT an independent parse-and-distance audit of the compressed persistence payloads. The previously validated GUDHI pilot covers only sample 0 of each new arm. No new-arm full-168 corrected persistence-distance means or factorial effects are claimed by this entry.

## LXXIX.2 Isolated five-arm corrected GUDHI sweep — prepared, pending Spark execution

The `Phase6_LXXIX_Corrected_PD_FullSweep_20261014.zip` workpack contains a Python driver that reuses **unchanged** the `parse()` and `distance()` routines from the validated LXXVII GUDHI pilot. It uses the frozen canonical parser in `$HOME/phire_runtime_audit_20260809_221548/recompute_pd/canonical_pd_pilot.py`, preserving the corrected finite D0/D1 pair policy and the three distance definitions:

- Bottleneck `dB`: GUDHI bottleneck with L-infinity ground metric and diagonal matches, aggregate maximum of D0 and D1.
- `W2inf`: GUDHI q=2 Wasserstein with L-infinity ground metric, aggregate Euclidean norm across D0 and D1.
- `W22`: GUDHI q=2 Wasserstein with Euclidean ground metric, aggregate Euclidean norm across D0 and D1.

Matched canonical PDs and sample indices 0–167 are mapped as follows:

| Method | PD source | Count |
|---|---|---:|
| GT | frozen Phase5E `phase5_GT_s<i>_..._pd_port_0.vtu` | 168 |
| CNN | frozen Phase5E `phase5_CNN_SR_s<i>_..._pd_port_0.vtu` | 168 |
| Gradient-only | LXXVII new `grad/pd/candidateF_grad_pairzero_expanded2688_SR_s<i>_...` | 168 |
| CV-only | LXXVII new `cv/pd/candidateF_grad_CV_only_expanded2688_SR_s<i>_...` | 168 |
| PERS-only | LXXVII new `pers/pd/candidateF_grad_PERS_only_expanded2688_SR_s<i>_...` | 168 |
| Full F1 | frozen Phase5E `phase5_F1_SR_s<i>_..._pd_port_0.vtu` | 168 |

The 504 new-arm + 336 frozen CNN/F1 comparisons total **840 GT-SR corrected distance rows**. Input fingerprinting covers 1,008 total PD VTUs (168 GT + 840 SR); only the 504 new-arm diagrams are newly extracted. The full F1 identification should be supported by the Phase5E source array/checkpoint provenance before attributing effects to a particular training objective in a manuscript.

The three workpack phases are `--mode preflight` (read-only all-input checksum inventory, pilot hash and numerical regression), `--mode sweep` (sample-wise resume with a complete five-model checkpoint per sample, isolated new CSV/summary/manifest outputs), and `--mode verify` (read-only hash and result-table verification). All writes are isolated to `ttk_runs_fixed/topology_finetuning/phase6_lxxix_corrected_pd_gudhi_20261014/`.

The planned 2×2 fixed loss ablation uses `grad` (0 CV, 0 PERS), `cv` (1 CV, 0 PERS), `pers` (0 CV, 1 PERS), and `f1` (1 CV, 1 PERS). For each lower-is-better metric Y, the script computes the descriptive per-field effects:

```text
CV main       = ((Y_cv   - Y_grad) + (Y_f1 - Y_pers)) / 2
PERS main     = ((Y_pers - Y_grad) + (Y_f1 - Y_cv))   / 2
CV x PERS     = Y_f1 - Y_cv - Y_pers + Y_grad
```

No statistical significance, causality or geographical generalization is implied: these are one-checkpoint measurements on 168 successive hourly fields, not independent climate draws. Merge-tree metrics and spatially constrained persistence/locality remain separate deferred stages.

**LXXIX workpack status:** Syntax and synthetic control-flow/CSV tests performed locally; complete Spark GUDHI sweep not run in chat. Do not report any dataset-wide corrected distance or ablation result until the real sweep and verification both exit 0.
