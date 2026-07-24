# Phase 2D-A Visual Review Record

**Project:** Unified wind-field super-resolution candidate analysis  
**Review stage:** Phase 2D-A qualitative audit  
**Review status:** Complete  
**Selection status:** All six deterministic primary samples accepted; no replacements required  
**Final-figure status:** Phase 2D-B not yet begun  

## 1. Purpose

This record documents the visual review of the six deterministic Phase-2D-A archetype selections and the contact sheet. The review is limited to the generated wind-speed fields, absolute speed-error maps, and the PD/MT annotations shown in the audit previews.

The review does **not** replace the quantitative conclusions from Phases 1, 2A, 2B, or 2C. The selected samples remain illustrative examples rather than random samples or population-level estimates.

## 2. Review rules

- `Accept`: the primary sample adequately represents the intended archetype.
- `Accept with presentation requirement`: the primary sample is scientifically appropriate, but the final figure must include additional topology-specific evidence to make the intended claim clear.
- `Replace`: the primary sample is visually unclear or misleading and should be replaced only by a precomputed alternate.
- No sample may be replaced merely because an alternate appears more visually dramatic.
- Any replacement must record the alternate rank and a concrete visual reason.
- Shared color scales must be preserved within each final composite.

## 3. Final decisions

| Archetype | Primary sample | Alternates | Decision | Visual-review conclusion |
|---|---:|---|---|---|
| Global PD/MT descriptor disagreement | **120** | 41, 143, 26 | **Accept** | Strong multi-method inversion: GAN is favored by PD but penalized by MT, while CNN and UV+E2 receive comparatively stronger MT assessments. The error maps also distinguish GAN from the fine-tuned candidates. |
| GAN-PD versus CNN-MT conflict | **34** | 33, 32, 27 | **Accept** | Clearest pairwise descriptor conflict. GAN has much better PD than CNN but substantially worse MT and visibly broader scalar-speed error. |
| F3-PD versus UV+E2-MT tradeoff | **119** | 71, 114, 115 | **Accept with presentation requirement** | The numerical PD/MT tradeoff is strong, but the speed fields and error maps are visually similar. The final figure must include PD and MT evidence, and preferably a zoomed structural region. |
| Balanced F2 improvement versus CNN | **25** | 135, 24, 22 | **Accept** | F2 improves both PD and MT over CNN and has a clearly darker error map. The final claim should present F2 as a balanced compromise, not as the universal winner. |
| Candidate-C continuity | **30** | 134, 107, 155 | **Accept** | Candidate C improves both reported topology distances and scalar-speed error relative to CNN. It provides a useful bridge between the submitted Candidate-C story and the expanded ablation findings. |
| Global PD/MT descriptor agreement | **19** | 20, 45, 79 | **Accept** | The stronger fine-tuned methods perform comparatively well under both descriptors. Agreement should be described as broad concordance rather than identical rankings. |

## 4. Per-sample review notes

### 4.1 Sample 120 — Global descriptor disagreement

Reported topology values:

| Method | PD | MT |
|---|---:|---:|
| CNN | 28.15 | 4.89 |
| GAN | 21.53 | 7.93 |
| Candidate C | 23.32 | 6.27 |
| F3: Grad+Crit | 23.30 | 6.62 |
| F2: Grad+Levelset+E2 | 24.38 | 5.62 |
| UV+E2 | 26.42 | 5.12 |

Review:

- GAN has the best PD but the worst MT.
- CNN has the worst PD among the displayed learned methods but the best MT.
- UV+E2 is comparatively weak under PD but nearly best under MT.
- GAN also shows more spatially extensive scalar-speed error than the fine-tuned candidates.

Final-figure requirement:

- Use this as the primary multi-method descriptor-disagreement case.
- Include a compact PD/MT comparison and topology-specific panels.
- Avoid implying that pointwise error alone explains the descriptor disagreement.

### 4.2 Sample 34 — GAN-PD versus CNN-MT conflict

Reported topology values:

| Method | PD | MT |
|---|---:|---:|
| CNN | 37.36 | 6.32 |
| GAN | 28.91 | 19.45 |

Review:

- GAN is strongly favored by PD.
- CNN is overwhelmingly favored by MT.
- GAN’s absolute speed-error map is brighter and more spatially extensive than CNN’s.

Final-figure requirement:

- Use as the clearest baseline conflict figure.
- Show GT, CNN, and GAN prominently; bicubic may remain as a reference.
- State explicitly that a lower PD distance does not guarantee better merge-tree or pointwise fidelity.

### 4.3 Sample 119 — F3 versus UV+E2 tradeoff

Reported topology values:

| Method | PD | MT |
|---|---:|---:|
| F3: Grad+Crit | 23.02 | 6.52 |
| UV+E2 | 27.08 | 4.85 |

Review:

- F3 is substantially better under PD.
- UV+E2 is substantially better under MT.
- Their full-field speed and error panels are visually similar.

Final-figure requirement:

- The final figure must include PD overlays or summaries and merge-tree evidence.
- Add a zoomed crop around a region with meaningful structural differences.
- Do not rely on speed/error panels alone to explain the topology result.

### 4.4 Sample 25 — Balanced F2 improvement

Reported topology values:

| Method | PD | MT |
|---|---:|---:|
| CNN | 36.58 | 11.37 |
| F2: Grad+Levelset+E2 | 31.48 | 6.64 |

Review:

- F2 substantially improves both topology descriptors over CNN.
- Its scalar-speed error is visibly reduced.
- F3 has a stronger PD value, while F2 is stronger in MT, consistent with the intended balanced-compromise interpretation.

Final-figure requirement:

- Compare CNN, F3, F2, and UV+E2.
- Present F2 as a balanced method rather than the absolute winner of every objective.

### 4.5 Sample 30 — Candidate-C continuity

Reported topology values:

| Method | PD | MT |
|---|---:|---:|
| CNN | 39.94 | 8.13 |
| Candidate C | 33.02 | 7.61 |

Review:

- Candidate C improves both topology distances relative to CNN.
- Candidate C’s speed-error map is visibly darker than those of bicubic, CNN, and GAN.
- Later methods may improve specific descriptors more strongly, but Candidate C remains a valid topology-inspired improvement.

Final-figure requirement:

- Frame Candidate C as continuity with the submitted-paper result.
- Explain that the expanded ablation study refines the mechanism: gradient supervision is more strongly associated with PD gains, while repaired E2 is more strongly associated with MT gains.

### 4.6 Sample 19 — Global descriptor agreement

Reported topology values:

| Method | PD | MT |
|---|---:|---:|
| CNN | 21.06 | 6.93 |
| GAN | 19.09 | 5.44 |
| Candidate C | 16.02 | 5.50 |
| F3: Grad+Crit | 15.60 | 5.24 |
| F2: Grad+Levelset+E2 | 17.51 | 5.31 |
| UV+E2 | 19.06 | 4.99 |

Review:

- The stronger fine-tuned candidates perform well under both descriptors.
- UV+E2 remains relatively more MT-oriented, so the rankings are not identical.
- The case provides a useful counterbalance to the disagreement examples.

Final-figure requirement:

- Describe this as broad descriptor concordance, not perfect rank agreement.
- Use it to show that PD/MT conflict is not universal.

## 5. Contact-sheet observations

The six cases cover multiple visual regimes rather than one repeated field pattern:

- samples 119 and 120 show lower-range fields with broad spatial structure;
- samples 25, 30, and 34 contain stronger filamentary and front-like features;
- sample 19 contains a broad diagonal transition with smoother large-scale organization.

At full-field scale, the fine-tuned candidates are often visually similar. This supports the need for topology-specific panels, zoomed regions, and exact metric annotations in Phase 2D-B.

## 6. Frozen selection decision

```text
global_descriptor_disagreement = 120
gan_pd_vs_cnn_mt_conflict      = 34
f3_pd_vs_uv_e2_mt_tradeoff     = 119
f2_balanced_vs_cnn             = 25
candidate_c_continuity         = 30
global_descriptor_agreement    = 19
```

No alternate is activated.

## 7. Phase 2D-B presentation requirements

1. Preserve these six selected samples.
2. Use human-readable method labels:
   - CNN
   - GAN
   - Candidate C
   - F3: Grad+Crit
   - F2: Grad+Levelset+E2
   - UV+E2
3. Use common field and error scales within each composite.
4. Include topology-specific evidence for samples 120, 34, and 119.
5. Include a zoomed structural region for sample 119.
6. Keep claims sample-specific.
7. Do not describe any selected case as a population estimate.
8. Do not claim automated TTK/ParaView rendering unless it is actually implemented.
9. Preserve a machine-readable figure-data CSV for every final figure.
10. Treat all Phase-1 through Phase-2D-A artifacts as frozen inputs.

## 8. Completion statement

Phase 2D-A visual review is complete. All six deterministic primary selections are accepted. Phase 2D-B final publication-quality figure production may proceed using the frozen sample set and the presentation requirements recorded above.

## Machine-readable acceptance gate

global_descriptor_disagreement sample_idx=120: ACCEPTED
gan_pd_vs_cnn_mt_conflict sample_idx=34: ACCEPTED
f3_pd_vs_uv_e2_mt_tradeoff sample_idx=119: ACCEPTED
f2_balanced_vs_cnn sample_idx=25: ACCEPTED
candidate_c_continuity sample_idx=30: ACCEPTED
global_descriptor_agreement sample_idx=19: ACCEPTED

No alternate was activated for any archetype.
