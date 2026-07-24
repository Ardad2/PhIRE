# Phase 2D-B: Draft Final Figure Captions

Draft captions only. Claims are sample-specific; no claim in this file generalizes beyond the sample and methods shown in that figure. See docs/unified_candidate_analysis_phase2db.md for the full quantitative-vs-illustrative framing.

## Figure 1: global_descriptor_disagreement (sample_idx=120)

For sample 120, PD and MT can produce strongly different cross-method preferences: GAN attains the best displayed PD distance but the worst MT distance, CNN shows the worst displayed PD distance but the best MT distance, and UV+E2 is comparatively MT-oriented among the remaining methods. Across the fixed benchmark, the quantitative analysis found this disagreement pattern is not universal (see Figure 6); this selected example visualizes one instance where it is pronounced.

## Figure 2: gan_pd_vs_cnn_mt_conflict (sample_idx=34)

For sample 34, a lower PD distance does not guarantee better merge-tree or pointwise fidelity: GAN improves on CNN's PD distance while CNN improves on GAN's MT distance. This illustrative case shows the tradeoff concretely; no claim is made that this ordering holds for every sample in the benchmark.

## Figure 3: f3_pd_vs_uv_e2_mt_tradeoff (sample_idx=119)

For sample 119, gradient-plus-critical supervision (F3) and repaired E2 supervision (UV+E2) influence different topology descriptors: F3 improves the PD distance relative to UV+E2, while UV+E2 improves the MT distance relative to F3, within the deterministically selected zoomed structural region shown. F2 is shown only as a compact contextual reference.

## Figure 4: f2_balanced_vs_cnn (sample_idx=25)

For sample 25, F2 provides a balanced PD/MT improvement over CNN rather than universally optimizing every objective: both the PD and MT distances improve over CNN in this selected example, illustrating the balanced-improvement archetype identified by the quantitative analysis.

## Figure 5: candidate_c_continuity (sample_idx=30)

For sample 30, Candidate C is a valid topology-inspired improvement over CNN in this selected example; the expanded ablation study (F3, F2, UV+E2) shown alongside it clarifies the more specific PD and MT mechanisms contributing to that improvement across the fixed benchmark.

## Figure 6: global_descriptor_agreement (sample_idx=19)

For sample 19, PD and MT disagreement is not universal: the displayed methods show broad descriptor concordance without necessarily sharing an identical ranking. This selected example visualizes a case of cross-method agreement, in contrast with the disagreement case in Figure 1.

