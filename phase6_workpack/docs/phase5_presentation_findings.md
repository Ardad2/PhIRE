# Phase 5E spatial persistence: compact presentation findings

**Status:** frozen Phase5E (168 fields), outside the v1.1 manuscript results. The four charts here visualize **existing** repository summaries, not newly computed topology. Figure-generating code uses the released raw `spatial_pd` aggregate CSVs. A source-column subset is included under `data/` solely to render the figures in this package.

## Primary claim (limited to the tested region and benchmark)
Topology-inspired F1 increases GT-side persistence-mass coverage under locality-constrained PD correspondence against both the pretrained CNN and reconstruction-only fine-tuning for all 168 fields at every tested radius. At extremely tight locality the symmetric persistence-cost penalty is larger, so higher GT-side coverage is not identical to better overall localized persistence agreement. Starting at moderate locality, F1 generally has lower constrained persistence cost than either baseline. This is **not** proof that individually identified meteorological structures are physically tracked or that F1 generalizes geographically.

## Panel captions

1. **`phase5_gt_coverage`:** The fraction of GT persistence mass participating in spatially admissible real-real pairings. At r=4, the mean GT-side coverage is 13.92% (CNN), 14.86% (UV), 23.80% (F1); at r=16 it is 59.13%, 49.40%, 71.09%. On a per-field basis F1 exceeds both comparators in 168/168 at each tested radius.
2. **`phase5_constrained_cost`:** The original W2,2 persistence objective with a maximum birth/death endpoint-displacement admissibility radius; diagonal assignments remain permitted. At r=4, mean cost is 46.06 (CNN), 43.93 (UV), **48.50 (F1)** (F1 worst); at r=16, 35.22, 35.43, **31.95 (F1)** (F1 best mean). At r=infinity the ordinary W2,2 results are recovered. This is a constrained cost, not a newly proven mathematical metric.
3. **`phase5_sr_coverage`:** Fraction of the prediction's *own* persistence mass represented by local real-real assignments. At r=16, CNN=84.26%, UV=89.28%, F1=85.43%. The coverage is directional; better GT-side coverage can coexist with lower SR-side coverage and more unmatched predicted persistence.
4. **`phase5_stable_advantage`:** Earliest threshold at which F1 has a lower constrained cost than both baselines and remains lower for all wider tested thresholds. 39 samples first stabilize at r=8; 118 at r=16; 7 at r=32; 2 at r=64; two never stabilize. Thus 157/168 (93.45%) are stably better by r=16; exceptions never stable are samples 162 and 163.

## Follow-on qualitative figure (not generated here)
One composite can show canonical 160x160 GT/CNN/UV/F1 wind speed plus critical-point endpoint markers for sample 69 (method exposition) and exception samples 162/163 (failure analysis); include GT->diagonal and diagonal->SR examples at r=4 versus r=16. Use exact Phase5E C-order provenance rather than older transposed VTI arrays. The 672 canonical VTI and VTU intermediate artifacts must be restored from Spark; they are not all distributed in the tag, so this qualitative spatial panel is **pending**.

## Source
`https://github.com/Ardad2/PhIRE/tree/topoinvis-2026-arxiv-v1.1/spatial_pd/phase5e_all168_locality_analysis` and `.../phase5e_all168_closeout_summary`.
