# Visual Inspection Plan for TopoAware SR

## Goal

Use the quantitative observation groups to perform a structured visual audit of the repaired PhIRE wind-field SR outputs. The goal is not to declare a universal winner between CNN and GAN, but to understand what each evaluator is responding to:

- CNN-facing direct-error metrics: PSNR, SSIM, WPD MAE/RMSE, gradient MAE.
- GAN-facing distributional/tail metrics: PSD log-L2, gradient W1, some upper-tail exceedance metrics.
- PD: strongly GAN-facing in the current bottleneck-distance setup.
- MT: intermediate; mostly CNN-facing but GAN-facing in 20 diagnostic samples.

## Main question

When GAN is favored by PD, MT, or distributional metrics, is it because GAN recovers real structure in the GT field, or because it creates visually plausible but spatially/hierarchically misaligned structure?

## Recommended inspection phases

### Phase 1 — Core contrast set

Inspect these first:

```text
8, 12, 25, 27, 31, 37, 39, 29, 32
```

Interpretation:

- 8, 12, 25 are near-tie topology-consensus-GAN / validator-disagreement cases.
- 27, 31, 37, 39, 29, 32 are MT-primary / CNN-control cases.

This gives a compact contrast between the most interesting edge cases and stable CNN-aligned controls.

### Phase 2 — MT-GAN diagnostic set

Inspect all MT-GAN samples:

```text
6, 8, 12, 16, 17, 18, 19, 20, 25, 48,
62, 63, 65, 68, 77, 79, 80, 82, 92, 154
```

Question:

> When MT favors GAN, does GAN better preserve the GT structure, or does it introduce sharper but misleading structures?

### Phase 3 — PD-GAN / MT-CNN disagreement set

Use the generated `group_candidate_structural_hallucination_signature.csv` and inspect the highest-ranked cases first.

Question:

> Does PD favor GAN because GAN has plausible feature lifetimes, while MT favors CNN because GAN's features are spatially or hierarchically misorganized?

### Phase 4 — GAN-distributional cases

Use `group_gan_distributional_cases.csv`.

Question:

> Do the spectral/distributional advantages of GAN correspond to visually meaningful wind structure?

### Phase 5 — Controls

Use `group_cnn_consensus_core.csv` and `group_topology_consensus_cnn.csv`.

Question:

> What do stable CNN-favoring examples look like, and how do they differ from GAN-favoring/topology-disagreement samples?

## What to look for in each panel

Each generated panel contains:

```text
GT speed | CNN speed | GAN speed | |CNN-GT| |GAN-GT|
```

Record observations about:

1. Whether GAN is sharper.
2. Whether GAN's sharp structures are present in GT.
3. Whether GAN introduces shifted ridges, extra peaks, or fragmented blobs.
4. Whether CNN is smoother but more spatially aligned.
5. Which output better preserves large-scale flow/ridge organization.
6. Whether the error maps support the visual impression.

## Suggested labels for notes

Use short labels in `visual_observation_template.csv`:

- `CNN clearly closer`
- `GAN visually sharper but suspicious`
- `GAN plausible structure`
- `GAN shifted/misaligned`
- `CNN oversmoothed`
- `ambiguous`
- `needs topology visualization`

## Scripts

Run from the `scripts/` directory:

```bash
cd ~/PhIRE/scripts

PYTHONNOUSERSITE=1 /usr/bin/python3 generate_visual_inspection_panels.py
```

Optional full-field panels:

```bash
cd ~/PhIRE/scripts

PYTHONNOUSERSITE=1 /usr/bin/python3 generate_visual_inspection_panels.py \
  --include-full-panels
```

Optional u/v component panels:

```bash
cd ~/PhIRE/scripts

PYTHONNOUSERSITE=1 /usr/bin/python3 generate_visual_inspection_panels.py \
  --include-uv-panels
```

Outputs are written to:

```text
ttk_runs_fixed/visual_inspection/
```
