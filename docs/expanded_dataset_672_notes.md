# Expanded Dataset 672 Notes

**Generated:** 2026-05-23

## Summary

- Total samples: 672 (4 windows × 168 hours)
- Output directory: `example_data_topology_expanded_672`
- HR shape: [672, 500, 500, 2]

## Spatial crop

- Target center: 39.5°N, -75.0°W
- Actual grid center: 39.4998°N, -75.0064°W
- WTK native rows: 720:1220, cols: 2102:2602
- HR patch: 500×500 (same as 168-sample benchmark)

## Seasonal windows

| Season | WTK start | Length | WTK end |
|--------|-----------|--------|---------|
| winter | 336 | 168 | 503 |
| spring | 2160 | 168 | 2327 |
| summer | 4344 | 168 | 4511 |
| fall | 6552 | 168 | 6719 |

## Resolution hierarchy

- HR: 500×500×2 — native WTK crop
- MR: 100×100×2 — 5× block-average from HR via `utils.downscale_image`
- LR: 10×10×2 — 10× block-average from MR via `utils.downscale_image`

## Statistics over expanded HR stack

- u: mean=2.1014  std=5.8320  min=-23.369  max=34.423
- v: mean=0.2081  std=5.9917  min=-25.524  max=36.243
- speed: mean=7.5806  std=4.1117  p50=7.102  p90=13.233  p95=15.363  p99=18.854

## Normalization compatibility with pretrained PhIRE CNN

Pretrained: mu=[0.7684, -0.4575]  sig=[5.02455, 5.9017]

- u mean z-score vs pretrained: 0.265
- v mean z-score vs pretrained: 0.113
- **compatible=True** (criterion: |z| < 3 for both channels)

The expanded dataset mean falls within 3σ of the pretrained normalization center.  The pretrained CNN checkpoint can be applied to this data without recomputing mu/sig.

## Benchmark non-overlap

Minimum WTK index in the expanded set: 336  (benchmark ends at 167).
None of the 672 samples overlap the original benchmark (WTK 0..167 = 2007-01-01 00:00 to 2007-01-07 23:00 UTC).

## Output files

| File | Description |
|------|-------------|
| `example_data_topology_expanded_672/wind_LR-MR.tfrecord` | PhIRE TFRecord LR=10×10→MR=100×100 |
| `example_data_topology_expanded_672/wind_MR-HR.tfrecord` | PhIRE TFRecord MR=100×100→HR=500×500 |
| `example_data_topology_expanded_672/hr_stack.npy` | (672, 500, 500, 2) float64 |
| `example_data_topology_expanded_672/mr_stack.npy` | (672, 100, 100, 2) float64 |
| `example_data_topology_expanded_672/lr_stack.npy` | (672, 10, 10, 2) float64 |
| `example_data_topology_expanded_672/manifest.csv` | Per-sample metadata (672 rows) |
| `example_data_topology_expanded_672/stats.json` | Channel stats and normalization comparison |
