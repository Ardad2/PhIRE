
# Part XCV — Phase 7: four-field global/locality development pilot verified; 2010 seasonal source expansion prepared (October 18, 2026)

## Prior-stage result verified on Spark
- XCIV zero-persistence compatibility finished: 28/28 diagrams' retained H0/H1 birth/death values and endpoint coordinates match the corrected numerical diagram convention, 24/24 unconstrained W22 values match XCIII, 192 aggregate locality and 384 dimension-level rows, four output SHA256 entries verified. No locked-region model predictions examined.
- XCIV four-field mean constrained W22 (lower is better):

| Model | r=4 | r=16 | r=32 | r=inf | GT persistence-mass coverage at r=16 |
|---|---:|---:|---:|---:|---:|
| cnn | 15.872628 | 13.065990 | 11.488728 | 9.560749 | 0.521896 |
| gan | 22.701018 | 16.753153 | 14.003156 | 11.388065 | 0.823593 |
| grad | 16.165423 | 11.531220 | 9.632986 | 7.412320 | 0.801454 |
| cv | 16.302017 | 12.256537 | 9.938167 | 7.806714 | 0.731901 |
| pers | 16.346778 | 11.956780 | 10.175161 | 7.919260 | 0.738989 |
| f1 | 16.568133 | 12.269268 | 10.029004 | 7.905475 | 0.709210 |

- The four fields are exploratory validation checks, **not** a conclusion that the historical 168-field trade-offs have reversed geographically. A low matching cost and high GT persistence coverage are distinct observations, not interchangeable estimates of correctly localized meteorological structures.

## Part XCV workpack plan
- Inputs pinned to LXXXVII 64-field screening CSV and NPZ checksums, LXXXIX locked region assignment, and XCI frozen protocol. Never selects new regions or timestamp windows based on model performance.
- Only 2010 **development** regions `gulf_inland`, `southwest_borderland`; February, May, August, November seven-day windows; predeclared UTC offsets 0 and 84 h within each window. Total 2 × 4 × 2 = **16 fields**, compared with the four-field numerical smoke test.
- Four February/August offset-zero native LXXXVII fields are copied byte-for-byte; the other 12 native 500×500 speed and wind-direction slices are re-read from WTK through read-only HSDS, converted to `[u,v]` using the original LXXXVII routine, and pooled to 100×100 inputs. Each resulting 500×500/100×100 pair must match its original screening mean and p99 speed within 2e-8 m/s, verify UTC index, region and grid footprint, and retain exact SHA256 integrity.
- No TensorFlow/model inference, GT-to-model metrics, topology extraction, merge-tree distances, 2012 replication data, or external-test predictions in XCV. This is only a data-availability/scaling pilot; **XCI's full 168 hourly fields per region-week are still the final predeclared benchmark**, not replaced by these two pilot hours per region-week.
- Expected isolated output: `ttk_runs_fixed/topology_finetuning/phase7_xcv_development_seasonal16_20261018/` with 16 NPZ, JSON manifest, markdown report and SHA256SUMS. Refuse overwrite and purge unfinished staging artifacts only.
- Current archive was syntax checked, hashed, mock-HSDS integration tested end-to-end for 12 remote-read equivalents plus four cached previews, and verified under checksum-tamper tests. **Real HSDS extraction is pending**.
- Follow-up after verification: process the fixed 16 development inputs through unchanged six-model checkpoints, then numerical/topological smoke checks using the already validated XC→XCIV methods before scaling to complete seasonal weeks. No changes to XCI frozen outcomes or locked region assignments.
