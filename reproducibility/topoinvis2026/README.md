# TopoInVis 2026 / arXiv v1 — paper-specific artifacts

This directory holds the material behind the numbers in
**Topology-Inspired Fine-Tuning for Wind-Field Super-Resolution** that previously
lived outside the repository. The full guide is
[`docs/topoinvis_2026_reproducibility.md`](../../docs/topoinvis_2026_reproducibility.md).

| Path | Contents |
|---|---|
| `pd_audit/recompute_pd/` | Diagram parser (`canonical_pd_pilot.py`), bottleneck/W2,inf sweep and GUDHI cross-check (code, CSV, summaries, environment, checksums) |
| `pd_audit/recompute_pd_w22/` | W2,2 sweep (`w22_full_sweep.py`), the frozen per-run corrected distance table `w22_full_sweep.csv`, GUDHI W2,2 cross-check, near-tie selection |
| `pd_audit/recompute_pd_w22/corrected_pd_mt/` | Output of `scripts/analyze_corrected_pd_mt_tradeoff.py`; `corrected_pd_mt_joined.csv` (18 methods x 168 fields) is the shared per-field source for quantitative topology tables and loss-study figures; the Sample 78 and conceptual figures have separate source dependencies |
| `manuscript_tools/` | `make_tables.py` / `make_figures.py`: regenerate quantitative topology tables and loss-study figures from `corrected_pd_mt_joined.csv` (numerical checks compare against fixed reference values; figure identity is not separately pixel-verified) |
| `manuscript_outputs/` | Tables, figures and verification log produced by those tools |
| `prepare_release_artifacts.sh` | One-time script that assembled this directory on Spark |
| `SHA256SUMS` | Checksums of Git-tracked package files, excluding the manifest itself |

Verify the paper's numbers (Python 3 with NumPy and Matplotlib):

```bash
python3 reproducibility/topoinvis2026/manuscript_tools/make_tables.py \
  --joined reproducibility/topoinvis2026/pd_audit/recompute_pd_w22/corrected_pd_mt/corrected_pd_mt_joined.csv \
  --out /tmp/topoinvis_tables
# expected: "All derived quantities match the values reported in the manuscript (...)"
```
