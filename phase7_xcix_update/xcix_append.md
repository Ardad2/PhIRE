# Part XCIX — Phase 7: verified 16-field corrected GUDHI results; original spatial-locality expansion prepared (October 18, 2026)

## Part XCVIII real Spark verification

- XCVIII `selftest`, `preflight`, `compute`, `verify` all completed successfully with exits 0. All **112** TTK diagrams numerically decoded using the archived corrected Phase6 parser; all **96** model/GT corrected `dB`, `W2inf`, `W22` comparisons recomputed and verified, including all **24** original XCIII four-field regression anchors; original Phase6 s0 regression of five arms passed. Eight region-week pairs × six arms = **48** summary rows. All **4 SHA256** result checks passed, no locked external-region predictions or scores accessed.
- Descriptive mean results on sixteen predeclared 160×160 top-left speed crops, across two development regions and four 2010 seasonal weeks with two hours per region-week (not a generalization estimate):

  | Model | Mean dB ↓ | Mean W2inf ↓ | Mean W22 ↓ |
  |---|---:|---:|---:|
  | cnn | 2.218899 | 8.349543 | 10.462440 |
  | gan | 3.371328 | 9.788407 | 12.211472 |
  | grad | **1.540991** | **6.892901** | **8.375874** |
  | cv | 1.597790 | 6.981211 | 8.502258 |
  | pers | 1.570721 | 6.955128 | 8.483273 |
  | f1 | 1.603079 | 7.033517 | 8.607247 |

- Gradient-only has the lowest global persistence distances on these descriptive means, but block-specific differences exist (e.g., PERS-only leads Gulf May and November in `W22`), and neither seasonality nor geographical generalization is established by these paired snapshots.

## Part XCIX locality expansion workpack (prepared; Spark not yet run)

- Scope: exactly the existing sixteen 2010 **development-only** fields and their **112** numerically audited XCVII canonical TTK diagrams. Reuse pinned original XCIV `compute_rows` and the unchanged Phase5E sparse matching solver (`SHA256 47c73decca4ba151fd8b250cae2911b4c2307471fe9c881fc42bfc968e211c3a`), including original radius grid **4, 8, 16, 32, 64, 96, 128, infinity** (grid cells, not kilometers).
- Before matching, independently decode **all 112** GT/model spatial birth/death critical coordinates using the original Phase5E reader. Require all H0/H1 nonzero features to match corrected canonical numeric persistence values, all coordinates finite and within the 160×160 crop, and the original `TOL=1e-10` zero/near-zero persistence counts to match the corrected canonical parser **for every diagram**. Do not modify archived reader, optimizer, XCIV/XCVIII data, or frozen XCI protocol.
- Compute **96** model-to-GT comparisons × **8** radii = **768** aggregate locality rows, and **1536** H0/H1 dimension rows. Require *all* infinite-radius `W22` values to match frozen XCVIII, and all **192** original XCIV aggregate rows plus **384** dimension-level rows to be reproduced for the four original fields mapped to seasonal16 indices 0, 4, 8, 12. Produce **48** region-week × model summary rows (two snapshots per block); report GT persistence-mass coverage separately from matching cost.
- Save isolated, atomic, overwrite-protected `ttk_runs_fixed/topology_finetuning/phase7_xcix_dev_seasonal16_locality_20261018/`, with five SHA-256-indexed result files plus `SHA256SUMS`, and full numerical verifier recomputation using the same original solver. This is not a second independent matching algorithm.
- Package integrity, Python syntax, flat-path extraction, master research-notes patch checks and synthetic regression/aggregation/overwrite/manifest tests are local-only; real Spark source/coordinate/matching gates are pending. If any one fails, stop without claiming locality rankings.
- No new inference, training, HSDS, topology hyperparameters, full500 TDA, merge trees, or locked test-region access. The benchmark's complete 2010 168-hour primary / 2012 replication full-week tests have not been conducted, and pretrained PhIRE training-set membership for these regions remains unknown.
