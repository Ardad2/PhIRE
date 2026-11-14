# Phase 6 paired corrected persistence-distance audit

Source: frozen LXXIX five-arm per-sample GUDHI CSV. All 168 sample indices matched.
Lower distances are better. No independent seed or statistical significance claims.

## dB

| Arm | Mean | Best-of-five count |
|---|---:|---:|
| cnn | 3.124108 | 2 |
| grad | 2.562506 | 20 |
| cv | 2.188473 | 34 |
| pers | 2.196119 | 53 |
| f1 | 2.149744 | 59 |
| ties | — | 0 |

| Comparison (A minus B) | Mean difference | Median | A better / worse / tied |
|---|---:|---:|---:|
| grad_minus_cnn | -0.561602 | -0.537288 | 130 / 38 / 0 |
| cv_minus_grad | -0.374034 | -0.304193 | 129 / 38 / 1 |
| pers_minus_grad | -0.366387 | -0.376694 | 126 / 42 / 0 |
| f1_minus_grad | -0.412762 | -0.341781 | 131 / 37 / 0 |
| f1_minus_cv | -0.038729 | -0.078996 | 98 / 69 / 1 |
| f1_minus_pers | -0.046375 | -0.075414 | 101 / 67 / 0 |
| pers_minus_cv | +0.007647 | -0.001503 | 84 / 84 / 0 |
| cv_minus_cnn | -0.935636 | -0.867059 | 151 / 17 / 0 |
| pers_minus_cnn | -0.927989 | -0.823263 | 152 / 14 / 2 |
| f1_minus_cnn | -0.974364 | -0.912582 | 151 / 17 / 0 |

| Factorial effect | Mean | Negative / positive / zero fields | 24-hour block means |
|---|---:|---:|---|
| cv | -0.210205 | 119 / 49 / 0 | -0.313, -0.189, -0.168, -0.147, -0.124, -0.251, -0.280 |
| pers | -0.202558 | 120 / 48 / 0 | -0.210, -0.299, -0.292, -0.256, -0.089, -0.172, -0.101 |
| interaction | +0.327658 | 51 / 117 / 0 | +0.482, +0.622, +0.303, +0.344, +0.025, +0.270, +0.248 |

## W2inf

| Arm | Mean | Best-of-five count |
|---|---:|---:|
| cnn | 19.152145 | 1 |
| grad | 14.889700 | 41 |
| cv | 14.407073 | 31 |
| pers | 14.153230 | 68 |
| f1 | 14.325676 | 27 |
| ties | — | 0 |

| Comparison (A minus B) | Mean difference | Median | A better / worse / tied |
|---|---:|---:|---:|
| grad_minus_cnn | -4.262445 | -4.476134 | 164 / 4 / 0 |
| cv_minus_grad | -0.482627 | -0.369548 | 117 / 51 / 0 |
| pers_minus_grad | -0.736470 | -0.479409 | 115 / 53 / 0 |
| f1_minus_grad | -0.564024 | -0.404159 | 101 / 67 / 0 |
| f1_minus_cv | -0.081397 | -0.028144 | 87 / 81 / 0 |
| f1_minus_pers | +0.172446 | +0.151641 | 49 / 119 / 0 |
| pers_minus_cv | -0.253844 | -0.142259 | 107 / 61 / 0 |
| cv_minus_cnn | -4.745072 | -4.995739 | 166 / 2 / 0 |
| pers_minus_cnn | -4.998915 | -5.004913 | 166 / 2 / 0 |
| f1_minus_cnn | -4.826469 | -4.790556 | 167 / 1 / 0 |

| Factorial effect | Mean | Negative / positive / zero fields | 24-hour block means |
|---|---:|---:|---|
| cv | -0.155090 | 103 / 65 / 0 | -0.444, -0.257, -0.012, +0.063, +0.060, -0.301, -0.195 |
| pers | -0.408934 | 110 / 58 / 0 | -0.712, -1.287, -0.047, +0.012, +0.320, -0.589, -0.559 |
| interaction | +0.655073 | 34 / 134 / 0 | +1.045, +1.117, +0.316, +0.518, +0.156, +0.813, +0.620 |

## W22

| Arm | Mean | Best-of-five count |
|---|---:|---:|
| cnn | 24.552151 | 1 |
| grad | 18.306367 | 53 |
| cv | 17.875645 | 24 |
| pers | 17.484939 | 66 |
| f1 | 17.831897 | 24 |
| ties | — | 0 |

| Comparison (A minus B) | Mean difference | Median | A better / worse / tied |
|---|---:|---:|---:|
| grad_minus_cnn | -6.245784 | -6.615243 | 164 / 4 / 0 |
| cv_minus_grad | -0.430722 | -0.344622 | 105 / 63 / 0 |
| pers_minus_grad | -0.821428 | -0.567661 | 105 / 63 / 0 |
| f1_minus_grad | -0.474469 | -0.267064 | 97 / 71 / 0 |
| f1_minus_cv | -0.043747 | +0.017702 | 82 / 86 / 0 |
| f1_minus_pers | +0.346958 | +0.330417 | 34 / 134 / 0 |
| pers_minus_cv | -0.390706 | -0.180479 | 108 / 60 / 0 |
| cv_minus_cnn | -6.676506 | -6.960783 | 166 / 2 / 0 |
| pers_minus_cnn | -7.067212 | -7.002674 | 167 / 1 / 0 |
| f1_minus_cnn | -6.720253 | -6.599013 | 166 / 2 / 0 |

| Factorial effect | Mean | Negative / positive / zero fields | 24-hour block means |
|---|---:|---:|---|
| cv | -0.041882 | 85 / 83 / 0 | -0.407, -0.181, +0.098, +0.268, +0.274, -0.202, -0.144 |
| pers | -0.432587 | 103 / 65 / 0 | -0.803, -1.671, +0.057, +0.227, +0.557, -0.662, -0.734 |
| interaction | +0.777680 | 36 / 132 / 0 | +1.217, +1.238, +0.441, +0.627, +0.178, +1.011, +0.733 |

## Interpretation boundaries

- Each value is a matched same-sample descriptive comparison; the fields are successive hours and are not independent replicates.
- Mean advantage alone does not prove systematic superiority; consult wins/losses, medians, and 24-hour blocks.
- A positive CV×PERS interaction in a lower-is-better metric means less-than-additive combination relative to the gradient-only 00 cell, not by itself failure of the combined model.
- F1's historical source/checkpoint continuity with the 2×2 training protocol remains a separate provenance gate.
- Unconstrained persistence distances do not establish correct spatial feature locations or merge-tree agreement.
