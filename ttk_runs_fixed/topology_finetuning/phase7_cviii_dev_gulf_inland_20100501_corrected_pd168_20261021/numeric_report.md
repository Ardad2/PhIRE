# Phase 7 Part CIII — corrected GUDHI distances, 168-hour development week

Gulf Inland, 2010-05 1–7 2010 UTC, 168 consecutive hourly samples.
**This is one development region-week, not 168 independent generalization replicates.**

1176 numerically decoded canonical TTK diagrams; 1008 model-to-GT comparisons,
using the exact Phase6 corrected GUDHI functions, with historical 2007 and XCVIII hour-0/84 regressions.

| Model | Mean hourly dB ↓ | Mean hourly W2inf ↓ | Mean hourly W22 ↓ |
|---|---:|---:|---:|
| cnn | 1.316839 | 6.304008 | 8.194879 |
| gan | 1.455013 | 5.922214 | 7.371158 |
| grad | 1.007760 | 5.062648 | 6.319847 |
| cv | 1.029228 | 5.219638 | 6.530926 |
| pers | 1.034698 | 5.226000 | 6.546583 |
| f1 | 1.047275 | 5.329780 | 6.699722 |

## Daily descriptive means

| Day UTC | Model | N hours | dB ↓ | W2inf ↓ | W22 ↓ |
|---|---|---:|---:|---:|---:|
| 2010-05-01 | cnn | 24 | 1.872433 | 7.529873 | 9.462943 |
| 2010-05-01 | gan | 24 | 1.383185 | 7.189068 | 9.058935 |
| 2010-05-01 | grad | 24 | 1.440767 | 6.574107 | 8.185516 |
| 2010-05-01 | cv | 24 | 1.388292 | 6.367787 | 7.844724 |
| 2010-05-01 | pers | 24 | 1.374463 | 6.372866 | 7.865256 |
| 2010-05-01 | f1 | 24 | 1.391257 | 6.476133 | 8.010348 |
| 2010-05-02 | cnn | 24 | 1.566244 | 7.993410 | 10.266473 |
| 2010-05-02 | gan | 24 | 1.571064 | 6.918772 | 8.416213 |
| 2010-05-02 | grad | 24 | 1.362540 | 6.996176 | 8.686149 |
| 2010-05-02 | cv | 24 | 1.371241 | 7.138647 | 8.887815 |
| 2010-05-02 | pers | 24 | 1.355596 | 7.099881 | 8.857723 |
| 2010-05-02 | f1 | 24 | 1.386422 | 7.283551 | 9.111110 |
| 2010-05-03 | cnn | 24 | 0.936946 | 4.674885 | 6.082130 |
| 2010-05-03 | gan | 24 | 1.627396 | 5.398656 | 6.717880 |
| 2010-05-03 | grad | 24 | 0.694002 | 3.563564 | 4.389911 |
| 2010-05-03 | cv | 24 | 0.701847 | 3.803291 | 4.737556 |
| 2010-05-03 | pers | 24 | 0.691299 | 3.817914 | 4.756453 |
| 2010-05-03 | f1 | 24 | 0.704037 | 3.857261 | 4.819199 |
| 2010-05-04 | cnn | 24 | 1.180260 | 5.172233 | 6.675844 |
| 2010-05-04 | gan | 24 | 1.935840 | 5.984349 | 7.494094 |
| 2010-05-04 | grad | 24 | 0.901335 | 4.079309 | 5.028001 |
| 2010-05-04 | cv | 24 | 0.951074 | 4.270829 | 5.273034 |
| 2010-05-04 | pers | 24 | 0.974729 | 4.279661 | 5.284625 |
| 2010-05-04 | f1 | 24 | 0.973246 | 4.308102 | 5.324324 |
| 2010-05-05 | cnn | 24 | 1.267751 | 4.555299 | 5.795216 |
| 2010-05-05 | gan | 24 | 1.533283 | 5.607760 | 7.015459 |
| 2010-05-05 | grad | 24 | 0.868700 | 3.467467 | 4.217209 |
| 2010-05-05 | cv | 24 | 0.927347 | 3.615331 | 4.433814 |
| 2010-05-05 | pers | 24 | 0.938948 | 3.586318 | 4.385460 |
| 2010-05-05 | f1 | 24 | 0.959989 | 3.707450 | 4.553237 |
| 2010-05-06 | cnn | 24 | 1.109042 | 6.560376 | 8.890319 |
| 2010-05-06 | gan | 24 | 1.125102 | 4.614150 | 5.718469 |
| 2010-05-06 | grad | 24 | 0.844136 | 4.866037 | 6.211481 |
| 2010-05-06 | cv | 24 | 0.855445 | 5.123276 | 6.579356 |
| 2010-05-06 | pers | 24 | 0.874586 | 5.181850 | 6.677841 |
| 2010-05-06 | f1 | 24 | 0.867106 | 5.236843 | 6.771707 |
| 2010-05-07 | cnn | 24 | 1.285195 | 7.641980 | 10.191227 |
| 2010-05-07 | gan | 24 | 1.009223 | 5.742744 | 7.177057 |
| 2010-05-07 | grad | 24 | 0.942839 | 5.891873 | 7.520665 |
| 2010-05-07 | cv | 24 | 1.009350 | 6.218307 | 7.960185 |
| 2010-05-07 | pers | 24 | 1.033268 | 6.243507 | 7.998723 |
| 2010-05-07 | f1 | 24 | 1.048871 | 6.439120 | 8.308130 |

**Scope and limitations:** numeric topology concerns only the unchanged top-left 160×160 scalar-speed crop,
not the full 500×500 vector field. Day-level summaries describe correlated hours, not independent statistical blocks.
Verification recomputes every value with the same trusted GUDHI decoder and distance implementation; this is not an
independently implemented numerical algorithm. No Phase5E locality, merge trees, training, model selection,
locked-region inference or external geographic generalization is performed in CIII.
Original PhIRE pretrained WTK spatial/temporal exposure is unverified.

