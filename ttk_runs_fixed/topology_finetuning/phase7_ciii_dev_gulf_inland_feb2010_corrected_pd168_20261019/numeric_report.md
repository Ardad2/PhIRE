# Phase 7 Part CIII — corrected GUDHI distances, 168-hour development week

Gulf inland, February 1–7 2010 UTC, 168 consecutive hourly samples.
**This is one development region-week, not 168 independent generalization replicates.**

1176 numerically decoded canonical TTK diagrams; 1008 model-to-GT comparisons,
using the exact Phase6 corrected GUDHI functions, with historical 2007 and XCVIII hour-0/84 regressions.

| Model | Mean hourly dB ↓ | Mean hourly W2inf ↓ | Mean hourly W22 ↓ |
|---|---:|---:|---:|
| cnn | 1.317913 | 5.192943 | 6.662873 |
| gan | 1.382009 | 5.327205 | 6.633682 |
| grad | 0.973180 | 4.081367 | 4.993435 |
| cv | 1.008254 | 4.164934 | 5.090495 |
| pers | 1.009105 | 4.194375 | 5.154307 |
| f1 | 1.024358 | 4.252456 | 5.228484 |

## Daily descriptive means

| Day UTC | Model | N hours | dB ↓ | W2inf ↓ | W22 ↓ |
|---|---|---:|---:|---:|---:|
| 2010-02-01 | cnn | 24 | 1.171930 | 5.286786 | 6.666202 |
| 2010-02-01 | gan | 24 | 1.127740 | 5.695691 | 7.068801 |
| 2010-02-01 | grad | 24 | 0.865535 | 3.993366 | 4.818149 |
| 2010-02-01 | cv | 24 | 0.840824 | 4.148776 | 5.017137 |
| 2010-02-01 | pers | 24 | 0.831476 | 4.076057 | 4.938293 |
| 2010-02-01 | f1 | 24 | 0.829154 | 4.171635 | 5.072048 |
| 2010-02-02 | cnn | 24 | 1.421075 | 5.892855 | 7.684686 |
| 2010-02-02 | gan | 24 | 1.275549 | 4.932678 | 6.029745 |
| 2010-02-02 | grad | 24 | 1.076426 | 4.156643 | 5.036779 |
| 2010-02-02 | cv | 24 | 1.166530 | 4.462041 | 5.407817 |
| 2010-02-02 | pers | 24 | 1.137399 | 4.446737 | 5.394156 |
| 2010-02-02 | f1 | 24 | 1.178362 | 4.534472 | 5.517940 |
| 2010-02-03 | cnn | 24 | 0.918483 | 4.015307 | 5.268312 |
| 2010-02-03 | gan | 24 | 0.923587 | 4.743632 | 6.094571 |
| 2010-02-03 | grad | 24 | 0.673612 | 3.416000 | 4.287849 |
| 2010-02-03 | cv | 24 | 0.716058 | 3.271994 | 4.057136 |
| 2010-02-03 | pers | 24 | 0.705096 | 3.467416 | 4.369886 |
| 2010-02-03 | f1 | 24 | 0.725807 | 3.325119 | 4.130586 |
| 2010-02-04 | cnn | 24 | 2.378134 | 7.508786 | 9.344390 |
| 2010-02-04 | gan | 24 | 2.907372 | 7.976500 | 9.826274 |
| 2010-02-04 | grad | 24 | 1.610249 | 6.114141 | 7.359081 |
| 2010-02-04 | cv | 24 | 1.710596 | 6.291342 | 7.623293 |
| 2010-02-04 | pers | 24 | 1.734535 | 6.340080 | 7.712394 |
| 2010-02-04 | f1 | 24 | 1.747568 | 6.416676 | 7.806125 |
| 2010-02-05 | cnn | 24 | 1.403541 | 5.738860 | 7.360806 |
| 2010-02-05 | gan | 24 | 1.309471 | 5.197019 | 6.404961 |
| 2010-02-05 | grad | 24 | 1.137500 | 4.655254 | 5.751908 |
| 2010-02-05 | cv | 24 | 1.079922 | 4.396178 | 5.335445 |
| 2010-02-05 | pers | 24 | 1.118812 | 4.450880 | 5.464261 |
| 2010-02-05 | f1 | 24 | 1.108995 | 4.500470 | 5.518704 |
| 2010-02-06 | cnn | 24 | 0.957889 | 3.434286 | 4.403196 |
| 2010-02-06 | gan | 24 | 1.248740 | 4.859494 | 6.193273 |
| 2010-02-06 | grad | 24 | 0.667369 | 2.772952 | 3.439493 |
| 2010-02-06 | cv | 24 | 0.711716 | 2.915615 | 3.633331 |
| 2010-02-06 | pers | 24 | 0.694347 | 2.887231 | 3.602271 |
| 2010-02-06 | f1 | 24 | 0.732371 | 3.050415 | 3.845248 |
| 2010-02-07 | cnn | 24 | 0.974342 | 4.473720 | 5.912520 |
| 2010-02-07 | gan | 24 | 0.881606 | 3.885425 | 4.818147 |
| 2010-02-07 | grad | 24 | 0.781567 | 3.461211 | 4.260783 |
| 2010-02-07 | cv | 24 | 0.832135 | 3.668591 | 4.559308 |
| 2010-02-07 | pers | 24 | 0.842067 | 3.692224 | 4.598887 |
| 2010-02-07 | f1 | 24 | 0.848247 | 3.768406 | 4.708739 |

**Scope and limitations:** numeric topology concerns only the unchanged top-left 160×160 scalar-speed crop,
not the full 500×500 vector field. Day-level summaries describe correlated hours, not independent statistical blocks.
Verification recomputes every value with the same trusted GUDHI decoder and distance implementation; this is not an
independently implemented numerical algorithm. No Phase5E locality, merge trees, training, model selection,
locked-region inference or external geographic generalization is performed in CIII.
Original PhIRE pretrained WTK spatial/temporal exposure is unverified.

