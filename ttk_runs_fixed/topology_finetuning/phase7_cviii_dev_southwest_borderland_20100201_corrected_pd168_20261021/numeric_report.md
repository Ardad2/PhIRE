# Phase 7 Part CIII — corrected GUDHI distances, 168-hour development week

Southwest Borderland, 2010-02 1–7 2010 UTC, 168 consecutive hourly samples.
**This is one development region-week, not 168 independent generalization replicates.**

1176 numerically decoded canonical TTK diagrams; 1008 model-to-GT comparisons,
using the exact Phase6 corrected GUDHI functions, with historical 2007 and XCVIII hour-0/84 regressions.

| Model | Mean hourly dB ↓ | Mean hourly W2inf ↓ | Mean hourly W22 ↓ |
|---|---:|---:|---:|
| cnn | 1.809163 | 8.820658 | 11.252322 |
| gan | 4.342029 | 10.623125 | 13.187466 |
| grad | 1.629562 | 7.366920 | 9.035153 |
| cv | 1.634258 | 7.399746 | 9.067438 |
| pers | 1.606318 | 7.355123 | 9.014435 |
| f1 | 1.642782 | 7.437585 | 9.126462 |

## Daily descriptive means

| Day UTC | Model | N hours | dB ↓ | W2inf ↓ | W22 ↓ |
|---|---|---:|---:|---:|---:|
| 2010-02-01 | cnn | 24 | 1.443455 | 7.522057 | 9.690320 |
| 2010-02-01 | gan | 24 | 3.305359 | 8.839677 | 11.047225 |
| 2010-02-01 | grad | 24 | 1.222048 | 6.135124 | 7.567805 |
| 2010-02-01 | cv | 24 | 1.205866 | 6.313755 | 7.823755 |
| 2010-02-01 | pers | 24 | 1.231162 | 6.331708 | 7.845311 |
| 2010-02-01 | f1 | 24 | 1.233858 | 6.354354 | 7.887144 |
| 2010-02-02 | cnn | 24 | 1.566723 | 7.723962 | 10.005569 |
| 2010-02-02 | gan | 24 | 2.605965 | 7.820031 | 9.665438 |
| 2010-02-02 | grad | 24 | 1.289302 | 6.248333 | 7.702815 |
| 2010-02-02 | cv | 24 | 1.282409 | 6.419607 | 7.961156 |
| 2010-02-02 | pers | 24 | 1.293342 | 6.433427 | 8.013736 |
| 2010-02-02 | f1 | 24 | 1.283719 | 6.481599 | 8.072818 |
| 2010-02-03 | cnn | 24 | 1.879875 | 10.573680 | 13.476024 |
| 2010-02-03 | gan | 24 | 4.459174 | 12.173010 | 15.246231 |
| 2010-02-03 | grad | 24 | 1.737024 | 8.689242 | 10.689700 |
| 2010-02-03 | cv | 24 | 1.798769 | 8.621094 | 10.466687 |
| 2010-02-03 | pers | 24 | 1.774466 | 8.420237 | 10.236181 |
| 2010-02-03 | f1 | 24 | 1.776868 | 8.605697 | 10.471734 |
| 2010-02-04 | cnn | 24 | 1.486173 | 8.189625 | 10.443233 |
| 2010-02-04 | gan | 24 | 5.028997 | 10.352999 | 12.569326 |
| 2010-02-04 | grad | 24 | 1.284183 | 6.897649 | 8.474387 |
| 2010-02-04 | cv | 24 | 1.357735 | 6.888402 | 8.438702 |
| 2010-02-04 | pers | 24 | 1.347089 | 6.841283 | 8.365968 |
| 2010-02-04 | f1 | 24 | 1.400932 | 6.951943 | 8.519711 |
| 2010-02-05 | cnn | 24 | 1.700474 | 8.312062 | 10.708194 |
| 2010-02-05 | gan | 24 | 3.387710 | 9.200470 | 11.388124 |
| 2010-02-05 | grad | 24 | 1.564644 | 6.867684 | 8.428290 |
| 2010-02-05 | cv | 24 | 1.598022 | 7.095409 | 8.760110 |
| 2010-02-05 | pers | 24 | 1.547782 | 7.087427 | 8.757318 |
| 2010-02-05 | f1 | 24 | 1.604032 | 7.133308 | 8.816800 |
| 2010-02-06 | cnn | 24 | 2.047230 | 8.896179 | 11.267751 |
| 2010-02-06 | gan | 24 | 5.363807 | 11.839443 | 14.750653 |
| 2010-02-06 | grad | 24 | 1.899555 | 7.752769 | 9.519606 |
| 2010-02-06 | cv | 24 | 1.920377 | 7.788312 | 9.564334 |
| 2010-02-06 | pers | 24 | 1.853239 | 7.735544 | 9.474861 |
| 2010-02-06 | f1 | 24 | 1.925603 | 7.810053 | 9.582255 |
| 2010-02-07 | cnn | 24 | 2.540210 | 10.527039 | 13.175163 |
| 2010-02-07 | gan | 24 | 6.243188 | 14.136247 | 17.645268 |
| 2010-02-07 | grad | 24 | 2.410178 | 8.977639 | 10.863468 |
| 2010-02-07 | cv | 24 | 2.276626 | 8.671646 | 10.457322 |
| 2010-02-07 | pers | 24 | 2.197147 | 8.636232 | 10.407671 |
| 2010-02-07 | f1 | 24 | 2.274459 | 8.726138 | 10.534776 |

**Scope and limitations:** numeric topology concerns only the unchanged top-left 160×160 scalar-speed crop,
not the full 500×500 vector field. Day-level summaries describe correlated hours, not independent statistical blocks.
Verification recomputes every value with the same trusted GUDHI decoder and distance implementation; this is not an
independently implemented numerical algorithm. No Phase5E locality, merge trees, training, model selection,
locked-region inference or external geographic generalization is performed in CIII.
Original PhIRE pretrained WTK spatial/temporal exposure is unverified.

