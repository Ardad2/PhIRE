# Phase 7 Part CIII — corrected GUDHI distances, 168-hour development week

Southwest Borderland, 2010-05 1–7 2010 UTC, 168 consecutive hourly samples.
**This is one development region-week, not 168 independent generalization replicates.**

1176 numerically decoded canonical TTK diagrams; 1008 model-to-GT comparisons,
using the exact Phase6 corrected GUDHI functions, with historical 2007 and XCVIII hour-0/84 regressions.

| Model | Mean hourly dB ↓ | Mean hourly W2inf ↓ | Mean hourly W22 ↓ |
|---|---:|---:|---:|
| cnn | 2.326211 | 9.853938 | 12.273587 |
| gan | 4.638909 | 12.946154 | 16.260901 |
| grad | 2.009753 | 8.375479 | 10.176955 |
| cv | 1.964134 | 8.363117 | 10.161532 |
| pers | 1.893822 | 8.366893 | 10.188375 |
| f1 | 1.987631 | 8.432030 | 10.258091 |

## Daily descriptive means

| Day UTC | Model | N hours | dB ↓ | W2inf ↓ | W22 ↓ |
|---|---|---:|---:|---:|---:|
| 2010-05-01 | cnn | 24 | 2.843104 | 9.023078 | 10.956488 |
| 2010-05-01 | gan | 24 | 6.646957 | 15.062371 | 19.538030 |
| 2010-05-01 | grad | 24 | 1.970319 | 7.665486 | 9.241969 |
| 2010-05-01 | cv | 24 | 2.042865 | 7.641525 | 9.151673 |
| 2010-05-01 | pers | 24 | 2.000618 | 7.540183 | 9.011131 |
| 2010-05-01 | f1 | 24 | 2.078847 | 7.655561 | 9.160667 |
| 2010-05-02 | cnn | 24 | 2.546132 | 10.470926 | 13.090971 |
| 2010-05-02 | gan | 24 | 5.498461 | 14.350639 | 18.184565 |
| 2010-05-02 | grad | 24 | 2.249363 | 8.561383 | 10.375289 |
| 2010-05-02 | cv | 24 | 2.334198 | 8.768622 | 10.576646 |
| 2010-05-02 | pers | 24 | 2.232684 | 8.670848 | 10.495246 |
| 2010-05-02 | f1 | 24 | 2.359430 | 8.899132 | 10.785143 |
| 2010-05-03 | cnn | 24 | 2.151299 | 9.765983 | 12.167873 |
| 2010-05-03 | gan | 24 | 4.764228 | 13.523143 | 16.659398 |
| 2010-05-03 | grad | 24 | 2.096657 | 8.449588 | 10.236273 |
| 2010-05-03 | cv | 24 | 1.928624 | 8.298552 | 10.073134 |
| 2010-05-03 | pers | 24 | 1.949855 | 8.302856 | 10.143832 |
| 2010-05-03 | f1 | 24 | 1.962087 | 8.360989 | 10.184355 |
| 2010-05-04 | cnn | 24 | 2.550928 | 10.977373 | 13.376680 |
| 2010-05-04 | gan | 24 | 4.624294 | 14.228052 | 17.611639 |
| 2010-05-04 | grad | 24 | 2.022903 | 9.317938 | 11.271180 |
| 2010-05-04 | cv | 24 | 2.034706 | 9.312642 | 11.212993 |
| 2010-05-04 | pers | 24 | 1.971047 | 9.378554 | 11.267798 |
| 2010-05-04 | f1 | 24 | 2.072192 | 9.405236 | 11.263903 |
| 2010-05-05 | cnn | 24 | 1.837685 | 8.772612 | 11.143187 |
| 2010-05-05 | gan | 24 | 3.165142 | 10.722705 | 13.523747 |
| 2010-05-05 | grad | 24 | 1.754199 | 7.385635 | 9.112019 |
| 2010-05-05 | cv | 24 | 1.577583 | 7.357869 | 9.105319 |
| 2010-05-05 | pers | 24 | 1.560842 | 7.411864 | 9.197223 |
| 2010-05-05 | f1 | 24 | 1.581054 | 7.389288 | 9.183357 |
| 2010-05-06 | cnn | 24 | 2.357852 | 10.303906 | 12.944745 |
| 2010-05-06 | gan | 24 | 4.277423 | 11.892389 | 14.746806 |
| 2010-05-06 | grad | 24 | 2.036313 | 8.775167 | 10.685215 |
| 2010-05-06 | cv | 24 | 2.001112 | 8.765898 | 10.722990 |
| 2010-05-06 | pers | 24 | 1.886532 | 8.854352 | 10.854175 |
| 2010-05-06 | f1 | 24 | 2.054718 | 8.885749 | 10.866513 |
| 2010-05-07 | cnn | 24 | 1.996479 | 9.663690 | 12.235164 |
| 2010-05-07 | gan | 24 | 3.495857 | 10.843782 | 13.562123 |
| 2010-05-07 | grad | 24 | 1.938514 | 8.473155 | 10.316740 |
| 2010-05-07 | cv | 24 | 1.829847 | 8.396711 | 10.287967 |
| 2010-05-07 | pers | 24 | 1.655177 | 8.409593 | 10.349219 |
| 2010-05-07 | f1 | 24 | 1.805090 | 8.428256 | 10.362700 |

**Scope and limitations:** numeric topology concerns only the unchanged top-left 160×160 scalar-speed crop,
not the full 500×500 vector field. Day-level summaries describe correlated hours, not independent statistical blocks.
Verification recomputes every value with the same trusted GUDHI decoder and distance implementation; this is not an
independently implemented numerical algorithm. No Phase5E locality, merge trees, training, model selection,
locked-region inference or external geographic generalization is performed in CIII.
Original PhIRE pretrained WTK spatial/temporal exposure is unverified.

