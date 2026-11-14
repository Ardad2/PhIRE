# Phase 7 Part CIII — corrected GUDHI distances, 168-hour development week

Gulf Inland, 2010-11 1–7 2010 UTC, 168 consecutive hourly samples.
**This is one development region-week, not 168 independent generalization replicates.**

1176 numerically decoded canonical TTK diagrams; 1008 model-to-GT comparisons,
using the exact Phase6 corrected GUDHI functions, with historical 2007 and XCVIII hour-0/84 regressions.

| Model | Mean hourly dB ↓ | Mean hourly W2inf ↓ | Mean hourly W22 ↓ |
|---|---:|---:|---:|
| cnn | 1.833918 | 7.674460 | 9.998829 |
| gan | 1.672425 | 6.657927 | 8.265875 |
| grad | 1.501902 | 6.128449 | 7.635318 |
| cv | 1.529366 | 6.336833 | 7.919711 |
| pers | 1.538040 | 6.256334 | 7.800484 |
| f1 | 1.529942 | 6.372009 | 7.982836 |

## Daily descriptive means

| Day UTC | Model | N hours | dB ↓ | W2inf ↓ | W22 ↓ |
|---|---|---:|---:|---:|---:|
| 2010-11-01 | cnn | 24 | 2.047987 | 9.152453 | 11.929254 |
| 2010-11-01 | gan | 24 | 1.758228 | 6.831044 | 8.331621 |
| 2010-11-01 | grad | 24 | 1.712628 | 6.739802 | 8.229865 |
| 2010-11-01 | cv | 24 | 1.750587 | 6.985457 | 8.582079 |
| 2010-11-01 | pers | 24 | 1.758920 | 7.121365 | 8.756954 |
| 2010-11-01 | f1 | 24 | 1.746197 | 7.171113 | 8.868847 |
| 2010-11-02 | cnn | 24 | 3.176694 | 10.094551 | 12.707128 |
| 2010-11-02 | gan | 24 | 2.970873 | 9.815303 | 12.079307 |
| 2010-11-02 | grad | 24 | 2.560869 | 7.838554 | 9.410164 |
| 2010-11-02 | cv | 24 | 2.574962 | 7.821668 | 9.337242 |
| 2010-11-02 | pers | 24 | 2.647164 | 7.951706 | 9.527265 |
| 2010-11-02 | f1 | 24 | 2.567779 | 7.890602 | 9.454667 |
| 2010-11-03 | cnn | 24 | 2.085136 | 8.359563 | 10.645888 |
| 2010-11-03 | gan | 24 | 1.972744 | 7.180092 | 8.733598 |
| 2010-11-03 | grad | 24 | 1.972094 | 7.220211 | 8.914793 |
| 2010-11-03 | cv | 24 | 1.956233 | 7.151456 | 8.743583 |
| 2010-11-03 | pers | 24 | 1.922851 | 6.951968 | 8.486724 |
| 2010-11-03 | f1 | 24 | 1.940473 | 7.185034 | 8.821076 |
| 2010-11-04 | cnn | 24 | 2.147812 | 7.493494 | 9.629994 |
| 2010-11-04 | gan | 24 | 1.967946 | 7.315473 | 9.142795 |
| 2010-11-04 | grad | 24 | 1.767073 | 6.335618 | 7.875865 |
| 2010-11-04 | cv | 24 | 1.752927 | 6.591439 | 8.247544 |
| 2010-11-04 | pers | 24 | 1.780116 | 6.220145 | 7.658741 |
| 2010-11-04 | f1 | 24 | 1.752683 | 6.460312 | 8.074094 |
| 2010-11-05 | cnn | 24 | 1.373417 | 5.715310 | 7.580986 |
| 2010-11-05 | gan | 24 | 1.277809 | 5.595884 | 7.087445 |
| 2010-11-05 | grad | 24 | 0.859323 | 4.301763 | 5.414314 |
| 2010-11-05 | cv | 24 | 0.962555 | 4.650827 | 5.894254 |
| 2010-11-05 | pers | 24 | 0.953113 | 4.421377 | 5.548878 |
| 2010-11-05 | f1 | 24 | 0.995615 | 4.619958 | 5.837794 |
| 2010-11-06 | cnn | 24 | 1.079681 | 5.983032 | 8.086655 |
| 2010-11-06 | gan | 24 | 0.904886 | 4.659915 | 5.895867 |
| 2010-11-06 | grad | 24 | 0.856762 | 4.805553 | 6.210626 |
| 2010-11-06 | cv | 24 | 0.894886 | 5.076568 | 6.604690 |
| 2010-11-06 | pers | 24 | 0.879895 | 5.034655 | 6.560459 |
| 2010-11-06 | f1 | 24 | 0.888518 | 5.130003 | 6.691433 |
| 2010-11-07 | cnn | 24 | 0.926698 | 6.922814 | 9.411898 |
| 2010-11-07 | gan | 24 | 0.854489 | 5.207774 | 6.590489 |
| 2010-11-07 | grad | 24 | 0.784567 | 5.657643 | 7.391599 |
| 2010-11-07 | cv | 24 | 0.813413 | 6.080414 | 8.028584 |
| 2010-11-07 | pers | 24 | 0.824222 | 6.093123 | 8.064366 |
| 2010-11-07 | f1 | 24 | 0.818330 | 6.147042 | 8.131940 |

**Scope and limitations:** numeric topology concerns only the unchanged top-left 160×160 scalar-speed crop,
not the full 500×500 vector field. Day-level summaries describe correlated hours, not independent statistical blocks.
Verification recomputes every value with the same trusted GUDHI decoder and distance implementation; this is not an
independently implemented numerical algorithm. No Phase5E locality, merge trees, training, model selection,
locked-region inference or external geographic generalization is performed in CIII.
Original PhIRE pretrained WTK spatial/temporal exposure is unverified.

