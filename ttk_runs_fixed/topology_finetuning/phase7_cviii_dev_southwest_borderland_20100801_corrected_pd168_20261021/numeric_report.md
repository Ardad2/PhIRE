# Phase 7 Part CIII — corrected GUDHI distances, 168-hour development week

Southwest Borderland, 2010-08 1–7 2010 UTC, 168 consecutive hourly samples.
**This is one development region-week, not 168 independent generalization replicates.**

1176 numerically decoded canonical TTK diagrams; 1008 model-to-GT comparisons,
using the exact Phase6 corrected GUDHI functions, with historical 2007 and XCVIII hour-0/84 regressions.

| Model | Mean hourly dB ↓ | Mean hourly W2inf ↓ | Mean hourly W22 ↓ |
|---|---:|---:|---:|
| cnn | 2.056869 | 8.940267 | 11.269066 |
| gan | 3.710275 | 11.488708 | 14.396649 |
| grad | 1.834464 | 7.551453 | 9.173748 |
| cv | 1.728312 | 7.527978 | 9.200345 |
| pers | 1.743987 | 7.592379 | 9.291722 |
| f1 | 1.756754 | 7.594873 | 9.300173 |

## Daily descriptive means

| Day UTC | Model | N hours | dB ↓ | W2inf ↓ | W22 ↓ |
|---|---|---:|---:|---:|---:|
| 2010-08-01 | cnn | 24 | 2.010324 | 8.678729 | 10.990957 |
| 2010-08-01 | gan | 24 | 3.394133 | 10.604217 | 13.445963 |
| 2010-08-01 | grad | 24 | 1.743953 | 7.258240 | 8.853921 |
| 2010-08-01 | cv | 24 | 1.618229 | 7.225913 | 8.867254 |
| 2010-08-01 | pers | 24 | 1.641377 | 7.407735 | 9.114444 |
| 2010-08-01 | f1 | 24 | 1.640664 | 7.325007 | 8.990379 |
| 2010-08-02 | cnn | 24 | 2.041949 | 7.774076 | 9.773849 |
| 2010-08-02 | gan | 24 | 3.836059 | 11.587315 | 14.565100 |
| 2010-08-02 | grad | 24 | 1.709163 | 6.657147 | 8.087574 |
| 2010-08-02 | cv | 24 | 1.743027 | 6.568954 | 8.021186 |
| 2010-08-02 | pers | 24 | 1.736127 | 6.666742 | 8.141023 |
| 2010-08-02 | f1 | 24 | 1.759438 | 6.623342 | 8.095625 |
| 2010-08-03 | cnn | 24 | 1.883544 | 8.583021 | 10.814144 |
| 2010-08-03 | gan | 24 | 3.977276 | 11.469650 | 14.337935 |
| 2010-08-03 | grad | 24 | 1.766198 | 7.354423 | 8.918021 |
| 2010-08-03 | cv | 24 | 1.523317 | 7.214030 | 8.838770 |
| 2010-08-03 | pers | 24 | 1.570235 | 7.230927 | 8.869916 |
| 2010-08-03 | f1 | 24 | 1.562251 | 7.225380 | 8.858241 |
| 2010-08-04 | cnn | 24 | 2.061924 | 8.989198 | 11.253898 |
| 2010-08-04 | gan | 24 | 3.316000 | 11.963502 | 14.974190 |
| 2010-08-04 | grad | 24 | 1.747400 | 7.742410 | 9.387818 |
| 2010-08-04 | cv | 24 | 1.617384 | 7.608803 | 9.260638 |
| 2010-08-04 | pers | 24 | 1.617133 | 7.612890 | 9.276809 |
| 2010-08-04 | f1 | 24 | 1.638258 | 7.640398 | 9.316484 |
| 2010-08-05 | cnn | 24 | 2.240171 | 9.316599 | 11.664931 |
| 2010-08-05 | gan | 24 | 4.802864 | 12.745758 | 15.892309 |
| 2010-08-05 | grad | 24 | 2.091803 | 8.019688 | 9.697245 |
| 2010-08-05 | cv | 24 | 1.941225 | 7.972514 | 9.680315 |
| 2010-08-05 | pers | 24 | 1.923611 | 7.937770 | 9.641416 |
| 2010-08-05 | f1 | 24 | 2.022457 | 8.025194 | 9.766956 |
| 2010-08-06 | cnn | 24 | 2.337925 | 10.138567 | 12.822615 |
| 2010-08-06 | gan | 24 | 3.206771 | 11.074291 | 13.727667 |
| 2010-08-06 | grad | 24 | 2.155771 | 8.519433 | 10.322045 |
| 2010-08-06 | cv | 24 | 2.050149 | 8.580959 | 10.456172 |
| 2010-08-06 | pers | 24 | 2.068946 | 8.610668 | 10.520591 |
| 2010-08-06 | f1 | 24 | 2.059619 | 8.664006 | 10.593666 |
| 2010-08-07 | cnn | 24 | 1.822248 | 9.101679 | 11.563066 |
| 2010-08-07 | gan | 24 | 3.438822 | 10.976223 | 13.833380 |
| 2010-08-07 | grad | 24 | 1.626958 | 7.308829 | 8.949614 |
| 2010-08-07 | cv | 24 | 1.604850 | 7.524673 | 9.278080 |
| 2010-08-07 | pers | 24 | 1.650478 | 7.679921 | 9.477855 |
| 2010-08-07 | f1 | 24 | 1.614591 | 7.660787 | 9.479862 |

**Scope and limitations:** numeric topology concerns only the unchanged top-left 160×160 scalar-speed crop,
not the full 500×500 vector field. Day-level summaries describe correlated hours, not independent statistical blocks.
Verification recomputes every value with the same trusted GUDHI decoder and distance implementation; this is not an
independently implemented numerical algorithm. No Phase5E locality, merge trees, training, model selection,
locked-region inference or external geographic generalization is performed in CIII.
Original PhIRE pretrained WTK spatial/temporal exposure is unverified.

