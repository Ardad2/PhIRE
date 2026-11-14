# Phase 7 Part CIII — corrected GUDHI distances, 168-hour development week

Gulf Inland, 2010-08 1–7 2010 UTC, 168 consecutive hourly samples.
**This is one development region-week, not 168 independent generalization replicates.**

1176 numerically decoded canonical TTK diagrams; 1008 model-to-GT comparisons,
using the exact Phase6 corrected GUDHI functions, with historical 2007 and XCVIII hour-0/84 regressions.

| Model | Mean hourly dB ↓ | Mean hourly W2inf ↓ | Mean hourly W22 ↓ |
|---|---:|---:|---:|
| cnn | 1.322646 | 6.128661 | 8.091253 |
| gan | 1.529304 | 5.149404 | 6.316753 |
| grad | 1.030532 | 4.688777 | 5.842914 |
| cv | 1.027313 | 4.839941 | 6.071065 |
| pers | 1.022742 | 4.823607 | 6.044826 |
| f1 | 1.032587 | 4.917682 | 6.188861 |

## Daily descriptive means

| Day UTC | Model | N hours | dB ↓ | W2inf ↓ | W22 ↓ |
|---|---|---:|---:|---:|---:|
| 2010-08-01 | cnn | 24 | 1.103173 | 6.315413 | 8.482186 |
| 2010-08-01 | gan | 24 | 1.355297 | 4.963294 | 6.206442 |
| 2010-08-01 | grad | 24 | 0.768573 | 4.664336 | 5.904251 |
| 2010-08-01 | cv | 24 | 0.825740 | 4.922515 | 6.270727 |
| 2010-08-01 | pers | 24 | 0.807062 | 4.879669 | 6.223617 |
| 2010-08-01 | f1 | 24 | 0.831255 | 4.979662 | 6.352192 |
| 2010-08-02 | cnn | 24 | 1.110666 | 4.483054 | 5.889154 |
| 2010-08-02 | gan | 24 | 1.761058 | 4.956166 | 6.131591 |
| 2010-08-02 | grad | 24 | 0.826464 | 3.491446 | 4.336325 |
| 2010-08-02 | cv | 24 | 0.832650 | 3.621189 | 4.533611 |
| 2010-08-02 | pers | 24 | 0.819502 | 3.629448 | 4.548851 |
| 2010-08-02 | f1 | 24 | 0.839350 | 3.660477 | 4.589962 |
| 2010-08-03 | cnn | 24 | 1.152746 | 5.659223 | 7.438259 |
| 2010-08-03 | gan | 24 | 1.336715 | 4.781893 | 5.905589 |
| 2010-08-03 | grad | 24 | 0.938435 | 4.405759 | 5.493327 |
| 2010-08-03 | cv | 24 | 0.944959 | 4.511407 | 5.658874 |
| 2010-08-03 | pers | 24 | 0.927137 | 4.474763 | 5.598036 |
| 2010-08-03 | f1 | 24 | 0.937921 | 4.535333 | 5.699918 |
| 2010-08-04 | cnn | 24 | 1.436665 | 6.394840 | 8.481140 |
| 2010-08-04 | gan | 24 | 1.453412 | 4.967942 | 5.995681 |
| 2010-08-04 | grad | 24 | 1.126054 | 4.925738 | 6.110552 |
| 2010-08-04 | cv | 24 | 1.101008 | 5.058628 | 6.327799 |
| 2010-08-04 | pers | 24 | 1.103470 | 5.009162 | 6.257618 |
| 2010-08-04 | f1 | 24 | 1.096206 | 5.120910 | 6.445618 |
| 2010-08-05 | cnn | 24 | 1.321447 | 5.567166 | 7.372038 |
| 2010-08-05 | gan | 24 | 1.395216 | 4.814997 | 5.943084 |
| 2010-08-05 | grad | 24 | 1.102783 | 4.316500 | 5.403510 |
| 2010-08-05 | cv | 24 | 1.056914 | 4.456823 | 5.602097 |
| 2010-08-05 | pers | 24 | 1.060586 | 4.425942 | 5.549249 |
| 2010-08-05 | f1 | 24 | 1.067733 | 4.538044 | 5.723633 |
| 2010-08-06 | cnn | 24 | 1.430383 | 6.280588 | 8.246999 |
| 2010-08-06 | gan | 24 | 1.582773 | 5.391848 | 6.539064 |
| 2010-08-06 | grad | 24 | 1.100685 | 4.933681 | 6.144126 |
| 2010-08-06 | cv | 24 | 1.190199 | 5.105039 | 6.389897 |
| 2010-08-06 | pers | 24 | 1.192566 | 5.097166 | 6.364297 |
| 2010-08-06 | f1 | 24 | 1.206685 | 5.228455 | 6.561482 |
| 2010-08-07 | cnn | 24 | 1.703444 | 8.200346 | 10.728994 |
| 2010-08-07 | gan | 24 | 1.820657 | 6.169688 | 7.495823 |
| 2010-08-07 | grad | 24 | 1.350728 | 6.083979 | 7.508306 |
| 2010-08-07 | cv | 24 | 1.239721 | 6.203986 | 7.714452 |
| 2010-08-07 | pers | 24 | 1.248873 | 6.249103 | 7.772115 |
| 2010-08-07 | f1 | 24 | 1.248962 | 6.360896 | 7.949221 |

**Scope and limitations:** numeric topology concerns only the unchanged top-left 160×160 scalar-speed crop,
not the full 500×500 vector field. Day-level summaries describe correlated hours, not independent statistical blocks.
Verification recomputes every value with the same trusted GUDHI decoder and distance implementation; this is not an
independently implemented numerical algorithm. No Phase5E locality, merge trees, training, model selection,
locked-region inference or external geographic generalization is performed in CIII.
Original PhIRE pretrained WTK spatial/temporal exposure is unverified.

