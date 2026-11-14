# Phase 7 Part CII — 168-hour canonical TTK persistence extraction

Frozen Gulf inland 2010-02-01 00:00 through 2010-02-07 23:00 UTC.
**One development region-week; not an external geographic generalization benchmark.**

168 sequential WTK hours × GT and six fixed model outputs = **1176** canonical VTI/port-0 VTU pairs.
Existing hour 0 and hour 84 inputs/diagrams (XCVII indices 0 and 1) are bitwise reused (14 pairs); **1162** pairs are newly generated.

| Source | Reused XCVII | Newly extracted | Total |
|---|---:|---:|---:|
| gt | 2 | 166 | 168 |
| cnn | 2 | 166 | 168 |
| gan | 2 | 166 | 168 |
| grad | 2 | 166 | 168 |
| cv | 2 | 166 | 168 |
| pers | 2 | 166 | 168 |
| f1 | 2 | 166 | 168 |

All model/GT scalar-speed crops are top-left 160×160, VTK x=columns, y=rows.
Unmodified canonical ASCII Float32 converter and `ttkPersistenceDiagramCmd -t 1`.
**Only VTI values and PD VTU XML structure are validated here. Compressed TTK numerical persistence pairs are NOT decoded.**
Corrected GUDHI numerical decoding, XCIII/XCVIII anchor regression, and Phase5E locality matching are later steps.
No WTK reads, TensorFlow inference, model changes, locked-region access, or hyperparameter selection.
Original pretrained PhIRE training-geography/time exposure remains unknown.
