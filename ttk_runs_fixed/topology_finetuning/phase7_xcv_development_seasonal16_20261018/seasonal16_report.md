# Phase 7 XCV — sixteen-field 2010 development source expansion

Two fixed development regions × four primary 2010 weeks × two predeclared hours (0, 84).
**This 16-field tranche is for pipeline/scaling checks, not the complete 168-hour-per-week evaluation.**

| Region | Season UTC | Offset h | WTK time index | Source |
|---|---|---:|---:|---|
| gulf_inland | 2010-02-01 | 0 | 27048 | cached LXXXVII |
| gulf_inland | 2010-02-01 | 84 | 27132 | HSDS |
| gulf_inland | 2010-05-01 | 0 | 29184 | HSDS |
| gulf_inland | 2010-05-01 | 84 | 29268 | HSDS |
| gulf_inland | 2010-08-01 | 0 | 31392 | cached LXXXVII |
| gulf_inland | 2010-08-01 | 84 | 31476 | HSDS |
| gulf_inland | 2010-11-01 | 0 | 33600 | HSDS |
| gulf_inland | 2010-11-01 | 84 | 33684 | HSDS |
| southwest_borderland | 2010-02-01 | 0 | 27048 | cached LXXXVII |
| southwest_borderland | 2010-02-01 | 84 | 27132 | HSDS |
| southwest_borderland | 2010-05-01 | 0 | 29184 | HSDS |
| southwest_borderland | 2010-05-01 | 84 | 29268 | HSDS |
| southwest_borderland | 2010-08-01 | 0 | 31392 | cached LXXXVII |
| southwest_borderland | 2010-08-01 | 84 | 31476 | HSDS |
| southwest_borderland | 2010-11-01 | 0 | 33600 | HSDS |
| southwest_borderland | 2010-11-01 | 84 | 33684 | HSDS |

No model inference, topology metrics, new hyperparameter search, or external-test model scores.
Each file stores 500×500×2 HR and 100×100×2 MR [u,v] with exact region, grid rectangle and UTC metadata.
All extracted winds agree with previously frozen LXXXVII screening speed statistics within 2e-8 m/s.
These locations are outside documented fine-tuning footprint; original pretrained-model WTK membership is unknown.
