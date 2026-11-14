# Phase 7 XCI — frozen evaluation protocol

**Predeclared before any locked external-region model inference.**

This protocol is immutable. Read `evaluation_protocol.json` for all definitions.

## Roles

- Development: gulf_inland and southwest_borderland.
- Locked external: upper_midwest and pacific_inland.
- Historical reference: mid_atlantic_legacy (not a clean external test).

## Time windows

- Primary: 2010 Feb/May/Aug/Nov, 168 hourly fields per region-week.
- Replication: 2012 Feb/May/Aug/Nov, same 168-hour windows per region.
- Aggregation: paired within-field; block-level region-week means first; n=8 region-weeks per role per year.

## Predeclared outcomes

- Full 500x500 reconstruction UV RMSE and speed MAE.
- **Primary topology:** original top-left 160x160 speed crop; corrected GUDHI dB, W2inf, W22, frozen Phase5E W22 locality radius grid [4,8,16,32,64,96,128,infinity], GT mass coverage.
- **Secondary:** nine disjoint 160x160 tiles (starting x/y 0,160,320), reported separately.
- **Development-only exploratory:** full 500x500 topology feasibility and independently checked merge-tree distances.
- Six frozen checkpoints: cnn, gan, grad, cv, pers, f1.

## Rules

- No model/metric/window selection in response to external test scores.
- If development gates require code changes, issue a versioned protocol amendment **before** external inference; never rewrite this freeze.
- No pooled-hour independence claims. Document errors and missing records rather than silently excluding them.
- No claim that the original PhIRE pretraining has excluded the new geography.

## Provenance

- Original split SHA256: `34bd066349b67ca9fbe55c4dd1ee397946ea14b4e2c96a530dcb98c3abb29876`
- XC inference-report SHA256: `6a7af67a3fe2658dfedc0f5fa0503eaae6ab69460284074d1dcd6b6238318d09`
- XCI plan SHA256: `2de6234bd92ebaa09bf46988a9d3026261260f843ed916b682d3d207d39b26ac`
