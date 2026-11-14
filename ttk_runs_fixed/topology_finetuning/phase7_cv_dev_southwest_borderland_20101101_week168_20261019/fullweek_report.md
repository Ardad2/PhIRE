# Phase 7 Part C — complete development-only WTK source week

- Frozen region: `southwest_borderland`, rectangle `(173, 673, 572, 1072)` (WTK rows/columns).
- UTC: `2010-11-01T00:00:00` through `2010-11-07T23:00:00` inclusive; WTK indices `33600` to `33767`.
- 168 verified native HR `500×500×2` and factor-five MR `100×100×2` `[u,v]` float64 fields.
- Original XCV hours 0 and 84 copied bit-for-bit; 166 additional hourly WTK slices.
- Raw uncompressed HR+MR payload 698,880,000 bytes (666.50 MiB).
- Wind-speed and pooling validations applied to each hour; precise SHA256 provenance in manifest.
- **Source-data only.** No six-model predictions, PD, locality, training or locked external-region access.
- No conclusions about pretrained PhIRE data exposure, seasonal robustness, or independent hourly inference.
