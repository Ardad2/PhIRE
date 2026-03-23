# Wind MR→HR pairing audit

TFRecord: `example_data_extension_500/wind_MR-HR.tfrecord`

Candidate comparison set: rows [0, 1, 2, 3, 4, 50, 100, 120, 150, 167].

## Pairing result summary
- Samples audited: 6
- Paired MR rank-1 by RMSE: 5/6
- Paired MR rank-1 by MAE: 2/6
- Median paired-rank by RMSE: 1
- Mean paired RMSE: 6.0479
- Median paired RMSE: 5.90539
- Mean best-wrong minus paired RMSE gap: 0.151429
- Min best-wrong minus paired RMSE gap: -0.782151

## Orientation sanity check
- Representative samples checked: 3
- Cases where a non-identity transform beat identity: 0
- Identity was best on all representative orientation checks.

## Interpretation guide
- If the paired MR is usually rank-1 or near-rank-1 against wrong MR candidates, that strongly supports valid per-sample pairing.
- If the best-wrong minus paired gap is consistently positive, the paired MR is meaningfully closer to its own downsampled HR than to wrong samples.
- If identity beats transpose/flip/channel-swap checks, that argues against an obvious layout/orientation bug.
