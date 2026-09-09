Sample-69 PhIRE vs colleague PD compatibility audit
====================================================

Bundle contents:
  run_sample69_pd_compatibility.py

This script performs:

Phase A
-------
Same TTK canonical finite D0/D1 point arrays:
  - PhIRE/reference exact GUDHI metrics
  - colleague wrapper bottleneck
  - colleague wrapper Wasserstein order=1
  - colleague wrapper Wasserstein order=2

Expected:
  - parsed TTK points reproduce frozen PhIRE dB/W2inf/W22
  - colleague native bottleneck agrees with exact bottleneck to numerical tolerance
  - colleague order=2 Wasserstein agrees with PhIRE W2inf

The current colleague wrapper does not expose `internal_p=2`, so W22 is reported
as a parity extension rather than as a native wrapper result.

Phase B
-------
Uses the colleague's actual:
    tda_toolkit.persistence.compute_cubical_persistence()

on the authoritative sample-69 160x160 wind-speed fields.

It reports:
  - finite D0/D1 cardinalities
  - GT->CNN / UV / F1 dB, W2inf, W22
  - method rankings
  - same-field TTK-vs-cubical descriptor distances

Important:
Phase B is not expected to reproduce the TTK diagrams exactly because TTK and
GUDHI CubicalComplex use different filtered-complex/cell conventions.

Suggested server layout
-----------------------
Extract the colleague repository to:

    ~/PhIRE/third_party/tda-toolkit-mapper/

so that this exists:

    ~/PhIRE/third_party/tda-toolkit-mapper/src/tda_toolkit/

Then run:

    /usr/bin/python3 \
      "$W22/run_sample69_pd_compatibility.py" \
      2>&1 | tee "$W22/run_sample69_pd_compatibility.log"

If the colleague source is elsewhere:

    /usr/bin/python3 \
      "$W22/run_sample69_pd_compatibility.py" \
      --toolkit-root /path/to/tda-toolkit-mapper \
      2>&1 | tee "$W22/run_sample69_pd_compatibility.log"

Outputs:
    $W22/pd_colleague_compatibility/sample_069/
