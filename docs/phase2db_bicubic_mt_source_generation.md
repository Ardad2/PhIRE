# Phase 2D-B Bicubic Merge-Tree Source Generation

## Purpose

The Phase 2D-B manual topology checklist requires a Bicubic merge-tree
visualization for Figure 2, sample 34. Bicubic has no saved `dataSR.npy`, so
the field was reconstructed from CNN's canonical low-resolution input using
the project's frozen Bicubic convention.

This generated source is used only for the manual merge-tree geometry panel.
It does not modify the frozen metric evaluation. Bicubic PD and MT distances
therefore remain displayed as `N/A`.

## Reconstruction

- Source: `data_out_fixed/wind_mrhr_cnn/dataIN.npy`
- Target-shape reference:
  `data_out_fixed/wind_mrhr_cnn/dataGT.npy`
- Sample index: 34
- Input shape: `100 × 100 × 2`
- Reconstructed shape: `500 × 500 × 2`
- Interpolation:
  - applied independently to the u and v channels
  - `scipy.ndimage.zoom`
  - `order=3`
  - `mode='reflect'`
  - `prefilter=True`
- SciPy version: 1.17.1

## Scalar VTI

- Scalar: `wind_speed = sqrt(u² + v²)`
- Patch: `160 × 160`
- Patch origin: `x0=0, y0=0`
- Observed scalar range: `[0.1760, 23.9726]`
- Converter: `scripts/convert_phire_to_vti.py`
- VTK version: 9.6.0
- Temporary reconstruction and VTI artifacts were removed after extraction.

## TTK extraction

The source was processed with:

```text
ttkMergeTreeCmd -t 20 -i <bicubic-vti> -a wind_speed -o <output-prefix>
