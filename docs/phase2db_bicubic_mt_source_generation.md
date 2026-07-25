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
```

The command reported:

- tree type: Join
- scalar array: `wind_speed`
- thread count: 20

## Structural readback

### Port 0 — merge-tree nodes

- VTK type: `vtkUnstructuredGrid`
- Points: 332
- Cells: 332
- Scalar range: `[0.1760354489, 23.9726219177]`
- Includes `NodeId`, `Scalar`, `VertexId`, `CriticalType`,
  `RegionSize`, and `RegionSpan`.

### Port 1 — merge-tree arcs

- VTK type: `vtkUnstructuredGrid`
- Points: 332
- Cells: 331
- Scalar range: `[0.1760354489, 23.9726219177]`
- Includes `SegmentationId`, `upNodeId`, `downNodeId`,
  `RegionSize`, and `RegionSpan`.

### Port 2 — segmentation field

- VTK type: `vtkImageData`
- Points: 25,600
- Cells: 25,281
- Includes `wind_speed`, `SegmentationId`, `RegionSize`,
  `RegionSpan`, and `RegionType`.

## Generated artifacts

| Artifact | SHA-256 |
|---|---|
| `ttk_runs_fixed/bicubic/mt/SR/bicubic_SR_s34_speed_p160_x0_y0_mt_port_0.vtu` | `fcaab80a52a536cd69de2bb24bb7ce15418011c8ed7e3d1553e916a443df9cc7` |
| `ttk_runs_fixed/bicubic/mt/SR/bicubic_SR_s34_speed_p160_x0_y0_mt_port_1.vtu` | `5bd6f198e41ba31cad169e93c9a435169addb6fe66dcda921ff7cbb331e35119` |
| `ttk_runs_fixed/bicubic/mt/SR/bicubic_SR_s34_speed_p160_x0_y0_mt_port_2.vti` | `5dbecf19b0742cd981179a3970d82dc54bb2cd797d13f2d494097a617ec7f5e1` |

Port 1 is expected to provide the displayed arc geometry and port 0 the node
geometry. The exact displayed sources will be confirmed in ParaView and
recorded in the Figure 2 manual-panel metadata.
