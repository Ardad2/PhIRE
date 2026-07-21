# Topology-aware fine-tuning audit

This document records pre-implementation findings for the topology/physics-aware
CNN fine-tuning experiment. Each section is added as new decisions are confirmed.

---

## Scalar vs vector model alignment

**Audit date:** 2026-05-12  
**Auditor:** claude/audit-phire-wind-magnitude-sfqdw  

### Question

The repo contains two distinct model families:

| Family | Checkpoint path | Channels | Normalization |
|---|---|---|---|
| Vector [u,v] | `models/wind_mr-hr/trained_cnn/cnn` | c=2 | per-channel mu/sigma |
| Scalar speed | `models_fixed/wind_speed_mr-hr/trained_cnn/cnn/cnn` | c=1 | scalar mu/sigma |

Which family produced the paper's fixed outputs (`data_out_fixed/wind_mrhr_cnn/`,
`data_out_fixed/wind_mrhr_gan/`)?

### Evidence

#### 1. Inference scripts are definitive

`run_paired_wind_mr_hr_cnn_fixed.py` (the script that generated `data_out_fixed/wind_mrhr_cnn/`):

```python
data_path  = 'example_data_fixed/wind_MR-HR.tfrecord'   # vector TFRecord
model_path = 'models/wind_mr-hr/trained_cnn/cnn'          # VECTOR model
data_out_path = 'data_out_fixed/wind_mrhr_cnn'
mu_sig = [[0.7684, -0.4575], [5.02455, 5.9017]]           # per-channel [u, v]
```

`run_paired_wind_mr_hr_gan_fixed.py` is identical except `model_path = 'models/wind_mr-hr/trained_gan/gan'`.

The scalar-speed script (`run_paired_wind_speed_mr_hr_cnn_fixed.py`) writes to
`data_out_fixed/wind_speed_mrhr_cnn/` — a **different directory** not used by the
paper evaluation pipeline.

#### 2. `generate_baseline_visual_panels.py` confirms c=2

The visualization script contains:

```python
def _speed(uv: np.ndarray) -> np.ndarray:
    """Compute wind speed magnitude from (H, W, 2) [u, v] array."""
    return np.sqrt(uv[..., 0] ** 2 + uv[..., 1] ** 2).astype(np.float32)
```

It explicitly converts [u, v] → speed before rendering, confirming the npy files
are two-channel.

#### 3. `docs/wind_representation_audit.md` confirms c=2

> "TFRecords for wind are documented as `[ua, va]` channels."
> "The current topology pipeline explicitly converts arrays to speed (`sqrt(u^2+v^2)`) before VTI/TTK."

#### 4. `analysis_fixed/vector_scalar_selected_samples/summary.md` shows both were run

A separate scalar-CNN experiment was also run for comparison and produced
**lower speed MAE** than the vector CNN (`0.6154` vs `0.7148`). This makes physical
sense: the scalar model is trained end-to-end on scalar speed, not on [u,v]
components. However, the scalar outputs live in `data_out_fixed/wind_speed_mrhr_cnn/`
and were **not** used in the paper's topology evaluation.

#### 5. `data_out_fixed/` is absent on this machine

The npy files exist only on Spark (`/home/adadhwal/PhIRE/`). Inference
re-verification from checkpoint was therefore not run on this machine.
The script-level evidence above is conclusive without it.

### Conclusion

| Output directory | Model | Channels | mu | sigma |
|---|---|---|---|---|
| `data_out_fixed/wind_mrhr_cnn/` | `models/wind_mr-hr/trained_cnn/cnn` | **c=2 [u,v]** | [0.7684, −0.4575] | [5.02455, 5.9017] |
| `data_out_fixed/wind_mrhr_gan/` | `models/wind_mr-hr/trained_gan/gan` | **c=2 [u,v]** | [0.7684, −0.4575] | [5.02455, 5.9017] |
| `data_out_fixed/wind_speed_mrhr_cnn/` | `models_fixed/wind_speed_mr-hr/trained_cnn/cnn/cnn` | c=1 (scalar) | ~9.628 | ~4.296 |

**The paper evaluation uses the vector [u,v] model.** All topology (PD/MT), physics,
PSNR, and SSIM results in the paper are computed on scalar speed derived from
the vector outputs.

### Implication for fine-tuning

The fine-tuning experiment must target the **vector model**:

```
Checkpoint: models/wind_mr-hr/trained_cnn/cnn
Training data: example_data_fixed/wind_MR-HR.tfrecord  (c=2 [u,v])
mu_sig: [[0.7684, -0.4575], [5.02455, 5.9017]]
```

This has the following consequences:

1. **Loss operates on normalized [u,v] tensors** — `x_SR` shape `[batch, H, W, 2]`.
   Physical speed requires denormalization of both channels before squaring and summing.

2. **Denormalization for speed-based losses:**
   ```python
   mu  = tf.constant([0.7684, -0.4575], dtype=tf.float32)   # shape [2]
   sig = tf.constant([5.02455, 5.9017], dtype=tf.float32)    # shape [2]
   u_phys = sig[0] * x_SR[..., 0] + mu[0]
   v_phys = sig[1] * x_SR[..., 1] + mu[1]
   speed  = tf.sqrt(u_phys**2 + v_phys**2 + 1e-8)
   ```

3. **Physical thresholds (5, 10, 15 m/s)** cannot be applied as a simple cut on
   either channel — they must be applied to the derived `speed` tensor above.

4. **Scalar-speed-only metrics (PSNRspeed) are consistent** — both GT and SR are
   converted identically, so physics losses defined on speed are meaningful and
   comparable to existing paper metrics.

5. **PSNRuv (direct component metric)** is also computable. However, since the
   paper's primary evaluation is speed-based (consistent with topology), speed-based
   physics losses are the correct target for producing paper-consistent results.

6. **The scalar model (`models_fixed/wind_speed_mr-hr/`) is not the correct baseline.**
   Fine-tuning it would produce results incomparable to the existing CNN/GAN paper outputs.
   It may still be useful as a sanity reference experiment, but is not the primary target.

### Checkpoint epoch count

The vector CNN checkpoint file listing shows the model was trained for **≥ 94 epochs**
(`all_model_checkpoint_paths` in `checkpoint` file runs through `cnn00094`). The
final checkpoint is `models/wind_mr-hr/trained_cnn/cnn` (no epoch suffix).

The vector GAN was trained for **≥ 25 epochs** of adversarial fine-tuning after CNN
pretraining.
