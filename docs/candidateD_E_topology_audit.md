# Candidate D/Dpd and E/E2 Topology-Alignment Audit

**Date:** 2026-07-05
**Branch:** `claude/audit-phire-wind-magnitude-sfqdw`
**Scope:** Code-level audit of Candidate D/Dpd (differentiable PD-Wasserstein
residual refiner) and Candidate E/E2 (TTK critical-pair residual refiner),
requested to explain why neither improved the final TTK PD/MT metrics.

---

## 0. Environment constraints on this audit

This session's sandbox has **no numpy, no matplotlib, no vtk, no torch, no
running Docker daemon**, and the repo's own `.mamba_candidateD_pd` env is
built for a different CPU architecture (`Exec format error`). It also has
**no trained checkpoints, logs, or result CSVs** for Candidate D, Dpd, E, or
E2 anywhere in this checkout — only the pilot infrastructure (scripts +
design notes) and one small, non-representative constraints CSV
(`ttk_runs_fixed/topology_finetuning/candidateE_constraints/ttk_pd_critical_pairs.csv`,
5 samples, likely a stale dev artifact predating the real 168-sample E-pilot
run — see §4).

Consequently:

- **Phase 1 (code audit)** is complete and high-confidence — verified by
  direct reading of the training/loss/vertex-extraction code, cross-checked
  against `PhIREGANs.py`, `sr_network.py`, and `convert_phire_to_vti.py`, and
  in one case (§2.3) verified with a standalone reproducible simulation.
- **Phase 2/3 (diagnostic metrics, visual overlays)** could not be executed
  against real trained outputs, because none exist in this environment. What
  is delivered instead: (a) diagnostics run against the one real (if
  non-representative) CSV that does exist, and (b) ready-to-run scripts that
  compute everything Phase 2/3 asked for, meant to be run on the machine
  that holds the actual checkpoints/data (referred to as "Spark" in the
  repo's own notes).

---

## 1. Where each candidate actually lives

| Candidate | Training code | Architecture | Loss weights |
|---|---|---|---|
| B / C / UV | `sr_network.py` (`PhysicsLossConfig`, `_build_aux_losses`, `_critical_value_loss`) inside the TF1 `PhIREGANs` graph | **Fine-tunes the CNN generator itself** | `lambda_speed=0.01, lambda_grad=0.05, lambda_levelset=0.25, lambda_crit=0.001` |
| D (pilot + expanded672) | `scripts/run_candidateD_pd_refiner.py`, `scripts/run_candidateD_expanded672_pd_refiner.py` | PyTorch `RefinerNet`, **post-processes the frozen pretrained CNN's output** | Same numeric values as C, `lambda_pd=0.0` (monitoring only) |
| Dpd (expanded672) | `scripts/run_candidateDpd_expanded672_pd_refiner.py` | Same `RefinerNet` | Same as D, `lambda_pd=0.001904` (genuine) |
| E (pilot) | `scripts/run_candidateE_ttkcrit_refiner.py` + `scripts/extract_ttk_pd_critical_pairs.py` | Same `RefinerNet` | C's weights + `lambda_ttkcv=0.0, lambda_ttkpers=0.0` (calibrate-only) |
| E2 (expanded672) | `scripts/run_candidateE2_expanded672_ttkcrit_refiner.py` + `scripts/build_candidateE2_expanded672_ttk_constraints.py` | Same `RefinerNet` | C's weights + `lambda_ttkcv=0.04, lambda_ttkpers=0.02` |

**This is the first thing worth flagging on its own**: Candidates B/C/UV
fine-tune the actual CNN generator inside TF1. Candidates D/Dpd/E/E2 do **not**
touch the CNN at all — they freeze it and train a small separate PyTorch
residual network on top:

```python
# scripts/run_candidateDpd_expanded672_pd_refiner.py, run_candidateE2_expanded672_ttkcrit_refiner.py
class RefinerNet(nn.Module):
    """SR_D = SR_CNN + residual_scale * body(SR_CNN); body's last conv is
    zero-initialized so SR_D == SR_CNN at t=0."""
```

This is a legitimate design choice (documented, low-risk, avoids re-deriving
TF1 gradients), but it means D/Dpd/E/E2 results are not directly comparable
in *mechanism* to B/C/UV — they are testing "how much can a small
post-hoc residual correction improve topology," not "how much can
fine-tuning the generator improve topology." Keep this distinction in mind
when interpreting why D/Dpd/E/E2 might underperform C: they have far less
representational freedom to begin with (a small, zero-initialized residual
head vs. the full CNN).

---

## 2. Phase 1 findings, ranked by likely impact

### 2.1 — CONFIRMED, HIGH IMPACT: `L_uv` is computed in physical (denormalized)
units in every PyTorch RefinerNet candidate, not normalized units

**User's stated design intent:** *"L_uv should compare normalized predicted
u,v to normalized GT u,v. Scalar topology losses should denormalize u,v
first."*

**What the TF1 candidates (B/C/UV) actually do** (`sr_network.py`):
the TF1 graph operates on **normalized** `x_HR`/`x_SR` end-to-end;
`content_loss = tf.reduce_mean((x_HR - x_SR)**2, ...)` is computed on these
normalized tensors, and `_denorm_speed(x_HR, mu_tf, sig_tf)` is called
*specifically* to produce physical-unit speed for the auxiliary
(`L_speed`, `L_grad`, `L_crit`, `L_levelset`) terms. This matches the
user's intended design exactly.

**What the PyTorch RefinerNet candidates (D/Dpd/E/E2) actually do**:

```python
# PhIREGANs.py, test_paired() — this is what produced dataSR.npy / dataGT.npy
batch_LR_phys = self.mu_sig[1]*batch_LR + self.mu_sig[0]
batch_HR_phys = self.mu_sig[1]*batch_HR + self.mu_sig[0]
batch_SR_phys = self.mu_sig[1]*batch_SR + self.mu_sig[0]
...
np.save(self.data_out_path + '/dataSR.npy', all_SR)   # already denormalized
np.save(self.data_out_path + '/dataGT.npy', all_HR)   # already denormalized
```

```python
# run_candidateDpd_expanded672_pd_refiner.py / run_candidateE2_expanded672_ttkcrit_refiner.py
def load_data(data_dir):
    sr = _to_chw(np.load(data_dir / 'dataSR.npy'))   # physical units, no renormalization
    gt = _to_chw(np.load(data_dir / 'dataGT.npy'))   # physical units, no renormalization
    ...

def l_uv(sr, gt):
    return F.mse_loss(sr, gt)     # <- MSE on RAW PHYSICAL u,v, not normalized

def _speed_t(uv):
    return torch.sqrt(uv[:, 0]**2 + uv[:, 1]**2 + _EPS)   # also physical, this part is fine
```

`dataSR.npy`/`dataGT.npy` are **already denormalized** by `PhIREGANs.test_paired()`
before being written to disk (confirmed above, `PhIREGANs.py` lines 482–509).
There is **no `mu_sig` renormalization anywhere** in
`run_candidateDpd_expanded672_pd_refiner.py` or
`run_candidateE2_expanded672_ttkcrit_refiner.py` — `mu_sig=[[0.7684, -0.4575],
[5.02455, 5.9017]]` appears **only inside a docstring/comment** showing the
command used to *generate* the expanded672 arrays, never in the training loop
itself. So `l_uv` in every RefinerNet candidate is MSE on raw m/s-scale u,v,
while `_speed_t`/`l_speed`/`l_grad`/`l_crit` are also on raw m/s-scale (which
is correct for those terms, since they are supposed to operate on physical
speed — but `l_uv` should not be).

**Why this matters:** in the TF1 design, `L_uv` is intentionally kept at an
`O(1)` normalized scale while the auxiliary physical-scale losses are
down-weighted by small lambdas (`0.01`, `0.05`, `0.001`) so that, after
weighting, they land in roughly the same order of magnitude as `L_uv`. In the
PyTorch RefinerNet candidates, `L_uv` was moved into the *same* large
physical-unit scale as the auxiliary terms (since `sig ≈ 5–6`, raw physical
MSE is roughly `sig² ≈ 25–36×` larger than normalized-space MSE for
comparable relative error) — but the same small lambdas from Candidate C were
reused unchanged. The practical effect: relative to `L_uv`'s new, larger raw
scale, the auxiliary topology terms (`L_crit`, `L_PD`, `L_ttkcv`,
`L_ttkpers`) contribute a **smaller fraction of the total gradient than they
did in the TF1 candidates they were copied from** — the residual refiner is
pulled disproportionately hard toward simply reproducing the frozen CNN's
output, at the expense of the topology-oriented terms.

This is compounded by a second issue: the `lambda_pd = 0.001904` calibration
(documented in `docs/candidateD_pd_refiner_notes.md`) was derived from a
diagnostic run explicitly marked `*(synthetic data — real data not
found)*` — i.e. the "10% of L_uv" target was calibrated on **randomly
generated placeholder arrays**, not real physical wind fields, so even its
own stated calibration is unverified on real data.

**Recommendation:** before re-running Dpd/E2, renormalize `sr`/`gt` with the
same `mu_sig` used to generate the arrays before computing `l_uv`, and
re-derive `lambda_pd`/`lambda_ttkcv`/`lambda_ttkpers` from a diagnostic pass
on **real** (not synthetic) physical fields.

---

### 2.2 — CONFIRMED, HIGH IMPACT: Candidate Dpd trains a *superlevel* PD loss
against a pipeline that evaluates *sublevel* PD

```python
# run_candidateDpd_expanded672_pd_refiner.py
def l_pd(sr, gt, cubical, w_loss_fn, crop=PD_CROP_SIZE):
    """PD Wasserstein-2 loss on GT-normalized scalar speed (superlevel-set)."""
    ...
# and, in the actual training setup (not just the smoke test):
cubical = CubicalComplex(superlevel=True)
```

This was previously established (see the prior TTK-filtration audit in this
conversation, backed by `ttkMergeTreeCmd`'s own `--help` output:
`[-T <Tree type {0: JT, 1: ST} (default: 0)>]`, JT = Join Tree = sublevel,
and no such flag is ever passed) that the **final evaluation pipeline**
(`ttkPersistenceDiagramCmd`, invoked with no filtration flag anywhere in
`scripts/run_candidate_topology_pipeline.sh`) computes persistence diagrams
under TTK's **default sublevel-set convention**.

Candidate Dpd's training loss, by contrast, explicitly requests
`CubicalComplex(superlevel=True)` — i.e. it computes and matches
**superlevel**-set persistence diagrams during training. Sublevel and
superlevel persistence diagrams of the same scalar field are generally
**different objects** (dual filtrations pair different critical points
together). Training to minimize Wasserstein distance between *superlevel*
diagrams does not directly minimize the *sublevel* bottleneck distance that
is actually reported in Tables 2–4 of the paper. This is a direct
train/eval objective mismatch, and a strong candidate explanation for why
genuine `L_PD` training (Dpd) failed to reliably improve the final TTK PD
metric even though the loss is a "real" differentiable PD-Wasserstein loss
per se.

**Recommendation:** re-run the L_PD training with `CubicalComplex(superlevel=False)`
(the library's sublevel default) so the training-time PD convention matches
the evaluation-time convention, before drawing further conclusions about
whether differentiable PD-Wasserstein training helps or hurts.

---

### 2.3 — CONFIRMED, HIGH IMPACT (likely the primary answer to the user's
central question): a coordinate transpose bug in the VTI writer silently
corrupts Candidate E/E2's fixed-index critical-value supervision

This is the most consequential and least obvious finding, so it is presented
with full derivation and a standalone reproducible proof (§2.3.4).

#### 2.3.1 — The bug, in the VTI writer

```python
# scripts/convert_phire_to_vti.py
H, W = scalar_2d.shape  # rows, cols
img = vtk.vtkImageData()
img.SetDimensions(W, H, 1)          # VTK point id = ix + iy*W  (x fastest)
img.SetExtent(0, W - 1, 0, H - 1, 0, 0)
...
# VTK expects data ordered with x varying fastest; ravel(order="F") is reliable here.
flat = np.ascontiguousarray(scalar_2d).ravel(order="F")
```

`ravel(order="F")` on a `(H, W)` array varies the **first axis (row/H)
fastest** — i.e. it produces `flat[i] = scalar_2d[i % H, i // H]`. But
`SetDimensions(W, H, 1)` tells VTK that the flat buffer varies **x (the
first declared dimension, W) fastest**, i.e. VTK will read
`img(ix, iy) = flat[ix + iy*W]`.

For the square 160×160 patches used everywhere in this pipeline (H=W), these
two conventions combine to give:

```
img(ix, iy) = scalar_2d[ix, iy]      (as actually written)
```

instead of the intended

```
img(ix, iy) = scalar_2d[iy, ix]      (x=column, y=row — the comment's own stated intent)
```

**This is a transpose.** The comment on line 97 ("Uses x=width (cols),
y=height (rows)") states the intended convention; the `ravel(order="F")` +
`SetDimensions(W,H,1)` combination does not implement it — it implements the
transposed field instead.

#### 2.3.2 — Why this does *not* invalidate the paper's published PD/MT numbers

A transpose of a square grid is a graph automorphism (it maps the 4- or
8-connected pixel adjacency structure onto itself). Persistence diagrams and
merge trees are computed purely from the topology of sublevel/superlevel sets
and depend only on the **multiset of scalar values and their connectivity**,
not on which axis is labeled "x" vs "y". Because the *same* transpose is
applied identically to every GT and every SR/CNN/GAN VTI (same script, same
call pattern, always square 160×160 patches), the bottleneck and merge-tree
**distances** reported by `compute_composite_tree_distance.py` — and hence
Tables 2–4 of the paper — are unaffected. This bug is invisible at the level
of aggregate distance metrics.

#### 2.3.3 — Why this *does* corrupt Candidate E/E2's spatial supervision

`extract_ttk_pd_critical_pairs.py` and
`build_candidateE2_expanded672_ttk_constraints.py` decode TTK's flat vertex
ID back into `(y, x)` using:

```python
# extract_ttk_pd_critical_pairs.py, lines 485-486, 703-704
birth_y, birth_x = bvid // W, bvid % W

# build_candidateE2_expanded672_ttk_constraints.py, lines 18-20, 296-303
# vid // PATCH = iy (first numpy index); vid % PATCH = ix (second numpy index)
# corrected_val = gt_speed[iy, ix]
b_iy, b_ix = bvids // W, bvids % W
corrected_bval = gt_speed_crop[b_iy, b_ix]
```

Given TTK's vertex id `vid = ix_vtk + iy_vtk*W` (VTK's own x-fastest
convention, which both scripts' comments correctly describe), this decoding
recovers `iy = iy_vtk`, `ix = ix_vtk`. But §2.3.1 established that the VTI's
actual scalar value at VTK point `(ix_vtk, iy_vtk)` equals
`scalar_2d[ix_vtk, iy_vtk]` — **not** `scalar_2d[iy_vtk, ix_vtk]`. So
`gt_speed[iy, ix] = gt_speed[iy_vtk, ix_vtk]` reads the **diagonally
transposed pixel** relative to the one TTK actually evaluated, for every
critical point that is not exactly on the main diagonal of the patch.

The "corrected value" step (`corrected_bval = gt_speed_crop[b_iy, b_ix]`) is
numerically self-consistent — it always produces *some* valid float, and the
downstream `_sanity_check()` in `build_candidateE2_expanded672_ttk_constraints.py`
(lines 320–388) **recomputes the exact same expression** (`gt_crop[b_iy, b_ix]`)
and compares it to the stored value, so it always passes. This sanity check is
tautological — it checks that the code is internally consistent with itself,
not that `(b_iy, b_ix)` is the true spatial location of the critical point
TTK identified. It cannot detect this bug.

**Practical consequence:** Candidate E/E2's `L_ttkcv` and `L_ttkpers` losses
select a spatial pixel `(iy_vtk, ix_vtk)` in the numpy array that is, in
general, **not** where TTK found a real high-persistence critical point in
the correctly-oriented field — it is the mirror-image pixel across the
patch's diagonal. Since wind fields are not diagonally symmetric, this
scatters the "critical-value" supervision across essentially arbitrary,
mostly non-critical pixel locations. This directly undermines the entire
premise of Candidate E/E2 (spatially precise persistence-relevant
supervision) while leaving the loss numerically well-behaved (no NaNs, no
crashes, no failed abort-checks) — which is exactly the kind of bug that
would produce "trains fine, doesn't help the final metric" behavior with no
obvious symptom during training.

#### 2.3.4 — Standalone reproducible proof

Verified computationally (pure Python, no VTK/numpy dependency, using VTK's
documented `pointId = ix + iy*dimX` convention) in
`scripts/verify_vti_transpose_bug.py` (new file, added by this audit, not
touching any existing script). Running it reproduces:

```
img[px][py] == scalar_2d[px][py]   (TRANSPOSED vs. intended)?  True
img[px][py] == scalar_2d[py][px]   (CORRECT/intended)?          False
```

for an asymmetric 4×4 test field, confirming the derivation above without
requiring VTK, Docker, or any trained model output.

**Recommendation:** fix `convert_phire_to_vti.py` to use
`scalar_2d.ravel(order="C")` (the numpy default) instead of `order="F"` — this
alone makes the existing `img(ix,iy) = flat[ix+iy*W]` VTK convention agree
with the intended `scalar_2d[iy, ix]` mapping. Do **not** change the vertex
decoding formulas in `extract_ttk_pd_critical_pairs.py` /
`build_candidateE2_expanded672_ttk_constraints.py` — those already assume the
correct (non-transposed) convention; they only need the writer fixed to match
them. After fixing the writer, **re-run the extraction/constraint-building
scripts from scratch** (the existing NPZ/CSV constraint files were built from
transposed VTIs and must be regenerated, not reused). Note again that this
fix will **not** change any already-published bottleneck/MT distance number
(§2.3.2) — it only affects E/E2 training-time supervision.

---

### 2.4 — Confirmed correct / not a concern

- **Candidate B/C/UV (TF1) denormalization order**: correct. `L_uv` on
  normalized tensors, `L_speed`/`L_grad`/`L_crit`/`L_levelset` on
  `_denorm_speed()`-denormalized physical speed, exactly as intended.
- **Candidate C is entirely unaffected by the §2.3 transpose bug.**
  `_critical_value_loss()` in `sr_network.py` operates purely on in-graph TF
  tensors (3×3 max-pooling directly on `speed_hr`/`speed_sr`), with **no VTI
  round-trip and no TTK dependency at all** (`sr_network.py` contains zero
  references to vti/vtu/ttk). This is an important contrast: Candidate C's
  real, measured MT improvement (102/168 samples, per
  `docs/candidateD_pd_loss_feasibility.md`) is not suspect on this basis.
- **`L_ttkpers` sign/direction convention**: `persistence = |death − birth|`
  is stored and consumed unsigned throughout
  (`docs/candidateE_ttkcrit_refiner_notes.md`, "Persistence Sign Convention"),
  and both `L_ttkcv`/`L_ttkpers` are documented and implemented as
  direction-agnostic. This part of the design is sound *given* correct
  vertex coordinates — it is downstream of, and only matters once, §2.3 is
  fixed.
- **Candidate D pilot vs. Dpd `lambda_pd`**: confirmed exactly as the user
  described — `run_candidateD_expanded672_pd_refiner.py` uses
  `LAMBDA_PD = 0.0` (monitoring only, "pilot-faithful"),
  `run_candidateDpd_expanded672_pd_refiner.py` uses `LAMBDA_PD = 0.001904`
  (genuine training term), confirmed by direct diff of the two scripts'
  constants and by `docs/candidateDpd_expanded672_notes.md`.
- **Crop alignment**: the 160×160 top-left crop (`x0=0, y0=0`) selects the
  same *set* of pixels regardless of the §2.3 transpose (a diagonal
  reflection of a symmetric, corner-anchored square region maps the region
  onto itself) — crop alignment is not an additional, separate bug on top of
  §2.3.

---

## 3. Diagnostic scripts added by this audit

Two new, standalone scripts were added (no existing file was modified):

1. **`scripts/verify_vti_transpose_bug.py`** — pure-Python (stdlib only,
   no numpy/vtk needed) regression test that reproduces the §2.3 proof.
   Safe to run anywhere; exits non-zero if the bug is present, so it can be
   wired into CI once the writer is fixed.

2. **`scripts/diagnose_candidate_topology_alignment.py`** — the Phase 2/3
   diagnostic tool requested by the user. Designed to run on the machine
   that actually holds the trained checkpoints and `phase_c_results.csv`
   files (numpy + optionally matplotlib required). Given any subset of
   `{CNN, C, Dpd, E2}` data directories that exist, it computes:
   - TTK PD/MT means, count(PD<CNN), count(MT<CNN) per candidate, by reading
     each candidate's `phase_c_results.csv` (schema-compatible with
     `compute_composite_tree_distance.py`'s output) if present;
   - `L_crit` (Candidate C-style) and `L_ttkcv`/`L_ttkpers` (Candidate
     E2-style) proxy losses recomputed directly from `dataSR.npy`/`dataGT.npy`
     plus the E2 constraints NPZ, if present;
   - the requested "does lowering the proxy loss correlate with lowering the
     final TTK distance" scatter/correlation check;
   - overlay plots (GT speed field + Candidate C maxima + E2 birth/death
     vertices) for any sample index, if matplotlib is available.

   This session's sandbox cannot execute it against real data (§0): there
   are no `dataSR.npy`/`dataGT.npy` arrays for any candidate in this
   checkout, and the one existing constraints file
   (`ttk_pd_critical_pairs.csv`, §4) is a flat per-pair CSV, not the
   per-sample NPZ schema (`sample_idx`/`sample_start`/`sample_count`/...)
   the script's E2 loader expects, so it could not be run against that file
   either. Instead, `numpy` was installed in this sandbox and every core
   function (`speed`, `local_max_mask`, `l_crit_per_sample`,
   `e2_proxy_per_sample`, `pearson`) was smoke-tested directly against
   fabricated synthetic arrays of the correct shapes/schemas and confirmed
   to execute without error and return sane values. This confirms the
   script is mechanically correct and ready to run; it does **not** confirm
   anything about real Candidate C/Dpd/E2 behavior — that requires running
   it where the real checkpoints and `phase_c_results.csv` files live.

Both scripts live under `scripts/` and do not modify or delete anything
under `data_out*/`, `ttk_runs*/`, `models_fixed/`, or any existing script.

---

## 4. What could actually be run in this environment (Phase 2, partial)

The only real (non-synthetic) topology-adjacent data present in this
checkout is
`ttk_runs_fixed/topology_finetuning/candidateE_constraints/ttk_pd_critical_pairs.csv`
(513 lines). Inspecting it directly:

- It contains **5 unique samples** (`sample_idx` 0–4), not the 168 the real
  Candidate E pilot requires (`run_candidateE_ttkcrit_refiner.py` aborts with
  `NPZ has fewer than 168 samples` otherwise) — this is almost certainly a
  leftover dev/smoke-test artifact from before the pilot's real run, not the
  actual E-pilot constraint set.
- `sample_name` values (`gan_GT_s0...`) show it was built from a `gan_GT`
  labeled VTI, i.e. from the shared GT field via the baseline `gan`/`cnn`
  topology run, not from a `candidateD_GT`-labeled run as the later E-pilot
  notes describe.
- Sample 0 has 256 stored pairs; samples 1–4 each have exactly 64 (`top_k`),
  suggesting sample 0 predates a top-k cap being applied consistently.

Given this file is not representative of a real 168-sample run, and its
flat per-pair schema does not match the per-sample NPZ schema the new
diagnostic script's E2 loader expects, no diagnostic conclusions are drawn
from it and it was not used as a script input. It is retained exactly as-is
(not overwritten, per instructions). `diagnose_candidate_topology_alignment.py`
was instead smoke-tested against fabricated synthetic arrays of the correct
shapes/schemas (see §3, item 2) to confirm it runs end-to-end.

---

## 5. Phase 4 — Proposed next experiments (not launched)

In priority order:

1. **Fix §2.3 (VTI transpose) and §2.1 (L_uv normalization) first, and
   re-run E2 and Dpd from scratch with otherwise-identical hyperparameters
   before trying any new ablation.** Until these two bugs are fixed, any
   ablation built on top of the existing E2/Dpd pipeline is testing a
   confounded system — a negative result could mean "the idea doesn't work"
   or "the idea was never actually implemented as intended." This is the
   single highest-value next step and should happen before any of the
   ablations below.
2. **Re-run Dpd with `CubicalComplex(superlevel=False)`** (§2.2) to align
   training-time and evaluation-time filtration convention. Cheapest
   possible test of the superlevel/sublevel-mismatch hypothesis (one-line
   change).
3. Once §2.1/§2.2/§2.3 are fixed, re-run E2 with **`L_TTK-CV` only** and
   **`L_TTK-pers` only** (ablating the two terms independently) to see which
   (if either) drives improvement once vertex coordinates are correct.
4. **Only high-persistence pairs**: restrict E2's supervised vertex set to
   pairs above a higher persistence threshold (current `persistence_frac=0.01`
   is quite permissive) — reduces the chance of supervising noisy,
   low-persistence critical points even after the coordinate fix.
5. **Match predicted extrema to GT extrema within a radius** instead of
   fixed-index supervision — this is a more robust design in general (fixes
   the class of bug found in §2.3 by construction, since it no longer
   depends on any external coordinate system agreeing exactly with the numpy
   array's indexing), and is worth adopting regardless of whether §2.3 is
   patched, as defense in depth.
6. **Lower `lambda_ttkcv`/`lambda_ttkpers` weights**: only worth revisiting
   after §2.1 is fixed, since the current weights were calibrated under a
   different (physical-unit) `L_uv` scale than intended.

---

## 6. Answers to the four specific questions

**Was Dpd likely hurt by Wasserstein-training vs. bottleneck-evaluation
mismatch?**
Yes, plausibly, and for two independent reasons found in this audit: (a) it
trains against a *superlevel* PD Wasserstein loss while evaluation computes
*sublevel* PD bottleneck distance (§2.2) — these are different objects; and
(b) its `L_uv` anchor term is computed in physical rather than normalized
units (§2.1), so the calibrated `lambda_pd` weight is not landing at its
intended relative strength. Either alone could suppress the benefit of
genuine PD-Wasserstein training; together they make it unsurprising that
Dpd's real-data behavior would diverge from the (synthetic-data) diagnostic
that motivated it.

**Was E/E2 likely hurt by fixed-index critical-value supervision?**
Yes — and more specifically, it was very likely hurt by a **coordinate bug
in that fixed-index supervision**, not by the fixed-index design itself
being conceptually flawed. §2.3 shows the actual pixel locations used for
`L_ttkcv`/`L_ttkpers` are diagonally transposed relative to where TTK truly
found the critical points, for any non-diagonal pixel. This would make E2's
topology-oriented supervision largely decorrelated from real persistence
structure, which is consistent with training-time proxy-loss improvements
failing to translate into final TTK PD/MT improvement — exactly the pattern
the user asked to explicitly test for.

**Were there any implementation bugs or convention mismatches?**
Yes, three, described above: (1) `L_uv` computed in physical rather than
normalized units in all four PyTorch RefinerNet candidates (D/Dpd/E/E2);
(2) Dpd's PD loss uses the superlevel convention against a sublevel
evaluation pipeline; (3) a transpose bug in `convert_phire_to_vti.py`'s VTI
writer that corrupts E/E2's vertex-to-pixel coordinate mapping (but does
**not** affect any already-published bottleneck/MT distance number).

**What is the most promising next experiment?**
Fix §2.3 and §2.1, then re-run Candidate E2 (TTK-CV + TTK-pers) from scratch
with correct coordinates and correctly-scaled losses before drawing any
further conclusion about whether TTK-guided fixed-index supervision can
work. This is a low-effort, high-information experiment: the E2 pipeline
infrastructure, Docker constraint-builder, and refiner training script are
already complete and validated in every other respect (abort-checks, shape
validation, NaN guards) — the fix is localized to one `ravel()` call and one
missing renormalization step, not a redesign.
