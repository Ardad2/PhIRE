# Candidate D documentation update

## Purpose

Candidate D tested whether a true differentiable persistence-diagram (PD) loss could improve topology-aware super-resolution beyond Candidates B and C.

The key question was:

> If we add a differentiable PD-based training signal to a CNN-like output, does the post-hoc TTK PD/MT evaluation improve?

## Setup

Candidate D was implemented as a PyTorch residual refiner on top of the frozen pretrained CNN output:

```text
CNN SR [u,v] -> small residual refiner -> Candidate D [u,v]
```

Scalar wind speed was computed from the refined vector field:

```text
s = sqrt(u^2 + v^2)
```

The loss was:

```text
L_D = L_uv + lambda_speed L_speed + lambda_grad L_grad + lambda_crit L_crit + lambda_PD L_PD
```

Candidate D used:

```text
lambda_PD = 0.2
PD crop size = 100 x 100
training = 3 epochs
learning rate = 1e-4
```

The real-data diagnostic gave:

```text
L_PD / L_uv ≈ 0.46
```

so `lambda_PD = 0.2` made the weighted PD term roughly a conservative 10% contribution relative to `L_uv`.

## Why PyTorch PD loss was used instead of TTK bottleneck distance

TTK bottleneck distance is excellent for evaluation, but it is not directly usable as a training loss because it returns a scalar distance after external TTK/ParaView computation and does not expose gradients with respect to the scalar field or model weights.

Candidate D therefore used a PyTorch cubical-complex / Wasserstein-style PD loss so that gradients could flow through the scalar speed tensor and update the residual refiner.

## Topology results

Candidate D was evaluated with the same TTK PD/MT pipeline as CNN, GAN, Candidate B, and Candidate C.

| Metric | CNN | GAN | Candidate B | Candidate C | Candidate D |
|---|---:|---:|---:|---:|---:|
| PD distance mean | 27.4063 | 20.8641 | 26.1794 | 27.0021 | 27.9510 |
| MT distance mean | 5.8678 | 8.3481 | 6.0186 | 5.7141 | 5.9837 |
| PD lower than CNN | - | 166/168 | 136/168 | 120/168 | 1/168 |
| MT lower than CNN | - | 20/168 | 66/168 | 102/168 | 50/168 |
| Original MT-GAN cases recovered | - | - | 4/20 | 11/20 | 1/20 |

Candidate D recovered only one original MT-GAN case: sample 25.

## Interpretation

Candidate D is a negative but useful result.

It shows that adding a differentiable PD loss is technically feasible, but it does not automatically improve the TTK PD or MT metrics. The likely reasons include:

- training used a PyTorch Wasserstein-style PD loss, while evaluation used TTK bottleneck distance;
- training used a 100 x 100 PD crop, while TTK evaluation used 160 x 160 patches;
- the PD weight was conservative;
- the residual refiner may have been too limited to substantially change topology;
- PD and MT reward different structural properties.

The main conclusion is:

> Differentiable PD supervision is feasible, but the training objective must be better aligned with the TTK topology evaluator.

## Comparison to Kissi et al. (2025)

After inspecting the Kissi et al. code, their practical differentiable topology loss appears to work differently from simply backpropagating through a TTK Wasserstein distance.

Their code uses TTK/ParaView to:

1. read a precomputed persistence diagram `.vtu`,
2. extract persistence-pair birth and death vertices,
3. map those vertices back to scalar-field indices using `ttkVertexScalarField`,
4. store the target scalar values at those critical vertices.

The differentiable training loss is then a PyTorch loss on the neural output values at those fixed critical-point positions.

So TTK identifies the important topological vertices, while PyTorch provides the gradient by indexing into the output tensor.

This is closer to Candidate C than Candidate D:

```text
Candidate C:
  detects high-speed local maxima and penalizes SR error at those locations

Kissi-style critical-point loss:
  uses actual TTK persistence-pair birth/death vertices and penalizes SR error there

Candidate D:
  uses a PyTorch PD Wasserstein-style loss, but it was not well aligned with TTK PD/MT evaluation
```

## Next proposed candidate

The next topology-aware experiment should use a Kissi-style TTK-derived critical-pair loss:

```text
Candidate E = Candidate C + TTK persistence-pair critical-value loss
```

Proposed loss:

```text
L_TTK-CV = mean_p w_p * (s_SR(p) - s_GT(p))^2
```

where `p` are birth/death vertices from high-persistence GT persistence pairs, and `w_p` can weight pairs by persistence.

This is more principled than Candidate C's local-maxima heuristic because the points come from the actual TTK persistence diagram. It is still differentiable because the critical locations are fixed during training and the loss is just indexed scalar-value error.

## Suggested visual inspection update

The visual inspection page should include Candidate D with columns:

```text
GT | CNN | Candidate B | Candidate C | Candidate D | GAN
|CNN-GT| | |B-GT| | |C-GT| | |D-GT| | |GAN-GT|
|D-B| | |D-C|
```

The key visual question is whether Candidate D makes visible structural changes or remains very close to Candidate B/C while failing to improve TTK PD/MT.
