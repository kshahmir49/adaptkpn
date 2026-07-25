# Method summary

The proposed method is a zero-shot denoising framework based on range-gated adaptive kernel prediction.

Given a noisy image `y`, the model generates noisy views and trains an image-specific denoiser without using a clean target. During inference, the optimized model is applied to the original noisy image.

## Adaptive filtering stage

At stage `t`, the current estimate is `u^(t-1)`. A lightweight kernel predictor estimates a learned spatial kernel `S^(t)` for each local neighborhood.

An intensity-based range gate computes:

```math
R^{(t)}(p,q) =
\exp\left(
-\frac{\|u^{(t-1)}(p)-u^{(t-1)}(q)\|_2^2}
{2(\sigma_r^{(t)})^2}
\right).
```

The final local kernel is:

```math
K^{(t)}(p,q)=
\frac{S^{(t)}(p,q)R^{(t)}(p,q)}
{\sum_{q'\in\mathcal{N}(p)}S^{(t)}(p,q')R^{(t)}(p,q')}.
```

The filtered estimate is:

```math
\tilde{u}^{(t)}(p)=
\sum_{q\in\mathcal{N}(p)}K^{(t)}(p,q)u^{(t-1)}(q).
```

A residual smoothing update is then applied:

```math
u^{(t)}=(1-\eta)u^{(t-1)}+\eta\tilde{u}^{(t)}.
```

The spatial kernel is learned by the CNN. The range kernel is computed from intensity similarity. The range bandwidth is trainable in the provided implementation.
