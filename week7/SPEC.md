# Week 7 — Part 1: Forward Diffusion

This specification covers **Part 1 only**. Training a denoising network and
reverse sampling are out of scope and must not be implemented in this part.

## Data distribution

Use a two-dimensional four-component Gaussian mixture:

| Component | Center | Weight | Covariance |
|-----------|--------|-------:|------------|
| 0 | `( 3,  0)` | `1/4` | `0.09 I` |
| 1 | `( 0,  3)` | `1/4` | `0.09 I` |
| 2 | `(-3,  0)` | `1/4` | `0.09 I` |
| 3 | `( 0, -3)` | `1/4` | `0.09 I` |

Sampling must be reproducible through an explicit JAX PRNG key.

## Forward diffusion

The forward process uses `T = 1000` steps and a linear variance schedule

```text
beta_t = 0.0001 + (t - 1) * (0.02 - 0.0001) / (T - 1),   t = 1, ..., T
alpha_t = 1 - beta_t
alpha_bar_0 = 1
alpha_bar_t = product_{s=1..t} alpha_s,   t = 1, ..., T
q(x_t | x_0) = N(sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)
```

`alpha_bar` is therefore stored as an array of length `T + 1`, indexed directly
by the integer diffusion time `t`.

The implementation must expose the schedule and the reparameterized
closed-form forward sample in a reusable Python module.

## VERIFY (1): forward diffusion

1. Generate at least 2000 samples from the mixture.
2. Render a forward-diffusion animation with **fixed axis limits** on both
   coordinates.
3. At `t = 1000`, print the mean and variance of both coordinates.
4. Verify that each mean is within `0.1` of `0` and each variance is within
   `0.1` of `1`.

## Deliverables

- Reusable JAX implementation and pytest tests.
- Reproducible experiment entry point using CPU JAX and an explicit seed.
- Evidence in `week7/evidence/`:
  - forward-diffusion animation,
  - keyframe figure,
  - printed VERIFY (1) log.
- Raw numerical results in `week7/artifacts/` (ignored by Git).

## Out of scope

- Neural-network training.
- Reverse diffusion / sampling.
- Use or modification of the provided `week7/data/` resources.

The provided CIFAR-10 resources are left untouched and are reserved for later
parts of the exercise.
