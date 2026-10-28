# Week 7 — Part 1: Diffusion Training and VERIFY (1)–(3)

This specification covers **Part 1 only**: the Gaussian-mixture forward process,
the time-conditioned noise predictor, DDPM reverse sampling, and VERIFY (1),
(2), and (3). Part 2 and later high-dimensional diffusion work are out of scope.

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

`alpha_bar` is stored as an array of length `T + 1`, indexed directly by the
integer diffusion time `t`.

## Model and training configuration

The exact configuration is in `diffusion.toml`:

- Hidden layers `[128, 128, 128]`, SiLU activation.
- 32 sinusoidal time features with offset `-1`.
- Normal weight initialization with standard deviation `1/sqrt(input_width)`.
- Adam with learning rate `0.001` and cosine decay to zero.
- Batch size `512`, `25000` updates, seed `2026`, `float32`.
- Checkpoints at
  `[0, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 25000]`.

The time features are 16 sine and 16 cosine values at fixed log-spaced
frequencies. Time is normalized by `T` after applying the offset, so `t=1`
maps to normalized time zero. The model input is `[x_t, time_features]`, and
the output is the predicted noise for `x_t`.

Training minimizes the mean squared error between predicted and true
forward-diffusion noise on batches with uniformly sampled diffusion levels
`t in {1, ..., T}`. The noise-free forward sample is generated in closed form.

## DDPM reverse sampling

Reverse sampling follows DDPM Algorithm 2:

```text
x_(t-1) = 1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_theta(x_t, t))
          + sqrt(beta_t) * z
```

The reverse-step variance is `beta_t`. The final step `t=1` adds no noise, and
no clipping is applied.

## VERIFY (1): forward diffusion

1. Generate at least 2000 samples from the mixture.
2. Render a forward-diffusion animation with fixed axis limits.
3. At `t = 1000`, print the mean and variance of both coordinates.
4. Verify that each mean is within `0.1` of `0` and each variance is within
   `0.1` of `1`.

## VERIFY (2): training and checkpoint trajectories

1. Train the noise predictor to every checkpoint in `diffusion.toml`.
2. Save the parameter checkpoint and the raw loss history for every checkpoint.
3. Generate 10 DDPM trajectories at every checkpoint.
4. Reuse exactly the same initial Gaussian samples and reverse-step noise at
   every checkpoint.
5. Save an animation showing how the generated endpoints evolve during
   training, with the training-loss history and a label for every checkpoint.
6. Save a figure showing the 10 reverse trajectories at every checkpoint.

## VERIFY (3): trained-model sampling

1. Load the saved update-25000 checkpoint. Do not retrain.
2. Generate 2000 samples using the trained noise predictor and 1000 DDPM
   reverse steps, with an explicit reproducible seed.
3. Save an animation of the reverse process from Gaussian noise to the four
   generated clusters.
4. Plot the final generated samples beside 2000 exact target samples drawn from
   the Gaussian mixture.
5. Assign each generated sample to its nearest target center.
6. For every cluster, compute:
   - population fraction, target `0.25` with tolerance `0.04`;
   - estimated center, with each coordinate within `0.05` of the target
     center;
   - standard deviation in x and y, target `0.30` with tolerance `0.03`.
7. Print a PASS/FAIL table covering every criterion, save the figures and
   animation in `evidence/`, and save the raw samples, trajectory, and metrics
   in `artifacts/`.

## Deliverables

- Reusable JAX implementation and pytest tests.
- Reproducible entry points using CPU JAX and explicit seeds.
- `diffusion.toml` with the page-3 training configuration.
- Evidence in `week7/evidence/`:
  - VERIFY (1) animation, keyframes, and log;
  - VERIFY (2) endpoint-evolution animation, checkpoint grid, training-loss
    figure, and log;
  - VERIFY (3) reverse-diffusion animation, reverse keyframes,
    generated-vs-target figure, and PASS/FAIL log.
- Raw checkpoints, losses, trajectories, samples, and numerical results in
  `week7/artifacts/` (ignored by Git).

## Out of scope

- Part 2 and later tasks.
- CIFAR-10 diffusion or any high-dimensional image training.
- Use or modification of the provided `week7/data/` resources.

The provided CIFAR-10 resources are left untouched and are reserved for later
parts of the exercise.
