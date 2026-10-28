# Week 7 — Parts 1–5 (through Part 5 VERIFY (1))

This specification covers **Part 1**, **Part 2 VERIFY (1)**, **Part 3 VERIFY
(1)–(2)**, **Part 4 VERIFY (1)**, and **Part 5 VERIFY (1)**: the Gaussian-mixture
forward process, the time-conditioned noise predictor, DDPM reverse sampling,
VERIFY (1)–(3), deterministic DDIM sampling, Flow Matching, the sampling-step
comparison, pretrained CIFAR-10 DDPM sampling, and RePaint. Part 6 and later
work are out of scope.

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

## Part 2 VERIFY (1): deterministic DDIM vs DDPM

1. Reuse the trained update-25000 noise predictor. Do not retrain.
2. Implement deterministic DDIM (eta = 0):

   ```text
   x0_pred = (x_t - sqrt(1 - alpha_bar_t) * eps_theta(x_t, t)) / sqrt(alpha_bar_t)
   x_prev = sqrt(alpha_bar_prev) * x0_pred
            + sqrt(1 - alpha_bar_prev) * eps_theta(x_t, t)
   ```

   The final transition to `t=0` is implicit. Support any strictly decreasing
   subset of noise levels in `1, ..., 1000`.
3. Implement the exact conditional-mean noise predictor
   `E[epsilon | x_t]` for the equal-weight Gaussian mixture, using the
   posterior responsibilities of the four components.
4. Run DDIM over 1000 levels twice from the same starting Gaussian noise.
5. Run DDPM twice from the same starting Gaussian noise with two different
   reverse-step noise draws.
6. Plot the DDIM trajectory and the two DDPM trajectories for a shared sample.
7. Generate 2000 DDIM samples and 2000 DDPM samples, and compare them with 2000
   exact target samples.
8. Print the largest difference between the two DDIM runs.
9. Report whether DDIM cluster fractions are within `0.015` of `0.25`.
10. Report whether DDIM and DDPM widths are between `0.29` and `0.32`. A
    cluster width is `sqrt((std_x^2 + std_y^2) / 2)`.
11. Save figures in `evidence/` and raw samples/metrics in `artifacts/`.

## Part 3 VERIFY (1): Flow Matching velocity fields

1. Create `flow.toml` with the page-11 settings:
   - hidden layers `[128, 128, 128]`, SiLU;
   - 32 sinusoidal time features, time scale `1000`;
   - Adam with learning rate `0.001` and cosine decay to zero;
   - batch size `512`, `25000` updates, seed `2029`, `float32`.
2. Train a new velocity network from scratch:
   - sample `x0` from the four-component Gaussian mixture;
   - sample `z ~ N(0, I)` and `s ~ Uniform[0, 1]`;
   - construct `x_s = (1-s) x0 + s z`;
   - target velocity `u = z - x0`;
   - minimize mean squared velocity prediction error.
3. Implement Euler sampling backward from `s=1` to `s=0`.
4. Implement the exact analytical marginal velocity
   `E[z - x0 | x_s]` for the Gaussian mixture.
5. For `s = 0.1, 0.5, 0.9`, plot learned and exact velocity fields on exactly
   the same spatial grid. The grid is component-adapted: each mixture
   component contributes a `7 x 7` local grid around its forward-marginal mean
   `(1-s) * center`, with extent `±2` marginal standard deviations.
6. Compute the relative RMS field difference
   `||v_learned - v_exact||_RMS / ||v_exact||_RMS` at each `s`, and verify all
   three values are below `0.1`.
7. Print a PASS/FAIL table. Save velocity-field and training-loss figures in
   `evidence/`; save checkpoints, loss history, field grids, and raw metrics in
   `artifacts/`.

## Part 3 VERIFY (2): sampling-step comparison

1. Reuse the trained diffusion and Flow Matching checkpoints. Do not retrain.
2. Compare DDPM, DDIM, and Flow Matching at sampling steps `1, 2, 5, 10, 20,
   50, 100`, plus `1000` steps for DDPM/DDIM.
3. Use evenly spaced time/noise levels and 4000 generated samples per run.
4. For coarse DDPM jumps from `t` to `t_prev`, use Equation (17):

   ```text
   x0_pred = (x_t - sqrt(1 - alpha_bar_t) * eps_theta(x_t, t)) / sqrt(alpha_bar_t)
   jump_beta = 1 - alpha_bar_t / alpha_bar_prev
   mean = sqrt(alpha_bar_prev) * jump_beta / (1 - alpha_bar_t) * x0_pred
          + sqrt(alpha_bar_t / alpha_bar_prev)
            * (1 - alpha_bar_prev) / (1 - alpha_bar_t) * x_t
   jump_variance = (1 - alpha_bar_prev) / (1 - alpha_bar_t) * jump_beta
   ```

   The final jump to `t_prev=0` has zero variance.
5. For every run, compute cluster populations, centers, x/y standard
   deviations, isotropic cluster width `sqrt((std_x^2 + std_y^2)/2)`, and
   energy distance against 4000 exact target samples.
6. Estimate the energy-distance null scale from 20 independent pairs of exact
   sample sets (4000 samples each), including the 95% quantile.
7. Plot energy distance with the null-scale band, maximum cluster population
   error, and maximum cluster width error versus sampling steps.
8. Verify Flow Matching at 20 steps has energy distance `< 0.005` and maximum
   width error `< 0.03`.
9. Include exact-predictor DDIM at 1 and 2 steps to demonstrate coarse-grid
   limitations.
10. Save the figure in `evidence/`, raw metrics/samples in `artifacts/`, and
    print a PASS/FAIL table.

## Part 4 VERIFY (1): pretrained CIFAR-10 DDPM

1. Download the frozen checkpoint with the pinned command:

   ```sh
   uv run --with huggingface-hub==1.33.0 hf download google/ddpm-cifar10-32 \
     config.json diffusion_pytorch_model.safetensors \
     --revision 267b167dc01f0e4e61923ea244e8b988f84deb80 \
     --local-dir inputs/cifar-model
   ```

   The weights must not be committed to Git.
2. Implement the diffusers v0.40.0 `UNet2DModel` in JAX, load the 328
   safetensors tensors, and validate every expected parameter shape. Do not
   retrain.
3. Generate four RGB `32x32` images with 1000-step DDPM sampling:
   - float32;
   - pass model timestep `t-1` for integer reverse step `t`;
   - no clipping during sampling;
   - map to display with `clip((x+1)/2, 0, 1)` only.
4. Visualize reverse diffusion at several noise levels, show all four final
   images, and compare with real CIFAR-10 images using the Week 6
   nearest-neighbour gallery conventions/script.
5. Document what is pretrained (frozen U-Net weights) versus computed during
   sampling (initial noise, reverse updates, final samples, display mapping).
6. Check the final image shape, finite outputs, display range, and visual
   recognizability. Save figures in `evidence/`, raw samples/results in
   `artifacts/`, and print a PASS/FAIL table.

## Part 5 VERIFY (1): RePaint inpainting

1. Load the four CIFAR-10 test images from `inputs/images.npz` (the verified
   Week 7 image bundle).
2. Implement RePaint Algorithm 6:
   - hide rows and columns `6:26` in all RGB channels;
   - use 50 reverse jumps `1000, 980, ..., 20, 0`;
   - use two attempts per jump except the final jump, for exactly 99 network
     evaluations;
   - forward-noise between attempts and replace known pixels with
     forward-noised observations;
   - clip only the predicted clean image `x0` to `[-1,1]`.
3. Generate four completions for each of the four images using seeds `7201`,
   `7202`, `7203`, `7204`.
4. Show original, masked image, four completions, and pixelwise variance.
5. Report hidden-pixel MSE for every completion, verify maximum observed-pixel
   change `< 1e-5`, and confirm completions differ inside the masked region.
6. Save figures in `evidence/`, raw results in `artifacts/`, and print a
   PASS/FAIL table.

## Deliverables

- Reusable JAX implementation and pytest tests.
- Reproducible entry points using CPU JAX and explicit seeds.
- `diffusion.toml` with the page-3 training configuration.
- Evidence in `week7/evidence/`:
  - VERIFY (1) animation, keyframes, and log;
  - VERIFY (2) endpoint-evolution animation, checkpoint grid, training-loss
    figure, and log;
  - VERIFY (3) reverse-diffusion animation, reverse keyframes,
    generated-vs-target figure, and PASS/FAIL log;
  - Part 2 trajectory comparison, DDIM/DDPM/target sample comparison, exact
    predictor comparison, and PASS/FAIL log;
  - Part 3 learned/exact velocity-field figure, training-loss figure, and
    PASS/FAIL log;
  - Part 3 VERIFY (2) sampling-step metrics figure and PASS/FAIL log;
  - Part 4 reverse-diffusion, final-sample, generated-vs-real, and Week 6 real
    CIFAR gallery figures plus PASS/FAIL log;
  - Part 5 RePaint completion/variance figure and PASS/FAIL log.
- Raw checkpoints, losses, trajectories, samples, velocity grids, sampling
  metrics, CIFAR samples, RePaint results, and numerical results in
  `week7/artifacts/` (ignored by Git).
- Pinned CIFAR model config in `inputs/cifar-model/`; weights remain ignored.

## Out of scope

- Part 6 and later tasks.
- CIFAR-10 model training or fine-tuning.
- Modification of the provided `week7/data/` resources (they are read-only
  comparison inputs).
