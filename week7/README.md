# Week 7 — Part 1 and Part 2 VERIFY (1)

This week implements **Part 1** and **Part 2 VERIFY (1)** of the Week 7
diffusion exercise:

1. the four-component Gaussian mixture and its forward diffusion process,
2. a time-conditioned noise predictor trained with MSE,
3. DDPM reverse sampling,
4. the VERIFY (1), VERIFY (2), and VERIFY (3) studies, and
5. deterministic DDIM sampling with comparison to stochastic DDPM and the
   exact conditional-mean Gaussian-mixture predictor.

Part 3, high-dimensional CIFAR-10 diffusion, and other later work are not
included.

## Setup and commands

The environment is managed with `uv` and uses CPU JAX.

```sh
cd week7
uv sync
uv run pytest -q
uv run python run_forward_diffusion.py
uv run python run_verify2.py
uv run python run_verify3.py
uv run python run_part2_verify1.py
```

To regenerate all checked-in VERIFY logs exactly:

```sh
cd week7
set -o pipefail
JAX_PLATFORMS=cpu uv run python run_forward_diffusion.py 2>&1 | tee evidence/verify1.txt
JAX_PLATFORMS=cpu uv run python run_verify2.py 2>&1 | tee evidence/verify2.txt
JAX_PLATFORMS=cpu uv run python run_verify3.py 2>&1 | tee evidence/verify3.txt
JAX_PLATFORMS=cpu uv run python run_part2_verify1.py 2>&1 | tee evidence/part2_verify1.txt
```

The VERIFY (2) run is deterministic: training uses seed `2026`, and all
checkpoints reuse the same initial Gaussian samples and reverse-step noise.
VERIFY (3) does not retrain; it loads the update-25000 checkpoint and uses seed
`5315` for the 2000 initial Gaussian samples, reverse-step noise, and exact
target samples.

Part 2 VERIFY (1) also loads the update-25000 checkpoint without retraining. It
uses start seed `3`, DDPM reverse-noise seeds `1000` and `1003`, and 2000 samples
for each method.

## Distribution and forward process

- Centers: `(3, 0)`, `(0, 3)`, `(-3, 0)`, `(0, -3)`.
- Equal component weights, `1/4` each.
- Isotropic covariance `0.09 I`, i.e. component standard deviation `0.3`.
- `T = 1000` steps with a linear beta schedule from `0.0001` to `0.02`.
- `q(x_t | x_0) = N(sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)`.
- `alpha_bar` is indexed by `t = 0, ..., 1000`, with `alpha_bar_0 = 1`.

The forward samples are generated in closed form with a fixed noise draw per
sample, so each point follows a continuous path as the cloud spreads out.

## `diffusion.toml`

| Setting | Value |
|---------|-------|
| Hidden layers | `[128, 128, 128]` |
| Activation | SiLU |
| Time features | 32 sinusoidal features, offset `-1` |
| Weight init | normal, std `1/sqrt(input_width)` |
| Optimizer | Adam |
| Learning rate | `0.001`, cosine decay to `0` |
| Batch size | `512` |
| Updates | `25000` |
| Seed | `2026` |
| Dtype | `float32` |
| Checkpoints | `[0, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 25000]` |

The time input is normalized by `T` after applying the offset, so `t=1` maps
to normalized time zero. The 32 features are 16 sine and 16 cosine values at
fixed log-spaced frequencies. The MLP input is `[x_t, time_features]`
(`2 + 32 = 34` values), followed by three SiLU hidden layers of width 128 and a
2-value noise output. Adam uses the standard `beta1=0.9`, `beta2=0.999`, and
`epsilon=1e-8`.

## DDPM reverse sampling

Reverse sampling follows Algorithm 2 with variance `beta_t`:

```text
x_(t-1) = 1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_theta(x_t, t))
          + sqrt(beta_t) * z
```

There is no clipping. The additional noise is omitted at the final step
`t = 1`.

## VERIFY (1) result

Run settings: `2000` samples, seed `5315`, CPU (`cpu:0`), `T = 1000`.

| Quantity | Value | Required | Result |
|----------|------:|---------:|:------:|
| `beta_1` | `0.00010000` | linear start `0.0001` | pass |
| `beta_1000` | `0.02000000` | linear end `0.02` | pass |
| `alpha_bar_1000` | `4.03583e-05` | — | — |
| mean of x at `t=1000` | `+0.008245` | within `0.1` of `0` | pass |
| mean of y at `t=1000` | `-0.022044` | within `0.1` of `0` | pass |
| variance of x at `t=1000` | `1.047978` | within `0.1` of `1` | pass |
| variance of y at `t=1000` | `0.996061` | within `0.1` of `1` | pass |

**VERIFY (1): PASS.**

## VERIFY (2) result

Ten DDPM trajectories are generated at every checkpoint. The same initial
Gaussian `x_T` samples and the same reverse-step noise array are reused for all
checkpoints, so differences between frames come only from the trained weights.

| Update | Mean loss since previous checkpoint | Last-update loss |
|-------:|------------------------------------:|-----------------:|
| 0 | — (initial weights) | — |
| 50 | `0.666248` | `0.571829` |
| 100 | `0.496017` | `0.455348` |
| 200 | `0.362325` | `0.290499` |
| 500 | `0.291832` | `0.248902` |
| 1000 | `0.263296` | `0.302516` |
| 2000 | `0.257364` | `0.268763` |
| 5000 | `0.253851` | `0.250859` |
| 10000 | `0.251218` | `0.264171` |
| 25000 | `0.247490` | `0.253094` |

**Final training loss (last update): `0.253094`.**

**Final 50-update mean loss: `0.248167`.**

**VERIFY (2): COMPLETE.** All trajectories are finite. By update 25000 the ten
endpoints land near the four mixture centers.

## VERIFY (3) result

VERIFY (3) loads `artifacts/checkpoints/params-025000.npz` and does **not**
retrain. It generates 2000 samples with 1000 DDPM reverse steps (seed `5315`)
and assigns each generated sample to its nearest target center.

| Cluster | n | Population | Pop | Estimated center | max\|Δcenter\| | Center | std x | std y | max\|Δstd\| | Std | Overall |
|---:|---:|---:|:---:|---|---:|:---:|---:|---:|---:|:---:|:---:|
| 0 | 484 | 0.2420 | PASS | `(+2.9930, +0.0108)` | 0.0108 | PASS | 0.2908 | 0.3008 | 0.0092 | PASS | PASS |
| 1 | 507 | 0.2535 | PASS | `(+0.0176, +3.0086)` | 0.0176 | PASS | 0.2971 | 0.2776 | 0.0224 | PASS | PASS |
| 2 | 488 | 0.2440 | PASS | `(-2.9923, +0.0080)` | 0.0080 | PASS | 0.3011 | 0.2995 | 0.0011 | PASS | PASS |
| 3 | 521 | 0.2605 | PASS | `(-0.0031, -3.0021)` | 0.0031 | PASS | 0.2981 | 0.3118 | 0.0118 | PASS | PASS |

Criteria: population fraction `0.25 ± 0.04`, center coordinates `±0.05`, and
standard deviations `0.30 ± 0.03`.

**Overall VERIFY (3): PASS.**

## Part 2 VERIFY (1) result

Part 2 uses the same trained checkpoint, with deterministic DDIM (eta = 0)
over all 1000 noise levels. The DDIM update is

```text
x_prev = sqrt(alpha_bar_prev) * x0_pred
         + sqrt(1 - alpha_bar_prev) * eps_theta(x_t, t)
x0_pred = (x_t - sqrt(1 - alpha_bar_t) * eps_theta(x_t, t)) / sqrt(alpha_bar_t)
```

and the final transition to `t=0` is implicit. The sampler accepts any strictly
decreasing subset of levels in `1, ..., 1000`.

| Method | max \|fraction - 0.25\| | width range | Result |
|--------|----------------------------:|------------:|:------:|
| DDIM (learned predictor) | 0.0085 | `[0.2928, 0.3018]` | PASS |
| DDPM noise draw 1 | 0.0115 | `[0.2997, 0.3077]` | PASS |
| DDPM noise draw 2 | 0.0120 | `[0.2920, 0.2962]` | PASS |
| DDIM (exact conditional mean) | 0.0120 | `[0.2984, 0.3009]` | PASS |

- Largest difference between the two DDIM runs: **`0.0`**.
- DDIM fractions are within `0.015` of `0.25`: **PASS**.
- DDIM and DDPM widths are between `0.29` and `0.32`: **PASS** for DDIM and both
  DDPM noise draws. Width here is the per-cluster isotropic measure
  `sqrt((std_x^2 + std_y^2) / 2)`.
- Exact-predictor comparison on 20,000 forward samples: learned MSE `0.247183`,
  exact conditional-mean MSE `0.245741`, mean absolute predictor difference
  `0.015499`.

**Part 2 VERIFY (1): PASS.**

## Evidence

- `evidence/forward_diffusion.mp4` — VERIFY (1) forward animation.
- `evidence/forward_diffusion_keyframes.png` — VERIFY (1) snapshots.
- `evidence/verify1.txt` — VERIFY (1) printed log.
- `evidence/verify2_endpoint_evolution.mp4` — 10-frame animation of the ten
  generated endpoints at every checkpoint, with the training-loss history and
  checkpoint labels. Off-screen early endpoints are marked at the fixed axis
  boundary.
- `evidence/verify2_checkpoints.png` — all 10 reverse trajectories at every
  checkpoint, with fixed axes. `x` marks `x_T`; dots mark generated endpoints;
  off-screen endpoints are marked at the boundary.
- `evidence/training_loss.png` — full training-loss history with checkpoints.
- `evidence/verify2.txt` — VERIFY (2) printed numerical log.
- `evidence/reverse_diffusion.mp4` — 101-frame reverse-diffusion animation from
  Gaussian noise (`step = 0`) to the four generated clusters (`step = 1000`),
  with fixed axes and colors based on the final nearest-center assignment.
- `evidence/reverse_diffusion_keyframes.png` — reverse-process snapshots at
  steps `0, 200, 400, 600, 800, 1000`.
- `evidence/verify3_generated_vs_target.png` — final generated samples beside
  2000 exact Gaussian-mixture target samples.
- `evidence/verify3.txt` — VERIFY (3) numerical PASS/FAIL table.
- `evidence/part2_trajectory_comparison.png` — one shared-`x_T` sample's DDIM
  trajectory and the two stochastic DDPM trajectories.
- `evidence/part2_samples_comparison.png` — 2000 DDIM samples, 2000 DDPM
  samples, and 2000 exact target samples.
- `evidence/part2_exact_predictor_comparison.png` — DDIM with the learned
  predictor, DDIM with the exact conditional-mean predictor, and target
  samples.
- `evidence/part2_verify1.txt` — Part 2 numerical PASS/FAIL log.

Raw results (ignored by Git, regenerated by the command above):

- `artifacts/checkpoints/params-XXXXXX.npz` — one parameter checkpoint per
  listed update.
- `artifacts/verify2_results.json` — full settings, checkpoint losses, endpoint
  statistics, and evidence paths.
- `artifacts/verify2_losses.npz` — all 25,000 per-update losses.
- `artifacts/verify2_trajectories.npz` — all reverse trajectories, endpoints,
  shared `x_T`, and shared reverse-step noise.
- `artifacts/verify3_results.json` — VERIFY (3) settings, checkpoint path,
  metrics, assignment distances, and evidence paths.
- `artifacts/verify3_samples.npz` — generated samples, exact target samples,
  assigned labels/distances, shared `x_T`, and the full reverse trajectory.
- `artifacts/part2_results.json` — DDIM/DDPM/exact-predictor metrics, predictor
  MSE comparison, seeds, and verification flags.
- `artifacts/part2_samples.npz` — initial `x_T`, final DDIM/DDPM/exact samples,
  target samples, labels, and selected saved trajectories.

## Files

| File | Purpose |
|------|---------|
| `diffusion.toml` | Page-3 training configuration |
| `forward_diffusion.py` | Mixture, forward schedule, `q_sample`, VERIFY (1) |
| `diffusion_model.py` | Config loader, time features, MLP, Adam, training, DDPM sampler, checkpoint I/O |
| `diffusion_verify2_plot.py` | Checkpoint animation, checkpoint grid, loss figure |
| `verify3.py` | Nearest-center assignment, cluster metrics, PASS/FAIL table |
| `verify3_plot.py` | Reverse animation, reverse keyframes, generated-vs-target figure |
| `ddim.py` | Deterministic DDIM, level-subset validation, exact conditional-mean predictor |
| `part2_verify.py` | Part 2 fraction/width metrics and PASS/FAIL table |
| `part2_plot.py` | DDIM/DDPM trajectory and sample-comparison figures |
| `run_forward_diffusion.py` | VERIFY (1) entry point |
| `run_verify2.py` | Training and VERIFY (2) entry point |
| `run_verify3.py` | Checkpoint-only sampling and VERIFY (3) entry point |
| `run_part2_verify1.py` | DDIM vs DDPM Part 2 VERIFY (1) entry point |
| `tests/` | Forward-process, model, training, DDPM, DDIM, verification, and plotting tests |
| `SPEC.md` | Part 1 specification and out-of-scope items |

## Provided resources

`data/images.npz` and `data/sources.json` are the supplied Week 7 resources
(CIFAR-10 images with provenance). They are **not used or modified** in Part 1
and are reserved for later stages of the exercise. The local `.venv/` and
`artifacts/` are excluded from Git.
