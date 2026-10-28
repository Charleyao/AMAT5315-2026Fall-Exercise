# Week 7 — Part 1: Diffusion Training and VERIFY (2)

This week implements **Part 1** of the Week 7 diffusion exercise:

1. the four-component Gaussian mixture and its forward diffusion process,
2. a time-conditioned noise predictor trained with MSE,
3. DDPM reverse sampling, and
4. the VERIFY (2) checkpoint study.

No VERIFY (3), Part 2, high-dimensional CIFAR-10 diffusion, or other later work
is included.

## Setup and commands

The environment is managed with `uv` and uses CPU JAX.

```sh
cd week7
uv sync
uv run pytest -q
uv run python run_forward_diffusion.py
uv run python run_verify2.py
```

To regenerate both checked-in VERIFY logs exactly:

```sh
cd week7
set -o pipefail
JAX_PLATFORMS=cpu uv run python run_forward_diffusion.py 2>&1 | tee evidence/verify1.txt
JAX_PLATFORMS=cpu uv run python run_verify2.py 2>&1 | tee evidence/verify2.txt
```

The VERIFY (2) run is deterministic: training uses seed `2026`, and all
checkpoints reuse the same initial Gaussian samples and reverse-step noise.

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

Raw results (ignored by Git, regenerated by the command above):

- `artifacts/checkpoints/params-XXXXXX.npz` — one parameter checkpoint per
  listed update.
- `artifacts/verify2_results.json` — full settings, checkpoint losses, endpoint
  statistics, and evidence paths.
- `artifacts/verify2_losses.npz` — all 25,000 per-update losses.
- `artifacts/verify2_trajectories.npz` — all reverse trajectories, endpoints,
  shared `x_T`, and shared reverse-step noise.

## Files

| File | Purpose |
|------|---------|
| `diffusion.toml` | Page-3 training configuration |
| `forward_diffusion.py` | Mixture, forward schedule, `q_sample`, VERIFY (1) |
| `diffusion_model.py` | Config loader, time features, MLP, Adam, training, DDPM sampler, checkpoint I/O |
| `diffusion_verify2_plot.py` | Checkpoint animation, checkpoint grid, loss figure |
| `run_forward_diffusion.py` | VERIFY (1) entry point |
| `run_verify2.py` | Training and VERIFY (2) entry point |
| `tests/` | Forward-process, model, training, DDPM, and plotting tests |
| `SPEC.md` | Part 1 specification and out-of-scope items |

## Provided resources

`data/images.npz` and `data/sources.json` are the supplied Week 7 resources
(CIFAR-10 images with provenance). They are **not used or modified** in Part 1
and are reserved for later stages of the exercise. The local `.venv/` and
`artifacts/` are excluded from Git.
