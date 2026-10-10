# Week 5 — Seeing underground: differentiable acoustic wave propagation

This week builds a small seismic imaging stack from scratch and verifies every
derivative with Enzyme (Rust) and JAX (Python).

1. **Part 1 — autodiff fundamentals.** Manual node-by-node forward and reverse
   AD of the Lennard-Jones energy with JAX, JAX computational graphs, and a
   forward-versus-reverse scaling benchmark.
2. **Part 2 — forward solver.** A damped 2-D acoustic wave equation in Rust,
   with the numerical timestep isolated so Enzyme can differentiate it.
3. **Part 3 — Born and adjoint.** Enzyme-generated timestep JVPs for Born
   scattered data `d = J m`, and timestep VJPs for the full-history adjoint
   `Jᵀw`, verified by the transpose identity `<Jm, Jm> = <m, JᵀJm>`.
4. **Part 4 — Treeverse checkpointing.** A Rust port of
   [GiggleLiu/TreeverseAlgorithm.jl](https://github.com/GiggleLiu/TreeverseAlgorithm.jl)
   that trades forward recomputation for saved memory without changing the RTM
   image, plus a checkpointed Marmousi migration.

## Repository layout

```
week5/
  README.md                 this file
  seismic.design.toml       CLI/output contract for every mode
  inputs/                   official experiment JSON (download, not committed)
  data/                     official supporting data and MARMOUSI-LICENSE
  checker/                  official numerical gate (run: checker/check week5)
  sim.py, viewer/           official reference simulator and viewer
  seismic/                  Rust crate (solver + isolated Enzyme kernel)
  scripts/                  Python plotting and verification scripts
  artifacts/                generated evidence (gitignored; small files committed)
```

## Required software and dependencies

- **Python** 3.14 with `jax`, `numpy`, `matplotlib` in `week5/.venv`
  (validated: jax 0.11.2, numpy 2.5.3, matplotlib 3.11.2).
- **Rust nightly-2026-09-05 with the `enzyme` component.** This is pinned by
  `week5/seismic/rust-toolchain.toml`, so `cargo` in `week5/seismic/` uses it
  automatically. The rest of the repository is unaffected.
- **Official Week 5 input archive.** The experiment JSON files are course
  material and are **not committed**. Download the archive and place
  `reflector.json` and `marmousi.json` under `week5/inputs/`, and
  `MARMOUSI-LICENSE` under `week5/data/`. Without those files the commands
  below cannot run.

## Part 1 — reproduce the JAX results

```sh
week5/.venv/bin/python week5/scripts/lj_ad.py              # artifacts/ad/derivatives.json
week5/.venv/bin/python week5/scripts/compare_ad_modes.py   # artifacts/ad/modes.png
week5/.venv/bin/python week5/scripts/ad_graphs.py          # artifacts/ad/graph.png, grad-graph.png
week5/.venv/bin/python week5/scripts/ad_scaling.py         # artifacts/ad/scaling.png
```

`lj_ad.py` differentiates `U(r) = 4 (r⁻¹² − r⁻⁶)` at `r = 1.3` node by node
and checks the tangents, adjoints and `jax.grad` against the analytic
derivative. `ad_scaling.py` benchmarks forward vs reverse AD for Lennard-Jones
clusters; on this CPU-only box reverse cost grows once the `N²` cotangents
exceed cache, which the script documents while still passing the required
accuracy and the >100× forward/reverse separation.

## Part 2 — build and run the Rust seismic solver

```sh
cd week5/seismic
cargo build --release
cargo test --release          # 17 unit tests
```

The crate links a `no_std` kernel (`src/kernel.rs`) compiled by `build.rs`
with `-Zautodiff=Enable`; the crate itself never sees the autodiff pass. The
kernel exposes a primal timestep plus its Enzyme JVP and VJP through a C ABI.
Forward run of the reflector experiment:

```sh
week5/seismic/target/release/seismic \
  --experiment week5/inputs/reflector.json \
  --mode forward --every 3 \
  --out week5/artifacts/forward
```

Then generate and verify the Part 2 figures:

```sh
week5/.venv/bin/python week5/scripts/plot_inputs.py     # artifacts/inputs.png
week5/.venv/bin/python week5/scripts/verify_forward.py  # checks traces.npy
week5/.venv/bin/python week5/scripts/plot_gathers.py    # artifacts/forward/gathers.png
week5/.venv/bin/python week5/scripts/plot_snapshots.py  # wavefield.png, echo.png
```

## Part 3 — Born and full-history adjoint

```sh
# Born scattered data d = J m (Enzyme JVPs)
week5/seismic/target/release/seismic \
  --experiment week5/inputs/reflector.json --mode born \
  --out week5/artifacts/born

# Full-history adjoint Jᵀw with w = born_data (Enzyme VJPs)
week5/seismic/target/release/seismic \
  --experiment week5/inputs/reflector.json --mode adjoint \
  --data week5/artifacts/born/born_data.npy --every 3 \
  --out week5/artifacts/adjoint

week5/.venv/bin/python week5/scripts/plot_rtm_image.py        # adjoint/image.png
week5/.venv/bin/python week5/scripts/plot_adjoint_wavefield.py # adjoint/wavefield.png
```

The transpose identity is checked automatically inside `checker/check` using
the stored `born_data.npy` and `image.npy`.

## Part 4 — Treeverse checkpointing

```sh
for b in 1 3 5 10; do
  week5/seismic/target/release/seismic \
    --experiment week5/inputs/reflector.json --mode adjoint \
    --data week5/artifacts/born/born_data.npy \
    --storage treeverse --checkpoints "$b" \
    --out "week5/artifacts/checkpoint-$b"
done

week5/.venv/bin/python week5/scripts/verify_treeverse.py        # image errors + audits
week5/.venv/bin/python week5/scripts/plot_checkpoint_actions.py # checkpoint-actions.png
week5/.venv/bin/python week5/scripts/plot_checkpoint_work.py    # checkpoint-work.png
```

`--storage full` (the default) keeps every state; `--storage treeverse
--checkpoints δ` keeps at most `δ + 1` complete states and replays the rest.
The budget-1 schedule is by far the most expensive (28 680 forward steps per
shot); budget 10 needs only 642.

### Marmousi (large experiment)

```sh
# Born d = J m  (9 shots, 1200 steps, written as [9, 1200, 91] float64)
week5/seismic/target/release/seismic \
  --experiment week5/inputs/marmousi.json --mode born \
  --out week5/artifacts/marmousi-born

# Checkpointed RTM image — ALWAYS treeverse, never full history
week5/seismic/target/release/seismic \
  --experiment week5/inputs/marmousi.json --mode adjoint \
  --data week5/artifacts/marmousi-born/born_data.npy \
  --storage treeverse --checkpoints 5 \
  --out week5/artifacts/marmousi-image

week5/.venv/bin/python week5/scripts/plot_marmousi.py   # artifacts/marmousi.png
```

Full-history Marmousi would need `1201` states (≈4.16 GB); budget 5 needs 6
states (20 788 320 bytes), which is why it is the required configuration.

## Full verification

```sh
week5/.venv/bin/python week5/checker/check week5
cd week5/seismic && cargo test --release
```

`checker/check` runs the Part 1 AD checks, the forward trace checks, the Born
norm and transpose identity, the four checkpoint image/action checks and the
Marmousi norm and storage checks. The `scripts/verify_*.py` files re-check the
same data independently.

### Evidence inventory

| Part | Evidence | Generator |
| --- | --- | --- |
| 1 | `artifacts/ad/derivatives.json` | `scripts/lj_ad.py` |
| 1 | `artifacts/ad/modes.png` | `scripts/compare_ad_modes.py` |
| 1 | `artifacts/ad/graph.png`, `grad-graph.png` | `scripts/ad_graphs.py` |
| 1 | `artifacts/ad/scaling.png` | `scripts/ad_scaling.py` |
| 2 | `artifacts/inputs.png` | `scripts/plot_inputs.py` |
| 2 | `artifacts/forward/{run,result}.json`, `gathers.png` | Rust forward + `plot_gathers.py` |
| 2 | `artifacts/forward/{wavefield,echo}.png` | Rust forward + `plot_snapshots.py` |
| 3 | `artifacts/born/{run,result}.json` | Rust `--mode born` |
| 3 | `artifacts/adjoint/{run,result}.json`, `image.png` | Rust `--mode adjoint` + `plot_rtm_image.py` |
| 3 | `artifacts/adjoint/wavefield.png` | `plot_adjoint_wavefield.py` |
| 4 | `artifacts/checkpoint-actions.png` | `plot_checkpoint_actions.py` |
| 4 | `artifacts/checkpoint-work.png` | `plot_checkpoint_work.py` |
| 4 | `artifacts/checkpoint-{1,3,5,10}/{run,result}.json` | Rust treeverse adjoint |
| 4 | `artifacts/checkpoint-{1,3,5,10}/actions-*.json` | Rust treeverse adjoint |
| 4 | `artifacts/marmousi-born/{run,result}.json` | Rust Born |
| 4 | `artifacts/marmousi-image/{run,result}.json`, `actions-*.json` | Rust treeverse adjoint |
| 4 | `artifacts/marmousi.png` | `plot_marmousi.py` |

The `.npy` arrays (`traces`, `born_data`, `image`, `wavefield`, …) stay local
and are deliberately not committed.

## Numerical verification targets

| Check | Measured | Expected | Tolerance |
| --- | --- | --- | --- |
| AD energy / tangents / adjoints / `jax_grad` | exact | analytic | < 1e-12 |
| Forward trace L2 | 11.574769504 | 11.574770 | rel < 1e-4 |
| Forward first arrival | step 83 | step 83 | exact |
| Born squared L2 | 0.0348478902152 | 0.0348479 | ~1e-15 |
| Transpose identity `<Jm,Jm> = <m,JᵀJm>` | 3.98e-16 | 0 | rel < 1e-9 |
| Reflector RTM profile peak | z = 21 (2.1 km) | z = 21 | ±1 cell |
| Treeverse image vs full history, δ = 1/3/5/10 | 0 | 0 | rel < 1e-9 |
| Treeverse forward calls/shot, δ = 1/3/5/10 | 28680 / 1695 / 990 / 642 | same | exact |
| Treeverse peak states, δ = 1/3/5/10 | 2 / 4 / 6 / 11 | same | exact |
| Treeverse audit errors | 0 / 0 / 0 | 0 | exact |
| Marmousi image L2 | 6.703774060378e-4 | 6.7037741e-4 | rel < 1e-4 |
| Marmousi peak states / bytes | 6 / 20 788 320 | same | exact |

## The memory-versus-recomputation trade-off

The forward simulation is a chain `s_{n+1} = H_n(s_n)`. Reverse-mode AD needs
every `s_n`, but storing all N+1 states is expensive. Treeverse keeps only
`δ + 1` complete states (the initial `s_0` is never freed) and recomputes the
others by replaying forward steps from the nearest checkpoint. The repetition
number `τ` is the smallest integer with `binomial(τ+δ, τ) ≥ N`, and a recursive
split rule distributes the checkpoints. More budget means fewer replays: for
the reflector, budget 1 replays 28 680 steps per shot while budget 10 replays
642; for Marmousi, budget 5 replays 70 956 steps total but holds only 20.8 MB
instead of the ≈4.16 GB a full trajectory would need. This is a pure
time/space trade-off — the arithmetic performed by every VJP is identical.

## Why the RTM image differs from the true perturbation

The derivatives are exact (`<Jm,Jm>` matches `<m,JᵀJm>` to machine precision
and Treeverse reproduces the full-history image bit for bit), yet the image is
`JᵀJm`, not `m`. `J` is a band-limited, limited-aperture forward scattering
operator: it only "sees" the part of the perturbation that is illuminated by
the 9 shots and 91 surface receivers and that lies inside the 5 Hz source
bandwidth. `JᵀJ` is therefore a smoothing, band-limiting, geometry-dependent
filter — it preserves the dipping and curved reflection bands that are well
illuminated (especially in the upper ≈3 km) and attenuates the rest with
depth. The image also carries the wavelet's phase, producing the characteristic
opposite-sign side lobes around each reflector. Limited source–receiver
coverage, finite bandwidth, uneven illumination and the action of `JᵀJ` all
mean the migrated image can be numerically exact and still differ from the
original velocity perturbation.

## Implementation notes

- **Why a restart state holds two fields.** The damped wave update is
  second-order in time, so the next pressure depends on both `u_{n}` and
  `u_{n-1}`. A complete restart state is the pair `(u_previous, u_current)`;
  storing only one field would not allow the simulation to be restarted.
- **How one Enzyme timestep derivative is used.** The kernel defines the
  timestep once and `#[autodiff_forward]`/`#[autodiff_reverse]` generate a JVP
  and a VJP for it. Born composes the JVP forward in time with the velocity
  perturbation as the tangent; the adjoint composes the VJP backward in time,
  seeding the receiver weights into the newly computed pressure. The Rust loop
  only sequences these local derivatives; it never hand-differentiates the
  physics.
- **Why Treeverse replay does not change the image.** Replayed forward steps
  recreate `s_n` from a checkpoint with the same deterministic timestep. Each
  VJP receives the exact same primal state and writes into the same
  accumulators in the same descending order, so the velocity sensitivity is
  identical to full history; only *where the states come from* changes.
