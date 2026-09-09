# Lennard-Jones Fluid CLI — Design (md crate, Part 4)

- **Date:** 2026-09-09
- **Status:** Approved
- **Scope:** `week2/md` (Rust crate `md`)

## Goal

Extend the `md` crate into a CLI tool with three subcommands — `md run`,
`md check`, `md video` — that simulates a 2D Lennard-Jones fluid
(N = 100, ρ = 0.8, target T = 0.5, shifted potential with rc = 2.5,
Velocity Verlet, dt = 0.01, 2000 equilibration + 10000 production steps,
200 saved frames), writes `run.json` + `traj.jsonl`, re-analyzes raw frames in
`check`, and renders an MP4 in `video`. All Part 1/2/3 code and tests are
preserved and must stay green.

## Current state (context)

`week2/md/src/lib.rs` already contains:

- `greeting` (Part 1) and its test.
- `energy(r)` / `force(r)` (Part 2) and their tests.
- `System` / `Integrator` / `ForwardEuler` / `VelocityVerlet` /
  `run_dimer_experiment` (Part 3) and the dimer test.
- 4 passing tests; `Cargo.toml` has no dependencies.

## Approach (approved)

Hybrid dependency strategy:

- Add `serde` + `serde_json` (JSON read/write) and `rand` (seeded Gaussian
  velocities).
- Hand-roll the flat CLI parsing and the video frames (PPM) + an `ffmpeg`
  call. No `clap`, no image/plotting crate.

## Module layout

```
src/lib.rs          greeting, energy, force (unchanged); + RC const;
                    + energy_shifted/force_cut wrappers;
                    System/VelocityVerlet extended for periodic box + cutoff;
                    module declarations and re-exports
src/simulation.rs   lattice, Gaussian velocities, run driver
src/io.rs           RunMeta / TrajFrame / SimulationOutput serde types; read/write
src/check.rs        physics re-analysis (drift, T_speed, chi2)
src/cli.rs          manual arg parsing -> Command enum
src/video.rs        g(r), PPM frame rendering, ffmpeg call
src/main.rs         CLI dispatch (replaces the greeting print)
src/bin/dimer.rs    unchanged
src/bin/field_grid.rs  unchanged
tests/cli.rs        integration tests (contract run, CLI JSON readability)
```

## Reuse of Part 2 / Part 3

- Part 2 `energy` / `force` are reused unchanged through two thin wrappers:

```rust
pub fn energy_shifted(r: f64, rc: f64) -> f64 {
    if r < rc { energy(r) - energy(rc) } else { 0.0 }
}

pub fn force_cut(r: f64, rc: f64) -> f64 {
    if r < rc { force(r) } else { 0.0 }
}
```

- Part 3 `System` gains two optional fields while keeping the existing
  `System::new(positions, velocities)` signature (dimer behavior unchanged):

```rust
struct System {
    positions: Vec<[f64; 2]>,
    velocities: Vec<[f64; 2]>,
    accelerations: Vec<[f64; 2]>,
    box_len: Option<[f64; 2]>,   // None => open boundary (dimer)
    rc: Option<f64>,             // None => no cutoff (dimer)
}
```

- New constructor `System::new_periodic(positions, velocities, box_len, rc)`.
- `update_accelerations` / `potential_energy` apply minimum-image + cutoff
  only when `box_len` / `rc` are `Some`; otherwise they behave exactly as
  today (dimer).
- `VelocityVerlet::step` calls `apply_periodic_wrap()` (a no-op when
  `box_len` is `None`) immediately after the position update, before
  `update_accelerations()`.
- `System` gains `thermostat_temperature()`, `subtract_com_velocity()`, and
  `rescale_to_temperature(target)` (see below).
- `ForwardEuler` and `run_dimer_experiment` are unchanged; all four existing
  tests are untouched.

## Initial system (lattice and box)

With `h = (√3/2)·a` and `ρ = 1/(a·h)`:

- `a = √(2/(√3·ρ)) ≈ 1.201405`, `h = (√3/2)·a ≈ 1.04045`.
- `Lx = 10·a ≈ 12.01405`, `Ly = 10·h ≈ 10.40450`.
- For `i, j = 0..9`:
  - `x = (i + 0.5·(j mod 2))·a`
  - `y = j·h`
- All positions lie in `[0, Lx) × [0, Ly)`. Ten rows is even, so the
  alternating row offsets are compatible with the periodic boundary across
  the top/bottom.
- `--n` is honored minimally: `n` must be an even perfect square
  (`n = m²`, `m` even); the default `100 → m = 10`. Anything else is rejected
  with a clear error.

## Forces and energy

- Minimum-image convention in x and y.
- Cutoff `rc = 2.5 < min(Lx, Ly)/2 ≈ 5.2`, so only nearest images interact
  (no double counting).
- Force: `force_cut(r, rc)` for `r < rc`, zero outside; Newton's third law
  applied exactly per pair (so the total internal force sums to zero).
- Potential energy: `energy_shifted(r, rc)` summed over pairs.

## Initial velocities and equilibration

- Gaussian velocity components: mean 0, **variance T = 0.5**, generated with
  `rand::rngs::StdRng::seed_from_u64(2026)` + `StandardNormal` scaled by
  `sqrt(T)`.
- Then `subtract_com_velocity()`, then rescale to the target temperature.
- **Thermostat temperature (course definition):** after removing the
  center-of-mass velocity, two independent degrees of freedom are gone, so

  `T_thermo = 2·E_kin / (2N − 2) = E_kin / (N − 1)`.

  The rescaling uses `T_thermo`, **not** `E_kin / N`.
- Equilibration = Velocity Verlet with the thermostat ON, matching the
  reference `week2/sim.py` cadence: after step `s` (for `s = 0, 50, 100, …`)
  rescale velocities to the target `T` using `T_thermo` (no per-rescale COM
  subtraction). After equilibration, subtract the center-of-mass velocity once
  more before production.
- `T_thermo` is computed from `System::kinetic_energy()`.

## Production and sampling

- Velocity Verlet, `dt = 0.01`, thermostat OFF, 10000 steps.
- Wrap positions into the box every step (inside the integrator).
- Save when `step % sample_every == 0` for `step = 1..=steps` →
  steps 50, 100, …, 10000 = exactly 200 frames. Step 0 is not saved.

## Output files

`run.json` (RunMeta) contains **only** the required course fields:

```json
{"n":100,"rho":0.8,"box":[12.01405,10.4045],"dt":0.01,"temperature":0.5,
 "eq_steps":2000,"steps":10000,"sample_every":50,"seed":2026,
 "integrator":"velocity-verlet"}
```

No extra `rc` field. The cutoff is the fixed course constant `rc = 2.5`;
`md check` and `md video` use that constant directly.

`traj.jsonl` — one JSON object per saved production frame:

```json
{"step":50,"t":0.5,"pos":[[x,y],...],"vel":[[vx,vy],...],"E_pot":...,"E_kin":...}
```

`md run` writes both into `--out DIR` (default `artifacts`), creating the
directory if needed.

## CLI surface (manual parsing)

- `md run [--n N] [--rho R] [--temperature T] [--dt D] [--eq-steps E]
  [--steps S] [--sample-every K] [--seed S] [--out DIR]` — defaults exactly
  as the required contract (n=100, rho=0.8, temperature=0.5, dt=0.01,
  eq-steps=2000, steps=10000, sample-every=50, seed=2026, out=artifacts).
- `md check [DIR]` — DIR default `artifacts`.
- `md video [DIR] [--out FILE]` — DIR default `artifacts`,
  `--out` default `artifacts/run.mp4`.
- No/invalid subcommand or flag → usage on stderr, exit 2.

## `md check` (exact course formulas)

Reads `run.json` + `traj.jsonl` and **recomputes** E_pot (shifted LJ,
minimum-image, rc = 2.5) and E_kin from the raw `pos`/`vel` of every frame —
never the stored energies. Then:

1. **Secular drift.** `E0` = recomputed total energy of the first saved
   frame; `k = max(1, floor(frames/10))`; drift =
   `|mean(last k) − mean(first k)| / |E0|`. PASS if `< 2e-3`.
2. **T_speed.** Pool the speeds from all saved frames (M = 20000 in the
   default run); `T_speed = ⟨v²⟩ / 2`. PASS if
   `|T_speed − 0.5| < 0.05` (the target is the fixed course constant 0.5,
   matching the checker's `TEMP_EXPECT`).
3. **Speed distribution.** 24 equal-probability bins:
   `b_k = √(−2·T_speed·ln(1 − k/24))` for k = 0..23, `b_24 = ∞`;
   `E_b = M/24`; `χ²/22 = (1/22)·Σ(O_b − E_b)²/E_b`. PASS if `< 2`.

Prints each check with value + threshold + PASS/FAIL; exit 0 only if all
pass, else 1.

## `md video` (MP4 via ffmpeg)

- One video frame per saved trajectory frame (200). Canvas 960×480 PPM (P6):
  left 480×480 = dot plot of positions in the box; right 480×480 = g(r) plot.
- g(r): running average over all saved frames up to and including the current
  one, minimum-image pair distances, `dr = 0.05`,
  `r ∈ [0, min(Lx,Ly)/2 ≈ 5.20]`, normalized
  `g(r) = (O_bin / N_pairs) · (box_area / annulus_area)` with
  `annulus_area = π((r+dr)² − r²)`. Axes + curve + `g = 1` reference line.
- Encode: `ffmpeg -y -framerate 10 -i frame_%05d.ppm -c:v libx264
  -pix_fmt yuv420p -crf 28 -movflags +faststart <out>`, then delete the
  temporary frames.
- Verify `ffmpeg` exists (clear error if not) and that the MP4 is `< 2 MB`
  (exit 1 with a message if `≥ 2 MB`; at 960×480/crf 28 it is far below).

## Error handling

All I/O failures, missing `ffmpeg`, invalid `--n`, and oversize MP4 produce a
clear stderr message and a non-zero exit (2 for usage, 1 for runtime). No
panics on user-facing paths; `System` keeps its internal `assert_eq!` for
programming errors.

## Compatibility with the course checker and viewer

The extracted course references (`week2/sim.py`, `week2/checker/check`,
`week2/checker/test_contract.py`, `week2/viewer/index.html`) are authoritative
for the artifact format. The Rust CLI is compatible as follows:

- **`run.json` schema** — exactly the 10 required keys (`n`, `rho`, `box`,
  `dt`, `temperature`, `eq_steps`, `steps`, `sample_every`, `seed`,
  `integrator`), matching `require_run_schema`. No extra keys.
- **`traj.jsonl` schema** — exactly `step`, `t`, `pos`, `vel`, `E_pot`,
  `E_kin` per frame; positions wrapped into `[0, Lx) × [0, Ly)` as the checker
  requires.
- **Self-consistency (check 3)** — we log full-precision `f64` positions and
  velocities and compute `E_pot`/`E_kin` from those exact same values, so the
  checker's `rel < 1e-6` recomputation matches without the 9-significant-digit
  rounding used by the reference Python solver.
- **Chronology** — frames are saved at `step = 50, 100, …, 10000` with
  `t = step·dt`, exactly `steps // sample_every = 200` frames, satisfying
  check 2.
- **Physics gate** — `md check` recomputes the same three quantities with the
  same thresholds as `week2/checker/check`: secular drift `< 2e-3`
  (`k = max(1, ⌊frames/10⌋)`), `|T_speed − 0.5| < 0.05`, and
  `χ²/22 < 2` with 24 equal-probability Rayleigh bins. The Python checker
  remains the authoritative full gate (it additionally validates schema,
  config, chronology, and logged-vs-recomputed energy consistency).
- **Viewer** — `week2/viewer/index.html` reads `run.json` (`box`) and
  `traj.jsonl` (`pos`, `vel`, `E_pot`, `E_kin`, `t`, `step`); our files load
  unchanged.
- **Seed** — the reference uses 42, but the course requirement fixes
  `seed = 2026`; the checker validates only that `seed` is an integer, so
  2026 is accepted.
- **Running the official checker** — produce artifacts at the repository root
  (`md run --out artifacts` run from the repo root), then
  `python3 week2/checker/check .` (expects `PASS`).

## Testing (TDD — tests written before implementation)

Preserved: `greeting_is_correct`, `well_depth`,
`force_matches_energy_derivative`, `dimer_energy_errors`.

New tests (each written RED before its implementation):

1. **Total internal force sums to zero** — periodic multi-atom system with
   cutoff: `|Σ aᵢ| ≈ 0`.
2. **Shifted potential continuous at the cutoff** — `energy_shifted(rc)=0`,
   `energy_shifted(rc+δ)=0`, `|energy_shifted(rc−1e-6)| < 1e-6` (just inside),
   and `force_cut` is zero outside / equals `force` inside.
3. **Contract/default run passes the physics limits** — run the default
   config and `check` it: all three PASS conditions hold.
4. **CLI writes readable JSON** — `md run` writes `run.json` + `traj.jsonl`
   that parse back with the correct fields and one object per saved frame.
5. (Extra) **Seed reproducibility** — same seed ⇒ identical initial
   velocities; COM velocity ≈ 0 after subtraction.
6. (Extra) **Lattice correctness** — 100 positions, all in-box, density 0.8.
7. (Extra) **g(r) and PPM sanity** — g(r) peaks at the right separation for a
   known pair; PPM header/bytes are correct; MP4 encode only when ffmpeg is
   present.

## Out of scope

No neighbor/cell lists (N = 100 is fine with O(N²)); no other thermostats; no
`ForwardEuler` for the fluid; no changes to `dimer.rs`/`field_grid.rs` or the
plotting Python scripts; no `--n` values other than even `m×m`; no g(r)
output beyond the video plot.
