# Week 2 review — Lennard-Jones fluid `md` crate

Reviewed against the approved designs/plans in
`docs/superpowers/specs/` and `docs/superpowers/plans/`, the official course
checker (`week2/checker/check`), and the reference solver `week2/sim.py`.

## Verification performed after fixes

```sh
cd week2/md && cargo test --release          # 45 passed, 0 failed
cd week2 && make reproduce                    # writes week2/artifacts/{run.json,traj.jsonl}
week2/md/target/release/md check week2/artifacts   # PASS
python3 week2/checker/check week2             # PASS (official course gate)
```

The official checker result after the fixes:

```
secular energy drift = 1.537e-04  (< 2e-03)
T_meas = 0.4945                   (|.-0.5| < 0.05)
Maxwell-Boltzmann chi2/dof = 0.801 (< 2.0)
PASS
```

---

## Findings

### F1 — Missing post-equilibration center-of-mass subtraction — FIXED

**Area:** physics / disagreement with the committed design and reference.

The committed CLI design
(`docs/superpowers/specs/2026-09-09-md-cli-design.md`, "Initial velocities and
equilibration") and the plan (Task 8) both require:

> After equilibration, subtract the center-of-mass velocity once more before
> production.

The reference solver does exactly this (`vel -= vel.mean(axis=0)` after the
equilibration loop in `week2/sim.py`). The implementation omitted it and
replaced the comment with an argument that uniform rescaling makes it
unnecessary.

Physically the omission is small (the symmetric pair-force update conserves
linear momentum to machine precision, and the initial COM is already zero), but
it is still an implementation behavior that disagrees with the committed spec
and the reference workflow, and it leaves any accumulated COM drift in the
production run.

**Fix:** added `system.subtract_com_velocity();` immediately after the
equilibration loop in `week2/md/src/simulation.rs`.

**Commit:** `25c9e7e3ed7ecb0d572c9c09533d6f689ad3fdcd`

---

### F2 — `gaussian_velocities` deviates from the design and carries a false comment — FIXED (comment only)

**Area:** implementation vs design / documentation.

The committed design prescribes `rand::distributions::StandardNormal` scaled by
`sqrt(T)`. The implementation instead hand-rolls a Box–Muller transform and
claims:

> `rand` 0.8 has no StandardNormal, so we generate the Gaussian pair directly.

That claim is factually wrong — `rand` 0.8 does provide `StandardNormal`. The
Box–Muller output is distributionally equivalent (two independent
standard-normal draws), so this is **not a physics defect** and the course
checker does not pin the exact RNG stream (it only checks that `seed` is an
integer and that the resulting velocities are Maxwell–Boltzmann).

**Fix:** corrected the misleading comment. The sampler itself is left unchanged
because switching to `StandardNormal` would change the seed→velocity mapping and
churn every seeded artifact (`docs/traj.jsonl`, the published videos, and all
reproduced trajectories) for zero physical benefit.

**Commit:** `25c9e7e3ed7ecb0d572c9c09533d6f689ad3fdcd`

---

### F3 — Release profile does not carry debug info used by the profiling step — FIXED

**Area:** performance claims / reproducibility.

`week2/README.md` says the flame graphs come from a "release build with debug
info" (`samply … release build with debug info`). The committed `Cargo.toml`
had no `[profile.release]`, so a plain `cargo build --release` produced a
stripped binary and the documented profiling step could not resolve symbols.

**Fix:** added `[profile.release] debug = true` to `week2/md/Cargo.toml`.

**Commit:** `25c9e7e3ed7ecb0d572c9c09533d6f689ad3fdcd`

---

### F4 — `--ramp-to` / `ramp_to` is outside the three approved design docs — NOT FIXED (accepted)

**Area:** implementation vs design.

`RunArgs.ramp_to`, `RunConfig.ramp_to`, and `RunMeta.ramp_to` (the heating /
melting feature) are not described in any of the three committed design specs.
`RunMeta.ramp_to` is annotated with `#[serde(default, skip_serializing_if =
"Option::is_none")]`, so:

- the default contract run still writes exactly the 10 required `run.json`
  keys (verified: `week2/artifacts/run.json` has 10 keys and no `ramp_to`), and
- only heating runs carry the extra key.

The published viewer (`docs/index.html`, lines 712–716) explicitly reads
`ramp_to` to draw the temperature-ramp target line, so the extra key is an
intentional, viewer-supported extension rather than an accidental schema
change.

**Reason not fixed:** the default contract is unaffected, the official checker
ignores extra keys, and removing `ramp_to` would break the published heating
workflow. The real gap is that the feature is undocumented in the design docs;
it should be written up in a design/plan if the feature is kept.

---

### F5 — `md check` cannot validate heating/melting artifacts — NOT FIXED (by design)

**Area:** heating/melting workflow.

`check.rs` hardcodes `TARGET_TEMP = 0.5` and the three course-gate formulas.
A heating run (`--temperature 0.2 --ramp-to 1.2`) necessarily fails the
`|T_speed − 0.5| < 0.05` gate, so `md check` must not be used on heating
trajectories — only the viewer validates/presents them.

**Reason not fixed:** this matches the official checker, which is authoritative
and intentionally pins `TEMP_EXPECT = 0.5` for the NVE contract run. Making
`md check` temperature-aware would change the contract gate and is out of scope.

---

### F6 — README timing/profile claims are not backed by committed measurements — NOT FIXED

**Area:** performance claims.

The README's Timing and Profile tables have no committed backing data, and they
disagree with the course reference measurements present in the environment:

| Claim | README | Reference data in env | Notes |
| --- | ---: | ---: | --- |
| NumPy contract run | 17.833 s | 10.857 s (`week2/data/timing.json`) | untracked data |
| Rust debug | 5.926 s | 7.726 s (`week2/data/timing.json`) | untracked data |
| Rust release | 0.914 s | 0.672 s (`week2/data/timing.json`) | untracked data |
| Naive force share, N=400 | 98.37 % | 98.68 % (`week2/data/profile.json`) | untracked data |
| Cell-list force share, N=400 | 93.85 % | 95.82 % (`week2/data/profile.json`) | untracked data |
| Naive elapsed, N=400 | 0.7631 s | 0.7729 s (`week2/data/profile.json`) | untracked data |
| Cell-list elapsed, N=400 | 0.1758 s | 0.2093 s (`week2/data/profile.json`) | untracked data |
| Cell-list speedup, N=400 | ~4.3× | 3.69× (0.7729/0.2093) | untracked data |

The README's Profile row is labelled as samply (sampling-profiler) data while
`week2/data/profile.json` is wall-clock stage timing, so the two are not
directly comparable; but the committed repository contains **no** samply profile
data (only the two rendered PNGs), so the `0.7631 / 0.1758 / ~4.3×` figures
cannot be reproduced or audited from the repo.

The Benchmark (scaling) table **is** supported: it matches the committed
`week2/data/scaling-md.json` (speedups 1.50 / 4.52 / 21.68).

**Reason not fixed:** these are machine-bound measurements. Editing the numbers
by hand would be fabrication; they must be re-measured on the actual machine
with the profiling step now reproducible after F3, then the README updated from
the fresh data.

---

### F7 — README "Reproduce" references untracked files — NOT FIXED

**Area:** reproducibility of the documented workflow.

`git ls-files` shows the repo tracks only `week2/data/scaling-md.json` under
`week2/data/`, and does **not** track:

- `week2/data/benchmark.py` (used by the README Timing step),
- `week2/data/profile/` (used by the README Profile step),
- `week2/viewer/index.html` (used by the README heating step).

So in a fresh clone the Timing, Profile, and `cp week2/viewer/index.html
docs/index.html` steps fail. The self-contained, reproducible paths are
`make reproduce`, `md check artifacts`, and `python3 week2/benchmark_scaling.py`.

**Reason not fixed:** these are course-extracted reference files (some with
`:Zone.Identifier` sidecars) that do not belong in the student repository. The
README should either track them deliberately or clearly label those steps as
provenance notes rather than "Reproduce" instructions.

---

### F8 — `benchmark_scaling.py` "seconds per step" includes fixed overhead — NOT FIXED (minor)

**Area:** naive/cell-list comparison methodology.

`benchmark_scaling.py` divides the full `md run` wall time by
`EQ_STEPS + STEPS = 600`. That full time includes process startup, the
equilibration phase, and writing 10 trajectory frames, so the plotted
"seconds per step" is a full-run average, not a pure per-step force cost. The
startup/output overhead is most visible at N=100 (naive 0.0397 s for 600 steps),
which slightly flattens the naive curve's apparent slope.

The speedup ratios themselves are fair because both methods pay the same fixed
overhead and use the same seed/config. This is a presentation caveat, not a
correctness bug, so the benchmark is left as is.

---

### F9 — Heating workflow: viewer target line and sampling cadence — NOT FIXED (minor)

**Area:** heating/melting workflow.

Two cosmetic caveats in the published heating trajectory:

1. `ramp_target(t0, tf, step, total_steps)` uses `step/total_steps`, while the
   viewer (`docs/index.html`) draws the ramp as
   `t0 + (tf - t0) * frame_index/(frames - 1)`. At the first saved frame
   (step 100 of 20000) the run is at T ≈ 0.205 while the viewer's target line
   shows 0.200. The mismatch is ≤ 0.005 and vanishes at the last frame.
2. The production thermostat rescales every 50 steps regardless of
   `--sample-every`. The published run uses `--sample-every 100`, which is a
   multiple of 50, so every saved frame sits exactly on a rescale step and the
   per-frame temperature matches the ramp target. A non-multiple `--sample-every`
   would sample mid-ramp temperatures and the existing heating test's
   exact-equality assumption would no longer hold.

**Reason not fixed:** the physics of the ramp is correct and the published
trajectory is valid; (1) is a viewer-side convention and (2) is a documented
usage constraint rather than a defect.

---

### F10 — CLI does not validate positive `--temperature` / `--ramp-to` — NOT FIXED (minor)

**Area:** heating/melting workflow robustness.

`parse_run` accepts any `f64` for `--temperature` and `--ramp-to`. A negative
`--temperature` makes `temperature.sqrt()` NaN in `gaussian_velocities`, and a
negative `--ramp-to` makes `(target / t).sqrt()` NaN in
`rescale_to_temperature`, corrupting the trajectory. The run then fails late at
JSON serialization rather than with a clear usage error.

**Reason not fixed:** the contract run never uses these flags, and the review
focused on physics/performance/workflow correctness. This should be fixed as a
small validation pass (reject non-finite or non-positive temperatures) if the
CLI is meant to be robust to arbitrary user input.

---

## Test coverage notes (weak / missing tests)

- **Post-equilibration COM subtraction (F1)** has no dedicated test. A unit test
  would be weak because momentum conservation already keeps COM near zero, so a
  "COM ≈ 0" assertion passes with or without the subtraction. The behavior is
  instead pinned by the official checker's `T_meas`/drift gates and by matching
  `sim.py`.
- **Cell-list energy vs force pass** — `System::update_accelerations` discards
  the energy returned by `CellList::accelerations_and_energy`, and
  `System::potential_energy` recomputes with a scratch `CellList`. There is no
  test asserting the returned energy equals `potential_energy`. The existing
  naive-vs-cells agreement tests compare both paths' energies and forces, so a
  divergence here would be caught indirectly, but a direct assertion would be
  cheaper to debug.
- **Default-run `run.json` schema** — the exactly-10-keys assertion lives in
  `force_flags_run_and_keep_part4_run_json_contract` for small `--force` runs;
  the actual default contract run is only checked via `md check`. The `io.rs`
  unit test covers `ramp_to` omission, so the schema is effectively pinned, but
  an integration assertion on the default run's `run.json` would be more direct.
- **Dynamic (multi-step) naive-vs-cells agreement** is not tested; only static
  configurations are compared. This is acceptable because the two force paths
  are proven equal per configuration and the integrator is deterministic, but a
  short two-method run with a trajectory-difference check would add end-to-end
  confidence.
