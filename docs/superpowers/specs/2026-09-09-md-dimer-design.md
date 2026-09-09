# Two-Atom Lennard-Jones Molecular Dynamics — Design (md crate, Part 3)

- **Date:** 2026-09-09
- **Status:** Approved
- **Scope:** `week2/md` (Rust crate `md`)

## Goal

Extend the `md` crate with a two-atom (dimer) Lennard-Jones molecular dynamics
simulation. Implement two integrators — explicit forward Euler and
velocity-Verlet — behind one shared `Integrator` trait, and run the *same*
dimer experiment through both, recording the relative total-energy error
`(E(t) − E₀) / |E₀|`.

## Current state (context)

`week2/md/src/lib.rs` already contains:

- `pub fn energy(r: f64) -> f64` — Lennard-Jones pair potential
  `U(r) = 4 (r⁻¹² − r⁻⁶)`.
- `pub fn force(r: f64) -> f64` — radial scalar force `F(r) = −dU/dr`
  `= 24/r · (2 r⁻¹² − r⁻⁶)`.

These are reused unchanged. `week2/md/src/bin/field_grid.rs` shows the
established sign convention: for a test atom at displacement `(x, y)` from an
atom at the origin, `fx = F(r)·x/r`, `fy = F(r)·y/r`.

## Requirements

1. Forward Euler and velocity-Verlet integrators share one Rust trait.
2. The same experiment runs with either integrator.
3. Initial positions: atom 1 at `(0.0, 0.0)`, atom 2 at `(1.2, 0.0)`.
4. Both initial velocities `(0.0, 0.0)`.
5. Mass = 1 for both atoms.
6. Use the Lennard-Jones force already implemented in Part 2.
7. `dt = 0.01`.
8. Run both integrators for 500 steps.
9. Also run velocity-Verlet for 5000 steps.
10. Record relative total-energy error `(E(t) − E₀) / |E₀|`.
11. Velocity-Verlet maximum |relative error| must be `< 1e-3` over 500 steps.
12. Forward Euler final relative error must be `> 0.5`.
13. Follow the prescribed `Integrator` trait design.
14. TDD: write the dimer test before the implementation.

## Prescribed interfaces (verbatim, unchanged)

```rust
trait Integrator {
    fn step(&self, system: &mut System, dt: f64);
}

fn advance(method: &impl Integrator, system: &mut System, dt: f64) {
    method.step(system, dt);
}
```

Base `System` (prescribed, extended below):

```rust
struct System {
    positions: Vec<[f64; 2]>,
    velocities: Vec<[f64; 2]>,
}

impl System {
    fn n_atoms(&self) -> usize {
        self.positions.len()
    }
}
```

## Design

### System extension

`System` owns the position and velocity arrays (as prescribed) and is extended
with a cached acceleration array (needed by velocity-Verlet, whose algorithm
computes the initial acceleration before the first step and retains the new
acceleration for the next step):

```rust
struct System {
    positions: Vec<[f64; 2]>,
    velocities: Vec<[f64; 2]>,
    accelerations: Vec<[f64; 2]>,   // cached: always a(x) at current positions
}

impl System {
    fn n_atoms(&self) -> usize { self.positions.len() }

    fn new(positions: Vec<[f64; 2]>, velocities: Vec<[f64; 2]>) -> System
    //   - asserts positions.len() == velocities.len()
    //   - allocates accelerations as zeros, then calls update_accelerations()
    //     so a₀ is available before the first step

    fn update_accelerations(&mut self)   // LJ pair forces; mass = 1, so a = F
    fn kinetic_energy(&self) -> f64      // Σ 0.5·|v|²  (unit mass)
    fn potential_energy(&self) -> f64    // Σ_{i<j} energy(r_ij)
    fn total_energy(&self) -> f64        // kinetic + potential
}
```

**Mass:** hardcoded unit mass. The assignment fixes `m = 1` and the prescribed
base `System` has no `masses` field, so no `masses: Vec<f64>` is added.

### Invariant (correctness anchor)

After `System::new` and after every `step`, `system.accelerations` equals the
acceleration at the *current* positions. Both integrators maintain this.

### Pair-force sign convention

For a pair `(i, j)`: let `d = positions[j] − positions[i]`, `r = |d|`, and
`f = force(r)` (the existing `−dU/dr` scalar). Then:

- force on `j` = `f · d/r`
- force on `i` = `−f · d/r` (Newton's third law)
- acceleration = force (unit mass)

Sanity check: atom 0 at the origin, atom 1 at `(1.2, 0)` →
`force(1.2) ≈ −2.22`, so atom 1 accelerates in `−x` (toward atom 0) and atom 0
in `+x`.

Note: a sign flip makes the force non-conservative with respect to `U`, so the
velocity-Verlet `< 1e-3` energy check (below) catches it — the reference value
for a flipped force is ≈ 2.0, far above the bound.

### Integrators

`struct ForwardEuler;` and `struct VelocityVerlet;` both implement
`Integrator`.

**Forward Euler — explicit form (as specified in the course):**

Using the cached `aₙ`:

1. `xₙ₊₁ = xₙ + vₙ·dt`
2. `vₙ₊₁ = vₙ + aₙ·dt` — uses the *old* acceleration `aₙ`, not a recomputed one
3. `update_accelerations()` → cache `aₙ₊₁` for the next step

This is the explicit (non-symplectic) form. The symplectic/semi-implicit
variant (update `x`, recompute `a`, then update `v` with the new `a`) would
keep the energy error around ~0.009 and would *fail* the `> 0.5` requirement.

**Velocity Verlet:**

Snapshot `aₙ`, then:

1. `xₙ₊₁ = xₙ + vₙ·dt + ½·aₙ·dt²`
2. `update_accelerations()` → `aₙ₊₁`
3. `vₙ₊₁ = vₙ + ½·(aₙ + aₙ₊₁)·dt`

### Experiment runner

```rust
fn run_dimer_experiment(method: &impl Integrator, steps: usize, dt: f64) -> Vec<f64>
```

- Builds the dimer: positions `[[0.0, 0.0], [1.2, 0.0]]`,
  velocities `[[0.0, 0.0], [0.0, 0.0]]`.
- Records `E₀ = total_energy()` before stepping.
- Returns `steps + 1` values: the relative error `(E(t) − E₀) / |E₀|` at
  `t = 0, 1, …, steps`.
- Each step calls `advance(method, &mut system, dt)`.

## Acceptance criteria (dimer test)

One test `dimer_energy_errors`, written before implementation (RED first):

| Check | Expectation | Reference value |
|---|---|---|
| ForwardEuler, 500 steps, `dt = 0.01` | final relative error (`errors[500]`) `> 0.5` | ≈ +1.93 |
| VelocityVerlet, 500 steps, `dt = 0.01` | max \|relative error\| `< 1e-3` | ≈ 3.3e−4 |
| VelocityVerlet, 5000 steps, `dt = 0.01` | energy error stays bounded: max \|relative error\| `< 0.01` (loose long-time sanity check, **not** a strict `1e-3` acceptance) | ≈ 3.3e−4 |

The test also prints the three recorded values (visible with
`cargo test -- --nocapture`).

## Out of scope

- No changes to `energy`/`force`, `main.rs`, `field_grid.rs`, `Cargo.toml`, or
  `plot_field.py`.
- No `pub` visibility changes; everything lives in `lib.rs` and is tested by
  the in-file unit test module (consistent with the prescribed non-`pub`
  `System`/`Integrator` design).
- No temperature/thermostat, no neighbor lists, no general N-body driver
  beyond what the `System` representation naturally supports.
