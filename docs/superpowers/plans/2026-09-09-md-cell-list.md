# Cell-List Force Method (md Part 5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an O(N) cell-list force/energy path to the `md` crate behind `--force naive|cells`, make `cells` the default for `md run`, keep the low-level `System` naive by default, and prove cell-list forces/energy agree with the naive path within rounding tolerance. The `run.json` output schema stays exactly as in Part 4.

**Architecture:** `ForceMethod { Naive, Cells }` (default `Naive`) lives in a new `cell_list` module. `System` gains a `force_method` field and an owned, reused `CellList` buffer, plus a `with_force_method(method)` builder. `update_accelerations()` and `potential_energy()` dispatch on the method; the existing naive bodies move verbatim into private `*_naive` helpers. `RunArgs.force` / `RunConfig.force_method` default to `Cells`; `run_simulation` applies the chosen method. `run.json` is not modified.

**Tech Stack:** Rust (edition 2024), crates `serde` + `serde_json` + `rand` only; manual CLI parsing (unchanged style).

**Spec:** `docs/superpowers/specs/2026-09-09-md-cell-list-design.md`

## Global Constraints

- Work only in `week2/md/`; do **not** touch `week1/`, Python scripts, or unrelated files.
- Preserve all Part 1–4 code and tests: `greeting`, `energy`, `force`, `energy_shifted`, `force_cut`, `System::new`, `System::new_periodic`, `run_dimer_experiment`, and every existing test must stay green and bit-for-bit unchanged in behavior.
- Low-level default: `System::new` / `System::new_periodic` keep `ForceMethod::Naive`.
- Run-level default: `RunArgs.force` and `RunConfig.force_method` are `ForceMethod::Cells`.
- `run.json` contract is **unchanged**: `RunMeta` keeps exactly its Part 4 fields (`n, rho, box, dt, temperature, eq_steps, steps, sample_every, seed, integrator`). Do **not** add a `force` field anywhere in `io.rs`.
- Fixed cutoff `rc = 2.5` (`pub const RC: f64 = 2.5;`). Grid: `nx = max(1, floor(Lx / rc))`, `ny = max(1, floor(Ly / rc))`; cell width `Lx / nx` (≥ rc) and `Ly / ny`; atom→cell `cx = floor((x / Lx) * nx)` clamped to `nx - 1`.
- Neighbor search: own cell + 8 wrapped neighbors, offsets `{-1,0,1}²`, wrap via `(c as isize + d).rem_euclid(n as isize)`, then `sort_unstable()` + `dedup()`.
- Visit every atom pair exactly once using the `j > i` test inside the neighbor loop.
- Cell list uses the same minimum-image expression (`dx -= Lx * (dx / Lx).round()`) and the same cutoff physics as naive: accept pair iff `r2 < rc²`, force coefficient `fac = 24·(2·inv6² − inv6)·inv2`, energy `4·(inv6² − inv6) − shift` with `shift = 4·(rc⁻¹² − rc⁻⁶)`.
- Agreement tolerance: per force component `|Δa| ≤ 1e-9·max(|a_naive|, |a_cells|, 1.0)`; energy `|ΔE| ≤ 1e-9·max(|E_naive|, |E_cells|, 1.0)`.
- Naive source code must not change behavior: only additive edits (new fields, new methods, moving existing bodies into private helpers) are allowed.
- TDD: every behavior task writes the failing test first (RED), runs it, then implements (GREEN), then commits. For brand-new APIs the expected RED is a **compile error** ("feature missing"); do not add stubs to force a runtime failure — implement the real thing in the GREEN step.
- Run commands from `week2/md` unless noted: `cargo test`. The full default-contract integration test is slow; prefer `cargo test --release` when running everything.

---

## Task 1: `ForceMethod` enum + `--force` CLI parsing + config plumbing (no physics change)

Adds the run-level switch end-to-end (CLI → `RunArgs.force` → `RunConfig.force_method`). Physics is still naive at the end of this task; the cells path lands in Task 2 and is *applied* in Task 3.

**Files:**
- Create: `week2/md/src/cell_list.rs` (contains only `ForceMethod` for now)
- Modify: `week2/md/src/lib.rs` (declare `pub mod cell_list;` + re-export `ForceMethod`)
- Modify: `week2/md/src/cli.rs` (`--force` parsing, `RunArgs.force`, `usage()`)
- Modify: `week2/md/src/simulation.rs` (`RunConfig.force_method`, default test)
- Modify: `week2/md/src/main.rs` (copy `args.force` into `RunConfig`)
- Test: `week2/md/src/cli.rs` (tests), `week2/md/src/simulation.rs` (tests)

**Interfaces:**
- Produces: `md::ForceMethod` (`Naive`/`Cells`, `Default = Naive`, `parse(&str) -> Result<Self, String>`); `md::cli::RunArgs.force: ForceMethod` (default `Cells`); `md::simulation::RunConfig.force_method: ForceMethod` (default `Cells`).

- [ ] **Step 1: Create `cell_list.rs` with `ForceMethod`**

Create `week2/md/src/cell_list.rs` (the enum is a type the CLI tests need; wiring the flag is the behavior under test and comes later):

```rust
//! Selectable force methods for the LJ fluid: the naive O(N^2) pair loop and a
//! cell list (Part 5). `ForceMethod` is the run-level switch; `CellList`
//! (Task 2) implements the neighbor search.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ForceMethod {
    #[default]
    Naive,
    Cells,
}

impl ForceMethod {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "naive" => Ok(Self::Naive),
            "cells" => Ok(Self::Cells),
            other => Err(format!("expected naive or cells, got {other:?}")),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Naive => "naive",
            Self::Cells => "cells",
        }
    }
}
```

- [ ] **Step 2: Declare and re-export in `lib.rs`**

In `week2/md/src/lib.rs`, next to the other module declarations add `pub mod cell_list;`, and in the `pub use` block add the re-export:

```rust
pub mod cell_list;

pub use cell_list::ForceMethod;
```

- [ ] **Step 3: Write the failing CLI tests**

Add the import at the top of `week2/md/src/cli.rs`:

```rust
use crate::ForceMethod;
```

Add these four tests to the `mod tests` block at the bottom of `week2/md/src/cli.rs` (keep the existing tests):

```rust
    #[test]
    fn parse_run_force_defaults_to_cells() {
        match parse(&args(&["md", "run"])).unwrap() {
            Command::Run(r) => assert_eq!(r.force, ForceMethod::Cells),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn parse_run_force_naive() {
        match parse(&args(&["md", "run", "--force", "naive"])).unwrap() {
            Command::Run(r) => assert_eq!(r.force, ForceMethod::Naive),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn parse_run_force_cells() {
        match parse(&args(&["md", "run", "--force", "cells"])).unwrap() {
            Command::Run(r) => assert_eq!(r.force, ForceMethod::Cells),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn parse_run_force_rejects_invalid() {
        match parse(&args(&["md", "run", "--force", "bogus"])) {
            Err(msg) => assert!(msg.contains("invalid --force"), "error was: {msg}"),
            Ok(_) => panic!("expected an error for an invalid --force value"),
        }
    }
```

- [ ] **Step 4: Run tests to verify RED**

Run: `cargo test parse_run_force --lib`
Expected: all four tests FAIL with `unknown flag: --force` (the parse error the code currently returns for an unrecognized `--force` flag).

- [ ] **Step 5: Wire `--force` into `cli.rs`**

Add the field to `RunArgs` and its `Default` impl:

```rust
pub struct RunArgs {
    pub n: usize,
    // ... existing fields unchanged ...
    pub force: ForceMethod,
    pub out: PathBuf,
}
```

```rust
            seed: 2026,
            force: ForceMethod::Cells,
            out: PathBuf::from("artifacts"),
```

Add the match arm in `parse_run` (next to the other `"--…"` arms):

```rust
            "--force" => run.force = ForceMethod::parse(value)
                .map_err(|_| format!("invalid --force: {value} (expected naive or cells)"))?,
```

Update `usage()` so the `md run` line reads:

```rust
    "usage:\n  md run [--n 100] [--rho 0.8] [--temperature 0.5] [--dt 0.01] [--eq-steps 2000] [--steps 10000] [--sample-every 50] [--seed 2026] [--force naive|cells] [--out artifacts]\n  md check [artifacts]\n  md video [artifacts] [--out artifacts/run.mp4]".to_string()
```

- [ ] **Step 6: Run CLI tests to verify GREEN**

Run: `cargo test parse_run_force --lib`
Expected: 4 passed.

- [ ] **Step 7: Write the `RunConfig` default test (RED)**

Add to `mod tests` in `week2/md/src/simulation.rs`:

```rust
    #[test]
    fn run_config_defaults_to_cells() {
        assert_eq!(RunConfig::default().force_method, ForceMethod::Cells);
    }
```

- [ ] **Step 8: Run the test to verify RED**

Run: `cargo test run_config_defaults_to_cells --lib`
Expected: compile error — `no field 'force_method'` on `RunConfig` (feature-missing RED; do not stub).

- [ ] **Step 9: Implement `RunConfig.force_method`**

In `week2/md/src/simulation.rs`:

- change the crate import to include `ForceMethod`:

```rust
use crate::{advance, ForceMethod, RC, System, VelocityVerlet};
```

- add the field to `RunConfig`:

```rust
pub struct RunConfig {
    pub n: usize,
    // ... existing fields unchanged ...
    pub seed: u64,
    pub force_method: ForceMethod,
}
```

- add it to `Default`:

```rust
            seed: 2026,
            force_method: ForceMethod::Cells,
```

- update the existing `small_run_produces_one_saved_frame` test literal so the crate compiles (add the field after `seed: 2026,`):

```rust
            seed: 2026,
            force_method: ForceMethod::Cells,
```

- [ ] **Step 10: Update `main.rs` to copy the flag**

In `week2/md/src/main.rs`, `run_cmd` builds `md::RunConfig`; add the field:

```rust
        seed: args.seed,
        force_method: args.force,
```

- [ ] **Step 11: Run the full suite**

Run: `cargo test`
Expected: all tests pass (naive physics unchanged; `RunConfig.force_method` is carried but `run_simulation` does not use it yet).

- [ ] **Step 12: Commit**

```bash
git add week2/md/src/cell_list.rs week2/md/src/lib.rs week2/md/src/cli.rs week2/md/src/simulation.rs week2/md/src/main.rs
git commit -m "feat(md): add ForceMethod enum and --force naive|cells CLI flag"
```

---

## Task 2: `CellList` + `System` dispatch (RED: agreement tests → GREEN)

The core task: implement the cell list and make `System` dispatch on `ForceMethod`, keeping the naive bodies byte-for-byte.

**Files:**
- Modify: `week2/md/src/cell_list.rs` (add `CellList` + agreement/unit tests)
- Modify: `week2/md/src/lib.rs` (System fields, `with_force_method`, dispatch, private naive helpers, re-export `CellList`)
- Test: `week2/md/src/cell_list.rs` (test module)

**Interfaces:**
- Consumes: `ForceMethod` from Task 1.
- Produces: `CellList::new()`, `CellList::accelerations_and_energy(&mut self, &[[f64;2]], [f64;2], f64) -> (Vec<[f64;2]>, f64)`, `CellList::potential_energy(&mut self, &[[f64;2]], [f64;2], f64) -> f64`, and `System::with_force_method(ForceMethod) -> System`.

- [ ] **Step 1: Write the failing agreement and grid tests**

Append a `#[cfg(test)] mod tests` block to `week2/md/src/cell_list.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{build_lattice, RC, System};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    // Build a naive system and an equivalent cell-list system from the same
    // positions, then assert forces and potential energy agree within the
    // rounding tolerance from the spec.
    fn assert_force_and_energy_agree(naive: &System, cells: &System) {
        let n = naive.n_atoms();
        assert_eq!(n, cells.n_atoms());
        for i in 0..n {
            for k in 0..2 {
                let a = naive.accelerations[i][k];
                let b = cells.accelerations[i][k];
                let tol = 1e-9 * a.abs().max(b.abs()).max(1.0);
                assert!(
                    (a - b).abs() <= tol,
                    "acceleration mismatch atom {i} dim {k}: naive={a} cells={b}"
                );
            }
        }
        let en = naive.potential_energy();
        let ec = cells.potential_energy();
        let tol = 1e-9 * en.abs().max(ec.abs()).max(1.0);
        assert!((en - ec).abs() <= tol, "energy mismatch: naive={en} cells={ec}");
    }

    fn pair_systems(pos: Vec<[f64; 2]>, box_len: [f64; 2]) -> (System, System) {
        let vel = vec![[0.0; 2]; pos.len()];
        let naive = System::new_periodic(pos.clone(), vel.clone(), box_len, RC);
        let cells = System::new_periodic(pos, vel, box_len, RC).with_force_method(ForceMethod::Cells);
        (naive, cells)
    }

    #[test]
    fn cells_and_naive_agree_on_unperturbed_lattice() {
        let (pos, box_len) = build_lattice(100, 0.8).unwrap();
        let (naive, cells) = pair_systems(pos.clone(), box_len);
        assert_force_and_energy_agree(&naive, &cells);
    }

    #[test]
    fn cells_and_naive_agree_on_perturbed_lattice() {
        let (mut pos, box_len) = build_lattice(100, 0.8).unwrap();
        let mut rng = StdRng::seed_from_u64(2026);
        for p in &mut pos {
            p[0] = (p[0] + rng.gen_range(-0.3f64..0.3)).rem_euclid(box_len[0]);
            p[1] = (p[1] + rng.gen_range(-0.3f64..0.3)).rem_euclid(box_len[1]);
        }
        let (naive, cells) = pair_systems(pos.clone(), box_len);
        assert_force_and_energy_agree(&naive, &cells);
    }

    #[test]
    fn cells_and_naive_agree_across_periodic_boundary() {
        // Pairs straddle the x and y edges, so minimum image matters.
        let box_len = [10.0, 10.0];
        let pos = vec![
            [0.4, 5.0], [9.6, 5.0], // |dx| = 0.8 across the x wrap
            [5.0, 0.3], [5.0, 9.7], // |dy| = 0.6 across the y wrap
            [0.3, 0.2], [9.7, 9.8], // diagonal across both wraps
            [4.2, 4.1], [4.9, 5.2], [7.1, 8.0], [1.5, 6.3],
        ];
        let (naive, cells) = pair_systems(pos, box_len);
        assert_force_and_energy_agree(&naive, &cells);
    }

    #[test]
    fn cells_and_naive_agree_at_cutoff() {
        // Box is wide enough (Lx = 16 > the 1..13.6 span) that no pair wraps.
        // A: exactly rc apart -> zero; B: rc - 1e-6 -> interacts;
        // C: rc + 0.1 -> zero. Inter-group gaps are >= rc, so they contribute
        // nothing either. Only pair B interacts.
        let box_len = [16.0, 10.0];
        let pos = vec![
            [1.0, 5.0], [3.5, 5.0],          // A: r = 2.5 exactly
            [6.0, 5.0], [8.499999, 5.0],     // B: r = 2.5 - 1e-6
            [11.0, 5.0], [13.6, 5.0],        // C: r = 2.6
        ];
        let (naive, cells) = pair_systems(pos, box_len);
        assert_force_and_energy_agree(&naive, &cells);
        // The configuration must be non-degenerate: only pair B interacts.
        let e = naive.potential_energy();
        assert!(e.abs() > 1e-12, "expected the just-inside pair to interact");
    }

    #[test]
    fn cells_and_naive_agree_two_cell_wide_box() {
        // nx = ny = 2: neighbor offsets -1 and +1 wrap onto the same cell, so
        // dedup is essential; without it cross-cell pairs double count.
        let box_len = [5.5, 5.5]; // floor(5.5/2.5) = 2 cells per side
        let mut pos = vec![
            [0.3, 2.75], [5.2, 2.75], // x-wrap pair
            [2.75, 0.4], [2.75, 5.1], // y-wrap pair
            [1.0, 1.0], [1.9, 2.3], [3.9, 4.3], [4.6, 4.9], [0.7, 4.4], [4.8, 0.6],
        ];
        let mut rng = StdRng::seed_from_u64(7);
        for p in &mut pos {
            p[0] = (p[0] + rng.gen_range(-0.2f64..0.2)).rem_euclid(box_len[0]);
            p[1] = (p[1] + rng.gen_range(-0.2f64..0.2)).rem_euclid(box_len[1]);
        }
        let (naive, cells) = pair_systems(pos, box_len);
        assert_force_and_energy_agree(&naive, &cells);
    }

    #[test]
    fn cell_list_grid_dimensions() {
        let mut cl = CellList::new();
        let pos = vec![[0.0, 0.0]];
        cl.build(&pos, [5.5, 5.5], RC);
        assert_eq!((cl.nx, cl.ny), (2, 2));
        cl.build(&pos, [10.0, 10.0], RC);
        assert_eq!((cl.nx, cl.ny), (4, 4));
        cl.build(&pos, [2.0, 10.0], RC); // Lx < rc -> clamped to 1
        assert_eq!((cl.nx, cl.ny), (1, 4));
    }

    #[test]
    fn two_cell_neighbors_are_sorted_and_deduplicated() {
        let mut cl = CellList::new();
        let pos = vec![[0.0, 0.0]];
        cl.build(&pos, [5.5, 5.5], RC); // 2x2 grid
        assert_eq!(cl.neighbors.len(), 4);
        for nbrs in &cl.neighbors {
            let mut deduped = nbrs.clone();
            deduped.dedup();
            assert_eq!(nbrs, &deduped, "neighbor list must be deduplicated");
        }
    }
}
```

- [ ] **Step 2: Run tests to verify RED**

Run: `cargo test --lib`
Expected: compile error — `with_force_method` / `CellList` do not exist (feature-missing RED). Do not stub; proceed to implement.

- [ ] **Step 3: Implement `CellList` in `cell_list.rs`**

Append to `week2/md/src/cell_list.rs` (after the `ForceMethod` impl):

```rust
/// Reusable cell grid for the O(N) force/energy path.
///
/// Grid: `nx = max(1, floor(Lx / rc))`, `ny = max(1, floor(Ly / rc))`, so each
/// cell is at least `rc` wide; a particle's cutoff sphere lies within its own
/// cell plus the 8 wrapped neighbours. `cells` and `neighbors` are reused
/// across timesteps (cleared, not reallocated, when the box is unchanged).
pub struct CellList {
    cells: Vec<Vec<usize>>,     // atom indices per cell (index cy*nx + cx)
    neighbors: Vec<Vec<usize>>, // sorted, deduplicated neighbour ids per home cell
    nx: usize,
    ny: usize,
}

impl CellList {
    pub fn new() -> Self {
        CellList { cells: Vec::new(), neighbors: Vec::new(), nx: 0, ny: 0 }
    }

    fn build(&mut self, positions: &[[f64; 2]], box_len: [f64; 2], rc: f64) {
        let nx = ((box_len[0] / rc).floor() as usize).max(1);
        let ny = ((box_len[1] / rc).floor() as usize).max(1);
        self.nx = nx;
        self.ny = ny;
        let ncells = nx * ny;

        if self.cells.len() != ncells {
            self.cells.resize(ncells, Vec::new());
        }
        for cell in &mut self.cells {
            cell.clear();
        }
        for (i, p) in positions.iter().enumerate() {
            let cx = (((p[0] / box_len[0]) * nx as f64) as usize).min(nx - 1);
            let cy = (((p[1] / box_len[1]) * ny as f64) as usize).min(ny - 1);
            self.cells[cy * nx + cx].push(i);
        }

        if self.neighbors.len() != ncells {
            self.neighbors.resize(ncells, Vec::new());
        }
        for nbrs in &mut self.neighbors {
            nbrs.clear();
        }
        for cy in 0..ny {
            for cx in 0..nx {
                let nbrs = &mut self.neighbors[cy * nx + cx];
                for dy in -1..=1isize {
                    for dx in -1..=1isize {
                        let x = (cx as isize + dx).rem_euclid(nx as isize) as usize;
                        let y = (cy as isize + dy).rem_euclid(ny as isize) as usize;
                        nbrs.push(y * nx + x);
                    }
                }
                nbrs.sort_unstable();
                nbrs.dedup();
            }
        }
    }

    /// Visit every atom pair that can possibly be within the cutoff exactly
    /// once: for each home cell and each unique neighbour cell, iterate
    /// `i in home`, `j in neighbour` and keep only `j > i`. The candidate
    /// displacement is minimum-imaged and the pair is rejected when
    /// `r2 >= rc2`. `visit(i, j, dx, dy)` receives the accepted pair.
    fn for_each_pair<F>(
        &self,
        positions: &[[f64; 2]],
        box_len: [f64; 2],
        rc: f64,
        mut visit: F,
    ) where
        F: FnMut(usize, usize, f64, f64),
    {
        let rc2 = rc * rc;
        for (home, home_cell) in self.cells.iter().enumerate() {
            for &ni in &self.neighbors[home] {
                let other = &self.cells[ni];
                for &i in home_cell {
                    for &j in other {
                        if j <= i {
                            continue;
                        }
                        let mut dx = positions[j][0] - positions[i][0];
                        let mut dy = positions[j][1] - positions[i][1];
                        dx -= box_len[0] * (dx / box_len[0]).round();
                        dy -= box_len[1] * (dy / box_len[1]).round();
                        if dx * dx + dy * dy >= rc2 {
                            continue;
                        }
                        visit(i, j, dx, dy);
                    }
                }
            }
        }
    }

    /// Cell-list accelerations and shifted potential energy in one pass.
    pub fn accelerations_and_energy(
        &mut self,
        positions: &[[f64; 2]],
        box_len: [f64; 2],
        rc: f64,
    ) -> (Vec<[f64; 2]>, f64) {
        self.build(positions, box_len, rc);
        let mut acc = vec![[0.0; 2]; positions.len()];
        let shift = 4.0 * (rc.powi(-12) - rc.powi(-6));
        let mut epot = 0.0;
        self.for_each_pair(positions, box_len, rc, |i, j, dx, dy| {
            let inv2 = 1.0 / (dx * dx + dy * dy);
            let inv6 = inv2 * inv2 * inv2;
            // fac = force(r)/r (see spec §6); energy is 4(inv6^2 - inv6) - shift.
            let fac = 24.0 * (2.0 * inv6 * inv6 - inv6) * inv2;
            acc[j][0] += fac * dx;
            acc[j][1] += fac * dy;
            acc[i][0] -= fac * dx;
            acc[i][1] -= fac * dy;
            epot += 4.0 * (inv6 * inv6 - inv6) - shift;
        });
        (acc, epot)
    }

    /// Cell-list shifted potential energy only.
    pub fn potential_energy(
        &mut self,
        positions: &[[f64; 2]],
        box_len: [f64; 2],
        rc: f64,
    ) -> f64 {
        self.build(positions, box_len, rc);
        let shift = 4.0 * (rc.powi(-12) - rc.powi(-6));
        let mut epot = 0.0;
        self.for_each_pair(positions, box_len, rc, |_i, _j, dx, dy| {
            let inv2 = 1.0 / (dx * dx + dy * dy);
            let inv6 = inv2 * inv2 * inv2;
            epot += 4.0 * (inv6 * inv6 - inv6) - shift;
        });
        epot
    }
}
```

- [ ] **Step 4: Wire `System` in `lib.rs`**

In `week2/md/src/lib.rs`:

1. Re-export `CellList` next to `ForceMethod`:

```rust
pub use cell_list::{CellList, ForceMethod};
```

2. Add the two fields to the `System` struct:

```rust
pub struct System {
    positions: Vec<[f64; 2]>,
    velocities: Vec<[f64; 2]>,
    accelerations: Vec<[f64; 2]>, // cached: always a(x) at current positions
    box_len: Option<[f64; 2]>,    // None => open boundary (dimer)
    rc: Option<f64>,              // None => no cutoff (dimer)
    force_method: ForceMethod,    // low-level default stays Naive
    cells: CellList,              // reusable cell-grid buffer
}
```

3. Initialize them in `System::build` (add two lines to the struct literal):

```rust
            box_len,
            rc,
            force_method: ForceMethod::Naive,
            cells: CellList::new(),
```

4. Add the builder after `new_periodic`:

```rust
    // Select the force method. Recomputes the initial accelerations so they
    // match the chosen method from step 0.
    pub fn with_force_method(mut self, method: ForceMethod) -> System {
        self.force_method = method;
        self.update_accelerations();
        self
    }
```

5. Rename the existing public bodies to private helpers. In `update_accelerations`, rename `pub fn update_accelerations(&mut self) {` to `fn update_accelerations_naive(&mut self) {` and leave the body **exactly as it is**. In `potential_energy`, rename `pub fn potential_energy(&self) -> f64 {` to `fn potential_energy_naive(&self) -> f64 {` and leave the body **exactly as it is**.

6. Add the new public dispatcher for `update_accelerations` (after `update_accelerations_naive`):

```rust
    pub fn update_accelerations(&mut self) {
        match (self.force_method, self.box_len, self.rc) {
            (ForceMethod::Cells, Some(box_len), Some(rc)) => {
                let (acc, _) = self.cells.accelerations_and_energy(&self.positions, box_len, rc);
                self.accelerations = acc;
            }
            _ => self.update_accelerations_naive(),
        }
    }
```

7. Add the new public dispatcher for `potential_energy` (after `potential_energy_naive`):

```rust
    pub fn potential_energy(&self) -> f64 {
        match (self.force_method, self.box_len, self.rc) {
            (ForceMethod::Cells, Some(box_len), Some(rc)) => {
                // `&self` signature is preserved (Part 1-4 call sites stay
                // unchanged); energy is not the hot path, so a scratch buffer
                // is fine here. The per-step force path reuses `self.cells`.
                let mut scratch = CellList::new();
                scratch.potential_energy(&self.positions, box_len, rc)
            }
            _ => self.potential_energy_naive(),
        }
    }
```

- [ ] **Step 5: Run the tests to verify GREEN**

Run: `cargo test`
Expected: all tests pass — the 5 agreement tests, both `CellList` unit tests, the Task 1 tests, and every untouched Part 1–4 test.

- [ ] **Step 6: Commit**

```bash
git add week2/md/src/cell_list.rs week2/md/src/lib.rs
git commit -m "feat(md): add CellList force/energy path and ForceMethod dispatch in System"
```

---

## Task 3: Apply `force_method` in `run_simulation` + end-to-end validation

Makes `cells` the real default for `md run` and validates both paths end-to-end. Note: naive ≡ cells within rounding tolerance by design, so no unit-level RED exists for the wiring itself; the externally observable guarantees are pinned by regression tests (run.json schema unchanged, invalid flag rejected) and by the end-to-end gates (full suite, `md check`, the official checker, and `--force naive` baseline reproduction).

**Files:**
- Modify: `week2/md/src/simulation.rs` (`run_simulation` applies the method)
- Modify: `week2/md/tests/cli.rs` (regression tests: run.json contract unchanged; invalid flag rejected)
- Test: `week2/md/tests/cli.rs`

**Interfaces:**
- Consumes: `System::with_force_method` (Task 2), `RunConfig.force_method` (Task 1).
- Produces: `md run` default uses the cell list; `run.json` still has exactly the 10 Part 4 keys.

- [ ] **Step 1: Write the regression tests**

Append to `week2/md/tests/cli.rs`:

```rust
#[test]
fn force_flags_run_and_keep_part4_run_json_contract() {
    for force in ["naive", "cells"] {
        let dir = std::env::temp_dir().join(format!("md_force_{force}_test"));
        let _ = std::fs::remove_dir_all(&dir);

        let status = Command::new(env!("CARGO_BIN_EXE_md"))
            .args([
                "run", "--n", "36", "--eq-steps", "0", "--steps", "50",
                "--sample-every", "50", "--force", force, "--out", dir.to_str().unwrap(),
            ])
            .status()
            .expect("failed to run md binary");
        assert!(status.success(), "--force {force} run failed");

        let run_json = std::fs::read_to_string(dir.join("run.json")).unwrap();
        let meta: serde_json::Value = serde_json::from_str(&run_json).unwrap();
        let part4_keys = [
            "n", "rho", "box", "dt", "temperature", "eq_steps",
            "steps", "sample_every", "seed", "integrator",
        ];
        for key in part4_keys {
            assert!(meta.get(key).is_some(), "run.json missing key {key}");
        }
        assert_eq!(meta.as_object().unwrap().len(), part4_keys.len(),
            "run.json schema changed: expected exactly the Part 4 keys");
        assert!(meta.get("force").is_none(), "run.json must not contain a force key");

        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[test]
fn invalid_force_flag_is_rejected() {
    let dir = std::env::temp_dir().join("md_force_invalid_test");
    let out = Command::new(env!("CARGO_BIN_EXE_md"))
        .args(["run", "--force", "bogus", "--out", dir.to_str().unwrap()])
        .output()
        .expect("failed to run md binary");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("invalid --force"), "stderr: {stderr}");
    let _ = std::fs::remove_dir_all(&dir);
}
```

- [ ] **Step 2: Run the tests to verify they pass (regression pins)**

Run: `cargo test --test cli force_flags invalid_force`
Expected: both pass already (flag parsing landed in Task 1; schema is unchanged) — they pin the externally visible contract so Task 3's wiring cannot accidentally change `run.json`.

- [ ] **Step 3: Apply the method in `run_simulation`**

In `run_simulation` (`week2/md/src/simulation.rs`), build the system with the chosen method:

```rust
    let mut system = System::new_periodic(positions, velocities, box_len, RC)
        .with_force_method(config.force_method);
```

(No `io.rs` change: `RunMeta` and `run.json` stay exactly as in Part 4.)

- [ ] **Step 4: Run the full suite**

Run: `cargo test --release`
Expected: all unit and integration tests pass, including the two new regression tests and the existing `default_run_and_check_pass_physics_limits` (which now runs the full contract with the **cells** default and still passes `md check`).

- [ ] **Step 5: Verify the default run against the course gate**

Run (from `week2/`):

```bash
make reproduce
python3 checker/check .
```

Expected: `make reproduce` runs the default contract with cells; the checker prints `PASS` (energy conserved, speeds Maxwell–Boltzmann). This proves the cells path satisfies the course gate that previously validated naive.

- [ ] **Step 6: Verify `--force naive` reproduces the baseline**

Run (from `week2/md`), writing into a fresh dir layout the checker understands:

```bash
cargo run --release -- run --out /tmp/naive-baseline/artifacts --force naive
python3 ../checker/check /tmp/naive-baseline
```

Expected: run succeeds and the checker passes — the untouched naive path reproduces the original baseline behavior.

- [ ] **Step 7: Commit**

```bash
git add week2/md/src/simulation.rs week2/md/tests/cli.rs
git commit -m "feat(md): make cells the default run force method"
```

---

## Post-plan verification (manual, not a task)

- `cargo test --release` (full suite, all green).
- Profiling note: after these tasks the naive numbers in `week2/README.md` are the committed baseline; re-profiling the cells path with samply and updating `README.md`'s Profile table is a **separate** follow-up step (out of scope for this plan).
