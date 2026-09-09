# Cell-List Force Method — Design (md crate, Part 5)

- **Date:** 2026-09-09
- **Status:** Approved
- **Scope:** `week2/md` (Rust crate `md`)

## Goal

Add an O(N) cell-list force/energy path to the `md` crate, selectable per run
with `--force naive|cells`, with `cells` the default for `md run`. The
low-level `System` keeps the naive O(N²) path as its default so all Part 1–4
tests and behavior stay unchanged; `--force naive` reproduces the original
baseline exactly. The `run.json` output contract stays exactly as in Part 4
(no new fields are added).

## Current state (context)

- `System::update_accelerations()` and `System::potential_energy()` are naive
  O(N²) loops (minimum image + shifted LJ, `rc = 2.5`). This is the only force
  path today.
- `run_simulation` builds the lattice → `System::new_periodic` →
  Velocity-Verlet, saving `E_pot` every `sample_every` steps.
- `cli.rs` parses `md run` flags by hand; there is no `--force`.
- The course checker (`checker/check`) recomputes E_pot naively from raw `pos`
  and compares logged E_pot at rel < 1e-6; it requires `run.json` to contain
  exactly its 10 known keys (extra keys are ignored, but none are added).
- Reference data (`week2/data/profile.json`): at N = 400, 1000 steps, the naive
  force stage is 98.4% of wall time; the cell-list force stage is 95.8%.

## Approach (approved)

Approach 1: a `CellList` buffer owned by `System`, dispatched through a
`ForceMethod` enum.

- `ForceMethod { Naive, Cells }`, `Default = Naive` (low-level backward compat).
- `System::with_force_method(method)` builder; `run_simulation` opts into cells
  when `RunConfig.force_method == Cells`.
- Both `update_accelerations` and `potential_energy` dispatch on the method.
- `md run` / `RunConfig` default to `Cells`; `--force naive` = baseline.
- `run.json` keeps its Part 4 fields; the selected method is **not** recorded.

## Module layout

```
src/lib.rs          + `pub mod cell_list;` + `pub use cell_list::{CellList, ForceMethod};`
                      + System.force_method, System.cells
                      + with_force_method builder
                      + dispatch in update_accelerations/potential_energy
                      (naive bodies moved verbatim to private *_naive helpers)
src/cell_list.rs    NEW: ForceMethod enum, CellList struct + build /
                      for_each_pair / accelerations_and_energy / potential_energy
src/simulation.rs   RunConfig.force_method (default Cells); pass to System
src/cli.rs          RunArgs.force (default Cells); --force naive|cells parsing
src/main.rs         map RunArgs.force -> RunConfig.force_method
```

## Detailed design

### 1. ForceMethod

```rust
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

`ForceMethod` lives in `cell_list.rs` and is re-exported at the crate root. It
needs no serde derives because the method is deliberately **not** persisted to
`run.json`.

### 2. CellList data structure

```rust
pub struct CellList {
    cells: Vec<Vec<usize>>,     // atom indices per cell
    neighbors: Vec<Vec<usize>>, // deduplicated wrapped neighbor ids per home cell
    nx: usize,
    ny: usize,
}

impl CellList {
    pub fn new() -> Self { /* empty vectors */ }
    fn build(&mut self, positions: &[[f64; 2]], box_len: [f64; 2], rc: f64) { /* below */ }
    fn for_each_pair<F>(&self, positions, box_len, rc, visit: F) where F: FnMut(usize, usize, f64, f64) { /* below */ }
    pub fn accelerations_and_energy(&mut self, positions, box_len, rc) -> (Vec<[f64; 2]>, f64);
    pub fn potential_energy(&mut self, positions, box_len, rc) -> f64;
}
```

**Grid.** `nx = max(1, floor(Lx / rc))`, `ny = max(1, floor(Ly / rc))`; the cell
width is `Lx / nx` (≥ rc) and `Ly / ny`. Atom → cell: `cx = ((x / Lx) * nx as
f64) as usize` clamped to `nx - 1`; `cy` analogous. `build()` clears and resizes
`cells` to `nx * ny`, buckets each atom, then precomputes `neighbors[c]` for
every home cell `c`.

### 3. Pair enumeration (avoiding double counting)

`for_each_pair` visits, for each home cell, each of its unique neighbor cells,
then `for i in home { for j in neighbor { if j > i { visit(i, j, dx, dy) } } }`.
The `j > i` test counts every unordered pair exactly once:

- intra-cell pair: seen only in the `home == neighbor` visit; `j > i` keeps it
  once;
- cross-cell pair: seen twice (once from each side as home), and `j > i` keeps
  exactly one orientation.

The candidate pair is then filtered with the exact minimum-image expression and
the cutoff test `r2 < rc2` before `visit` is called.

### 4. Periodic wrapping and deduplication

Neighbor offsets are `{-1, 0, 1} × {-1, 0, 1}`. Wrapping uses
`(cx as isize + dx).rem_euclid(nx as isize) as usize` (same for `cy`). The nine
wrapped ids are collected, `sort_unstable`-ed and `dedup`-ed so each neighbor
cell appears at most once per home cell. Ordering after dedup is irrelevant to
correctness (see §3).

### 5. Two-cell-wide edge case

When `nx == 2` (or `ny == 2`), offsets `-1` and `+1` wrap to the *same* cell, so
the nine raw ids collapse after dedup; without dedup, cross-cell pairs would be
counted twice (2× force and energy). When `nx == 1`, all offsets collapse to the
single cell and the loop covers the whole box. The `max(1, …)` clamp guards
`Lx < rc` (which cannot occur for the contract run, since the checker requires
each side ≥ 2·rc, but may occur in tiny test boxes).

### 6. Force and energy physics + agreement tolerances

The cell list uses the **identical** LJ expressions as the naive path:

- minimum image: `dx -= Lx * (dx / Lx).round()` (the same formula as
  `System::separation`, duplicated verbatim in `cell_list.rs` so the naive path
  stays untouched);
- per accepted pair: `inv2 = 1/r2`, `inv6 = inv2³`, force coefficient
  `fac = 24·(2·inv6² − inv6)·inv2` (= `force(r)/r`, matching naive
  `f·dx/r`), energy `4·(inv6² − inv6) − shift` where
  `shift = 4·(rc⁻¹² − rc⁻⁶)` (= `energy(rc)`, matching `energy_shifted`).

Because both paths compute the same quantities, they differ only by summation
order and a few last-bit roundings (~1e-13). The reference measurement recorded
max force diff 1.78e-15 and energy diff 2.27e-13.

Tests assert, for every configuration:

- force arrays: `|a_cells[i][k] − a_naive[i][k]| ≤ 1e-9 · max(1, |a_naive[i][k]|)`;
- energy: `|E_cells − E_naive| ≤ 1e-9 · max(1, |E_naive|)`.

This is ~4 orders above rounding noise and ~6 orders below a real error (a
missing or doubled pair changes energy by O(1)).

### 7. CLI behavior

`md run [--force naive|cells] [existing flags]`.

- No `--force` → `ForceMethod::Cells`.
- `--force naive` → naive (reproduces the original baseline bit-for-bit).
- `--force cells` → cell list.
- Any other value → `error: invalid --force: <value> (expected naive or cells)` + usage, exit 2.

`RunArgs.force` (default `Cells`), `RunConfig.force_method` (default `Cells`),
and `run_cmd` copies `args.force` into the config. `usage()` is updated.

### 8. run.json contract (unchanged)

The `RunMeta` schema is **not** touched: `run.json` keeps exactly its Part 4
fields (`n, rho, box, dt, temperature, eq_steps, steps, sample_every, seed,
integrator`). The selected force method is a run-time choice only and is not
recorded, so old and new artifacts share one schema.

## Dispatch and buffer reuse

```rust
// lib.rs — System::update_accelerations (naive body moved to
// update_accelerations_naive, verbatim)
pub fn update_accelerations(&mut self) {
    match (self.force_method, self.box_len, self.rc) {
        (ForceMethod::Cells, Some(box_len), Some(rc)) => {
            let (acc, _) = self.cells.accelerations_and_energy(&self.positions, box_len, rc);
            self.accelerations = acc;
        }
        _ => self.update_accelerations_naive(),
    }
}

pub fn potential_energy(&self) -> f64 {
    match (self.force_method, self.box_len, self.rc) {
        (ForceMethod::Cells, Some(box_len), Some(rc)) => {
            let mut scratch = CellList::new();
            scratch.potential_energy(&self.positions, box_len, rc)
        }
        _ => self.potential_energy_naive(),
    }
}
```

- The per-timestep hot path (`update_accelerations`) reuses `System::cells`
  across steps (no per-step reallocation).
- `potential_energy` keeps its `&self` signature — so `total_energy` and every
  Part 1–4 call site stay unchanged — and uses a temporary `CellList` for the
  cells branch. Energy is computed only when frames are saved (every
  `sample_every` steps) or during analysis, so the scratch build is negligible.
- `with_force_method(mut self, method)` sets `force_method` and calls
  `update_accelerations()` so the initial accelerations match the chosen method
  (positions are unchanged, so this only swaps naive → cells initial
  accelerations).

## Testing and TDD order

Tests are written before implementation; the plan expands each into RED →
GREEN steps.

1. **CLI parsing** (RED: `unknown flag: --force`).
   - `parse_run_force_defaults_to_cells`
   - `parse_run_force_naive`
   - `parse_run_force_cells`
   - `parse_run_force_rejects_invalid`
   - `run_config_defaults_to_cells`
2. **Cell-list agreement** (RED: compile error — `CellList` / `ForceMethod` /
   `with_force_method` do not exist yet; this is the expected "feature missing"
   failure).
   - `cells_and_naive_agree_on_unperturbed_lattice` (N = 100)
   - `cells_and_naive_agree_on_perturbed_lattice` (seeded deterministic jitter,
     then re-wrapped)
   - `cells_and_naive_agree_across_periodic_boundary` (atoms near opposite
     edges; minimum-image pairs)
   - `cells_and_naive_agree_at_cutoff` (pairs at r = rc, rc − ε, rc + ε)
   - `cells_and_naive_agree_two_cell_wide_box` (nx = 2, ny = 2; exercises
     neighbor dedup and wrap)
   - `cell_list_grid_dimensions`, `two_cell_neighbors_are_sorted_and_deduplicated`
3. **Preservation** — the full Part 1–4 suite stays green; the `md run` default
   still passes `md check`; `make reproduce` + `checker/check` stay green; the
   `run.json` schema is byte-identical to Part 4 (no `force` key).

## Out of scope

- Re-timing / re-profiling the cells path (separate step after implementation).
- Changing `check` / `video` / `io` physics; they keep the naive recompute.
- Making `potential_energy` cell-list-based in the `check` subcommand.
- Recording the force method in `run.json`.
