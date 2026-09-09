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
