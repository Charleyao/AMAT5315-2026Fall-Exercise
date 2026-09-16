//! Square lattice of +1/-1 spins with periodic boundaries.

/// Spins are stored row-major: index = y * l + x.
#[derive(Clone, Debug)]
pub struct Lattice {
    pub l: usize,
    pub spins: Vec<i8>,
}

impl Lattice {
    /// A lattice with every spin up.
    pub fn all_up(l: usize) -> Self {
        Self {
            l,
            spins: vec![1; l * l],
        }
    }

    /// Number of sites.
    pub fn n(&self) -> usize {
        self.l * self.l
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> i8 {
        self.spins[y * self.l + x]
    }

    /// Sum of the four nearest neighbours of (x, y) on the periodic lattice.
    #[inline]
    pub fn neighbour_sum(&self, x: usize, y: usize) -> i32 {
        let l = self.l;
        let left = (x + l - 1) % l;
        let right = (x + 1) % l;
        let up = (y + l - 1) % l;
        let down = (y + 1) % l;
        (self.get(left, y) as i32)
            + (self.get(right, y) as i32)
            + (self.get(x, up) as i32)
            + (self.get(x, down) as i32)
    }

    /// Energy per site, E = -(1/N) * sum over nearest-neighbour bonds of s_i*s_j,
    /// with each bond counted once (right and down neighbours only).
    pub fn energy_per_site(&self) -> f64 {
        let l = self.l;
        let mut bond_sum: i64 = 0;
        for y in 0..l {
            for x in 0..l {
                let s = self.get(x, y) as i64;
                let right = self.get((x + 1) % l, y) as i64;
                let down = self.get(x, (y + 1) % l) as i64;
                bond_sum += s * right + s * down;
            }
        }
        -(bond_sum as f64) / (self.n() as f64)
    }

    /// Signed mean spin, M = (1/N) * sum of spins.
    pub fn magnetization(&self) -> f64 {
        let sum: i64 = self.spins.iter().map(|&s| s as i64).sum();
        sum as f64 / self.n() as f64
    }

    #[inline]
    pub fn flip(&mut self, index: usize) {
        self.spins[index] = -self.spins[index];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_up_has_energy_minus_two() {
        // Every bond is +1 and there are 2*N bonds, so E = -2.
        for l in [2, 4, 8] {
            let lattice = Lattice::all_up(l);
            assert!((lattice.energy_per_site() + 2.0).abs() < 1e-12);
            assert!((lattice.magnetization() - 1.0).abs() < 1e-12);
            assert_eq!(lattice.neighbour_sum(0, 0), 4);
        }
    }

    #[test]
    fn checkerboard_has_energy_plus_two() {
        // Even side: all four neighbours of every site have the opposite spin.
        let l = 4;
        let mut lattice = Lattice::all_up(l);
        for y in 0..l {
            for x in 0..l {
                if (x + y) % 2 == 1 {
                    lattice.flip(y * l + x);
                }
            }
        }
        assert!((lattice.energy_per_site() - 2.0).abs() < 1e-12);
        assert!(lattice.magnetization().abs() < 1e-12);
    }

    #[test]
    fn single_flip_within_all_up_costs_eight() {
        // E_after - E_before = 2 * s_i * (sum of neighbours) = 2 * 1 * 4 = 8.
        let l = 4;
        let before = Lattice::all_up(l).energy_per_site();
        let mut lattice = Lattice::all_up(l);
        lattice.flip(0);
        let delta = lattice.energy_per_site() - before;
        assert!((delta - 8.0 / (l * l) as f64).abs() < 1e-12);
    }
}
