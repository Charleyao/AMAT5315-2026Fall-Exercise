//! Two-dimensional periodic grid on `[0, 2*pi) x [0, 2*pi)`.
//!
//! Exactly `n` points per side, `x_j = 2*pi*j/n` and `y_l = 2*pi*l/n` for
//! `j, l = 0, ..., n-1`; the endpoint `2*pi` is never duplicated. Fields are
//! flattened row-major with the x index varying fastest:
//!
//! ```text
//! idx = iy * n + ix
//! ```
//!
//! so `idx` runs through `(ix, iy)` in the order
//! `(0,0), (1,0), ..., (n-1,0), (0,1), ...`.

use std::f64::consts::PI;

use crate::grid::fft_wavenumbers;

/// A uniform periodic grid with `n x n` points on `[0, 2*pi)^2`.
pub struct PeriodicGrid2D {
    n: usize,
    dx: f64,
    x: Vec<f64>,
}

impl PeriodicGrid2D {
    /// The grid `x_j = y_j = 2*pi*j/n`, `j = 0, ..., n-1`.
    pub fn new(n: usize) -> Self {
        assert!(n >= 2, "a periodic grid needs at least two points per side, got {n}");
        let dx = 2.0 * PI / n as f64;
        let x = (0..n).map(|j| j as f64 * dx).collect();
        Self { n, dx, x }
    }

    /// Number of points per side.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Total number of points `n * n`.
    pub fn len(&self) -> usize {
        self.n * self.n
    }

    /// Always false (`n >= 2`), present for lint parity with `Vec`.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Spacing `dx = 2*pi/n`, the same in both directions.
    pub fn dx(&self) -> f64 {
        self.dx
    }

    /// The one-dimensional coordinate vector `x_0 ... x_{n-1}` (also used for y).
    pub fn x(&self) -> &[f64] {
        &self.x
    }

    /// The point `x_i` (equivalently `y_i`).
    pub fn point(&self, i: usize) -> f64 {
        self.x[i]
    }

    /// Row-major flat index, `iy * n + ix` (x fastest).
    pub fn index(&self, ix: usize, iy: usize) -> usize {
        assert!(ix < self.n && iy < self.n, "({ix}, {iy}) outside {0}x{0}", self.n);
        iy * self.n + ix
    }

    /// Inverse of [`Self::index`]: `(ix, iy)` of a flat index.
    pub fn coords(&self, idx: usize) -> (usize, usize) {
        assert!(idx < self.len(), "index {idx} outside {}", self.len());
        (idx % self.n, idx / self.n)
    }

    /// Integer wavenumbers in FFT order (`0, 1, ..., n/2-1, -n/2, ..., -1`)
    /// for either axis.
    pub fn wavenumbers(&self) -> Vec<f64> {
        fft_wavenumbers(self.n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_has_n_squared_points_without_the_endpoint() {
        let n = 8;
        let g = PeriodicGrid2D::new(n);
        assert_eq!(g.n(), n);
        assert_eq!(g.len(), n * n);
        assert!((g.dx() - 2.0 * PI / n as f64).abs() < 1e-15);
        for j in 0..n {
            let want = 2.0 * PI * j as f64 / n as f64;
            assert!((g.point(j) - want).abs() < 1e-15);
        }
        assert!((g.point(0) - 0.0).abs() < 1e-15);
        assert!(g.point(n - 1) < 2.0 * PI);
    }

    #[test]
    fn index_is_row_major_with_x_fastest() {
        let n = 4;
        let g = PeriodicGrid2D::new(n);
        // x runs fastest: index = iy*n + ix.
        assert_eq!(g.index(0, 0), 0);
        assert_eq!(g.index(1, 0), 1);
        assert_eq!(g.index(3, 0), 3);
        assert_eq!(g.index(0, 1), 4);
        assert_eq!(g.index(3, 2), 2 * n + 3);
        // Round trip over every point.
        for ix in 0..n {
            for iy in 0..n {
                assert_eq!(g.coords(g.index(ix, iy)), (ix, iy));
            }
        }
    }

    #[test]
    fn wavenumbers_are_the_fft_order() {
        let g = PeriodicGrid2D::new(8);
        assert_eq!(
            g.wavenumbers(),
            vec![0.0, 1.0, 2.0, 3.0, -4.0, -3.0, -2.0, -1.0]
        );
    }
}
