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
    pub fn spacing(&self) -> f64 {
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

    /// Wrap any integer index into `0..n`.
    pub fn wrap(&self, i: isize) -> usize {
        let n = self.n as isize;
        (((i % n) + n) % n) as usize
    }

    fn check(&self, u: &[f64]) {
        assert_eq!(u.len(), self.len(), "field has {} entries, grid has {}", u.len(), self.len());
    }

    /// First derivative in x by second-order centred differences, periodic:
    /// `(u_{i+1,j} - u_{i-1,j}) / (2 dx)`.
    pub fn dx(&self, u: &[f64]) -> Vec<f64> {
        self.check(u);
        let n = self.n;
        let inv = 1.0 / (2.0 * self.dx);
        let mut out = vec![0.0; self.len()];
        for iy in 0..n {
            for ix in 0..n {
                let i = self.index(ix, iy);
                let ip = self.index(self.wrap(ix as isize + 1), iy);
                let im = self.index(self.wrap(ix as isize - 1), iy);
                out[i] = (u[ip] - u[im]) * inv;
            }
        }
        out
    }

    /// First derivative in y by second-order centred differences, periodic.
    pub fn dy(&self, u: &[f64]) -> Vec<f64> {
        self.check(u);
        let n = self.n;
        let inv = 1.0 / (2.0 * self.dx);
        let mut out = vec![0.0; self.len()];
        for iy in 0..n {
            for ix in 0..n {
                let i = self.index(ix, iy);
                let jp = self.index(ix, self.wrap(iy as isize + 1));
                let jm = self.index(ix, self.wrap(iy as isize - 1));
                out[i] = (u[jp] - u[jm]) * inv;
            }
        }
        out
    }

    /// Second derivative in x by second-order centred differences, periodic:
    /// `(u_{i+1,j} - 2 u_{i,j} + u_{i-1,j}) / dx^2`.
    pub fn dxx(&self, u: &[f64]) -> Vec<f64> {
        self.check(u);
        let n = self.n;
        let inv = 1.0 / (self.dx * self.dx);
        let mut out = vec![0.0; self.len()];
        for iy in 0..n {
            for ix in 0..n {
                let i = self.index(ix, iy);
                let ip = self.index(self.wrap(ix as isize + 1), iy);
                let im = self.index(self.wrap(ix as isize - 1), iy);
                out[i] = (u[ip] - 2.0 * u[i] + u[im]) * inv;
            }
        }
        out
    }

    /// Second derivative in y by second-order centred differences, periodic.
    pub fn dyy(&self, u: &[f64]) -> Vec<f64> {
        self.check(u);
        let n = self.n;
        let inv = 1.0 / (self.dx * self.dx);
        let mut out = vec![0.0; self.len()];
        for iy in 0..n {
            for ix in 0..n {
                let i = self.index(ix, iy);
                let jp = self.index(ix, self.wrap(iy as isize + 1));
                let jm = self.index(ix, self.wrap(iy as isize - 1));
                out[i] = (u[jp] - 2.0 * u[i] + u[jm]) * inv;
            }
        }
        out
    }

    /// Mixed second derivative by the centred 2-D stencil, periodic:
    /// `(u_{i+1,j+1} - u_{i+1,j-1} - u_{i-1,j+1} + u_{i-1,j-1}) / (4 dx^2)`.
    pub fn dxdy(&self, u: &[f64]) -> Vec<f64> {
        self.check(u);
        let n = self.n;
        let inv = 1.0 / (4.0 * self.dx * self.dx);
        let mut out = vec![0.0; self.len()];
        for iy in 0..n {
            for ix in 0..n {
                let i = self.index(ix, iy);
                let ipp = self.index(self.wrap(ix as isize + 1), self.wrap(iy as isize + 1));
                let ipm = self.index(self.wrap(ix as isize + 1), self.wrap(iy as isize - 1));
                let imp = self.index(self.wrap(ix as isize - 1), self.wrap(iy as isize + 1));
                let imm = self.index(self.wrap(ix as isize - 1), self.wrap(iy as isize - 1));
                out[i] = (u[ipp] - u[ipm] - u[imp] + u[imm]) * inv;
            }
        }
        out
    }

    /// Centred-difference Laplacian, `dxx + dyy`, periodic.
    pub fn laplacian(&self, u: &[f64]) -> Vec<f64> {
        let a = self.dxx(u);
        let b = self.dyy(u);
        a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
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
        assert!((g.spacing() - 2.0 * PI / n as f64).abs() < 1e-15);
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

    fn g_field(g: &PeriodicGrid2D) -> Vec<f64> {
        let n = g.n();
        let mut out = vec![0.0; g.len()];
        for iy in 0..n {
            for ix in 0..n {
                out[g.index(ix, iy)] =
                    (3.0 * g.point(ix)).sin() * (2.0 * g.point(iy)).cos();
            }
        }
        out
    }

    /// Exact derivative of `g = sin(3x) cos(2y)` by name.
    fn g_exact(g: &PeriodicGrid2D, which: &str) -> Vec<f64> {
        let n = g.n();
        let mut out = vec![0.0; g.len()];
        for iy in 0..n {
            for ix in 0..n {
                let (x, y) = (g.point(ix), g.point(iy));
                out[g.index(ix, iy)] = match which {
                    "dx" => 3.0 * (3.0 * x).cos() * (2.0 * y).cos(),
                    "dxx" => -9.0 * (3.0 * x).sin() * (2.0 * y).cos(),
                    "dxdy" => -6.0 * (3.0 * x).cos() * (2.0 * y).sin(),
                    "laplacian" => -13.0 * (3.0 * x).sin() * (2.0 * y).cos(),
                    other => panic!("unknown exact derivative {other}"),
                };
            }
        }
        out
    }

    fn max_abs_err(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
    }

    /// Centred-difference errors at `n`, ordered `[dx, dxx, dxdy, laplacian]`.
    fn centred_errors(n: usize) -> [f64; 4] {
        let g = PeriodicGrid2D::new(n);
        let f = g_field(&g);
        [
            max_abs_err(&g.dx(&f), &g_exact(&g, "dx")),
            max_abs_err(&g.dxx(&f), &g_exact(&g, "dxx")),
            max_abs_err(&g.dxdy(&f), &g_exact(&g, "dxdy")),
            max_abs_err(&g.laplacian(&f), &g_exact(&g, "laplacian")),
        ]
    }

    #[test]
    fn centred_differences_are_second_order() {
        // error(n = 32) / error(n = 64) should be about 4 for each operator.
        let e32 = centred_errors(32);
        let e64 = centred_errors(64);
        for (name, i) in [("dx", 0), ("dxx", 1), ("dxdy", 2), ("laplacian", 3)] {
            let ratio = e32[i] / e64[i];
            assert!(
                (ratio - 4.0).abs() < 0.5,
                "{name}: ratio {ratio} is not second order (e32 = {}, e64 = {})",
                e32[i],
                e64[i]
            );
        }
    }
}
