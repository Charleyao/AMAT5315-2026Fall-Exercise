//! One-dimensional periodic grid on [0, 2*pi) and centred finite differences.
//!
//! The grid holds exactly `n` points, `x_j = 2*pi*j/n` for `j = 0, ..., n-1`.
//! There is no duplicated endpoint: `x_0 = 0` and `x_{n-1} = 2*pi*(n-1)/n`.
//! Derivatives use periodic wraparound, so point `0`'s left neighbour is `n-1`
//! and point `n-1`'s right neighbour is `0`.

use std::f64::consts::PI;

/// A uniform periodic grid with `n` points on `[0, 2*pi)`.
pub struct PeriodicGrid {
    n: usize,
    dx: f64,
    x: Vec<f64>,
}

impl PeriodicGrid {
    /// The grid `x_j = 2*pi*j/n`, `j = 0, ..., n-1`.
    pub fn new(n: usize) -> Self {
        assert!(n >= 2, "a periodic grid needs at least two points, got {n}");
        let dx = 2.0 * PI / n as f64;
        let x = (0..n).map(|j| j as f64 * dx).collect();
        Self { n, dx, x }
    }

    /// Number of grid points.
    pub fn len(&self) -> usize {
        self.n
    }

    /// True when the grid has no points (never, since `n >= 2`).
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Spacing `dx = 2*pi/n`.
    pub fn dx(&self) -> f64 {
        self.dx
    }

    /// All grid points `x_0 ... x_{n-1}`.
    pub fn x(&self) -> &[f64] {
        &self.x
    }

    /// The point `x_j`.
    pub fn point(&self, j: usize) -> f64 {
        self.x[j]
    }

    /// Wrap any integer index into `0..n` (negative indices wrap from the end).
    pub fn wrap(&self, j: isize) -> usize {
        let n = self.n as isize;
        (((j % n) + n) % n) as usize
    }

    /// Left neighbour of `j`; at `j = 0` this wraps to `n-1`.
    pub fn left(&self, j: usize) -> usize {
        self.wrap(j as isize - 1)
    }

    /// Right neighbour of `j`; at `j = n-1` this wraps to `0`.
    pub fn right(&self, j: usize) -> usize {
        self.wrap(j as isize + 1)
    }

    /// First derivative, second-order centred difference:
    /// `(u_{j+1} - u_{j-1}) / (2 dx)`, with periodic wrapping.
    pub fn d1(&self, u: &[f64]) -> Vec<f64> {
        assert_eq!(u.len(), self.n, "u has {} entries, grid has {}", u.len(), self.n);
        (0..self.n)
            .map(|j| (u[self.right(j)] - u[self.left(j)]) / (2.0 * self.dx))
            .collect()
    }

    /// Second derivative, second-order centred difference:
    /// `(u_{j+1} - 2 u_j + u_{j-1}) / dx^2`, with periodic wrapping.
    pub fn d2(&self, u: &[f64]) -> Vec<f64> {
        assert_eq!(u.len(), self.n, "u has {} entries, grid has {}", u.len(), self.n);
        (0..self.n)
            .map(|j| {
                (u[self.right(j)] - 2.0 * u[j] + u[self.left(j)]) / (self.dx * self.dx)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sample a function on the grid.
    fn sample<F: Fn(f64) -> f64>(g: &PeriodicGrid, f: F) -> Vec<f64> {
        g.x().iter().map(|&x| f(x)).collect()
    }

    fn max_abs_error(got: &[f64], want: &[f64]) -> f64 {
        got.iter()
            .zip(want.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn grid_holds_n_points_on_zero_to_two_pi() {
        let n = 8;
        let g = PeriodicGrid::new(n);
        assert_eq!(g.len(), n);
        assert_eq!(g.x().len(), n);
        assert_eq!(g.dx(), 2.0 * PI / n as f64);

        // Exactly the points x_j = 2*pi*j/n, and no extra endpoint x = 2*pi.
        for (j, &x) in g.x().iter().enumerate() {
            let want = 2.0 * PI * j as f64 / n as f64;
            assert!((x - want).abs() < 1e-15, "x_{j} = {x}, want {want}");
        }
        assert!((g.point(0) - 0.0).abs() < 1e-15);
        assert!(g.point(n - 1) < 2.0 * PI);
    }

    #[test]
    fn wrapping_maps_0_to_n_minus_1_and_n_minus_1_to_0() {
        let n = 8;
        let g = PeriodicGrid::new(n);

        // The explicit case the task calls out.
        assert_eq!(g.left(0), n - 1);
        assert_eq!(g.right(n - 1), 0);

        // Neighbour relations everywhere else, and the wrap helper.
        for j in 0..n {
            assert_eq!(g.left(j), (j + n - 1) % n);
            assert_eq!(g.right(j), (j + 1) % n);
        }
        assert_eq!(g.wrap(-1), n - 1);
        assert_eq!(g.wrap(n as isize), 0);
        assert_eq!(g.wrap((n + 3) as isize), 3);
    }

    #[test]
    fn d1_of_sin_is_cos() {
        let n = 64;
        let g = PeriodicGrid::new(n);
        let u = sample(&g, f64::sin);
        let want = sample(&g, f64::cos);
        let err = max_abs_error(&g.d1(&u), &want);
        println!("n = {n}, max |d1(sin) - cos| = {err:e}");
        assert!(err < 2e-3, "max |d1(sin) - cos| = {err}");
    }

    #[test]
    fn d2_of_sin_is_minus_sin() {
        let n = 64;
        let g = PeriodicGrid::new(n);
        let u = sample(&g, f64::sin);
        let want = sample(&g, |x| -x.sin());
        let err = max_abs_error(&g.d2(&u), &want);
        println!("n = {n}, max |d2(sin) + sin| = {err:e}");
        assert!(err < 1e-3, "max |d2(sin) + sin| = {err}");
    }

    #[test]
    fn d1_of_a_constant_is_zero_everywhere() {
        let g = PeriodicGrid::new(16);
        let u = vec![3.5; g.len()];
        assert!(g.d1(&u).iter().all(|v| v.abs() < 1e-12));
        assert!(g.d2(&u).iter().all(|v| v.abs() < 1e-12));
    }

    #[test]
    fn d1_is_second_order_in_dx() {
        // Halving dx should cut the error by about four.
        let coarse = PeriodicGrid::new(32);
        let fine = PeriodicGrid::new(64);
        let err_c = max_abs_error(&coarse.d1(&sample(&coarse, f64::sin)), &sample(&coarse, f64::cos));
        let err_f = max_abs_error(&fine.d1(&sample(&fine, f64::sin)), &sample(&fine, f64::cos));
        let ratio = err_c / err_f;
        assert!((ratio - 4.0).abs() < 0.5, "error ratio = {ratio}, expected ~4");
    }
}
