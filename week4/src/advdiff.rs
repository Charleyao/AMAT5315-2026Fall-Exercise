//! The one-dimensional periodic linear advection-diffusion operator
//!
//! `u_t + c u_x = nu u_xx`,  i.e.  `du/dt = F(u) = -c u_x + nu u_xx`.
//!
//! It reuses the derivative routines already implemented and tested on
//! [`PeriodicGrid`]: the spectral (Fourier) pair `fourier_d1` / `fourier_d2`
//! or the second-order centred-difference pair `d1` / `d2`. There is no second
//! FFT derivative anywhere in this module.

use crate::grid::PeriodicGrid;

/// Which spatial discretisation the advection-diffusion RHS uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpatialScheme {
    /// Spectral derivatives: [`PeriodicGrid::fourier_d1`] / [`PeriodicGrid::fourier_d2`].
    Fourier,
    /// Second-order centred differences: [`PeriodicGrid::d1`] / [`PeriodicGrid::d2`].
    FiniteDifference,
}

/// The linear operator `F(u) = -c u_x + nu u_xx` on a periodic grid.
pub struct AdvectionDiffusion<'a> {
    pub grid: &'a PeriodicGrid,
    pub c: f64,
    pub nu: f64,
    pub scheme: SpatialScheme,
}

impl<'a> AdvectionDiffusion<'a> {
    pub fn new(grid: &'a PeriodicGrid, c: f64, nu: f64, scheme: SpatialScheme) -> Self {
        Self { grid, c, nu, scheme }
    }

    /// One evaluation of `F(u)`, reusing the grid's derivative routines.
    pub fn rhs(&self, u: &[f64]) -> Vec<f64> {
        let (ux, uxx) = match self.scheme {
            SpatialScheme::Fourier => (self.grid.fourier_d1(u), self.grid.fourier_d2(u)),
            SpatialScheme::FiniteDifference => (self.grid.d1(u), self.grid.d2(u)),
        };
        ux.iter()
            .zip(uxx.iter())
            .map(|(&x, &xx)| -self.c * x + self.nu * xx)
            .collect()
    }

    /// The RHS as a closure, ready to hand to any [`Integrator`](crate::Integrator).
    pub fn rhs_fn(&self) -> impl Fn(&[f64]) -> Vec<f64> + '_ {
        move |u| self.rhs(u)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Euler, Integrator, Midpoint, RK4};

    const N: usize = 64;
    const K: f64 = 3.0;
    const C: f64 = 1.0;
    const NU: f64 = 0.05;

    const T_END: f64 = 0.1;
    const DT: f64 = 0.001;
    /// `T_END / DT` is exactly 100, so no final step is ever shortened.
    const STEPS: usize = 100;

    /// u(x) = sin(k x).
    fn wave(g: &PeriodicGrid, k: f64) -> Vec<f64> {
        g.x().iter().map(|&x| (k * x).sin()).collect()
    }

    /// The exact RHS of u = sin(k x):
    /// `-c k cos(k x) - nu k^2 sin(k x)`.
    fn exact_rhs(g: &PeriodicGrid, k: f64) -> Vec<f64> {
        g.x()
            .iter()
            .map(|&x| -C * k * (k * x).cos() - NU * k * k * (k * x).sin())
            .collect()
    }

    /// The exact solution `exp(-nu k^2 t) sin(k (x - c t))`.
    fn exact_solution(g: &PeriodicGrid, k: f64, t: f64) -> Vec<f64> {
        let decay = (-NU * k * k * t).exp();
        g.x()
            .iter()
            .map(|&x| decay * (k * (x - C * t)).sin())
            .collect()
    }

    fn max_abs_error(got: &[f64], want: &[f64]) -> f64 {
        got.iter()
            .zip(want.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    }

    /// March `u' = F(u)` for `STEPS` steps of size `DT` (an exact whole number).
    fn integrate<I, F>(method: &I, rhs: &F, u0: &[f64]) -> Vec<f64>
    where
        I: Integrator,
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let mut u = u0.to_vec();
        for _ in 0..STEPS {
            u = method.step(&u, DT, rhs);
        }
        u
    }

    fn check_step_count() {
        assert!(
            (STEPS as f64 * DT - T_END).abs() < 1e-12,
            "T_END / DT must be a whole number"
        );
    }

    #[test]
    fn fourier_rhs_matches_the_exact_single_wave_rhs() {
        let g = PeriodicGrid::new(N);
        let model = AdvectionDiffusion::new(&g, C, NU, SpatialScheme::Fourier);
        let err = max_abs_error(&model.rhs(&wave(&g, K)), &exact_rhs(&g, K));
        println!("n = {N}, Fourier RHS max error = {err:e}");
        assert!(err < 1e-10, "Fourier RHS error = {err}");
    }

    #[test]
    fn fd_rhs_has_the_expected_second_order_error() {
        let exact = |n: usize| {
            let g = PeriodicGrid::new(n);
            let model = AdvectionDiffusion::new(&g, C, NU, SpatialScheme::FiniteDifference);
            max_abs_error(&model.rhs(&wave(&g, K)), &exact_rhs(&g, K))
        };
        let err_64 = exact(64);
        let err_128 = exact(128);
        let ratio = err_64 / err_128;
        println!(
            "finite-difference RHS max error: n = 64 -> {err_64:e}, n = 128 -> {err_128:e}, \
             ratio = {ratio:.3}"
        );
        // Second order: halving dx cuts the error by about four.
        assert!(err_64 < 0.1, "n = 64 FD RHS error = {err_64}");
        assert!((ratio - 4.0).abs() < 1.0, "FD error ratio = {ratio}, expected ~4");
    }

    #[test]
    fn euler_advances_a_single_fourier_wave() {
        check_step_count();
        let g = PeriodicGrid::new(N);
        let model = AdvectionDiffusion::new(&g, C, NU, SpatialScheme::Fourier);
        let rhs = model.rhs_fn();
        let u = integrate(&Euler, &rhs, &wave(&g, K));
        let err = max_abs_error(&u, &exact_solution(&g, K, T_END));
        println!("Euler single-wave max error = {err:e}");
        assert!(err < 1e-2, "Euler single-wave error = {err}");
    }

    #[test]
    fn midpoint_advances_a_single_fourier_wave() {
        check_step_count();
        let g = PeriodicGrid::new(N);
        let model = AdvectionDiffusion::new(&g, C, NU, SpatialScheme::Fourier);
        let rhs = model.rhs_fn();
        let u = integrate(&Midpoint, &rhs, &wave(&g, K));
        let err = max_abs_error(&u, &exact_solution(&g, K, T_END));
        println!("Midpoint single-wave max error = {err:e}");
        assert!(err < 1e-3, "Midpoint single-wave error = {err}");
    }

    #[test]
    fn rk4_advances_a_single_fourier_wave() {
        check_step_count();
        let g = PeriodicGrid::new(N);
        let model = AdvectionDiffusion::new(&g, C, NU, SpatialScheme::Fourier);
        let rhs = model.rhs_fn();
        let u = integrate(&RK4, &rhs, &wave(&g, K));
        let err = max_abs_error(&u, &exact_solution(&g, K, T_END));
        println!("RK4 single-wave max error = {err:e}");
        assert!(err < 1e-6, "RK4 single-wave error = {err}");
    }

    #[test]
    fn the_three_steppers_rank_rk4_better_than_midpoint_better_than_euler() {
        check_step_count();
        let g = PeriodicGrid::new(N);
        let model = AdvectionDiffusion::new(&g, C, NU, SpatialScheme::Fourier);
        let rhs = model.rhs_fn();
        let u0 = wave(&g, K);
        let exact = exact_solution(&g, K, T_END);

        let euler = max_abs_error(&integrate(&Euler, &rhs, &u0), &exact);
        let midpoint = max_abs_error(&integrate(&Midpoint, &rhs, &u0), &exact);
        let rk4 = max_abs_error(&integrate(&RK4, &rhs, &u0), &exact);

        assert!(rk4 < midpoint, "RK4 {rk4} should beat midpoint {midpoint}");
        assert!(midpoint < euler, "midpoint {midpoint} should beat Euler {euler}");
    }
}
