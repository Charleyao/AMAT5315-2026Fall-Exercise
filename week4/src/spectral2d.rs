//! Two-dimensional Fourier (spectral) operators on `[0, 2*pi)^2`.
//!
//! The wavenumber ordering is exactly Part 1's (`0, 1, ..., n/2-1, -n/2, ...,
//! -1` in each axis, see [`crate::grid::fft_wavenumbers`]). The forward and
//! inverse 2-D FFTs are unnormalised in `rustfft`, so [`Spectral2D::inverse`]
//! divides by `n*n`. Odd first derivatives keep Part 1's Nyquist convention: the
//! `kx = -n/2` (resp. `ky = -n/2`) mode is annihilated by `dx` (resp. `dy`).
//!
//! Fields are flat, row-major, `idx = iy*n + ix`.

use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use crate::grid::fft_wavenumbers;
use crate::grid2d::PeriodicGrid2D;

/// Spectral machinery on an `n x n` periodic grid.
pub struct Spectral2D {
    grid: PeriodicGrid2D,
    fft: Arc<dyn Fft<f64>>,
    ifft: Arc<dyn Fft<f64>>,
    /// Integer wavenumber of FFT index `i`, the same list for both axes.
    k: Vec<f64>,
}

impl Spectral2D {
    pub fn new(n: usize) -> Self {
        let mut planner = FftPlanner::new();
        Self {
            grid: PeriodicGrid2D::new(n),
            fft: planner.plan_fft_forward(n),
            ifft: planner.plan_fft_inverse(n),
            k: fft_wavenumbers(n),
        }
    }

    pub fn n(&self) -> usize {
        self.grid.n()
    }

    pub fn len(&self) -> usize {
        self.grid.len()
    }

    /// The underlying grid (coordinates, indexing, centred differences).
    pub fn grid(&self) -> &PeriodicGrid2D {
        &self.grid
    }

    /// True for the Nyquist index `i = n/2` on an even grid.
    fn is_nyquist(&self, i: usize) -> bool {
        self.n() % 2 == 0 && i == self.n() / 2
    }

    /// First-derivative multiplier `i kx`, zero on the Nyquist mode.
    fn k1x(&self, ix: usize) -> Complex<f64> {
        if self.is_nyquist(ix) {
            Complex::new(0.0, 0.0)
        } else {
            Complex::new(0.0, self.k[ix])
        }
    }

    /// First-derivative multiplier `i ky`, zero on the Nyquist mode.
    fn k1y(&self, iy: usize) -> Complex<f64> {
        if self.is_nyquist(iy) {
            Complex::new(0.0, 0.0)
        } else {
            Complex::new(0.0, self.k[iy])
        }
    }

    /// Forward 2-D FFT of a real field (unnormalised): rows then columns.
    pub fn forward(&self, f: &[f64]) -> Vec<Complex<f64>> {
        assert_eq!(f.len(), self.len(), "field has {} entries, expected {}", f.len(), self.len());
        let n = self.n();
        let mut a: Vec<Complex<f64>> =
            f.iter().map(|&v| Complex::new(v, 0.0)).collect();
        for iy in 0..n {
            self.fft.process(&mut a[iy * n..(iy + 1) * n]);
        }
        let mut col = vec![Complex::new(0.0, 0.0); n];
        for ix in 0..n {
            for iy in 0..n {
                col[iy] = a[iy * n + ix];
            }
            self.fft.process(&mut col);
            for iy in 0..n {
                a[iy * n + ix] = col[iy];
            }
        }
        a
    }

    /// Inverse 2-D FFT, normalised by `1/n^2`, taking the real part.
    pub fn inverse(&self, hat: &[Complex<f64>]) -> Vec<f64> {
        assert_eq!(hat.len(), self.len(), "spectrum has {} entries, expected {}", hat.len(), self.len());
        let n = self.n();
        let mut a = hat.to_vec();
        for iy in 0..n {
            self.ifft.process(&mut a[iy * n..(iy + 1) * n]);
        }
        let mut col = vec![Complex::new(0.0, 0.0); n];
        for ix in 0..n {
            for iy in 0..n {
                col[iy] = a[iy * n + ix];
            }
            self.ifft.process(&mut col);
            for iy in 0..n {
                a[iy * n + ix] = col[iy];
            }
        }
        let norm = 1.0 / (n * n) as f64;
        a.iter().map(|c| c.re * norm).collect()
    }

    /// Multiply every Fourier mode by `mult(ix, iy)`, then invert.
    fn apply<G>(&self, f: &[f64], mult: G) -> Vec<f64>
    where
        G: Fn(usize, usize) -> Complex<f64>,
    {
        let n = self.n();
        let mut hat = self.forward(f);
        for iy in 0..n {
            for ix in 0..n {
                let i = iy * n + ix;
                hat[i] *= mult(ix, iy);
            }
        }
        self.inverse(&hat)
    }

    /// `df/dx`, multiplier `i kx` (Nyquist annihilated).
    pub fn dx(&self, f: &[f64]) -> Vec<f64> {
        self.apply(f, |ix, _| self.k1x(ix))
    }

    /// `df/dy`, multiplier `i ky` (Nyquist annihilated).
    pub fn dy(&self, f: &[f64]) -> Vec<f64> {
        self.apply(f, |_, iy| self.k1y(iy))
    }

    /// `d2f/dx2`, multiplier `-kx^2`.
    pub fn dxx(&self, f: &[f64]) -> Vec<f64> {
        self.apply(f, |ix, _| Complex::new(-self.k[ix] * self.k[ix], 0.0))
    }

    /// `d2f/dy2`, multiplier `-ky^2`.
    pub fn dyy(&self, f: &[f64]) -> Vec<f64> {
        self.apply(f, |_, iy| Complex::new(-self.k[iy] * self.k[iy], 0.0))
    }

    /// `d2f/dxdy`, multiplier `(i kx)(i ky) = -kx ky` (Nyquist annihilated on
    /// either axis).
    pub fn dxdy(&self, f: &[f64]) -> Vec<f64> {
        self.apply(f, |ix, iy| self.k1x(ix) * self.k1y(iy))
    }

    /// `laplacian(f)`, multiplier `-(kx^2 + ky^2)`.
    pub fn laplacian(&self, f: &[f64]) -> Vec<f64> {
        self.apply(f, |ix, iy| {
            Complex::new(-(self.k[ix] * self.k[ix] + self.k[iy] * self.k[iy]), 0.0)
        })
    }

    // -------------------------------------------------------------------------
    // Two-thirds dealiasing
    // -------------------------------------------------------------------------

    /// Two-thirds cutoff `floor(n/3)` in each signed wavenumber.
    pub fn k_cut(&self) -> usize {
        self.n() / 3
    }

    /// Whether FFT mode `(ix, iy)` survives the two-thirds rule:
    /// `|kx| <= k_cut` **and** `|ky| <= k_cut`, using the signed wavenumbers.
    /// The Nyquist mode is always outside (for `n >= 2`, `floor(n/3) < n/2`).
    pub fn is_retained(&self, ix: usize, iy: usize) -> bool {
        let cut = self.k_cut() as f64;
        self.k[ix].abs() <= cut && self.k[iy].abs() <= cut
    }

    /// Zero every Fourier coefficient outside the two-thirds band, in place.
    pub fn dealias_hat(&self, hat: &mut [Complex<f64>]) {
        assert_eq!(hat.len(), self.len(), "spectrum has {} entries, expected {}", hat.len(), self.len());
        let n = self.n();
        for iy in 0..n {
            for ix in 0..n {
                if !self.is_retained(ix, iy) {
                    hat[iy * n + ix] = Complex::new(0.0, 0.0);
                }
            }
        }
    }

    /// Real-space two-thirds dealiasing: FFT, cutoff, inverse FFT.
    pub fn dealias(&self, f: &[f64]) -> Vec<f64> {
        let mut hat = self.forward(f);
        self.dealias_hat(&mut hat);
        self.inverse(&hat)
    }

    // -------------------------------------------------------------------------
    // Incompressible vorticity-equation RHS
    // -------------------------------------------------------------------------

    /// `omega_t = -(u omega_x + v omega_y) + nu laplacian(omega)`.
    ///
    /// Every call re-derives `psi`, `u`, `v`, the gradients and the nonlinear
    /// term from the `omega` it is handed, so an integrator can drive it straight
    /// with `rk4.step(omega, dt, |stage| self.vorticity_rhs(stage, nu))` without
    /// any state cached between stages.
    pub fn vorticity_rhs(&self, omega: &[f64], nu: f64) -> Vec<f64> {
        // 1. the carried vorticity itself must obey the cutoff.
        let omega = self.dealias(omega);
        // 2-3. psi from -laplacian(psi) = omega, then u = psi_y, v = -psi_x.
        let (u, v) = self.velocity_from_vorticity(&omega);
        // 4. vorticity gradients.
        let omega_x = self.dx(&omega);
        let omega_y = self.dy(&omega);
        // 5. pointwise nonlinear product, computed in real space.
        let adv: Vec<f64> = u
            .iter()
            .zip(omega_x.iter())
            .zip(v.iter().zip(omega_y.iter()))
            .map(|((u, ox), (v, oy))| u * ox + v * oy)
            .collect();
        // 6. dealias the nonlinear product.
        let adv = self.dealias(&adv);
        // 7-8. diffusion and the full RHS.
        let diff = self.laplacian(&omega);
        let rhs: Vec<f64> = adv
            .iter()
            .zip(diff.iter())
            .map(|(a, d)| -a + nu * d)
            .collect();
        // 9. keep the RHS inside the band as well.
        self.dealias(&rhs)
    }

    // -------------------------------------------------------------------------
    // Diagnostics
    // -------------------------------------------------------------------------

    /// Energy `E = 0.5 * mean(u^2 + v^2)`.
    pub fn energy(&self, u: &[f64], v: &[f64]) -> f64 {
        assert_eq!(u.len(), self.len());
        assert_eq!(v.len(), self.len());
        0.5 * u.iter().zip(v.iter()).map(|(a, b)| a * a + b * b).sum::<f64>()
            / self.len() as f64
    }

    /// Enstrophy `Z = 0.5 * mean(omega^2)`.
    pub fn enstrophy(&self, omega: &[f64]) -> f64 {
        assert_eq!(omega.len(), self.len());
        0.5 * omega.iter().map(|w| w * w).sum::<f64>() / self.len() as f64
    }

    /// Solve `-laplacian(psi) = omega` on the periodic square:
    /// `psi_hat(k) = omega_hat(k) / (kx^2 + ky^2)`, with `psi_hat(0,0) = 0`
    /// (the streamfunction's additive constant is arbitrary).
    pub fn poisson(&self, omega: &[f64]) -> Vec<f64> {
        let n = self.n();
        let mut hat = self.forward(omega);
        for iy in 0..n {
            for ix in 0..n {
                let k2 = self.k[ix] * self.k[ix] + self.k[iy] * self.k[iy];
                let i = iy * n + ix;
                hat[i] = if k2 == 0.0 {
                    Complex::new(0.0, 0.0)
                } else {
                    hat[i] / k2
                };
            }
        }
        self.inverse(&hat)
    }

    /// Velocity from vorticity with the course convention `u = psi_y`,
    /// `v = -psi_x`, where `-laplacian(psi) = omega`.
    pub fn velocity_from_vorticity(&self, omega: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let psi = self.poisson(omega);
        let u = self.dy(&psi);
        let v: Vec<f64> = self.dx(&psi).iter().map(|x| -x).collect();
        (u, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn forward_inverse_round_trip_is_roundoff() {
        let s = Spectral2D::new(32);
        let g = s.grid();
        let n = s.n();
        let mut f = vec![0.0; s.len()];
        for iy in 0..n {
            for ix in 0..n {
                let (x, y) = (g.point(ix), g.point(iy));
                f[g.index(ix, iy)] = (3.0 * x).sin() * (2.0 * y).cos()
                    + 0.5 * (x + y).cos()
                    - 0.25 * (x * 4.0).sin() * (y * 3.0).sin();
            }
        }
        let back = s.inverse(&s.forward(&f));
        let err = max_abs_err(&f, &back);
        println!("2D FFT round-trip max error = {err:.3e}");
        assert!(err < 1e-12, "round-trip error = {err}");
    }

    #[test]
    fn fourier_derivatives_of_a_represented_wave_are_roundoff() {
        let s = Spectral2D::new(32);
        let g = s.grid();
        let f = g_field(g);
        let dx = max_abs_err(&s.dx(&f), &g_exact(g, "dx"));
        let dxx = max_abs_err(&s.dxx(&f), &g_exact(g, "dxx"));
        let dxdy = max_abs_err(&s.dxdy(&f), &g_exact(g, "dxdy"));
        let lap = max_abs_err(&s.laplacian(&f), &g_exact(g, "laplacian"));
        println!("Fourier n=32: dx {dx:.3e}, dxx {dxx:.3e}, dxdy {dxdy:.3e}, laplacian {lap:.3e}");
        assert!(dx < 1e-10, "dx error {dx}");
        assert!(dxx < 1e-10, "dxx error {dxx}");
        assert!(dxdy < 1e-10, "dxdy error {dxdy}");
        assert!(lap < 1e-10, "laplacian error {lap}");
    }

    #[test]
    fn prints_the_derivative_comparison_table() {
        // Centred differences at 32 and 64, and Fourier at 32, for the same wave.
        let fd = |n: usize| {
            let g = PeriodicGrid2D::new(n);
            let f = g_field(&g);
            [
                max_abs_err(&g.dx(&f), &g_exact(&g, "dx")),
                max_abs_err(&g.dxx(&f), &g_exact(&g, "dxx")),
                max_abs_err(&g.dxdy(&f), &g_exact(&g, "dxdy")),
                max_abs_err(&g.laplacian(&f), &g_exact(&g, "laplacian")),
            ]
        };
        let s = Spectral2D::new(32);
        let gf = g_field(s.grid());
        let fourier = [
            max_abs_err(&s.dx(&gf), &g_exact(s.grid(), "dx")),
            max_abs_err(&s.dxx(&gf), &g_exact(s.grid(), "dxx")),
            max_abs_err(&s.dxdy(&gf), &g_exact(s.grid(), "dxdy")),
            max_abs_err(&s.laplacian(&gf), &g_exact(s.grid(), "laplacian")),
        ];
        let e32 = fd(32);
        let e64 = fd(64);
        println!(
            "{:<12}{:>12}{:>12}{:>10}{:>14}",
            "Derivative", "FD n=32", "FD n=64", "ratio", "Fourier n=32"
        );
        for (i, name) in ["dx", "dxx", "dxdy", "laplacian"].iter().enumerate() {
            println!(
                "{:<12}{:>12.5}{:>12.5}{:>10.3}{:>14.3e}",
                name,
                e32[i],
                e64[i],
                e32[i] / e64[i],
                fourier[i]
            );
        }
        for i in 0..4 {
            assert!((e32[i] / e64[i] - 4.0).abs() < 0.5);
            assert!(fourier[i] < 1e-10);
        }
    }

    #[test]
    fn poisson_ignores_the_mean_mode() {
        // psi_hat(0,0) = 0, so a constant vorticity must give psi = 0.
        let s = Spectral2D::new(16);
        let omega = vec![2.5; s.len()];
        let psi = s.poisson(&omega);
        let max = psi.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        println!("poisson(constant) max |psi| = {max:.3e} (psi_hat(0,0) = 0)");
        assert!(max < 1e-12);
    }

    #[test]
    fn taylor_green_omega_recovers_velocity() {
        let n = 64;
        let s = Spectral2D::new(n);
        let g = s.grid();
        let mut omega = vec![0.0; s.len()];
        let mut u_exact = vec![0.0; s.len()];
        let mut v_exact = vec![0.0; s.len()];
        for iy in 0..n {
            for ix in 0..n {
                let (x, y) = (g.point(ix), g.point(iy));
                let i = g.index(ix, iy);
                omega[i] = -2.0 * x.cos() * y.cos();
                u_exact[i] = x.cos() * y.sin();
                v_exact[i] = -x.sin() * y.cos();
            }
        }
        let (u, v) = s.velocity_from_vorticity(&omega);
        let err_u = max_abs_err(&u, &u_exact);
        let err_v = max_abs_err(&v, &v_exact);
        let div: Vec<f64> = s
            .dx(&u)
            .iter()
            .zip(s.dy(&v).iter())
            .map(|(a, b)| a + b)
            .collect();
        let max_div = div.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        println!(
            "Taylor-Green omega -> u: err_u {err_u:.3e}, err_v {err_v:.3e}, \
             max |div| {max_div:.3e}"
        );
        assert!(err_u < 1e-10, "u error {err_u}");
        assert!(err_v < 1e-10, "v error {err_v}");
        assert!(max_div < 1e-10, "divergence {max_div}");
    }

    // ---- two-thirds dealiasing ----------------------------------------------

    /// Signed wavenumber `k` as an FFT index in `0..n`.
    fn mode_idx(k: i64, n: usize) -> usize {
        if k >= 0 {
            k as usize
        } else {
            (k + n as i64) as usize
        }
    }

    /// Largest `|hat|` outside the two-thirds band.
    fn max_outside(s: &Spectral2D, hat: &[Complex<f64>]) -> f64 {
        let n = s.n();
        let mut m = 0.0f64;
        for iy in 0..n {
            for ix in 0..n {
                if !s.is_retained(ix, iy) {
                    m = m.max(hat[iy * n + ix].norm());
                }
            }
        }
        m
    }

    #[test]
    fn two_thirds_cutoff_uses_signed_wavenumbers() {
        let n = 12; // floor(n/3) = 4
        let s = Spectral2D::new(n);
        assert_eq!(s.k_cut(), 4);

        let put = |hat: &mut Vec<Complex<f64>>, kx: i64, ky: i64, v: f64| {
            hat[mode_idx(ky, n) * n + mode_idx(kx, n)] = Complex::new(v, 0.0);
        };
        let get = |hat: &Vec<Complex<f64>>, kx: i64, ky: i64| {
            hat[mode_idx(ky, n) * n + mode_idx(kx, n)]
        };

        let mut hat = vec![Complex::new(0.0, 0.0); s.len()];
        put(&mut hat, 4, 4, 1.0); // retained
        put(&mut hat, 4, -4, 2.0); // retained
        put(&mut hat, 5, 0, 3.0); // |kx| > 4 -> removed
        put(&mut hat, 0, -5, 4.0); // |ky| > 4 -> removed
        put(&mut hat, -6, 0, 5.0); // Nyquist -> removed
        s.dealias_hat(&mut hat);

        assert_eq!(get(&hat, 4, 4), Complex::new(1.0, 0.0));
        assert_eq!(get(&hat, 4, -4), Complex::new(2.0, 0.0));
        assert_eq!(get(&hat, 5, 0), Complex::new(0.0, 0.0));
        assert_eq!(get(&hat, 0, -5), Complex::new(0.0, 0.0));
        assert_eq!(get(&hat, -6, 0), Complex::new(0.0, 0.0));
    }

    #[test]
    fn dealiasing_keeps_low_k_and_removes_high_k() {
        let n = 16; // floor(n/3) = 5
        let s = Spectral2D::new(n);
        let g = s.grid();
        assert_eq!(s.k_cut(), 5);
        let mut f = vec![0.0; s.len()];
        let mut low = vec![0.0; s.len()];
        for iy in 0..n {
            for ix in 0..n {
                let (x, y) = (g.point(ix), g.point(iy));
                let i = g.index(ix, iy);
                low[i] = (2.0 * x).sin() * (3.0 * y).cos();
                f[i] = low[i] + 0.5 * (6.0 * x).sin(); // kx = 6 > 5
            }
        }
        let d = s.dealias(&f);
        let err = max_abs_err(&d, &low);
        println!("dealiasing high-k removal error = {err:.3e}");
        assert!(err < 1e-12, "dealiased field differs from the low-k part: {err}");
    }

    // ---- vorticity RHS --------------------------------------------------------

    #[test]
    fn taylor_green_nonlinear_term_is_zero_and_rhs_is_diffusion() {
        let n = 64;
        let nu = 0.1;
        let s = Spectral2D::new(n);
        let g = s.grid();
        let mut omega = vec![0.0; s.len()];
        for iy in 0..n {
            for ix in 0..n {
                let (x, y) = (g.point(ix), g.point(iy));
                omega[g.index(ix, iy)] = -2.0 * x.cos() * y.cos();
            }
        }
        let (u, v) = s.velocity_from_vorticity(&omega);
        let omega_x = s.dx(&omega);
        let omega_y = s.dy(&omega);
        let adv: Vec<f64> = u
            .iter()
            .zip(omega_x.iter())
            .zip(v.iter().zip(omega_y.iter()))
            .map(|((u, ox), (v, oy))| u * ox + v * oy)
            .collect();
        let max_adv = adv.iter().fold(0.0f64, |m, x| m.max(x.abs()));

        let rhs = s.vorticity_rhs(&omega, nu);
        let target: Vec<f64> = omega.iter().map(|w| -2.0 * nu * w).collect();
        let max_rhs_err = max_abs_err(&rhs, &target);
        println!(
            "Taylor-Green: max |u omega_x + v omega_y| = {max_adv:.3e}, \
             max |rhs + 2 nu omega| = {max_rhs_err:.3e}"
        );
        assert!(max_adv < 1e-10, "nonlinear term {max_adv}");
        assert!(max_rhs_err < 1e-10, "rhs error {max_rhs_err}");
    }

    #[test]
    fn nonlinear_path_on_a_smooth_dealiased_field() {
        // A non-Taylor-Green field with modes near the cutoff, so the products
        // really do reach outside the band before dealiasing.
        let n = 32; // floor(n/3) = 10
        let s = Spectral2D::new(n);
        let g = s.grid();
        assert_eq!(s.k_cut(), 10);
        let mut omega = vec![0.0; s.len()];
        for iy in 0..n {
            for ix in 0..n {
                let (x, y) = (g.point(ix), g.point(iy));
                omega[g.index(ix, iy)] = (6.0 * x).sin() * (4.0 * y).cos()
                    + 0.8 * (3.0 * x).sin() * (5.0 * y).cos()
                    + 0.6 * (2.0 * x).cos() * (6.0 * y).sin();
            }
        }
        let omega = s.dealias(&omega); // carried field obeys the cutoff

        let (u, v) = s.velocity_from_vorticity(&omega);
        let omega_x = s.dx(&omega);
        let omega_y = s.dy(&omega);
        let adv: Vec<f64> = u
            .iter()
            .zip(omega_x.iter())
            .zip(v.iter().zip(omega_y.iter()))
            .map(|((u, ox), (v, oy))| u * ox + v * oy)
            .collect();
        let max_adv = adv.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let raw_outside = max_outside(&s, &s.forward(&adv));
        let dealt_outside = max_outside(&s, &s.forward(&s.dealias(&adv)));

        let rhs = s.vorticity_rhs(&omega, 0.05);
        let rhs_outside = max_outside(&s, &s.forward(&rhs));

        println!(
            "nonlinear: max |adv| = {max_adv:.3e}, raw out-of-band = {raw_outside:.3e}, \
             dealt out-of-band = {dealt_outside:.3e}, rhs out-of-band = {rhs_outside:.3e}"
        );
        assert!(max_adv > 1e-6, "nonlinear term vanished");
        assert!(raw_outside > 1e-6, "raw product had no out-of-band content");
        assert!(dealt_outside < 1e-9, "dealiased product still has out-of-band content");
        assert!(rhs_outside < 1e-9, "rhs still has out-of-band content");
    }

    #[test]
    fn taylor_green_energy_and_enstrophy() {
        let n = 64;
        let s = Spectral2D::new(n);
        let g = s.grid();
        let mut omega = vec![0.0; s.len()];
        let mut u = vec![0.0; s.len()];
        let mut v = vec![0.0; s.len()];
        for iy in 0..n {
            for ix in 0..n {
                let (x, y) = (g.point(ix), g.point(iy));
                let i = g.index(ix, iy);
                omega[i] = -2.0 * x.cos() * y.cos();
                u[i] = x.cos() * y.sin();
                v[i] = -x.sin() * y.cos();
            }
        }
        let e = s.energy(&u, &v);
        let z = s.enstrophy(&omega);
        println!("Taylor-Green E = {e:.16}, Z = {z:.16}");
        assert!((e - 0.25).abs() < 1e-12, "E = {e}");
        assert!((z - 0.5).abs() < 1e-12, "Z = {z}");
    }

    #[test]
    fn vorticity_rhs_is_stateless_in_omega() {
        // Each call re-derives psi, u, v from the omega it receives, so calling
        // the RHS on `a` after touching `b` must reproduce the first answer.
        let n = 32;
        let s = Spectral2D::new(n);
        let g = s.grid();
        let field = |kx: f64, ky: f64| {
            let mut f = vec![0.0; s.len()];
            for iy in 0..n {
                for ix in 0..n {
                    let (x, y) = (g.point(ix), g.point(iy));
                    f[g.index(ix, iy)] = (kx * x).sin() * (ky * y).cos();
                }
            }
            f
        };
        let a = field(2.0, 3.0);
        let b = field(4.0, 1.0);
        let ra1 = s.vorticity_rhs(&a, 0.05);
        let _rb = s.vorticity_rhs(&b, 0.05);
        let ra2 = s.vorticity_rhs(&a, 0.05);
        assert_eq!(ra1, ra2);
    }
}
