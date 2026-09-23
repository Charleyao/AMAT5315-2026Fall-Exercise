//! `field` — write a 2-D incompressible velocity field to stdout as JSON.
//!
//! Strictly follows `week4/field.design.toml`:
//!
//! ```text
//! field taylor-green --n 64 --nu 0.1 --t 1
//! field random --n 128 --seed 2026 --k-min 2 --k-max 6
//! ```
//!
//! stdout is one JSON object with `case`, `n`, `seed`, `k_band`, `u`, `v`; the
//! arrays are flat, row-major, `idx = iy*n + ix`. There are no unstated defaults:
//! `--n` is required for both cases, `--t` defaults to 0, `--nu` is required when
//! `--t > 0`, and `random` requires `--seed`, `--k-min`, `--k-max`.
//!
//! Debug/usage text goes to stderr so stdout stays machine-readable.

use std::env;
use std::f64::consts::PI;
use std::process::exit;

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use week4::grid2d::PeriodicGrid2D;

// =================================================================================
// Taylor-Green
// =================================================================================

/// Exact Taylor-Green field at time `t`:
/// `u = cos x sin y exp(-2 nu t)`, `v = -sin x cos y exp(-2 nu t)`.
fn taylor_green(n: usize, t: f64, nu: f64) -> (Vec<f64>, Vec<f64>) {
    let g = PeriodicGrid2D::new(n);
    let decay = (-2.0 * nu * t).exp();
    let mut u = vec![0.0; g.len()];
    let mut v = vec![0.0; g.len()];
    for iy in 0..n {
        let y = g.point(iy);
        for ix in 0..n {
            let x = g.point(ix);
            let idx = g.index(ix, iy);
            u[idx] = x.cos() * y.sin() * decay;
            v[idx] = -x.sin() * y.cos() * decay;
        }
    }
    (u, v)
}

// =================================================================================
// Random vorticity field
// =================================================================================

/// SplitMix64 step: a small deterministic generator so the phases depend only on
/// the seed, never on `n`.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform in `[0, 1)` from 53 random bits.
fn uniform01(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

/// Signed integer wavenumber `k` as an index in `0..n` (FFT order).
fn mode_index(k: i64, n: usize) -> usize {
    if k >= 0 {
        k as usize
    } else {
        (k + n as i64) as usize
    }
}

/// The canonical list of retained modes and their phases.
///
/// The enumeration depends only on `(seed, k_min, k_max)`, not on `n`: it walks
/// the independent half of the integer wavenumber plane in a fixed order, so a
/// mode present at two resolutions receives the same phase. The conjugate partner
/// of each mode gets the conjugate coefficient.
fn random_phases(seed: i64, k_min: i64, k_max: i64) -> Vec<(i64, i64, f64)> {
    let mut state = seed as u64;
    let mut modes = Vec::new();
    let k_lo = k_min as f64;
    let k_hi = k_max as f64;
    for ky in 0..=k_max {
        for kx in -k_max..=k_max {
            let r = ((kx * kx + ky * ky) as f64).sqrt();
            if r < k_lo || r > k_hi {
                continue;
            }
            if kx == 0 && ky == 0 {
                continue; // the mean mode carries no vorticity here
            }
            // Independent half: upper half plane, plus the +x axis.
            if !(ky > 0 || (ky == 0 && kx > 0)) {
                continue;
            }
            let phase = 2.0 * PI * uniform01(&mut state);
            modes.push((kx, ky, phase));
        }
    }
    modes
}

/// 2-D inverse FFT (unnormalised forward convention), divided by `n^2`.
fn ifft2(spectrum: &[Complex<f64>], n: usize) -> Vec<f64> {
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n);
    let mut a = spectrum.to_vec();

    // Transform along x (contiguous rows).
    for iy in 0..n {
        ifft.process(&mut a[iy * n..(iy + 1) * n]);
    }
    // Transform along y (strided columns).
    let mut col = vec![Complex::new(0.0, 0.0); n];
    for ix in 0..n {
        for iy in 0..n {
            col[iy] = a[iy * n + ix];
        }
        ifft.process(&mut col);
        for iy in 0..n {
            a[iy * n + ix] = col[iy];
        }
    }

    let norm = 1.0 / (n * n) as f64;
    a.iter().map(|c| c.re * norm).collect()
}

/// Random vorticity field with equal-amplitude modes in the band
/// `k_min <= |k| <= k_max`, recovered as a divergence-free velocity through the
/// streamfunction, then rescaled so `E(0) = 0.5 * mean(u^2 + v^2) = 0.5`.
fn random_field(n: usize, seed: i64, k_min: i64, k_max: i64) -> (Vec<f64>, Vec<f64>) {
    let g = PeriodicGrid2D::new(n);
    let size = g.len();
    let mut omega_hat = vec![Complex::new(0.0, 0.0); size];

    for (kx, ky, phase) in random_phases(seed, k_min, k_max) {
        let coeff = Complex::from_polar(1.0, phase);
        let (ix, iy) = (mode_index(kx, n), mode_index(ky, n));
        omega_hat[g.index(ix, iy)] = coeff;
        let (mxi, myi) = (mode_index(-kx, n), mode_index(-ky, n));
        omega_hat[g.index(mxi, myi)] = coeff.conj();
    }

    // Streamfunction: psi_hat = omega_hat / k^2, then u_hat = i ky psi_hat,
    // v_hat = -i kx psi_hat. Both are exactly divergence-free:
    // i kx u_hat + i ky v_hat = 0.
    let ks = g.wavenumbers();
    let mut u_hat = vec![Complex::new(0.0, 0.0); size];
    let mut v_hat = vec![Complex::new(0.0, 0.0); size];
    for iy in 0..n {
        let ky = ks[iy];
        for ix in 0..n {
            let kx = ks[ix];
            let k2 = kx * kx + ky * ky;
            if k2 > 0.0 {
                let w = omega_hat[g.index(ix, iy)];
                u_hat[g.index(ix, iy)] = Complex::new(0.0, ky) * w / k2;
                v_hat[g.index(ix, iy)] = Complex::new(0.0, -kx) * w / k2;
            }
        }
    }

    let u = ifft2(&u_hat, n);
    let v = ifft2(&v_hat, n);
    let mean = u
        .iter()
        .zip(v.iter())
        .map(|(a, b)| a * a + b * b)
        .sum::<f64>()
        / size as f64;
    let scale = 1.0 / mean.sqrt();
    (
        u.iter().map(|x| x * scale).collect(),
        v.iter().map(|x| x * scale).collect(),
    )
}

// =================================================================================
// JSON output
// =================================================================================

fn json_array(values: &[f64]) -> String {
    let mut s = String::with_capacity(values.len() * 24 + 2);
    s.push('[');
    for (i, v) in values.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("{v:.17e}"));
    }
    s.push(']');
    s
}

// =================================================================================
// Argument parsing
// =================================================================================

const USAGE: &str = "\
usage:
  field taylor-green --n N [--nu NU] [--t T]
  field random --n N --seed S --k-min A --k-max B

  taylor-green: t defaults to 0; --nu is required when t > 0.
  random:       seed, k-min and k-max are required.";

fn flag(args: &[String], name: &str) -> Result<Option<String>, String> {
    match args.iter().position(|a| a == name) {
        None => Ok(None),
        Some(i) => args
            .get(i + 1)
            .cloned()
            .map(Some)
            .ok_or_else(|| format!("missing value for {name}")),
    }
}

fn parse<T: std::str::FromStr>(args: &[String], name: &str) -> Result<Option<T>, String> {
    flag(args, name)?
        .map(|raw| raw.parse::<T>().map_err(|_| format!("bad value for {name}: {raw:?}")))
        .transpose()
}

fn required<T: std::str::FromStr>(args: &[String], name: &str) -> Result<T, String> {
    parse::<T>(args, name)?.ok_or_else(|| format!("missing required {name}"))
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    let sub = args.first().ok_or("missing subcommand")?;
    let rest = &args[1..];

    match sub.as_str() {
        "taylor-green" => {
            let n: usize = required(rest, "--n")?;
            let t: f64 = parse(rest, "--t")?.unwrap_or(0.0);
            let nu = parse::<f64>(rest, "--nu")?;
            if t > 0.0 && nu.is_none() {
                return Err("taylor-green: --nu is required when --t > 0".into());
            }
            let (u, v) = taylor_green(n, t, nu.unwrap_or(0.0));
            println!(
                "{{\"case\":\"taylor-green\",\"n\":{n},\"seed\":null,\"k_band\":null,\
                 \"u\":{},\"v\":{}}}",
                json_array(&u),
                json_array(&v)
            );
        }
        "random" => {
            let n: usize = required(rest, "--n")?;
            let seed: i64 = required(rest, "--seed")?;
            let k_min: i64 = required(rest, "--k-min")?;
            let k_max: i64 = required(rest, "--k-max")?;
            if k_min < 0 || k_max < k_min {
                return Err(format!("random: need 0 <= k-min <= k-max, got {k_min}..{k_max}"));
            }
            let (u, v) = random_field(n, seed, k_min, k_max);
            println!(
                "{{\"case\":\"random\",\"n\":{n},\"seed\":{seed},\"k_band\":[{k_min},{k_max}],\
                 \"u\":{},\"v\":{}}}",
                json_array(&u),
                json_array(&v)
            );
        }
        other => return Err(format!("unknown subcommand {other:?}")),
    }
    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("field: {err}\n\n{USAGE}");
        exit(2);
    }
}

// =================================================================================
// Tests
// =================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taylor_green_arrays_have_n_squared_entries() {
        let n = 8;
        let (u, v) = taylor_green(n, 0.0, 0.0);
        assert_eq!(u.len(), n * n);
        assert_eq!(v.len(), n * n);
    }

    #[test]
    fn taylor_green_hits_known_grid_points() {
        let n = 8;
        let g = PeriodicGrid2D::new(n);
        let (u, v) = taylor_green(n, 0.0, 0.0);
        // (x, y) = (0, pi/2): u = 1, v = 0.
        let i = g.index(0, 2);
        assert!((u[i] - 1.0).abs() < 1e-12, "u = {}", u[i]);
        assert!(v[i].abs() < 1e-12, "v = {}", v[i]);
        // (x, y) = (pi/2, 0): u = 0, v = -1.
        let i = g.index(2, 0);
        assert!(u[i].abs() < 1e-12, "u = {}", u[i]);
        assert!((v[i] + 1.0).abs() < 1e-12, "v = {}", v[i]);
    }

    #[test]
    fn taylor_green_decays_by_exp_minus_2_nu_t() {
        let n = 8;
        let nu = 0.1;
        let t: f64 = 1.3;
        let base = taylor_green(n, 0.0, 0.0);
        let decay = (-2.0 * nu * t).exp();
        let (u, v) = taylor_green(n, t, nu);
        for i in 0..n * n {
            assert!((u[i] - base.0[i] * decay).abs() < 1e-12);
            assert!((v[i] - base.1[i] * decay).abs() < 1e-12);
        }
    }

    #[test]
    fn taylor_green_uses_row_major_flattening() {
        let n = 6;
        let g = PeriodicGrid2D::new(n);
        let (u, v) = taylor_green(n, 0.0, 0.0);
        for iy in 0..n {
            for ix in 0..n {
                let idx = g.index(ix, iy); // == iy*n + ix
                assert_eq!(idx, iy * n + ix);
                let x = g.point(ix);
                let y = g.point(iy);
                assert!((u[idx] - x.cos() * y.sin()).abs() < 1e-12);
                assert!((v[idx] + x.sin() * y.cos()).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn random_field_is_reproducible() {
        let a = random_field(32, 2026, 2, 6);
        let b = random_field(32, 2026, 2, 6);
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, b.1);
    }

    #[test]
    fn random_field_changes_with_seed() {
        let a = random_field(32, 2026, 2, 6);
        let b = random_field(32, 2027, 2, 6);
        let diff: f64 = a
            .0
            .iter()
            .zip(b.0.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max);
        assert!(diff > 1e-6, "different seeds gave the same field");
    }

    #[test]
    fn random_field_has_n_squared_entries() {
        let n = 32;
        let (u, v) = random_field(n, 2026, 2, 6);
        assert_eq!(u.len(), n * n);
        assert_eq!(v.len(), n * n);
    }

    #[test]
    fn random_field_energy_is_one_half() {
        let n = 32;
        let (u, v) = random_field(n, 2026, 2, 6);
        let energy = 0.5 * u.iter().zip(v.iter()).map(|(a, b)| a * a + b * b).sum::<f64>()
            / (n * n) as f64;
        println!("random E(0) = {energy:.16}");
        assert!((energy - 0.5).abs() < 1e-12, "E(0) = {energy}");
    }

    #[test]
    fn random_field_is_divergence_free() {
        // Independent check with the Part 1 spectral derivative: du/dx along the
        // rows, dv/dy down the columns.
        let n = 32;
        let (u, v) = random_field(n, 2026, 2, 6);
        let g = week4::grid::PeriodicGrid::new(n);
        let mut div = vec![0.0; n * n];
        for iy in 0..n {
            let row: Vec<f64> = u[iy * n..(iy + 1) * n].to_vec();
            let du = g.fourier_d1(&row);
            for ix in 0..n {
                div[iy * n + ix] += du[ix];
            }
        }
        for ix in 0..n {
            let col: Vec<f64> = (0..n).map(|iy| v[iy * n + ix]).collect();
            let dv = g.fourier_d1(&col);
            for iy in 0..n {
                div[iy * n + ix] += dv[iy];
            }
        }
        let max = div.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        println!("random max |div u| = {max:.3e}");
        assert!(max < 1e-10, "divergence {max} is not round-off small");
    }

    #[test]
    fn random_phases_do_not_depend_on_n() {
        // The enumeration has no n argument, so the same seed/band gives the same
        // phases at any resolution; check the mode set matches the band.
        let modes = random_phases(2026, 2, 6);
        assert!(!modes.is_empty());
        for (kx, ky, phase) in &modes {
            let r = (((*kx) * (*kx) + (*ky) * (*ky)) as f64).sqrt();
            assert!(r >= 2.0 && r <= 6.0);
            assert!(*ky > 0 || (*ky == 0 && *kx > 0));
            assert!(*phase >= 0.0 && *phase < 2.0 * PI);
        }
    }
}
