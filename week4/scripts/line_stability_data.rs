//! Data generator for `week4/evidence/line-stability.png` (Part 1 stability check).
//!
//! It is a crate binary (see `[[bin]]` in `Cargo.toml`) whose source lives under
//! `scripts/`, so it links the *real* library: the RK4 growth map is measured by
//! stepping the library `RK4` on `y' = lambda y` written as the equivalent 2-D
//! real system, and the pulse runs use the library Fourier advection-diffusion
//! RHS. Nothing here re-implements an integrator or an FFT derivative.
//!
//! Run from `week4/` via `scripts/line_stability.py`, or directly:
//!
//!   cd week4
//!   cargo run --release --bin line-stability-data
//!
//! Writes generated data under `artifacts/line-stability/` (gitignored) and the
//! JSON summary there too; the figure itself is drawn by `scripts/line_stability.py`.

use std::f64::consts::{FRAC_PI_2, PI};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use week4::Integrator;
use week4::RK4;
use week4::advdiff::{AdvectionDiffusion, SpatialScheme};
use week4::grid::PeriodicGrid;

// The 1-D advection-diffusion line of Part 1.
const N: usize = 64;
const C: f64 = 1.0;
const NU: f64 = 0.05;
const SIGMA: f64 = 0.35;
const X0: f64 = FRAC_PI_2;
const T_END: f64 = 6.0;
const DTS: [f64; 2] = [0.045, 0.056];

// The complex-plane grid for the measured growth map.
const NX: usize = 271;
const NY: usize = 361;
const RE_MIN: f64 = -4.2;
const RE_MAX: f64 = 1.2;
const IM_MIN: f64 = -3.6;
const IM_MAX: f64 = 3.6;

/// `|R(lambda h)|` measured with one library-RK4 step on the equivalent 2-D real
/// system `a' = Re(lambda) a - Im(lambda) b`, `b' = Im(lambda) a + Re(lambda) b`,
/// starting from `(a, b) = (1, 0)`.
fn rk4_growth(re: f64, im: f64) -> f64 {
    let rhs = |s: &[f64]| vec![re * s[0] - im * s[1], im * s[0] + re * s[1]];
    let y1 = RK4.step(&[1.0, 0.0], 1.0, rhs);
    (y1[0] * y1[0] + y1[1] * y1[1]).sqrt()
}

/// Bisection for a root of `f` in `[lo, hi]`, where `f(lo)` and `f(hi)` have
/// opposite signs.
fn bisect<F: Fn(f64) -> f64>(mut lo: f64, mut hi: f64, f: F) -> f64 {
    let f_lo = f(lo);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if (f(mid) > 0.0) == (f_lo > 0.0) {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo).abs() < 1e-13 {
            break;
        }
    }
    0.5 * (lo + hi)
}

/// Eigenvalues `lambda_k` of the line operator for every Fourier mode.
///
/// Non-Nyquist: `lambda_k = -nu k^2 - i c k`. Nyquist `k = -n/2`: the first
/// derivative multiplier is zero (that is the library's `fourier_d1` convention),
/// so `lambda = -nu k^2` is real.
fn line_lambdas(g: &PeriodicGrid) -> Vec<(f64, f64)> {
    let nyquist = -(N as f64) / 2.0;
    g.wavenumbers()
        .iter()
        .map(|&k| {
            if (k - nyquist).abs() < 1e-9 {
                (-NU * k * k, 0.0)
            } else {
                (-NU * k * k, -C * k)
            }
        })
        .collect()
}

fn max_growth(lambdas: &[(f64, f64)], dt: f64) -> f64 {
    lambdas
        .iter()
        .filter(|&&(lr, li)| lr != 0.0 || li != 0.0) // the k = 0 mode has |R| = 1 exactly
        .map(|&(lr, li)| rk4_growth(lr * dt, li * dt))
        .fold(0.0, f64::max)
}

/// The wavenumber whose mode has the largest `|R(lambda_k dt)|`.
fn dominant_k(lambdas: &[(f64, f64)], ks: &[f64], dt: f64) -> (f64, f64) {
    let mut best = (0.0, f64::NEG_INFINITY);
    for (i, &(lr, li)) in lambdas.iter().enumerate() {
        if lr == 0.0 && li == 0.0 {
            continue; // neutral k = 0 mode
        }
        let gm = rk4_growth(lr * dt, li * dt);
        if gm > best.1 {
            best = (ks[i], gm);
        }
    }
    best
}

/// Periodised Gaussian: sum over periodic images so the initial condition is
/// smooth and periodic on `[0, 2*pi)`; the result is never a truncated
/// non-periodic Gaussian.
fn periodic_gaussian(g: &PeriodicGrid) -> Vec<f64> {
    g.x()
        .iter()
        .map(|&x| {
            let mut s = 0.0;
            for j in -4..=4 {
                let d = x - X0 - 2.0 * PI * j as f64;
                s += (-d * d / (2.0 * SIGMA * SIGMA)).exp();
            }
            s
        })
        .collect()
}

fn write_f64_bin(path: &PathBuf, values: &[f64]) {
    let file = File::create(path).expect("create binary data file");
    let mut w = BufWriter::new(file);
    for v in values {
        w.write_all(&v.to_le_bytes()).expect("write f64");
    }
    w.flush().expect("flush");
}

fn fmt_f64(v: f64) -> String {
    format!("{v:.17e}")
}

fn fmt_f64_vec(v: &[f64]) -> String {
    let mut s = String::from("[");
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&fmt_f64(*x));
    }
    s.push(']');
    s
}

fn fmt_pairs(v: &[(f64, f64)]) -> String {
    let mut s = String::from("[");
    for (i, (a, b)) in v.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("[{},{}]", fmt_f64(*a), fmt_f64(*b)));
    }
    s.push(']');
    s
}

fn main() {
    // Locate week4/ from the manifest, so the generator works from any cwd.
    let week4_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data_dir = week4_dir.join("artifacts").join("line-stability");
    fs::create_dir_all(&data_dir).expect("create artifacts/line-stability/");

    // ---- A: measured RK4 growth map -----------------------------------------
    let mut growth = Vec::with_capacity(NX * NY);
    for iy in 0..NY {
        let im = IM_MIN + (IM_MAX - IM_MIN) * iy as f64 / (NY - 1) as f64;
        for ix in 0..NX {
            let re = RE_MIN + (RE_MAX - RE_MIN) * ix as f64 / (NX - 1) as f64;
            growth.push(rk4_growth(re, im));
        }
    }
    write_f64_bin(&data_dir.join("line-stability-growth.bin"), &growth);

    // Axis crossings of the RK4 region, measured via the library RK4.
    let real_crossing = bisect(-2.0, -3.0, |x| rk4_growth(x, 0.0) - 1.0);
    let imag_crossing = bisect(2.0, 3.5, |y| rk4_growth(0.0, y) - 1.0);

    // ---- line spectrum and dt_crit ------------------------------------------
    let g = PeriodicGrid::new(N);
    let ks = g.wavenumbers();
    let lambdas = line_lambdas(&g);

    // Confirm the Nyquist convention through the library itself.
    let nyquist_field: Vec<f64> = (0..N).map(|j| (PI * j as f64).cos()).collect();
    let nyquist_d1_max = g
        .fourier_d1(&nyquist_field)
        .iter()
        .fold(0.0f64, |m, x| m.max(x.abs()));

    let dt_crit = bisect(0.01, 0.09, |dt| max_growth(&lambdas, dt) - 1.0);

    let dt_stats: Vec<(f64, f64, f64)> = DTS
        .iter()
        .map(|&dt| {
            let (k, mg) = dominant_k(&lambdas, &ks, dt);
            (dt, mg, k)
        })
        .collect();

    // ---- B: Gaussian pulse runs ---------------------------------------------
    let model = AdvectionDiffusion::new(&g, C, NU, SpatialScheme::Fourier);
    let rhs = model.rhs_fn();
    let ic = periodic_gaussian(&g);

    let mut pulse_json = String::new();
    for (i, &dt) in DTS.iter().enumerate() {
        let steps = (T_END / dt).floor() as usize; // constant dt, never shortened
        let mut u = ic.clone();
        let mut hist: Vec<f64> = Vec::with_capacity((steps + 1) * N);
        let mut times = Vec::with_capacity(steps + 1);
        hist.extend_from_slice(&u);
        times.push(0.0);
        for s in 1..=steps {
            u = RK4.step(&u, dt, &rhs);
            hist.extend_from_slice(&u);
            times.push(s as f64 * dt);
        }
        let finite = u.iter().all(|v| v.is_finite());
        let max_abs = u.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let nt = times.len();
        write_f64_bin(&data_dir.join(format!("line-pulse-{dt:.3}.bin")), &hist);

        if i > 0 {
            pulse_json.push_str(",\n");
        }
        pulse_json.push_str(&format!(
            "    \"{dt:.3}\": {{\"dt\": {}, \"steps\": {}, \"nt\": {}, \
             \"finite\": {}, \"max_abs_final\": {}, \"times\": {}}}",
            fmt_f64(dt),
            steps,
            nt,
            finite,
            fmt_f64(max_abs),
            fmt_f64_vec(&times),
        ));
    }

    // ---- JSON -----------------------------------------------------------------
    let mut j = String::new();
    j.push_str("{\n");
    j.push_str(&format!("  \"nx\": {NX}, \"ny\": {NY},\n"));
    j.push_str(&format!(
        "  \"re_min\": {}, \"re_max\": {},\n",
        fmt_f64(RE_MIN),
        fmt_f64(RE_MAX)
    ));
    j.push_str(&format!(
        "  \"im_min\": {}, \"im_max\": {},\n",
        fmt_f64(IM_MIN),
        fmt_f64(IM_MAX)
    ));
    j.push_str(&format!(
        "  \"n\": {N}, \"c\": {}, \"nu\": {},\n",
        fmt_f64(C),
        fmt_f64(NU)
    ));
    j.push_str(&format!(
        "  \"sigma\": {}, \"x0\": {}, \"t_end\": {},\n",
        fmt_f64(SIGMA),
        fmt_f64(X0),
        fmt_f64(T_END)
    ));
    j.push_str(&format!(
        "  \"rk4_real_crossing\": {},\n",
        fmt_f64(real_crossing)
    ));
    j.push_str(&format!(
        "  \"rk4_imag_crossing\": {},\n",
        fmt_f64(imag_crossing)
    ));
    j.push_str(&format!("  \"dt_crit\": {},\n", fmt_f64(dt_crit)));
    j.push_str(&format!(
        "  \"nyquist_d1_max\": {},\n",
        fmt_f64(nyquist_d1_max)
    ));
    j.push_str("  \"wavenumbers\": ");
    j.push_str(&fmt_f64_vec(&ks));
    j.push_str(",\n");

    j.push_str("  \"spectrum\": {\n");
    for (i, &dt) in DTS.iter().enumerate() {
        let pairs: Vec<(f64, f64)> = lambdas.iter().map(|&(lr, li)| (lr * dt, li * dt)).collect();
        j.push_str(&format!("    \"{dt:.3}\": {}", fmt_pairs(&pairs)));
        j.push_str(if i + 1 == DTS.len() { "\n" } else { ",\n" });
    }
    j.push_str("  },\n");

    j.push_str("  \"max_growth\": {\n");
    for (i, (dt, mg, k)) in dt_stats.iter().enumerate() {
        j.push_str(&format!(
            "    \"{dt:.3}\": {{\"max\": {}, \"dominant_k\": {}}}",
            fmt_f64(*mg),
            fmt_f64(*k)
        ));
        j.push_str(if i + 1 == dt_stats.len() { "\n" } else { ",\n" });
    }
    j.push_str("  },\n");

    j.push_str("  \"pulse\": {\n");
    j.push_str(&pulse_json);
    j.push_str("\n  }\n}\n");

    fs::write(data_dir.join("line-stability-data.json"), j).expect("write JSON");

    // ---- what the task asks us to print -------------------------------------
    println!("RK4 real-axis crossing  = {real_crossing:.6}");
    println!("RK4 imaginary crossing  = {imag_crossing:.6}");
    println!("RK4 line dt_crit        = {dt_crit:.6}");
    for (dt, mg, k) in &dt_stats {
        println!("dt = {dt:.3}: max_k |R(lambda_k dt)| = {mg:.6}, dominant k = {k}");
    }
    println!("Nyquist fourier_d1 max  = {nyquist_d1_max:.3e} (library convention check)");
}
