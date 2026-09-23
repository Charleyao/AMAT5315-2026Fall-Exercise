//! Data generator for the propagation-accuracy panel of
//! `week4/evidence/line-accuracy.png`.
//!
//! Three numerical cases advance the same periodised Gaussian pulse to
//! `t ~= 2*pi` using the real library, then the exact advection-diffusion
//! solution (advection *and* diffusion, periodic) is compared:
//!
//!   RK4   + Fourier derivatives,  dt = 0.02
//!   RK4   + centred differences,  dt = 0.02
//!   Euler + Fourier derivatives,  dt = 0.005
//!
//! `dt` is never shortened: each run integrates a whole number of steps and the
//! exact solution is evaluated at the time actually reached.
//!
//! Run from `week4/scripts/` via `line_accuracy.py`, or:
//!
//!   cd week4
//!   cargo run --release --bin line-accuracy-data
//!
//! Writes `artifacts/line-accuracy/line-accuracy-data.json`.

use std::f64::consts::{FRAC_PI_2, PI};
use std::fs;
use std::path::PathBuf;

use week4::advdiff::{AdvectionDiffusion, SpatialScheme};
use week4::grid::PeriodicGrid;
use week4::{Euler, Integrator, RK4};

const N: usize = 64;
const C: f64 = 1.0;
const NU: f64 = 0.002;
const SIGMA: f64 = 0.25;
const X0: f64 = FRAC_PI_2;
const T_END: f64 = 2.0 * PI;
/// Number of periodic images summed for a smooth periodic pulse.
const IMAGES: i32 = 5;

/// Periodised Gaussian, same construction as the stability experiment:
/// sum over periodic images, never a truncated non-periodic Gaussian.
fn periodic_gaussian(g: &PeriodicGrid) -> Vec<f64> {
    g.x()
        .iter()
        .map(|&x| {
            let mut s = 0.0;
            for j in -IMAGES..=IMAGES {
                let d = x - X0 - 2.0 * PI * j as f64;
                s += (-d * d / (2.0 * SIGMA * SIGMA)).exp();
            }
            s
        })
        .collect()
}

/// Exact periodic advection-diffusion solution at time `t`:
/// a sum of Gaussians with centre `x0 + c t` and widened variance `sigma^2 + 2 nu t`,
/// each with the mass-preserving amplitude `sigma / sqrt(sigma^2 + 2 nu t)`.
fn exact_solution(g: &PeriodicGrid, t: f64) -> Vec<f64> {
    let variance = SIGMA * SIGMA + 2.0 * NU * t;
    let amplitude = SIGMA / variance.sqrt();
    g.x()
        .iter()
        .map(|&x| {
            let mut s = 0.0;
            for j in -IMAGES..=IMAGES {
                let d = x - (X0 + C * t) - 2.0 * PI * j as f64;
                s += amplitude * (-d * d / (2.0 * variance)).exp();
            }
            s
        })
        .collect()
}

fn max_abs_error(got: &[f64], want: &[f64]) -> f64 {
    got.iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

/// Integrate the line model for a whole number of constant `dt` steps.
fn run<I: Integrator>(
    method: &I,
    g: &PeriodicGrid,
    scheme: SpatialScheme,
    dt: f64,
    u0: &[f64],
) -> (Vec<f64>, f64) {
    let model = AdvectionDiffusion::new(g, C, NU, scheme);
    let rhs = model.rhs_fn();
    let steps = (T_END / dt).floor() as usize;
    let mut u = u0.to_vec();
    for _ in 0..steps {
        u = method.step(&u, dt, &rhs);
    }
    (u, steps as f64 * dt)
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

fn main() {
    let week4_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data_dir = week4_dir.join("artifacts").join("line-accuracy");
    fs::create_dir_all(&data_dir).expect("create artifacts/line-accuracy/");

    let g = PeriodicGrid::new(N);
    let ic = periodic_gaussian(&g);

    // (label, integrator kind, scheme, dt)
    struct Case {
        label: &'static str,
        method: &'static str,
        scheme: &'static str,
        dt: f64,
    }
    let cases = [
        Case { label: "RK4 + Fourier", method: "rk4", scheme: "fourier", dt: 0.02 },
        Case { label: "RK4 + centred finite diff", method: "rk4", scheme: "finite-difference", dt: 0.02 },
        Case { label: "Euler + Fourier", method: "euler", scheme: "fourier", dt: 0.005 },
    ];

    let mut profiles: Vec<(&'static str, Vec<f64>, f64, f64)> = Vec::new();
    for c in &cases {
        let scheme = if c.scheme == "fourier" {
            SpatialScheme::Fourier
        } else {
            SpatialScheme::FiniteDifference
        };
        let (u, t_final) = if c.method == "rk4" {
            run(&RK4, &g, scheme, c.dt, &ic)
        } else {
            run(&Euler, &g, scheme, c.dt, &ic)
        };
        let exact = exact_solution(&g, t_final);
        let err = max_abs_error(&u, &exact);
        profiles.push((c.label, u, t_final, err));
    }

    // All runs must land on the same final time for one common exact profile.
    let t_final = profiles[0].2;
    for p in &profiles {
        assert!((p.2 - t_final).abs() < 1e-12, "final times differ: {} vs {}", p.2, t_final);
    }
    let exact = exact_solution(&g, t_final);

    // ---- JSON -----------------------------------------------------------------
    let mut j = String::new();
    j.push_str("{\n");
    j.push_str(&format!("  \"n\": {N}, \"c\": {}, \"nu\": {},\n", fmt_f64(C), fmt_f64(NU)));
    j.push_str(&format!(
        "  \"sigma\": {}, \"x0\": {}, \"t_end\": {}, \"t_final\": {},\n",
        fmt_f64(SIGMA),
        fmt_f64(X0),
        fmt_f64(T_END),
        fmt_f64(t_final)
    ));
    j.push_str("  \"x\": ");
    j.push_str(&fmt_f64_vec(g.x()));
    j.push_str(",\n");
    j.push_str("  \"initial\": ");
    j.push_str(&fmt_f64_vec(&ic));
    j.push_str(",\n");
    j.push_str("  \"exact\": ");
    j.push_str(&fmt_f64_vec(&exact));
    j.push_str(",\n");
    j.push_str("  \"cases\": [\n");
    for (i, c) in cases.iter().enumerate() {
        let (label, u, tf, err) = &profiles[i];
        j.push_str(&format!(
            "    {{\"label\": \"{label}\", \"method\": \"{}\", \"scheme\": \"{}\", \
             \"dt\": {}, \"t_final\": {}, \"max_abs_error\": {}, \"u\": {}}}",
            c.method,
            c.scheme,
            fmt_f64(c.dt),
            fmt_f64(*tf),
            fmt_f64(*err),
            fmt_f64_vec(u)
        ));
        j.push_str(if i + 1 == cases.len() { "\n" } else { ",\n" });
    }
    j.push_str("  ]\n}\n");
    fs::write(data_dir.join("line-accuracy-data.json"), j).expect("write JSON");

    // ---- table ----------------------------------------------------------------
    println!("n = {N}, c = {C}, nu = {NU}, sigma = {SIGMA}, t_final = {t_final:.6}");
    println!("{:<28} {:>14}", "method", "max abs error");
    for (label, _, _, err) in &profiles {
        println!("{label:<28} {err:>14.6e}");
    }
}
