//! Data generator for both panels of `week4/evidence/line-accuracy.png`.
//!
//! * propagation: three numerical cases advance one periodised Gaussian to
//!   `t ~= 2*pi` (RK4 + Fourier, RK4 + centred FD, Euler + Fourier) and are
//!   compared with the exact periodic advection-diffusion solution;
//! * convergence: Euler, midpoint, classical RK4 and the equal-weight four-stage
//!   control are run at four steps with Fourier derivatives only, and the
//!   temporal order is fitted from the actual errors.
//!
//! Everything uses the real library; no integrator or derivative is re-implemented
//! here. `cargo run --release --bin line-accuracy-data` (or `line_accuracy.py`)
//! writes `artifacts/line-accuracy/line-accuracy-data.json`.

use std::f64::consts::{FRAC_PI_2, PI};
use std::fs;
use std::path::PathBuf;

use week4::advdiff::{AdvectionDiffusion, SpatialScheme};
use week4::grid::PeriodicGrid;
use week4::{EqualWeightRK4, Euler, Integrator, Midpoint, RK4};

/// Number of periodic images summed for a smooth periodic pulse.
const IMAGES: i32 = 5;

// ---- left panel: propagation ------------------------------------------------
const P_N: usize = 64;
const P_C: f64 = 1.0;
const P_NU: f64 = 0.002;
const P_SIGMA: f64 = 0.25;
const P_T_END: f64 = 2.0 * PI;

// ---- right panel: temporal convergence --------------------------------------
const V_N: usize = 64;
const V_C: f64 = 1.0;
const V_NU: f64 = 0.05;
const V_SIGMA: f64 = 0.35;
const V_T_END: f64 = 1.0;
const V_DTS: [f64; 4] = [0.02, 0.01, 0.005, 0.0025];
const V_METHODS: [&str; 4] = ["Euler", "Midpoint", "RK4", "Equal-weight RK"];

/// Periodised Gaussian, same construction as the stability experiment:
/// sum over periodic images, never a truncated non-periodic Gaussian.
fn periodic_gaussian(g: &PeriodicGrid, x0: f64, sigma: f64) -> Vec<f64> {
    g.x()
        .iter()
        .map(|&x| {
            let mut s = 0.0;
            for j in -IMAGES..=IMAGES {
                let d = x - x0 - 2.0 * PI * j as f64;
                s += (-d * d / (2.0 * sigma * sigma)).exp();
            }
            s
        })
        .collect()
}

/// Exact periodic advection-diffusion solution at time `t`: a sum of Gaussians
/// with centre `x0 + c t` and widened variance `sigma^2 + 2 nu t`, each with the
/// mass-preserving amplitude `sigma / sqrt(sigma^2 + 2 nu t)`.
fn exact_solution(g: &PeriodicGrid, x0: f64, sigma: f64, c: f64, nu: f64, t: f64) -> Vec<f64> {
    let variance = sigma * sigma + 2.0 * nu * t;
    let amplitude = sigma / variance.sqrt();
    g.x()
        .iter()
        .map(|&x| {
            let mut s = 0.0;
            for j in -IMAGES..=IMAGES {
                let d = x - (x0 + c * t) - 2.0 * PI * j as f64;
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

/// Integrate for `steps` constant-`dt` steps; returns `(u, t_final)`.
fn run<I: Integrator>(
    method: &I,
    g: &PeriodicGrid,
    c: f64,
    nu: f64,
    scheme: SpatialScheme,
    dt: f64,
    steps: usize,
    u0: &[f64],
) -> (Vec<f64>, f64) {
    let model = AdvectionDiffusion::new(g, c, nu, scheme);
    let rhs = model.rhs_fn();
    let mut u = u0.to_vec();
    for _ in 0..steps {
        u = method.step(&u, dt, &rhs);
    }
    (u, steps as f64 * dt)
}

/// The same, dispatching on a convergence-method name.
fn run_named(
    method: &str,
    g: &PeriodicGrid,
    c: f64,
    nu: f64,
    dt: f64,
    steps: usize,
    u0: &[f64],
) -> (Vec<f64>, f64) {
    match method {
        "Euler" => run(&Euler, g, c, nu, SpatialScheme::Fourier, dt, steps, u0),
        "Midpoint" => run(&Midpoint, g, c, nu, SpatialScheme::Fourier, dt, steps, u0),
        "RK4" => run(&RK4, g, c, nu, SpatialScheme::Fourier, dt, steps, u0),
        "Equal-weight RK" => {
            run(&EqualWeightRK4, g, c, nu, SpatialScheme::Fourier, dt, steps, u0)
        }
        other => panic!("unknown method {other}"),
    }
}

/// Least-squares slope of `log(error)` against `log(dt)`.
fn loglog_slope(dts: &[f64], errors: &[f64]) -> f64 {
    let xm: f64 = dts.iter().map(|d| d.ln()).sum::<f64>() / dts.len() as f64;
    let ym: f64 = errors.iter().map(|e| e.ln()).sum::<f64>() / errors.len() as f64;
    let sxx: f64 = dts.iter().map(|d| (d.ln() - xm).powi(2)).sum();
    let sxy: f64 = dts
        .iter()
        .zip(errors.iter())
        .map(|(d, e)| (d.ln() - xm) * (e.ln() - ym))
        .sum();
    sxy / sxx
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

    // ---- left panel: propagation --------------------------------------------
    let g = PeriodicGrid::new(P_N);
    let ic = periodic_gaussian(&g, FRAC_PI_2, P_SIGMA);
    let cases: [(&str, &str, SpatialScheme, f64); 3] = [
        ("RK4 + Fourier", "RK4", SpatialScheme::Fourier, 0.02),
        (
            "RK4 + centred finite diff",
            "RK4",
            SpatialScheme::FiniteDifference,
            0.02,
        ),
        ("Euler + Fourier", "Euler", SpatialScheme::Fourier, 0.005),
    ];

    let mut propagation: Vec<(&str, &str, f64, Vec<f64>, f64, f64)> = Vec::new();
    for (label, method, scheme, dt) in cases.iter() {
        let steps = (P_T_END / dt).floor() as usize;
        let (u, t_final) = if *method == "RK4" {
            run(&RK4, &g, P_C, P_NU, *scheme, *dt, steps, &ic)
        } else {
            run(&Euler, &g, P_C, P_NU, *scheme, *dt, steps, &ic)
        };
        propagation.push((label, method, *dt, u, t_final, 0.0));
    }
    let p_t_final = propagation[0].4;
    for p in &propagation {
        assert!(
            (p.4 - p_t_final).abs() < 1e-12,
            "final times differ: {} vs {}",
            p.4,
            p_t_final
        );
    }
    let p_exact = exact_solution(&g, FRAC_PI_2, P_SIGMA, P_C, P_NU, p_t_final);

    // ---- right panel: temporal convergence ----------------------------------
    let cg = PeriodicGrid::new(V_N);
    let cic = periodic_gaussian(&cg, FRAC_PI_2, V_SIGMA);
    let cexact = exact_solution(&cg, FRAC_PI_2, V_SIGMA, V_C, V_NU, V_T_END);

    let mut conv: Vec<(&str, Vec<f64>, f64)> = Vec::new();
    for method in V_METHODS {
        let mut errors = Vec::with_capacity(V_DTS.len());
        for dt in V_DTS {
            let steps = (V_T_END / dt).round() as usize;
            let (u, t_final) = run_named(method, &cg, V_C, V_NU, dt, steps, &cic);
            assert!(
                (t_final - V_T_END).abs() < 1e-12,
                "{method}: t_final {t_final} != {V_T_END} (dt = {dt})"
            );
            errors.push(max_abs_error(&u, &cexact));
        }
        let slope = loglog_slope(&V_DTS, &errors);
        conv.push((method, errors, slope));
    }

    // ---- JSON -----------------------------------------------------------------
    let mut j = String::new();
    j.push_str("{\n");

    // propagation section
    j.push_str("  \"propagation\": {\n");
    j.push_str(&format!(
        "    \"n\": {P_N}, \"c\": {}, \"nu\": {}, \"sigma\": {}, \"x0\": {}, \
         \"t_end\": {}, \"t_final\": {},\n",
        fmt_f64(P_C),
        fmt_f64(P_NU),
        fmt_f64(P_SIGMA),
        fmt_f64(FRAC_PI_2),
        fmt_f64(P_T_END),
        fmt_f64(p_t_final)
    ));
    j.push_str("    \"x\": ");
    j.push_str(&fmt_f64_vec(g.x()));
    j.push_str(",\n    \"initial\": ");
    j.push_str(&fmt_f64_vec(&ic));
    j.push_str(",\n    \"exact\": ");
    j.push_str(&fmt_f64_vec(&p_exact));
    j.push_str(",\n    \"cases\": [\n");
    for (i, (label, _method, dt, u, tf, _err)) in propagation.iter().enumerate() {
        let err = max_abs_error(u, &p_exact);
        j.push_str(&format!(
            "      {{\"label\": \"{label}\", \"dt\": {}, \"t_final\": {}, \
             \"max_abs_error\": {}, \"u\": {}}}",
            fmt_f64(*dt),
            fmt_f64(*tf),
            fmt_f64(err),
            fmt_f64_vec(u)
        ));
        j.push_str(if i + 1 == propagation.len() { "\n" } else { ",\n" });
    }
    j.push_str("    ]\n  },\n");

    // convergence section
    j.push_str("  \"convergence\": {\n");
    j.push_str(&format!(
        "    \"n\": {V_N}, \"c\": {}, \"nu\": {}, \"sigma\": {}, \"x0\": {}, \
         \"t_end\": {},\n",
        fmt_f64(V_C),
        fmt_f64(V_NU),
        fmt_f64(V_SIGMA),
        fmt_f64(FRAC_PI_2),
        fmt_f64(V_T_END)
    ));
    j.push_str("    \"dts\": ");
    j.push_str(&fmt_f64_vec(&V_DTS));
    j.push_str(",\n    \"methods\": [\n");
    for (i, (method, errors, slope)) in conv.iter().enumerate() {
        j.push_str(&format!(
            "      {{\"label\": \"{method}\", \"errors\": {}, \"slope\": {}}}",
            fmt_f64_vec(errors),
            fmt_f64(*slope)
        ));
        j.push_str(if i + 1 == conv.len() { "\n" } else { ",\n" });
    }
    j.push_str("    ]\n  }\n}\n");
    fs::write(data_dir.join("line-accuracy-data.json"), j).expect("write JSON");

    // ---- tables ---------------------------------------------------------------
    println!("propagation: n = {P_N}, nu = {P_NU}, sigma = {P_SIGMA}, t_final = {p_t_final:.6}");
    println!("{:<28} {:>8} {:>16}", "method", "dt", "max abs error");
    for (label, _method, dt, u, _tf, _err) in &propagation {
        println!("{label:<28} {dt:>8} {:>16.6e}", max_abs_error(u, &p_exact));
    }

    println!();
    println!(
        "convergence: n = {V_N}, nu = {V_NU}, sigma = {V_SIGMA}, t_end = {V_T_END}"
    );
    println!("{:<18} {:>8} {:>16}", "method", "dt", "max error");
    for (method, errors, _slope) in &conv {
        for (dt, err) in V_DTS.iter().zip(errors.iter()) {
            println!("{method:<18} {dt:>8} {err:>16.6e}");
        }
    }
    println!();
    for (method, _errors, slope) in &conv {
        println!("fitted slope, {method:<18} p = {slope:.4}");
    }
}
