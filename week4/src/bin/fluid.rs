//! `fluid` — integrate a 2-D incompressible vorticity field on stdin.
//!
//! Strictly follows `week4/fluid.design.toml`:
//!
//! ```text
//! field taylor-green --n 64 | fluid --method rk4 --nu 0.1 --dt 0.01 \
//!     --t-end 1 --every 0.1 --out artifacts/taylor-green
//! ```
//!
//! stdin is the JSON object written by `field`. `fluid` builds the initial
//! vorticity `omega = dv/dx - du/dy` with the verified spectral derivatives,
//! dealiases it, and advances `omega_t = -(u omega_x + v omega_y) + nu lap(omega)`
//! with the Part 1 integrators. Snapshots are saved at step 0 and every
//! `round(every/dt)` steps; the step is never shortened for a snapshot.
//!
//! stdout: a `t<TAB>E<TAB>Z` header, then one such line per snapshot. It stops at
//! the first non-finite energy, prints that line and exits 1.
//! `<out>/run.json` and `<out>/fields.jsonl` follow the design contract; fields
//! are serialised to 6 decimals but integrated at full precision.

use std::fs;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::exit;

use serde_json::Value;
use week4::spectral2d::Spectral2D;
use week4::{Euler, Integrator, Midpoint, RK4};

const USAGE: &str = "\
usage: field ... | fluid --method {euler|rk2|rk4} --nu NU --dt DT \
--t-end T --every EVERY --out DIR";

// =================================================================================
// Input
// =================================================================================

struct FieldInput {
    case: String,
    n: usize,
    seed: Option<i64>,
    k_band: Value,
    u: Vec<f64>,
    v: Vec<f64>,
}

fn json_numbers(obj: &Value, key: &str) -> Result<Vec<f64>, String> {
    let arr = obj
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("input JSON is missing the array {key:?}"))?;
    arr.iter()
        .map(|x| x.as_f64().ok_or_else(|| format!("{key} contains a non-number")))
        .collect()
}

fn parse_input(text: &str) -> Result<FieldInput, String> {
    let obj: Value = serde_json::from_str(text).map_err(|e| format!("stdin is not JSON: {e}"))?;
    let case = obj
        .get("case")
        .and_then(Value::as_str)
        .ok_or("input JSON is missing \"case\"")?
        .to_string();
    let n = obj
        .get("n")
        .and_then(Value::as_u64)
        .ok_or("input JSON is missing \"n\"")? as usize;
    let seed = obj.get("seed").and_then(Value::as_i64);
    let k_band = obj.get("k_band").cloned().unwrap_or(Value::Null);
    let u = json_numbers(&obj, "u")?;
    let v = json_numbers(&obj, "v")?;
    if u.len() != n * n || v.len() != n * n {
        return Err(format!("u/v have {} / {} entries, expected n*n = {}", u.len(), v.len(), n * n));
    }
    Ok(FieldInput { case, n, seed, k_band, u, v })
}

// =================================================================================
// Integration
// =================================================================================

/// Advance one step with the Part 1 integrator; every stage re-derives velocity
/// from the stage's own omega inside `vorticity_rhs`.
fn step_once(method: &str, s: &Spectral2D, omega: &[f64], dt: f64, nu: f64) -> Vec<f64> {
    let rhs = |w: &[f64]| s.vorticity_rhs(w, nu);
    match method {
        "euler" => Euler.step(omega, dt, rhs),
        "rk2" => Midpoint.step(omega, dt, rhs),
        "rk4" => RK4.step(omega, dt, rhs),
        other => unreachable!("validated method {other}"),
    }
}

/// Number of steps that land on a snapshot; step 0 always is one.
fn is_snapshot(step: usize, stride: usize) -> bool {
    step % stride == 0
}

// =================================================================================
// Output helpers
// =================================================================================

fn arr6(values: &[f64]) -> String {
    let mut s = String::with_capacity(values.len() * 10 + 2);
    s.push('[');
    for (i, v) in values.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("{v:.6}"));
    }
    s.push(']');
    s
}

fn fields_line(t: f64, step: usize, u: &[f64], v: &[f64], omega: &[f64]) -> String {
    format!(
        "{{\"t\":{t:.6},\"step\":{step},\"u\":{},\"v\":{},\"omega\":{}}}",
        arr6(u),
        arr6(v),
        arr6(omega)
    )
}

// =================================================================================
// Recovery of physical fields at a snapshot
// =================================================================================

struct Diagnostics {
    u: Vec<f64>,
    v: Vec<f64>,
    energy: f64,
    enstrophy: f64,
}

fn diagnose(s: &Spectral2D, omega: &[f64]) -> Diagnostics {
    let (u, v) = s.velocity_from_vorticity(omega);
    let energy = s.energy(&u, &v);
    let enstrophy = s.enstrophy(omega);
    Diagnostics { u, v, energy, enstrophy }
}

// =================================================================================
// Argument parsing
// =================================================================================

fn arg_value(args: &[String], name: &str) -> Result<Option<String>, String> {
    match args.iter().position(|a| a == name) {
        None => Ok(None),
        Some(i) => args
            .get(i + 1)
            .cloned()
            .map(Some)
            .ok_or_else(|| format!("missing value for {name}")),
    }
}

fn required<T: std::str::FromStr>(args: &[String], name: &str) -> Result<T, String> {
    let raw = arg_value(args, name)?.ok_or_else(|| format!("missing required {name}"))?;
    raw.parse::<T>().map_err(|_| format!("bad value for {name}: {raw:?}"))
}

// =================================================================================
// Run
// =================================================================================

enum Outcome {
    Finished,
    Unstable,
}

fn run() -> Result<Outcome, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let method: String = required(&args, "--method")?;
    if !matches!(method.as_str(), "euler" | "rk2" | "rk4") {
        return Err(format!("invalid --method {method:?} (expected euler, rk2 or rk4)"));
    }
    let nu: f64 = required(&args, "--nu")?;
    let dt: f64 = required(&args, "--dt")?;
    let t_end: f64 = required(&args, "--t-end")?;
    let every: f64 = required(&args, "--every")?;
    let out: PathBuf = PathBuf::from(
        arg_value(&args, "--out")?.ok_or("missing required --out")?,
    );
    if dt <= 0.0 || every <= 0.0 || t_end < 0.0 {
        return Err("need dt > 0, every > 0, t-end >= 0".into());
    }

    let mut text = String::new();
    std::io::stdin()
        .read_to_string(&mut text)
        .map_err(|e| format!("reading stdin: {e}"))?;
    let input = parse_input(&text)?;

    let s = Spectral2D::new(input.n);
    // omega = dv/dx - du/dy from the piped velocity, then the carried vorticity
    // is put on the two-thirds band.
    let du_dy = s.dy(&input.u);
    let dv_dx = s.dx(&input.v);
    let omega0: Vec<f64> = dv_dx
        .iter()
        .zip(du_dy.iter())
        .map(|(a, b)| a - b)
        .collect();
    let mut omega = s.dealias(&omega0);

    fs::create_dir_all(&out).map_err(|e| format!("creating {}: {e}", out.display()))?;
    let run_json = serde_json::json!({
        "case": input.case,
        "n": input.n,
        "seed": input.seed,
        "k_band": input.k_band,
        "method": method,
        "nu": nu,
        "dt": dt,
        "t_end": t_end,
        "snapshot_every": every,
    });
    fs::write(
        out.join("run.json"),
        serde_json::to_string_pretty(&run_json).unwrap(),
    )
    .map_err(|e| format!("writing run.json: {e}"))?;

    let steps_total = (t_end / dt).round() as usize;
    let stride = (every / dt).round() as usize;
    if stride == 0 {
        return Err(format!("round(every/dt) = 0 (every = {every}, dt = {dt})"));
    }

    let fields_path = out.join("fields.jsonl");
    let mut fields = BufWriter::new(
        fs::File::create(&fields_path).map_err(|e| format!("creating fields.jsonl: {e}"))?,
    );

    println!("t\tE\tZ");
    for step in 0..=steps_total {
        if is_snapshot(step, stride) {
            let t = step as f64 * dt;
            let d = diagnose(&s, &omega);
            if !d.energy.is_finite() || !d.enstrophy.is_finite() {
                println!("{t:.6}\t{:.6}\t{:.6}", d.energy, d.enstrophy);
                fields.flush().ok();
                return Ok(Outcome::Unstable);
            }
            writeln!(fields, "{}", fields_line(t, step, &d.u, &d.v, &omega))
                .map_err(|e| format!("writing fields.jsonl: {e}"))?;
            println!("{t:.6}\t{:.6}\t{:.6}", d.energy, d.enstrophy);
        }
        if step < steps_total {
            omega = step_once(&method, &s, &omega, dt, nu);
        }
    }
    fields.flush().ok();
    Ok(Outcome::Finished)
}

fn main() {
    match run() {
        Ok(Outcome::Finished) => {}
        Ok(Outcome::Unstable) => exit(1),
        Err(msg) => {
            eprintln!("fluid: {msg}\n\n{USAGE}");
            exit(2);
        }
    }
}

// =================================================================================
// Tests
// =================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_field_json_object() {
        let text = r#"{"case":"taylor-green","n":2,"seed":null,"k_band":null,
            "u":[1.0,2.0,3.0,4.0],"v":[5.0,6.0,7.0,8.0]}"#;
        let f = parse_input(text).unwrap();
        assert_eq!(f.case, "taylor-green");
        assert_eq!(f.n, 2);
        assert_eq!(f.seed, None);
        assert_eq!(f.k_band, Value::Null);
        assert_eq!(f.u, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(f.v, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn taylor_green_input_has_omega_minus_2_cos_cos() {
        // Build the field JSON in the same shape `field taylor-green` emits.
        let n = 8;
        let s = Spectral2D::new(n);
        let g = s.grid();
        let mut u = vec![0.0; n * n];
        let mut v = vec![0.0; n * n];
        for iy in 0..n {
            for ix in 0..n {
                let (x, y) = (g.point(ix), g.point(iy));
                let i = g.index(ix, iy);
                u[i] = x.cos() * y.sin();
                v[i] = -x.sin() * y.cos();
            }
        }
        let du_dy = s.dy(&u);
        let dv_dx = s.dx(&v);
        let omega: Vec<f64> = dv_dx.iter().zip(du_dy.iter()).map(|(a, b)| a - b).collect();
        let exact: Vec<f64> = (0..n * n)
            .map(|i| {
                let (ix, iy) = g.coords(i);
                -2.0 * g.point(ix).cos() * g.point(iy).cos()
            })
            .collect();
        let err = omega
            .iter()
            .zip(exact.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-12, "omega error {err}");
    }

    #[test]
    fn snapshot_stride_is_round_every_over_dt() {
        assert!(is_snapshot(0, 10));
        assert!(is_snapshot(10, 10));
        assert!(!is_snapshot(9, 10));
        assert!(is_snapshot(100, 10));
    }

    #[test]
    fn arrays_serialise_to_six_decimals() {
        assert_eq!(arr6(&[0.0, 1.0 / 3.0]), "[0.000000,0.333333]");
    }
}
