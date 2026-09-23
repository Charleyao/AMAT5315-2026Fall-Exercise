//! Build a perturbed `field` JSON for the sensitivity experiment.
//!
//! The perturbation is the divergence-free velocity from the streamfunction
//! `delta_psi = -(eps M / 25) cos(3x) cos(4y)`, so its vorticity is exactly the
//! learning-sheet ripple `delta_omega = -eps M cos(3x) cos(4y)` with
//! `eps = 7e-5` and `M = max(max|u|, max|v|)` of the input field. The
//! reconstruction `dv/dx - du/dy` is checked with the library [`Spectral2D`].
//!
//! Usage: `sensitivity-prep ORIGINAL.json PERTURBED.json`
//! Metadata (`case`, `n`, `seed`, `k_band`) is copied unchanged.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::exit;

use serde_json::{Value, json};
use week4::spectral2d::Spectral2D;

const EPS: f64 = 7e-5;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 2 {
        eprintln!("usage: sensitivity-prep ORIGINAL.json PERTURBED.json");
        exit(2);
    }
    let src = PathBuf::from(&args[0]);
    let dst = PathBuf::from(&args[1]);

    let mut obj: Value = serde_json::from_str(&fs::read_to_string(&src).expect("read input"))
        .expect("input is JSON");
    let n = obj["n"].as_u64().expect("input has n") as usize;
    let mut u: Vec<f64> = obj["u"]
        .as_array()
        .expect("input has u")
        .iter()
        .map(|x| x.as_f64().expect("u is numeric"))
        .collect();
    let mut v: Vec<f64> = obj["v"]
        .as_array()
        .expect("input has v")
        .iter()
        .map(|x| x.as_f64().expect("v is numeric"))
        .collect();
    assert_eq!(u.len(), n * n);
    assert_eq!(v.len(), n * n);

    // M uses |u| and |v| separately, not the speed sqrt(u^2+v^2).
    let m = u
        .iter()
        .chain(v.iter())
        .map(|x| x.abs())
        .fold(0.0, f64::max);

    let s = Spectral2D::new(n);
    let g = s.grid();
    let mut du = vec![0.0; n * n];
    let mut dv = vec![0.0; n * n];
    let mut target = vec![0.0; n * n];
    for iy in 0..n {
        for ix in 0..n {
            let (x, y) = (g.point(ix), g.point(iy));
            let i = g.index(ix, iy);
            du[i] = (4.0 * EPS * m / 25.0) * (3.0 * x).cos() * (4.0 * y).sin();
            dv[i] = -(3.0 * EPS * m / 25.0) * (3.0 * x).sin() * (4.0 * y).cos();
            target[i] = -EPS * m * (3.0 * x).cos() * (4.0 * y).cos();
        }
    }

    // Independent check with the library spectral derivatives.
    let dv_dx = s.dx(&dv);
    let du_dy = s.dy(&du);
    let recon: Vec<f64> = dv_dx.iter().zip(du_dy.iter()).map(|(a, b)| a - b).collect();
    let err = recon
        .iter()
        .zip(target.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);

    for i in 0..n * n {
        u[i] += du[i];
        v[i] += dv[i];
    }
    obj["u"] = json!(u);
    obj["v"] = json!(v);
    fs::write(&dst, serde_json::to_string(&obj).expect("serialize"))
        .unwrap_or_else(|e| panic!("write {}: {e}", dst.display()));

    println!(
        "{}: n = {n}, M = {m:.6}, max |delta_omega| = {:.3e}, \
         spectral reconstruction max error = {err:.3e}",
        src.file_name().unwrap().to_string_lossy(),
        EPS * m
    );
    assert!(err < 1e-10, "reconstructed vorticity error {err}");
}
