//! Small helper that samples the Lennard-Jones pair potential/force field
//! on a rectangular grid and prints one row per grid point.
//!
//! It calls the actual `md::energy` and `md::force` functions from the
//! library crate, so the plotted field is exactly the field defined by the
//! Rust implementation.
//!
//! Usage:
//!   field_grid XMIN XMAX NX YMIN YMAX NY
//!
//! Output (one line per point, columns):
//!   x y U Fx Fy
//! where `U = energy(r)`, `F = force(r) * r_hat` (the radial force resolved
//! into Cartesian components).

use md::{energy, force};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 7 {
        eprintln!("usage: field_grid XMIN XMAX NX YMIN YMAX NY");
        std::process::exit(1);
    }

    let xmin: f64 = args[1].parse().expect("XMIN must be a number");
    let xmax: f64 = args[2].parse().expect("XMAX must be a number");
    let nx: usize = args[3].parse().expect("NX must be an integer");
    let ymin: f64 = args[4].parse().expect("YMIN must be a number");
    let ymax: f64 = args[5].parse().expect("YMAX must be a number");
    let ny: usize = args[6].parse().expect("NY must be an integer");

    for iy in 0..ny {
        let y = if ny == 1 {
            ymin
        } else {
            ymin + (ymax - ymin) * (iy as f64) / ((ny - 1) as f64)
        };

        for ix in 0..nx {
            let x = if nx == 1 {
                xmin
            } else {
                xmin + (xmax - xmin) * (ix as f64) / ((nx - 1) as f64)
            };

            let r = (x * x + y * y).sqrt();
            let u = energy(r);
            let f = force(r);

            // The Rust `force` returns the radial scalar force F(r) = -dU/dr.
            // Its sign already points outward (F > 0) for r < r0 and inward
            // (F < 0) for r > r0; multiplying by the unit vector r_hat turns
            // that scalar into a Cartesian vector.
            let (fx, fy) = if r > 0.0 {
                (f * x / r, f * y / r)
            } else {
                (0.0, 0.0)
            };

            println!("{x:.12} {y:.12} {u:.12} {fx:.12} {fy:.12}");
        }
    }
}
