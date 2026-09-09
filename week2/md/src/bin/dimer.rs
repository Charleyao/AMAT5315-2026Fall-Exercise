//! Emit the relative total-energy error histories for the two-atom
//! Lennard-Jones dimer experiment, using the actual integrators and the
//! experiment driver from the `md` library crate.
//!
//! This binary is only a thin printer around `md::run_dimer_experiment`; the
//! physics and the integrators live in `md/src/lib.rs`. The Python script
//! `week2/plot_dimer.py` builds and runs this binary and turns its output
//! into `week2/dimer.png`.
//!
//! Output format (one run per block, one sample per line):
//!   <method> <steps>
//!   <index> <relative_error>
//!   ...
//!
//! It prints three blocks:
//!   * ForwardEuler, 500 steps
//!   * VelocityVerlet, 500 steps
//!   * VelocityVerlet, 5000 steps
//! all with the shared initial state (positions [[0, 0], [1.2, 0]],
//! velocities [[0, 0], [0, 0]]) and dt = 0.01.

use md::{run_dimer_experiment, ForwardEuler, VelocityVerlet};

fn emit(name: &str, steps: usize, errors: &[f64]) {
    println!("{name} {steps}");
    for (index, error) in errors.iter().enumerate() {
        println!("{index} {error:.17e}");
    }
}

fn main() {
    let dt = 0.01;

    let euler_500 = run_dimer_experiment(&ForwardEuler, 500, dt);
    emit("ForwardEuler", 500, &euler_500);

    let vv_500 = run_dimer_experiment(&VelocityVerlet, 500, dt);
    emit("VelocityVerlet", 500, &vv_500);

    let vv_5000 = run_dimer_experiment(&VelocityVerlet, 5000, dt);
    emit("VelocityVerlet", 5000, &vv_5000);
}
