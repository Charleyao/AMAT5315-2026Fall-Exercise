pub fn greeting() -> &'static str {
    "Hello, world!"
}


// Lennard-Jones potential energy
// U(r) = 4 * (r^(-12) - r^(-6))
pub fn energy(r: f64) -> f64 {
    let inv_r = 1.0 / r;
    let inv_r6 = inv_r.powi(6);
    let inv_r12 = inv_r6 * inv_r6;

    4.0 * (inv_r12 - inv_r6)
}


// Lennard-Jones radial force
// F(r) = 24/r * (2*r^(-12) - r^(-6))
pub fn force(r: f64) -> f64 {
    let inv_r = 1.0 / r;
    let inv_r6 = inv_r.powi(6);
    let inv_r12 = inv_r6 * inv_r6;

    24.0 * inv_r * (2.0 * inv_r12 - inv_r6)
}


// ---------------------------------------------------------------------------
// Part 3 scaffolding (RED): minimal stubs so the dimer test compiles and
// FAILS. Real behavior is implemented in the GREEN step.
// ---------------------------------------------------------------------------

pub struct System;

pub trait Integrator {
    fn step(&self, system: &mut System, dt: f64);
}

pub fn advance(method: &impl Integrator, system: &mut System, dt: f64) {
    method.step(system, dt);
}

pub struct ForwardEuler;
pub struct VelocityVerlet;

impl Integrator for ForwardEuler {
    fn step(&self, _system: &mut System, _dt: f64) {}
}

impl Integrator for VelocityVerlet {
    fn step(&self, _system: &mut System, _dt: f64) {}
}

pub fn run_dimer_experiment(method: &impl Integrator, steps: usize, dt: f64) -> Vec<f64> {
    let mut system = System;
    for _ in 0..steps {
        advance(method, &mut system, dt);
    }
    vec![0.0; steps + 1]
}


#[cfg(test)]
mod tests {
    use super::*;

    // Part 1 test
    #[test]
    fn greeting_is_correct() {
        assert_eq!(greeting(), "Hello, world!");
    }


    // Part 2 test 1:
    // U(r0) = -1 at r0 = 2^(1/6)
    #[test]
    fn well_depth() {
        let r0 = 2.0_f64.powf(1.0 / 6.0);

        assert!(
            (energy(r0) + 1.0).abs() < 1e-12,
            "Expected U(r0) = -1 at r0 = {}, but got {}",
            r0,
            energy(r0)
        );
    }


    // Part 2 test 2:
    // F(r) = -dU/dr
    #[test]
    fn force_matches_energy_derivative() {
        let h = 1e-5;

        let distances = [1.0, 1.1, 1.2, 1.5, 2.0];

        for r in distances {
            let numerical_force =
                -(energy(r + h) - energy(r - h)) / (2.0 * h);

            let analytical_force = force(r);

            let tolerance =
                1e-6 * analytical_force.abs().max(1.0);

            assert!(
                (analytical_force - numerical_force).abs() < tolerance,
                "Force mismatch at r = {}: analytical = {}, numerical = {}",
                r,
                analytical_force,
                numerical_force
            );
        }
    }


    // Part 3: two-atom molecular dynamics

    fn max_abs(xs: &[f64]) -> f64 {
        xs.iter().map(|x| x.abs()).fold(0.0_f64, f64::max)
    }

    #[test]
    fn dimer_energy_errors() {
        let dt = 0.01;

        // Explicit forward Euler must blow up in energy.
        let euler = run_dimer_experiment(&ForwardEuler, 500, dt);
        assert!(
            euler[500] > 0.5,
            "Forward Euler final relative energy error should exceed 0.5, got {}",
            euler[500]
        );

        // Velocity-Verlet must stay bounded over 500 steps.
        let vv500 = run_dimer_experiment(&VelocityVerlet, 500, dt);
        let vv500_max = max_abs(&vv500);
        assert!(
            vv500_max < 1e-3,
            "Velocity-Verlet max |relative energy error| over 500 steps should be < 1e-3, got {}",
            vv500_max
        );

        // Long-time comparison: 5000 steps. Record the error history and
        // confirm it remains bounded (no divergence) — no hard threshold.
        let vv5000 = run_dimer_experiment(&VelocityVerlet, 5000, dt);
        let vv5000_max = max_abs(&vv5000);
        assert!(
            vv5000_max.is_finite(),
            "Velocity-Verlet 5000-step max |relative energy error| should stay bounded (finite), got {}",
            vv5000_max
        );

        println!(
            "dimer: euler final rel err = {}, vv500 max |err| = {}, vv5000 max |err| = {}",
            euler[500], vv500_max, vv5000_max
        );
    }
}
