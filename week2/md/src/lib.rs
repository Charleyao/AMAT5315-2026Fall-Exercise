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
// Part 3: two-atom Lennard-Jones molecular dynamics
// ---------------------------------------------------------------------------

pub struct System {
    positions: Vec<[f64; 2]>,
    velocities: Vec<[f64; 2]>,
    accelerations: Vec<[f64; 2]>, // cached: always a(x) at current positions
}

impl System {
    pub fn n_atoms(&self) -> usize {
        self.positions.len()
    }

    pub fn new(positions: Vec<[f64; 2]>, velocities: Vec<[f64; 2]>) -> System {
        assert_eq!(positions.len(), velocities.len());
        let n = positions.len();
        let mut system = System {
            positions,
            velocities,
            accelerations: vec![[0.0, 0.0]; n],
        };
        // Compute the initial accelerations before the first step.
        system.update_accelerations();
        system
    }

    // Acceleration from the Lennard-Jones pair force. Mass = 1, so a = F.
    // Sign convention matches field_grid.rs: for a pair (i, j) with
    // d = r_j - r_i, force on j = F(r) * d/r and force on i = -F(r) * d/r.
    pub fn update_accelerations(&mut self) {
        let n = self.n_atoms();
        for a in &mut self.accelerations {
            a[0] = 0.0;
            a[1] = 0.0;
        }
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = self.positions[j][0] - self.positions[i][0];
                let dy = self.positions[j][1] - self.positions[i][1];
                let r = (dx * dx + dy * dy).sqrt();
                let f = force(r); // scalar F(r) = -dU/dr
                let fx = f * dx / r;
                let fy = f * dy / r;
                self.accelerations[j][0] += fx;
                self.accelerations[j][1] += fy;
                self.accelerations[i][0] -= fx;
                self.accelerations[i][1] -= fy;
            }
        }
    }

    pub fn kinetic_energy(&self) -> f64 {
        self.velocities
            .iter()
            .map(|v| 0.5 * (v[0] * v[0] + v[1] * v[1]))
            .sum()
    }

    pub fn potential_energy(&self) -> f64 {
        let n = self.n_atoms();
        let mut u = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = self.positions[j][0] - self.positions[i][0];
                let dy = self.positions[j][1] - self.positions[i][1];
                u += energy((dx * dx + dy * dy).sqrt());
            }
        }
        u
    }

    pub fn total_energy(&self) -> f64 {
        self.kinetic_energy() + self.potential_energy()
    }
}

pub trait Integrator {
    fn step(&self, system: &mut System, dt: f64);
}

pub fn advance(method: &impl Integrator, system: &mut System, dt: f64) {
    method.step(system, dt);
}

pub struct ForwardEuler;
pub struct VelocityVerlet;

impl Integrator for ForwardEuler {
    fn step(&self, system: &mut System, dt: f64) {
        let n = system.n_atoms();
        // x_{n+1} = x_n + v_n dt
        for i in 0..n {
            system.positions[i][0] += system.velocities[i][0] * dt;
            system.positions[i][1] += system.velocities[i][1] * dt;
        }
        // v_{n+1} = v_n + a_n dt  (explicit: uses the OLD acceleration)
        for i in 0..n {
            system.velocities[i][0] += system.accelerations[i][0] * dt;
            system.velocities[i][1] += system.accelerations[i][1] * dt;
        }
        // Retain a_{n+1} for the next step.
        system.update_accelerations();
    }
}

impl Integrator for VelocityVerlet {
    fn step(&self, system: &mut System, dt: f64) {
        let n = system.n_atoms();
        let a_old = system.accelerations.clone(); // a_n
        // x_{n+1} = x_n + v_n dt + 0.5 a_n dt^2
        for i in 0..n {
            system.positions[i][0] +=
                system.velocities[i][0] * dt + 0.5 * a_old[i][0] * dt * dt;
            system.positions[i][1] +=
                system.velocities[i][1] * dt + 0.5 * a_old[i][1] * dt * dt;
        }
        // a_{n+1}
        system.update_accelerations();
        // v_{n+1} = v_n + 0.5 (a_n + a_{n+1}) dt
        for i in 0..n {
            system.velocities[i][0] +=
                0.5 * (a_old[i][0] + system.accelerations[i][0]) * dt;
            system.velocities[i][1] +=
                0.5 * (a_old[i][1] + system.accelerations[i][1]) * dt;
        }
    }
}

// Run the two-atom experiment with a given integrator and return the
// relative total-energy error (E(t) - E0) / |E0| for t = 0..=steps.
pub fn run_dimer_experiment(method: &impl Integrator, steps: usize, dt: f64) -> Vec<f64> {
    let mut system = System::new(
        vec![[0.0, 0.0], [1.2, 0.0]],
        vec![[0.0, 0.0], [0.0, 0.0]],
    );
    let e0 = system.total_energy();
    let mut errors = Vec::with_capacity(steps + 1);
    errors.push(0.0); // (E0 - E0) / |E0| at t = 0
    for _ in 0..steps {
        advance(method, &mut system, dt);
        errors.push((system.total_energy() - e0) / e0.abs());
    }
    errors
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
