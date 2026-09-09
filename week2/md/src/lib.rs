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


// Fixed course cutoff for the Lennard-Jones fluid (Part 4).
pub const RC: f64 = 2.5;

// Shifted Lennard-Jones pair potential with cutoff.
// U_cut(r) = U(r) - U(rc) for r < rc, and 0 for r >= rc.
pub fn energy_shifted(r: f64, rc: f64) -> f64 {
    if r < rc {
        energy(r) - energy(rc)
    } else {
        0.0
    }
}

// Lennard-Jones radial force with cutoff: the Part 2 force for r < rc,
// zero for r >= rc.
pub fn force_cut(r: f64, rc: f64) -> f64 {
    if r < rc {
        force(r)
    } else {
        0.0
    }
}


// ---------------------------------------------------------------------------
// Part 3: two-atom Lennard-Jones molecular dynamics
// ---------------------------------------------------------------------------

pub struct System {
    positions: Vec<[f64; 2]>,
    velocities: Vec<[f64; 2]>,
    accelerations: Vec<[f64; 2]>, // cached: always a(x) at current positions
    box_len: Option<[f64; 2]>,    // None => open boundary (dimer)
    rc: Option<f64>,              // None => no cutoff (dimer)
}

impl System {
    pub fn n_atoms(&self) -> usize {
        self.positions.len()
    }

    pub fn new(positions: Vec<[f64; 2]>, velocities: Vec<[f64; 2]>) -> System {
        Self::build(positions, velocities, None, None)
    }

    pub fn new_periodic(
        positions: Vec<[f64; 2]>,
        velocities: Vec<[f64; 2]>,
        box_len: [f64; 2],
        rc: f64,
    ) -> System {
        Self::build(positions, velocities, Some(box_len), Some(rc))
    }

    fn build(
        positions: Vec<[f64; 2]>,
        velocities: Vec<[f64; 2]>,
        box_len: Option<[f64; 2]>,
        rc: Option<f64>,
    ) -> System {
        assert_eq!(positions.len(), velocities.len());
        let n = positions.len();
        let mut system = System {
            positions,
            velocities,
            accelerations: vec![[0.0, 0.0]; n],
            box_len,
            rc,
        };
        // Compute the initial accelerations before the first step.
        system.update_accelerations();
        system
    }

    // Minimum-image separation of a raw displacement (dx, dy).
    fn separation(&self, dx: f64, dy: f64) -> (f64, f64) {
        match self.box_len {
            Some([lx, ly]) => {
                let dx = dx - lx * (dx / lx).round();
                let dy = dy - ly * (dy / ly).round();
                (dx, dy)
            }
            None => (dx, dy),
        }
    }

    // Wrap positions back into [0, Lx) x [0, Ly) when periodic.
    pub fn apply_periodic_wrap(&mut self) {
        if let Some([lx, ly]) = self.box_len {
            for p in &mut self.positions {
                p[0] = p[0].rem_euclid(lx);
                p[1] = p[1].rem_euclid(ly);
            }
        }
    }

    // Acceleration from the Lennard-Jones pair force. Mass = 1, so a = F.
    // With a periodic box, uses the minimum-image convention; with a cutoff,
    // pairs at r >= rc contribute no force.
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
                let (dx, dy) = self.separation(dx, dy);
                let r = (dx * dx + dy * dy).sqrt();
                let f = match self.rc {
                    Some(rc) => force_cut(r, rc),
                    None => force(r),
                };
                if f == 0.0 {
                    continue;
                }
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
                let (dx, dy) = self.separation(dx, dy);
                let r = (dx * dx + dy * dy).sqrt();
                u += match self.rc {
                    Some(rc) => energy_shifted(r, rc),
                    None => energy(r),
                };
            }
        }
        u
    }

    pub fn total_energy(&self) -> f64 {
        self.kinetic_energy() + self.potential_energy()
    }

    // Thermostat temperature: after removing the center-of-mass velocity,
    // 2 degrees of freedom are gone, so T_thermo = E_kin / (N - 1).
    pub fn thermostat_temperature(&self) -> f64 {
        let n = self.n_atoms();
        assert!(n > 1, "thermostat temperature needs N > 1");
        self.kinetic_energy() / (n - 1) as f64
    }

    pub fn subtract_com_velocity(&mut self) {
        let n = self.n_atoms();
        if n == 0 {
            return;
        }
        let mut sum = [0.0, 0.0];
        for v in &self.velocities {
            sum[0] += v[0];
            sum[1] += v[1];
        }
        let com = [sum[0] / n as f64, sum[1] / n as f64];
        for v in &mut self.velocities {
            v[0] -= com[0];
            v[1] -= com[1];
        }
    }

    // Rescale velocities to a target temperature using T_thermo.
    pub fn rescale_to_temperature(&mut self, target: f64) {
        let t = self.thermostat_temperature();
        if t <= 0.0 {
            return;
        }
        let scale = (target / t).sqrt();
        for v in &mut self.velocities {
            v[0] *= scale;
            v[1] *= scale;
        }
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
        // Periodic systems wrap positions back into the box here.
        system.apply_periodic_wrap();
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


    // Part 4: cutoff wrappers
    #[test]
    fn shifted_potential_continuous_at_cutoff() {
        let rc = 2.5;
        // Exactly zero at and beyond the cutoff.
        assert_eq!(energy_shifted(rc, rc), 0.0);
        assert_eq!(energy_shifted(rc + 0.5, rc), 0.0);
        assert_eq!(force_cut(rc, rc), 0.0);
        assert_eq!(force_cut(rc + 0.5, rc), 0.0);
        // Just inside the cutoff the shifted potential is ~continuous:
        // |U(rc - eps) - U(rc)| = |F(rc)| * eps + O(eps^2).
        let eps = 1e-6;
        assert!(
            energy_shifted(rc - eps, rc).abs() < 1e-6,
            "shifted potential should be ~continuous just inside rc, got {}",
            energy_shifted(rc - eps, rc)
        );
        // force_cut equals the Part 2 force inside the cutoff.
        assert_eq!(force_cut(rc - eps, rc), force(rc - eps));
    }

    // Part 4: periodic N-body checks
    #[test]
    fn total_internal_force_sums_to_zero() {
        let box_len = [10.0, 10.0];
        let rc = 2.5;
        // Distinct sites, some near the boundary so minimum image is used.
        let positions = vec![
            [0.5, 0.5],
            [1.2, 0.5],
            [9.8, 9.8],
            [0.5, 9.7],
            [4.0, 5.0],
        ];
        let velocities = vec![[0.0, 0.0]; positions.len()];
        let mut system = System::new_periodic(positions, velocities, box_len, rc);
        system.update_accelerations();
        let mut sum = [0.0, 0.0];
        for a in &system.accelerations {
            sum[0] += a[0];
            sum[1] += a[1];
        }
        assert!(
            sum[0].abs() < 1e-12 && sum[1].abs() < 1e-12,
            "net internal force should be zero, got {:?}",
            sum
        );
    }

    #[test]
    fn periodic_pair_uses_minimum_image() {
        let box_len = [10.0, 10.0];
        // Two atoms near opposite x edges: true distance is 1.0, not 9.0.
        let positions = vec![[0.5, 5.0], [9.5, 5.0]];
        let velocities = vec![[0.0, 0.0], [0.0, 0.0]];
        let system = System::new_periodic(positions, velocities, box_len, 2.5);
        let expected = energy_shifted(1.0, 2.5);
        assert!(
            (system.potential_energy() - expected).abs() < 1e-12,
            "potential energy should use the minimum-image distance, got {}",
            system.potential_energy()
        );
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
