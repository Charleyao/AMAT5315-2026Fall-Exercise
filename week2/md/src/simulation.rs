//! Lattice construction, velocity initialization, and the simulation driver
//! for the Lennard-Jones fluid (Part 4).

use crate::io::{RunMeta, SimulationOutput, TrajFrame};
use crate::{advance, ForceMethod, RC, System, VelocityVerlet};
use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand::SeedableRng;

// Nearest-neighbor distance from the density: rho = 1 / (a * h),
// h = (sqrt(3)/2) * a  =>  a = sqrt(2 / (sqrt(3) * rho)).
pub fn lattice_a(rho: f64) -> f64 {
    (2.0 / (3.0_f64.sqrt() * rho)).sqrt()
}

pub fn lattice_h(a: f64) -> f64 {
    3.0_f64.sqrt() / 2.0 * a
}

// 10x10 triangular lattice in a rectangular periodic box.
// Only n = m^2 with even m is supported (10x10 => m = 10).
pub fn build_lattice(n: usize, rho: f64) -> Result<(Vec<[f64; 2]>, [f64; 2]), String> {
    let m = (n as f64).sqrt().round() as usize;
    if m < 2 || m * m != n || m % 2 != 0 {
        return Err(format!(
            "n = {n} is not supported: need n = m^2 with even m (default 100 = 10x10)"
        ));
    }
    let a = lattice_a(rho);
    let h = lattice_h(a);
    let lx = m as f64 * a;
    let ly = m as f64 * h;
    let mut positions = Vec::with_capacity(n);
    for j in 0..m {
        for i in 0..m {
            let x = (i as f64 + 0.5 * ((j % 2) as f64)) * a;
            let y = j as f64 * h;
            positions.push([x, y]);
        }
    }
    Ok((positions, [lx, ly]))
}

// Independent Gaussian velocity components with mean 0 and variance
// `temperature`, drawn from a deterministic seeded RNG.
pub fn gaussian_velocities(n: usize, temperature: f64, seed: u64) -> Vec<[f64; 2]> {
    let mut rng = StdRng::seed_from_u64(seed);
    let sigma = temperature.sqrt();
    // Box-Muller transform: two independent uniform draws produce two
    // independent standard-normal draws. u1 is drawn from [1e-12, 1) so the
    // log is always finite.
    let positive = Uniform::new(1e-12_f64, 1.0_f64);
    let unit = Uniform::new(0.0_f64, 1.0_f64);
    (0..n)
        .map(|_| {
            let u1 = positive.sample(&mut rng);
            let u2 = unit.sample(&mut rng);
            let radius = (-2.0 * u1.ln()).sqrt();
            let angle = 2.0 * std::f64::consts::PI * u2;
            [sigma * radius * angle.cos(), sigma * radius * angle.sin()]
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct RunConfig {
    pub n: usize,
    pub rho: f64,
    pub temperature: f64,
    pub dt: f64,
    pub eq_steps: usize,
    pub steps: usize,
    pub sample_every: usize,
    pub seed: u64,
    pub force_method: ForceMethod,
    pub ramp_to: Option<f64>,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            n: 100,
            rho: 0.8,
            temperature: 0.5,
            dt: 0.01,
            eq_steps: 2000,
            steps: 10000,
            sample_every: 50,
            seed: 2026,
            force_method: ForceMethod::Cells,
            ramp_to: None,
        }
    }
}

// Run the full simulation: build the lattice, initialize velocities
// (Gaussian -> subtract COM -> rescale with T_thermo), equilibrate with a
// velocity-rescaling thermostat every 50 steps, then run production (thermostat
// OFF unless --ramp-to is set) and save every `sample_every` steps (step 0 is
// not saved).

// Target thermostat temperature at a production step for a linear ramp from
// `t0` (start of production) to `tf` (final production step).
fn ramp_target(t0: f64, tf: f64, step: usize, total_steps: usize) -> f64 {
    debug_assert!(step <= total_steps && total_steps > 0);
    let f = step as f64 / total_steps as f64;
    t0 + (tf - t0) * f
}

pub fn run_simulation(config: &RunConfig) -> Result<SimulationOutput, String> {
    let (positions, box_len) = build_lattice(config.n, config.rho)?;
    let velocities = gaussian_velocities(config.n, config.temperature, config.seed);
    let mut system = System::new_periodic(positions, velocities, box_len, RC)
        .with_force_method(config.force_method);

    // Initial: subtract the center-of-mass velocity immediately, then rescale
    // to the target T using T_thermo = E_kin / (N - 1).
    system.subtract_com_velocity();
    system.rescale_to_temperature(config.temperature);

    // Equilibration with the thermostat ON: rescale every 50 steps using the
    // same T_thermo definition. Uniform rescaling preserves zero COM velocity,
    // so no per-rescale COM subtraction is needed.
    for s in 0..config.eq_steps {
        advance(&VelocityVerlet, &mut system, config.dt);
        if s % 50 == 0 {
            system.rescale_to_temperature(config.temperature);
        }
    }
    // Zero the center-of-mass momentum once more before production, matching
    // the reference week2/sim.py cadence.
    system.subtract_com_velocity();

    // Production. By default the thermostat is OFF (NVE). With --ramp-to the
    // target temperature rises linearly from config.temperature (start of
    // production) to ramp_to (final production step) and velocities are
    // rescaled every 50 production steps, exactly like equilibration (same
    // T_thermo = E_kin / (N - 1) definition).
    let mut frames = Vec::new();
    for step in 1..=config.steps {
        advance(&VelocityVerlet, &mut system, config.dt);
        if let Some(tf) = config.ramp_to {
            if step % 50 == 0 {
                let target = ramp_target(config.temperature, tf, step, config.steps);
                system.rescale_to_temperature(target);
            }
        }
        if step % config.sample_every == 0 {
            frames.push(TrajFrame {
                step,
                t: step as f64 * config.dt,
                pos: system.positions.clone(),
                vel: system.velocities.clone(),
                e_pot: system.potential_energy(),
                e_kin: system.kinetic_energy(),
            });
        }
    }

    Ok(SimulationOutput {
        meta: RunMeta {
            n: config.n,
            rho: config.rho,
            box_len,
            dt: config.dt,
            temperature: config.temperature,
            eq_steps: config.eq_steps,
            steps: config.steps,
            sample_every: config.sample_every,
            seed: config.seed,
            integrator: "velocity-verlet".to_string(),
            ramp_to: config.ramp_to,
        },
        frames,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lattice_has_correct_density_and_size() {
        let (positions, [lx, ly]) = build_lattice(100, 0.8).unwrap();
        assert_eq!(positions.len(), 100);
        assert!((100.0 / (lx * ly) - 0.8).abs() < 1e-12);
        for p in &positions {
            assert!(p[0] >= 0.0 && p[0] < lx);
            assert!(p[1] >= 0.0 && p[1] < ly);
        }
        let a = lattice_a(0.8);
        assert!((a - 1.201405).abs() < 1e-4);
        assert!((lx - 10.0 * a).abs() < 1e-12);
        assert!((ly - 10.0 * lattice_h(a)).abs() < 1e-12);
    }

    #[test]
    fn lattice_rejects_unsupported_n() {
        assert!(build_lattice(50, 0.8).is_err());
        assert!(build_lattice(81, 0.8).is_err());
    }

    #[test]
    fn gaussian_velocities_are_seeded_and_zero_com_after_subtract() {
        let v1 = gaussian_velocities(100, 0.5, 2026);
        let v2 = gaussian_velocities(100, 0.5, 2026);
        assert_eq!(v1, v2); // same seed => identical velocities
        let v3 = gaussian_velocities(100, 0.5, 2027);
        assert_ne!(v1, v3); // different seed => different velocities

        let (positions, box_len) = build_lattice(100, 0.8).unwrap();
        let mut system = System::new_periodic(positions, v1, box_len, RC);
        system.subtract_com_velocity();
        let mut sum = [0.0, 0.0];
        for v in &system.velocities {
            sum[0] += v[0];
            sum[1] += v[1];
        }
        assert!(sum[0].abs() < 1e-12 && sum[1].abs() < 1e-12);
        // After subtracting COM and rescaling with T_thermo, the thermostat
        // temperature matches the target.
        system.rescale_to_temperature(0.5);
        assert!((system.thermostat_temperature() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn small_run_produces_one_saved_frame() {
        let config = RunConfig {
            n: 36,
            rho: 0.8,
            temperature: 0.5,
            dt: 0.01,
            eq_steps: 0,
            steps: 50,
            sample_every: 50,
            seed: 2026,
            force_method: ForceMethod::Cells,
            ramp_to: None,
        };
        let output = run_simulation(&config).unwrap();
        assert_eq!(output.frames.len(), 1);
        let f = &output.frames[0];
        assert_eq!(f.step, 50);
        assert!((f.t - 0.5).abs() < 1e-12);
        assert_eq!(f.pos.len(), 36);
        assert_eq!(f.vel.len(), 36);
        for p in &f.pos {
            assert!(p[0] >= 0.0 && p[0] < output.meta.box_len[0]);
            assert!(p[1] >= 0.0 && p[1] < output.meta.box_len[1]);
        }
        assert!(f.e_pot.is_finite() && f.e_kin.is_finite());
        assert_eq!(output.meta.integrator.as_str(), "velocity-verlet");
    }

    #[test]
    fn run_config_defaults_to_cells() {
        assert_eq!(RunConfig::default().force_method, ForceMethod::Cells);
        assert_eq!(RunConfig::default().ramp_to, None);
    }

    #[test]
    fn heating_ramp_temperature_linearly_every_50_production_steps() {
        // eq = 0 so production starts at --temperature; the target ramps
        // linearly 0.2 -> 1.2 over the 400 production steps and velocities are
        // rescaled every 50 steps. Frames are sampled every 50 steps (exactly
        // at each rescale), so each frame's thermostat temperature must equal
        // the ramp target at that production step.
        let config = RunConfig {
            n: 36,
            rho: 0.8,
            temperature: 0.2,
            dt: 0.01,
            eq_steps: 0,
            steps: 400,
            sample_every: 50,
            seed: 2026,
            force_method: ForceMethod::Naive,
            ramp_to: Some(1.2),
        };
        let output = run_simulation(&config).unwrap();
        assert_eq!(output.frames.len(), 8); // steps 50, 100, ..., 400
        let n = 36usize;
        for f in &output.frames {
            let measured = f.e_kin / (n - 1) as f64;
            let expected = 0.2 + 1.0 * f.step as f64 / config.steps as f64;
            let tol = 1e-6 * expected.abs().max(1.0);
            assert!(
                (measured - expected).abs() <= tol,
                "step {}: measured T = {measured}, expected ramp target {expected}",
                f.step
            );
        }
        // First and last targets: step 50 -> 0.325, final step 400 -> 1.2.
        let first = &output.frames[0];
        let last = &output.frames[output.frames.len() - 1];
        assert!((first.e_kin / (n - 1) as f64 - 0.325).abs() < 1e-6);
        assert!((last.e_kin / (n - 1) as f64 - 1.2).abs() < 1e-6);
    }

    #[test]
    fn ramp_target_is_linear_between_endpoints() {
        assert_eq!(ramp_target(0.2, 1.2, 0, 400), 0.2);
        assert!((ramp_target(0.2, 1.2, 400, 400) - 1.2).abs() < 1e-12);
        assert!((ramp_target(0.2, 1.2, 200, 400) - 0.7).abs() < 1e-12);
        // Monotonic: later rescale steps have strictly higher targets.
        let mut prev = ramp_target(0.2, 1.2, 50, 400);
        for step in (100..=400).step_by(50) {
            let t = ramp_target(0.2, 1.2, step, 400);
            assert!(t > prev, "ramp not monotonic at step {step}");
            prev = t;
        }
    }

    #[test]
    fn default_unheated_run_has_no_ramp_and_keeps_temperature() {
        let config = RunConfig {
            n: 36,
            rho: 0.8,
            temperature: 0.5,
            dt: 0.01,
            eq_steps: 1000,
            steps: 100,
            sample_every: 100,
            seed: 2026,
            force_method: ForceMethod::Naive,
            ramp_to: None,
        };
        let output = run_simulation(&config).unwrap();
        assert_eq!(output.meta.ramp_to, None);
        assert_eq!(output.frames.len(), 1);
        // Production thermostat stays OFF: the temperature is not pulled to any
        // ramp target and stays near the equilibrated 0.5 over 100 NVE steps.
        let measured = output.frames[0].e_kin / 35.0;
        assert!(measured > 0.3 && measured < 0.7, "measured T = {measured}");
    }
}
