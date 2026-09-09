//! Lattice construction, velocity initialization, and the simulation driver
//! for the Lennard-Jones fluid (Part 4).

use crate::{RC, System};
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
    // independent standard-normal draws. `rand` 0.8 has no StandardNormal,
    // so we generate the Gaussian pair directly.
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
}
