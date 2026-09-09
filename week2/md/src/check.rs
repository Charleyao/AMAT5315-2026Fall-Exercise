//! Re-analysis of saved production frames: secular energy drift, T_speed,
//! and the 24-bin 2D Maxwell-Boltzmann speed-distribution chi^2.

use crate::io::SimulationOutput;
use crate::{RC, System};

pub struct CheckReport {
    pub secular_drift: f64,
    pub t_speed: f64,
    pub chi2_over_22: f64,
    pub passed: bool,
}

// Course gate constants (match week2/checker/check).
const SECULAR_TOL: f64 = 2e-3;
const TEMP_TOL: f64 = 0.05;
const CHI2_DOF_TOL: f64 = 2.0;
const TARGET_TEMP: f64 = 0.5;

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

// Course definition: E0 = first frame total energy;
// k = max(1, floor(frames/10)); drift = |mean(last k) - mean(first k)| / |E0|.
pub fn secular_drift(energies: &[f64]) -> f64 {
    assert!(!energies.is_empty());
    let k = (energies.len() / 10).max(1);
    let first_mean = mean(&energies[..k]);
    let last_mean = mean(&energies[energies.len() - k..]);
    (last_mean - first_mean).abs() / energies[0].abs()
}

// T_speed = <v^2> / 2 over all pooled speeds.
pub fn t_speed(speeds: &[f64]) -> f64 {
    let m = speeds.len() as f64;
    speeds.iter().map(|s| s * s).sum::<f64>() / m / 2.0
}

// 24 equal-probability bins for the 2D Maxwell-Boltzmann speed distribution:
// b_k = sqrt(-2 t ln(1 - k/24)), k = 0..23, b_24 = inf.
// chi2_over_22 = (1/22) * sum_b (O_b - E_b)^2 / E_b, E_b = M/24.
pub fn chi2_over_22(speeds: &[f64], t: f64) -> f64 {
    let n_bins = 24;
    let m = speeds.len() as f64;
    let expected = m / n_bins as f64;

    let mut edges = Vec::with_capacity(n_bins - 1);
    for k in 1..n_bins {
        let edge = (-2.0 * t * (1.0 - (k as f64) / (n_bins as f64)).ln()).sqrt();
        edges.push(edge);
    }

    let mut observed = vec![0usize; n_bins];
    for s in speeds {
        let mut bin = n_bins - 1;
        for (idx, edge) in edges.iter().enumerate() {
            if *s < *edge {
                bin = idx; // s < b_{idx+1}
                break;
            }
        }
        observed[bin] += 1;
    }

    observed
        .iter()
        .map(|o| {
            let diff = *o as f64 - expected;
            diff * diff / expected
        })
        .sum::<f64>()
        / 22.0
}

// Recompute the total energy of every saved frame from raw positions and
// velocities (never the stored E_pot/E_kin).
fn recompute_energies(output: &SimulationOutput) -> Vec<f64> {
    output
        .frames
        .iter()
        .map(|f| {
            let system = System::new_periodic(f.pos.clone(), f.vel.clone(), output.meta.box_len, RC);
            system.potential_energy() + system.kinetic_energy()
        })
        .collect()
}

pub fn check(output: &SimulationOutput) -> Result<CheckReport, String> {
    if output.frames.is_empty() {
        return Err("no saved frames to check".to_string());
    }

    let energies = recompute_energies(output);
    let speeds: Vec<f64> = output
        .frames
        .iter()
        .flat_map(|f| f.vel.iter())
        .map(|v| (v[0] * v[0] + v[1] * v[1]).sqrt())
        .collect();

    let drift = secular_drift(&energies);
    let t = t_speed(&speeds);
    let chi2 = chi2_over_22(&speeds, t);

    let passed = drift < SECULAR_TOL && (t - TARGET_TEMP).abs() < TEMP_TOL && chi2 < CHI2_DOF_TOL;

    Ok(CheckReport {
        secular_drift: drift,
        t_speed: t,
        chi2_over_22: chi2,
        passed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secular_drift_matches_course_formula() {
        let mut e = vec![1.0; 30];
        for i in 27..30 {
            e[i] = 1.002;
        }
        assert!((secular_drift(&e) - 0.002).abs() < 1e-12);
    }

    #[test]
    fn t_speed_is_mean_squared_speed_over_two() {
        assert!((t_speed(&[1.0; 10]) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn chi2_all_in_one_bin() {
        let speeds = vec![0.1; 24];
        assert!((chi2_over_22(&speeds, 0.5) - 552.0 / 22.0).abs() < 1e-12);
    }
}
