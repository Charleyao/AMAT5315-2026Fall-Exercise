//! Single-site Metropolis update.

use crate::lattice::Lattice;
use rand::Rng;
use rand::rngs::StdRng;

/// One Monte Carlo sweep: exactly `L * L` proposals, each site chosen
/// independently and uniformly at random, with replacement.
///
/// Returns the number of accepted proposals.
pub fn sweep(lattice: &mut Lattice, temperature: f64, rng: &mut StdRng) -> usize {
    let n = lattice.n();
    let l = lattice.l;
    let mut accepted = 0usize;

    for _ in 0..n {
        let index = rng.gen_range(0..n);
        let x = index % l;
        let y = index / l;
        let spin = lattice.spins[index] as i32;
        let field = lattice.neighbour_sum(x, y); // sum of the four neighbours
        // Flipping one spin changes the energy by 2 * s_i * (sum of neighbours).
        let delta_e = 2 * spin * field;

        let accept = if delta_e <= 0 {
            true
        } else {
            // `gen` is a reserved keyword in edition 2024, so draw a uniform
            // value with `gen_range`.
            rng.gen_range(0.0..1.0) < (-(delta_e as f64) / temperature).exp()
        };

        if accept {
            lattice.flip(index);
            accepted += 1;
        }
    }

    accepted
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn cold_start_never_accepts_uphill_flips() {
        // From an all-up lattice every flip costs +8, and exp(-8 / 0.01) is
        // numerically zero, so a sweep at a very low temperature accepts nothing.
        let mut lattice = Lattice::all_up(8);
        let mut rng = StdRng::seed_from_u64(2026);
        let accepted = sweep(&mut lattice, 0.01, &mut rng);
        assert_eq!(accepted, 0);
        assert!((lattice.magnetization() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn high_temperature_accepts_roughly_half_the_proposals() {
        // At large T the acceptance probability of an uphill move tends to 1,
        // so nearly every proposal is accepted.
        let mut lattice = Lattice::all_up(16);
        let mut rng = StdRng::seed_from_u64(7);
        let accepted = sweep(&mut lattice, 100.0, &mut rng);
        assert!(accepted > 200);
    }

    #[test]
    fn same_seed_replays_the_same_sweep() {
        let mut a = Lattice::all_up(8);
        let mut b = Lattice::all_up(8);
        let mut rng_a = StdRng::seed_from_u64(11);
        let mut rng_b = StdRng::seed_from_u64(11);
        for _ in 0..5 {
            sweep(&mut a, 2.3, &mut rng_a);
            sweep(&mut b, 2.3, &mut rng_b);
        }
        assert_eq!(a.spins, b.spins);
    }
}
