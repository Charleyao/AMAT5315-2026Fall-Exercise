//! Wolff single-cluster update.

use crate::lattice::Lattice;
use rand::Rng;
use rand::rngs::StdRng;

/// One Wolff cluster flip: grow one cluster from a uniformly random seed site,
/// adding each same-spin neighbour with probability `1 - exp(-2 / T)`, then
/// flip every spin in the cluster.
///
/// Returns the number of spins flipped (the cluster size).
pub fn cluster_flip(lattice: &mut Lattice, temperature: f64, rng: &mut StdRng) -> usize {
    let l = lattice.l;
    let n = lattice.n();
    let add_prob = 1.0 - (-2.0 / temperature).exp();

    let seed = rng.gen_range(0..n);
    let cluster_spin = lattice.spins[seed];

    let mut in_cluster = vec![false; n];
    let mut cluster = Vec::new();
    let mut stack = vec![seed];
    in_cluster[seed] = true;
    cluster.push(seed);

    while let Some(index) = stack.pop() {
        let x = index % l;
        let y = index / l;
        let neighbours = [
            ((x + l - 1) % l, y),
            ((x + 1) % l, y),
            (x, (y + l - 1) % l),
            (x, (y + 1) % l),
        ];
        for (nx, ny) in neighbours {
            let j = ny * l + nx;
            if !in_cluster[j]
                && lattice.spins[j] == cluster_spin
                && rng.gen_range(0.0..1.0) < add_prob
            {
                in_cluster[j] = true;
                stack.push(j);
                cluster.push(j);
            }
        }
    }

    for &index in &cluster {
        lattice.flip(index);
    }

    cluster.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn cold_cluster_covers_the_whole_lattice() {
        // At low T the add probability is essentially 1, so the cluster is the
        // entire connected up-spin component, i.e. every site.
        let mut lattice = Lattice::all_up(8);
        let mut rng = StdRng::seed_from_u64(2026);
        let size = cluster_flip(&mut lattice, 0.01, &mut rng);
        assert_eq!(size, 64);
        assert!((lattice.magnetization() + 1.0).abs() < 1e-12);
    }

    #[test]
    fn hot_cluster_is_small_on_average() {
        // At high T clusters are tiny; a single flip never exceeds the lattice.
        let mut lattice = Lattice::all_up(32);
        let mut rng = StdRng::seed_from_u64(3);
        let mut total = 0usize;
        let flips = 50;
        for _ in 0..flips {
            total += cluster_flip(&mut lattice, 100.0, &mut rng);
        }
        let mean = total as f64 / flips as f64;
        assert!(mean < 4.0, "mean cluster size was {mean}");
    }
}
