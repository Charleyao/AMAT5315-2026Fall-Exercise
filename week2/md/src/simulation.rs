//! Lattice construction, velocity initialization, and the simulation driver
//! for the Lennard-Jones fluid (Part 4).

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
}
