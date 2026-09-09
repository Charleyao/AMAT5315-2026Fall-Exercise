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
}
