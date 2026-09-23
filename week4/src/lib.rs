pub mod advdiff;
pub mod grid;

pub trait Integrator {
    fn step<F>(&self, y: &[f64], dt: f64, rhs: F) -> Vec<f64>
    where
        F: Fn(&[f64]) -> Vec<f64>;
}

// ============================================================
// Forward Euler
// ============================================================

pub struct Euler;

impl Integrator for Euler {
    fn step<F>(&self, y: &[f64], dt: f64, rhs: F) -> Vec<f64>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let k1 = rhs(y);

        y.iter()
            .zip(k1.iter())
            .map(|(yi, k1i)| yi + dt * k1i)
            .collect()
    }
}

// ============================================================
// Explicit midpoint / RK2
// ============================================================

pub struct Midpoint;

impl Integrator for Midpoint {
    fn step<F>(&self, y: &[f64], dt: f64, rhs: F) -> Vec<f64>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let k1 = rhs(y);

        let y_mid: Vec<f64> = y
            .iter()
            .zip(k1.iter())
            .map(|(yi, k1i)| yi + 0.5 * dt * k1i)
            .collect();

        let k2 = rhs(&y_mid);

        y.iter()
            .zip(k2.iter())
            .map(|(yi, k2i)| yi + dt * k2i)
            .collect()
    }
}

// ============================================================
// Classical RK4
// ============================================================

pub struct RK4;

impl Integrator for RK4 {
    fn step<F>(&self, y: &[f64], dt: f64, rhs: F) -> Vec<f64>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let k1 = rhs(y);

        let y2: Vec<f64> = y
            .iter()
            .zip(k1.iter())
            .map(|(yi, k1i)| yi + 0.5 * dt * k1i)
            .collect();

        let k2 = rhs(&y2);

        let y3: Vec<f64> = y
            .iter()
            .zip(k2.iter())
            .map(|(yi, k2i)| yi + 0.5 * dt * k2i)
            .collect();

        let k3 = rhs(&y3);

        let y4: Vec<f64> = y
            .iter()
            .zip(k3.iter())
            .map(|(yi, k3i)| yi + dt * k3i)
            .collect();

        let k4 = rhs(&y4);

        y.iter()
            .zip(k1.iter())
            .zip(k2.iter())
            .zip(k3.iter())
            .zip(k4.iter())
            .map(|((((yi, k1i), k2i), k3i), k4i)| {
                yi + dt / 6.0 * (k1i + 2.0 * k2i + 2.0 * k3i + k4i)
            })
            .collect()
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// The test ODE: y' = -y, y(0) = 1, i.e. F(y) = -y.
    fn neg(y: &[f64]) -> Vec<f64> {
        y.iter().map(|v| -v).collect()
    }

    fn assert_close(got: &[f64], want: &[f64], tol: f64) {
        assert_eq!(got.len(), want.len(), "lengths differ: {got:?} vs {want:?}");
        for (g, w) in got.iter().zip(want.iter()) {
            assert!(
                (g - w).abs() < tol,
                "got {g}, want {w} (tol {tol})"
            );
        }
    }

    #[test]
    fn euler_takes_one_step_of_y_prime_equals_minus_y() {
        let dt = 0.1;
        let y1 = Euler.step(&[1.0], dt, neg);
        // y1 = y0 + dt * F(y0) = 1 - dt
        assert_close(&y1, &[1.0 - dt], 1e-12);
    }

    #[test]
    fn midpoint_takes_one_step_of_y_prime_equals_minus_y() {
        let dt = 0.1;
        let y1 = Midpoint.step(&[1.0], dt, neg);
        // 1 - dt + dt^2 / 2
        assert_close(&y1, &[1.0 - dt + 0.5 * dt * dt], 1e-12);
    }

    #[test]
    fn rk4_takes_one_step_of_y_prime_equals_minus_y() {
        let dt = 0.1;
        let y1 = RK4.step(&[1.0], dt, neg);
        // 1 - dt + dt^2/2 - dt^3/6 + dt^4/24, the 4th-order Taylor of exp(-dt)
        let want = 1.0 - dt + dt * dt / 2.0 - dt * dt * dt / 6.0 + dt.powi(4) / 24.0;
        assert_close(&y1, &[want], 1e-12);
    }

    #[test]
    fn euler_calls_the_rhs_once() {
        use std::cell::Cell;
        let calls = Cell::new(0);
        let rhs = |y: &[f64]| {
            calls.set(calls.get() + 1);
            y.iter().map(|v| -v).collect()
        };
        let _ = Euler.step(&[1.0], 0.1, rhs);
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn midpoint_evaluates_at_the_formula_points() {
        use std::cell::RefCell;
        let y0 = 2.0;
        let dt = 0.1;
        let seen = RefCell::new(Vec::new());
        let rhs = |y: &[f64]| {
            seen.borrow_mut().push(y[0]);
            vec![-y[0]]
        };
        let _ = Midpoint.step(&[y0], dt, rhs);
        // k1 = F(y0); k2 = F(y0 + dt*k1/2)
        let p1 = y0;
        let p2 = y0 - 0.5 * dt * y0;
        let seen = seen.into_inner();
        assert_close(&seen, &[p1, p2], 1e-12);
    }

    #[test]
    fn rk4_evaluates_at_the_formula_points() {
        use std::cell::RefCell;
        let y0 = 2.0;
        let dt = 0.1;
        let seen = RefCell::new(Vec::new());
        let rhs = |y: &[f64]| {
            seen.borrow_mut().push(y[0]);
            vec![-y[0]]
        };
        let _ = RK4.step(&[y0], dt, rhs);
        // k1 = F(y0); k2 = F(y0 + dt*k1/2); k3 = F(y0 + dt*k2/2); k4 = F(y0 + dt*k3)
        let p1 = y0;
        let p2 = y0 - 0.5 * dt * y0; // k1 = -y0
        let p3 = y0 - 0.5 * dt * p2; // k2 = -p2
        let p4 = y0 - dt * p3;       // k3 = -p3
        let seen = seen.into_inner();
        assert_close(&seen, &[p1, p2, p3, p4], 1e-12);
    }

    #[test]
    fn rk4_is_more_accurate_than_euler_at_the_same_step() {
        let dt: f64 = 0.1;
        let exact = (-dt).exp();
        let euler = Euler.step(&[1.0], dt, neg)[0];
        let rk4 = RK4.step(&[1.0], dt, neg)[0];
        assert!((rk4 - exact).abs() < (euler - exact).abs());
    }
}
