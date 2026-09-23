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
