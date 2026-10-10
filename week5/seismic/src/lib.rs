//! Forward solver for the damped 2-D acoustic wave equation.
//!
//! All quantities are in the reduced units of the experiment JSON. Arrays are
//! indexed `[z, x]` and flattened C-order as `z * nx + x`. The numerical scheme
//! matches the accompanying NumPy baseline exactly:
//!
//! ```text
//! u_next = [ 2 u - (1 - sigma dt) u_prev
//!            + dt^2 (c^2 Laplacian(u) + q) ] / (1 + sigma dt)
//! ```
//!
//! with a five-point Laplacian, zero outer Dirichlet boundary, a quadratic
//! sponge, a Ricker source and unit-peak Gaussian footprints.

pub mod adjoint;
pub mod born;
pub mod treeverse;

use serde::Deserialize;
use serde_json::Value;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

/// The experiment file, exactly as documented in `seismic.design.toml`.
#[derive(Debug, Clone, Deserialize)]
pub struct Experiment {
    #[serde(default)]
    pub schema: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    pub nx: usize,
    pub nz: usize,
    pub dx: f64,
    pub dt: f64,
    pub steps: usize,
    pub source_frequency: f64,
    pub source_peak_time: f64,
    pub source_amplitude: f64,
    pub sponge_width: f64,
    pub sponge_strength: f64,
    /// Shot positions as `[x, z]` grid coordinates.
    pub shots: Vec<[f64; 2]>,
    /// Receiver positions as `[x, z]` grid indices.
    pub receivers: Vec<[i64; 2]>,
    /// Wave speed `c`, `[z][x]`.
    pub background: Vec<Vec<f64>>,
    /// Velocity perturbation, `[z][x]`, used for the forward echo.
    pub perturbation: Vec<Vec<f64>>,
    #[serde(default)]
    pub length_unit_m: Option<f64>,
    #[serde(default)]
    pub time_unit_s: Option<f64>,
    #[serde(default)]
    pub provenance: Option<Value>,
}

/// Result of one forward shot.
#[derive(Debug, Clone)]
pub struct ForwardResult {
    /// Receiver pressures, `steps * receivers` in `[step, receiver]` order.
    pub traces: Vec<f64>,
    /// Recorded wavefields, `frame_steps.len() * nz * nx`, C-order `[z, x]`.
    pub frames: Vec<f64>,
    /// Step index `n` of every recorded frame (`u^n` after step `n`).
    pub frame_steps: Vec<usize>,
}

impl Experiment {
    /// Read and validate an experiment file.
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        let experiment: Experiment = serde_json::from_str(&text)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        experiment.validate()?;
        Ok(experiment)
    }

    /// Reject malformed or out-of-range models instead of guessing.
    pub fn validate(&self) -> Result<(), String> {
        if self.nx == 0 || self.nz == 0 {
            return Err("nx and nz must be positive".into());
        }
        if self.dt <= 0.0 || self.dx <= 0.0 {
            return Err("dx and dt must be positive".into());
        }
        if self.sponge_width <= 0.0 {
            return Err("sponge_width must be positive".into());
        }
        if self.background.len() != self.nz
            || self.background.iter().any(|row| row.len() != self.nx)
        {
            return Err(format!(
                "background must have shape [{}][{}]",
                self.nz, self.nx
            ));
        }
        if self.perturbation.len() != self.nz
            || self.perturbation.iter().any(|row| row.len() != self.nx)
        {
            return Err(format!(
                "perturbation must have shape [{}][{}]",
                self.nz, self.nx
            ));
        }
        if self.shots.is_empty() || self.receivers.is_empty() {
            return Err("at least one shot and receiver are required".into());
        }
        for [x, z] in &self.receivers {
            if *x < 0 || *z < 0 || *x as usize >= self.nx || *z as usize >= self.nz {
                return Err(format!("receiver [{x}, {z}] is outside the grid"));
            }
        }
        Ok(())
    }

    /// Background wave speed flattened as `[z, x]`.
    pub fn background_flat(&self) -> Vec<f64> {
        self.background.iter().flatten().copied().collect()
    }

    /// Perturbation flattened as `[z, x]`.
    pub fn perturbation_flat(&self) -> Vec<f64> {
        self.perturbation.iter().flatten().copied().collect()
    }

    /// Sponge profile `sigma = strength * max(0, 1 - distance / width)^2`.
    pub fn sponge(&self) -> Vec<f64> {
        let mut sigma = vec![0.0; self.nx * self.nz];
        for z in 0..self.nz {
            for x in 0..self.nx {
                let distance = x.min(self.nx - 1 - x).min(z).min(self.nz - 1 - z) as f64;
                let taper = (self.sponge_width - distance).max(0.0) / self.sponge_width;
                sigma[z * self.nx + x] = self.sponge_strength * taper * taper;
            }
        }
        sigma
    }

    /// Unit-peak Gaussian footprint `exp(-((x-xs)^2 + (z-zs)^2)/2)`.
    pub fn footprint(&self, shot: [f64; 2]) -> Vec<f64> {
        let mut footprint = vec![0.0; self.nx * self.nz];
        for z in 0..self.nz {
            for x in 0..self.nx {
                let dx = x as f64 - shot[0];
                let dz = z as f64 - shot[1];
                footprint[z * self.nx + x] = (-0.5 * (dx * dx + dz * dz)).exp();
            }
        }
        footprint
    }
}

/// Ricker pulse `amp * (1 - 2 theta^2) exp(-theta^2)`, `theta = pi f0 (t - t0)`.
pub fn ricker(t: f64, frequency: f64, peak_time: f64, amplitude: f64) -> f64 {
    let theta = std::f64::consts::PI * frequency * (t - peak_time);
    amplitude * (1.0 - 2.0 * theta * theta) * (-theta * theta).exp()
}

/// Five-point Laplacian into `out`; the outer boundary of `out` stays zero.
pub fn laplacian(u: &[f64], nx: usize, nz: usize, dx: f64, out: &mut [f64]) {
    out.fill(0.0);
    let inv_dx2 = 1.0 / (dx * dx);
    for z in 1..nz - 1 {
        for x in 1..nx - 1 {
            let index = z * nx + x;
            out[index] = (u[index + 1] + u[index - 1] + u[index + nx]
                + u[index - nx]
                - 4.0 * u[index])
                * inv_dx2;
        }
    }
}

/// Force every outer-boundary cell to zero (Dirichlet).
pub fn apply_boundary(u: &mut [f64], nx: usize, nz: usize) {
    for x in 0..nx {
        u[x] = 0.0;
        u[(nz - 1) * nx + x] = 0.0;
    }
    for z in 0..nz {
        u[z * nx] = 0.0;
        u[z * nx + nx - 1] = 0.0;
    }
}

/// Sample a wavefield at the experiment receivers.
pub fn sample_receivers(u: &[f64], nx: usize, receivers: &[[i64; 2]]) -> Vec<f64> {
    receivers
        .iter()
        .map(|[x, z]| u[*z as usize * nx + *x as usize])
        .collect()
}

/// Integrate one shot. `record` keeps the `every`-th frame of `u^n`.
///
/// `c` is the flattened background (or background plus perturbation) wave
/// speed. Both initial fields `u^(-1)` and `u^(0)` are zero.
pub fn forward(
    experiment: &Experiment,
    c: &[f64],
    shot: [f64; 2],
    every: Option<usize>,
    record: bool,
) -> ForwardResult {
    let nx = experiment.nx;
    let nz = experiment.nz;
    let n = nx * nz;
    let dt = experiment.dt;

    let sigma = experiment.sponge();
    let denominator: Vec<f64> = sigma.iter().map(|s| 1.0 + dt * s).collect();
    let footprint = experiment.footprint(shot);

    let mut previous = vec![0.0; n];
    let mut current = vec![0.0; n];
    let mut next = vec![0.0; n];
    let mut lap = vec![0.0; n];

    let mut traces = vec![0.0; experiment.steps * experiment.receivers.len()];
    let mut frames = Vec::new();
    let mut frame_steps = Vec::new();

    for step in 0..experiment.steps {
        laplacian(&current, nx, nz, experiment.dx, &mut lap);
        let source = ricker(
            step as f64 * dt,
            experiment.source_frequency,
            experiment.source_peak_time,
            experiment.source_amplitude,
        );
        for z in 0..nz {
            for x in 0..nx {
                let index = z * nx + x;
                let force =
                    c[index] * c[index] * lap[index] + source * footprint[index];
                next[index] = (2.0 * current[index]
                    - (1.0 - dt * sigma[index]) * previous[index]
                    + dt * dt * force)
                    / denominator[index];
            }
        }
        apply_boundary(&mut next, nx, nz);

        let sampled = sample_receivers(&next, nx, &experiment.receivers);
        let base = step * experiment.receivers.len();
        traces[base..base + sampled.len()].copy_from_slice(&sampled);

        if record && let Some(every) = every
            && (step + 1) % every == 0
        {
            frames.extend_from_slice(&next);
            frame_steps.push(step + 1);
        }

        std::mem::swap(&mut previous, &mut current);
        std::mem::swap(&mut current, &mut next);
    }

    ForwardResult {
        traces,
        frames,
        frame_steps,
    }
}

/// Write a NumPy `.npy` file (format 1.0, C order, little endian).
fn write_npy_header<W: Write>(writer: &mut W, descr: &str, shape: &[usize]) -> io::Result<()> {
    let shape_text = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        let parts: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        format!("({})", parts.join(", "))
    };
    let mut header = format!(
        "{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_text}, }}"
    );
    let unpadded = 10 + header.len() + 1;
    let padding = (64 - (unpadded % 64)) % 64;
    header.push_str(&" ".repeat(padding));
    header.push('\n');

    writer.write_all(b"\x93NUMPY")?;
    writer.write_all(&[1, 0])?;
    writer.write_all(&(header.len() as u16).to_le_bytes())?;
    writer.write_all(header.as_bytes())?;
    Ok(())
}

/// Write a `float64` array as `.npy`.
pub fn write_npy_f64(path: &Path, shape: &[usize], data: &[f64]) -> io::Result<()> {
    let expected: usize = shape.iter().product();
    assert_eq!(expected, data.len(), "shape {shape:?} does not match data");
    let mut writer = BufWriter::new(File::create(path)?);
    write_npy_header(&mut writer, "<f8", shape)?;
    let mut bytes = Vec::with_capacity(data.len() * 8);
    for value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    writer.write_all(&bytes)?;
    writer.flush()
}

/// Write a `float32` array as `.npy`.
pub fn write_npy_f32(path: &Path, shape: &[usize], data: &[f32]) -> io::Result<()> {
    let expected: usize = shape.iter().product();
    assert_eq!(expected, data.len(), "shape {shape:?} does not match data");
    let mut writer = BufWriter::new(File::create(path)?);
    write_npy_header(&mut writer, "<f4", shape)?;
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    writer.write_all(&bytes)?;
    writer.flush()
}

/// Parse the shape tuple out of a `.npy` header dictionary.
fn npy_shape(header: &str) -> Result<Vec<usize>, String> {
    let key = header
        .find("'shape'")
        .or_else(|| header.find("\"shape\""))
        .ok_or("missing 'shape' field")?;
    let open = header[key..].find('(').ok_or("missing shape tuple")? + key;
    let close = header[open..].find(')').ok_or("unterminated shape tuple")? + open;
    header[open + 1..close]
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<usize>()
                .map_err(|error| format!("bad shape element {part:?}: {error}"))
        })
        .collect()
}

/// Read a little-endian, C-order `.npy` array of `float64`.
///
/// Supports NumPy format 1.0 and 2.0. Returns the parsed shape and the flat
/// data; the caller checks the shape against the experiment.
pub fn read_npy_f64(path: &Path) -> Result<(Vec<usize>, Vec<f64>), String> {
    let bytes = std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    if bytes.len() < 10 || &bytes[0..6] != b"\x93NUMPY" {
        return Err(format!("{}: not a NumPy .npy file", path.display()));
    }
    let major = bytes[6];
    let (header_start, header_len) = if major == 1 {
        (10usize, u16::from_le_bytes([bytes[8], bytes[9]]) as usize)
    } else {
        (
            12usize,
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
        )
    };
    let header_end = header_start + header_len;
    if header_end > bytes.len() {
        return Err(format!("{}: truncated .npy header", path.display()));
    }
    let header = std::str::from_utf8(&bytes[header_start..header_end])
        .map_err(|error| format!("{}: header is not UTF-8: {error}", path.display()))?;
    if !header.contains("<f8") {
        return Err(format!("{}: expected '<f8' data", path.display()));
    }
    if !header.contains("'fortran_order': False") && !header.contains("\"fortran_order\": False") {
        return Err(format!("{}: expected C-order data", path.display()));
    }
    let shape = npy_shape(header)?;
    let payload = &bytes[header_end..];
    if payload.len() % 8 != 0 {
        return Err(format!("{}: payload is not a whole number of f64", path.display()));
    }
    let values: Vec<f64> = payload
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let expected: usize = shape.iter().product();
    if expected != values.len() {
        return Err(format!(
            "{}: shape {shape:?} does not match {} values",
            path.display(),
            values.len()
        ));
    }
    Ok((shape, values))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_experiment() -> Experiment {
        Experiment {
            schema: None,
            name: Some("tiny".into()),
            nx: 6,
            nz: 6,
            dx: 1.0,
            dt: 0.2,
            steps: 10,
            source_frequency: 0.08,
            source_peak_time: 15.0,
            source_amplitude: 1.0,
            sponge_width: 2.0,
            sponge_strength: 0.4,
            shots: vec![[2.0, 2.0]],
            receivers: vec![[1, 1], [4, 1], [2, 4]],
            background: vec![vec![1.8; 6]; 6],
            perturbation: vec![vec![0.0; 6]; 6],
            length_unit_m: None,
            time_unit_s: None,
            provenance: None,
        }
    }

    #[test]
    fn five_point_laplacian_of_x_squared() {
        let (nx, nz, dx) = (5, 5, 1.0);
        let mut u = vec![0.0; nx * nz];
        for z in 0..nz {
            for x in 0..nx {
                u[z * nx + x] = (x * x) as f64;
            }
        }
        let mut out = vec![0.0; nx * nz];
        laplacian(&u, nx, nz, dx, &mut out);
        for z in 1..nz - 1 {
            for x in 1..nx - 1 {
                assert!((out[z * nx + x] - 2.0 / (dx * dx)).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn laplacian_of_linear_field_is_zero() {
        let (nx, nz) = (4, 4);
        let u: Vec<f64> = (0..nz)
            .flat_map(|z| (0..nx).map(move |x| 3.0 * x as f64 - 2.0 * z as f64))
            .collect();
        let mut out = vec![0.0; nx * nz];
        laplacian(&u, nx, nz, 1.0, &mut out);
        for z in 1..nz - 1 {
            for x in 1..nx - 1 {
                assert!(out[z * nx + x].abs() < 1e-12);
            }
        }
    }

    #[test]
    fn boundary_is_zeroed_and_interior_kept() {
        let (nx, nz) = (4, 4);
        let mut u: Vec<f64> = (0..nx * nz).map(|i| i as f64 + 1.0).collect();
        apply_boundary(&mut u, nx, nz);
        for x in 0..nx {
            assert_eq!(u[x], 0.0);
            assert_eq!(u[(nz - 1) * nx + x], 0.0);
        }
        for z in 0..nz {
            assert_eq!(u[z * nx], 0.0);
            assert_eq!(u[z * nx + nx - 1], 0.0);
        }
        assert_eq!(u[nx + 1], 6.0);
    }

    #[test]
    fn sponge_quadratic_profile() {
        let mut experiment = tiny_experiment();
        experiment.nx = 5;
        experiment.nz = 5;
        experiment.sponge_width = 2.0;
        experiment.sponge_strength = 0.4;
        let sigma = experiment.sponge();
        assert!((sigma[0] - 0.4).abs() < 1e-12); // corner: distance 0
        assert!((sigma[6] - 0.1).abs() < 1e-12); // (z=1,x=1): distance 1
        assert!((sigma[12] - 0.0).abs() < 1e-12); // (z=2,x=2): distance 2
        assert!((sigma[2] - 0.4).abs() < 1e-12); // top edge: distance 0
    }

    #[test]
    fn ricker_peak_and_troughs() {
        let (frequency, peak_time, amplitude) = (0.08, 15.0, 1.0);
        assert!((ricker(peak_time, frequency, peak_time, amplitude) - 1.0).abs() < 1e-15);
        let theta = 1.5f64.sqrt();
        let trough_t = peak_time + theta / (std::f64::consts::PI * frequency);
        let expected = (1.0 - 2.0 * theta * theta) * (-theta * theta).exp();
        assert!((ricker(trough_t, frequency, peak_time, amplitude) - expected).abs() < 1e-12);
        assert!((expected + 0.44626).abs() < 1e-4);
    }

    #[test]
    fn gaussian_footprint_peaks_at_the_shot() {
        let experiment = tiny_experiment();
        let footprint = experiment.footprint([2.0, 3.0]);
        assert!((footprint[3 * 6 + 2] - 1.0).abs() < 1e-15);
        let one_away = footprint[3 * 6 + 3];
        assert!((one_away - (-0.5f64).exp()).abs() < 1e-15);
        assert!(one_away < 1.0);
    }

    #[test]
    fn receiver_sampling_reads_cells() {
        let field: Vec<f64> = (0..25).map(|i| i as f64).collect();
        let receivers = [[1, 2], [4, 4]];
        assert_eq!(sample_receivers(&field, 5, &receivers), vec![11.0, 24.0]);
    }

    #[test]
    fn forward_dimensions_match_contract() {
        let experiment = tiny_experiment();
        let c = experiment.background_flat();
        let result = forward(&experiment, &c, experiment.shots[0], Some(3), true);
        assert_eq!(
            result.traces.len(),
            experiment.steps * experiment.receivers.len()
        );
        assert_eq!(result.frame_steps, vec![3, 6, 9]);
        assert_eq!(result.frames.len(), 3 * experiment.nz * experiment.nx);
    }

    #[test]
    fn forward_is_deterministic_and_finite() {
        let experiment = tiny_experiment();
        let c = experiment.background_flat();
        let first = forward(&experiment, &c, experiment.shots[0], None, false);
        let second = forward(&experiment, &c, experiment.shots[0], None, false);
        assert_eq!(first.traces, second.traces);
        assert!(first.traces.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn zero_initial_fields_stay_zero_without_a_source() {
        let mut experiment = tiny_experiment();
        experiment.source_amplitude = 0.0;
        let c = experiment.background_flat();
        let result = forward(&experiment, &c, experiment.shots[0], None, false);
        assert!(result.traces.iter().all(|v| v.abs() < 1e-15));
    }

    /// A small non-zero velocity perturbation for the Born tangent tests.
    fn tiny_perturbation(nx: usize, nz: usize) -> Vec<Vec<f64>> {
        (0..nz)
            .map(|z| {
                (0..nx)
                    .map(|x| 0.1 * (0.7 * x as f64 + 1.3 * z as f64).sin())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn kernel_primal_matches_the_validated_forward_solver() {
        let experiment = tiny_experiment();
        let c = experiment.background_flat();
        let computed = crate::born::kernel_traces(&experiment, &c);
        let n_receivers = experiment.receivers.len();
        let size = experiment.steps * n_receivers;
        for (shot_index, shot) in experiment.shots.iter().enumerate() {
            let reference = forward(&experiment, &c, *shot, None, false);
            let base = shot_index * size;
            for (offset, (actual, expected)) in computed[base..base + size]
                .iter()
                .zip(reference.traces.iter())
                .enumerate()
            {
                let tolerance = 1e-12 * expected.abs().max(1.0);
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "trace {offset}: Enzyme kernel {actual} vs forward {expected}"
                );
            }
        }
    }

    #[test]
    fn born_jvp_matches_central_finite_differences() {
        let mut experiment = tiny_experiment();
        experiment.perturbation = tiny_perturbation(experiment.nx, experiment.nz);
        let c = experiment.background_flat();
        let m = experiment.perturbation_flat();
        let n_receivers = experiment.receivers.len();
        let n_shots = experiment.shots.len();
        let size = experiment.steps * n_receivers;

        let jvp = crate::born::born(&experiment, &c, &m);
        assert_eq!(jvp.len(), n_shots * size);

        let epsilon = 1e-6;
        let mut finite = vec![0.0; n_shots * size];
        for (shot_index, shot) in experiment.shots.iter().enumerate() {
            let plus: Vec<f64> = c.iter().zip(m.iter()).map(|(c, m)| c + epsilon * m).collect();
            let minus: Vec<f64> = c.iter().zip(m.iter()).map(|(c, m)| c - epsilon * m).collect();
            let forward_plus = forward(&experiment, &plus, *shot, None, false);
            let forward_minus = forward(&experiment, &minus, *shot, None, false);
            let base = shot_index * size;
            for i in 0..size {
                finite[base + i] =
                    (forward_plus.traces[i] - forward_minus.traces[i]) / (2.0 * epsilon);
            }
        }

        let numerator: f64 = jvp
            .iter()
            .zip(finite.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        let denominator: f64 = finite.iter().map(|v| v * v).sum();
        let relative = (numerator / denominator).sqrt();
        assert!(
            relative < 1e-6,
            "Born JVP vs central finite differences relative error {relative}"
        );
    }

    /// Relative error of the transpose identity `<Jm, Jm> = <m, J^T Jm>`.
    fn transpose_relative_error(experiment: &Experiment) -> f64 {
        let c = experiment.background_flat();
        let m = experiment.perturbation_flat();
        let data = crate::born::born(experiment, &c, &m);
        let result = crate::adjoint::adjoint(experiment, &c, &data, None);
        let left: f64 = data.iter().map(|v| v * v).sum();
        let right: f64 = m
            .iter()
            .zip(result.image.iter())
            .map(|(m, image)| m * image)
            .sum();
        (left - right).abs() / left.abs().max(right.abs()).max(1e-30)
    }

    #[test]
    fn adjoint_satisfies_the_transpose_identity() {
        let mut experiment = tiny_experiment();
        experiment.perturbation = tiny_perturbation(experiment.nx, experiment.nz);
        experiment.source_peak_time = 1.0;
        experiment.steps = 40;
        let relative = transpose_relative_error(&experiment);
        assert!(
            relative < 1e-9,
            "transpose identity relative error {relative}"
        );
    }

    #[test]
    fn adjoint_accumulates_duplicate_receivers() {
        let mut experiment = tiny_experiment();
        experiment.perturbation = tiny_perturbation(experiment.nx, experiment.nz);
        experiment.source_peak_time = 1.0;
        experiment.steps = 20;
        // Two receivers share the cell [1, 1]; their weights must accumulate.
        experiment.receivers = vec![[1, 1], [1, 1], [4, 1], [2, 4]];
        let relative = transpose_relative_error(&experiment);
        assert!(
            relative < 1e-9,
            "duplicate-receiver transpose identity relative error {relative}"
        );
    }

    fn treeverse_data(experiment: &Experiment) -> (Vec<f64>, Vec<f64>) {
        let c = experiment.background_flat();
        let m = experiment.perturbation_flat();
        let data = crate::born::born(experiment, &c, &m);
        (c, data)
    }

    #[test]
    fn treeverse_small_example_schedule_matches_the_reference() {
        let mut experiment = tiny_experiment();
        experiment.perturbation = tiny_perturbation(experiment.nx, experiment.nz);
        experiment.source_peak_time = 1.0;
        experiment.steps = 6;
        let (c, data) = treeverse_data(&experiment);

        let result = crate::treeverse::adjoint_treeverse(&experiment, &c, &data, 2);
        let actions = &result.actions[0];
        let calls = actions.iter().filter(|a| a.action == "call").count();
        let grads: Vec<usize> = actions
            .iter()
            .filter(|a| a.action == "grad")
            .map(|a| a.step)
            .collect();
        let first_store = actions
            .iter()
            .find(|a| a.action == "store")
            .expect("at least one store")
            .step;

        // Learning Sheet reference for N = 6, delta = 2.
        assert_eq!(calls, 8, "forward calls");
        assert_eq!(grads, vec![5, 4, 3, 2, 1, 0], "reverse order");
        assert_eq!(first_store, 3, "initial interior checkpoint");
        assert_eq!(result.stats.peak_saved_states, 3, "peak saved states");
        assert_eq!(result.stats.reverse_calls, 6, "reverse calls");
        assert_eq!(actions.last().unwrap().saved_states, 1, "only s_0 left");
        crate::treeverse::validate_schedule(actions, 6, 2).expect("valid schedule");
    }

    #[test]
    fn treeverse_image_matches_full_history() {
        let mut experiment = tiny_experiment();
        experiment.perturbation = tiny_perturbation(experiment.nx, experiment.nz);
        experiment.source_peak_time = 1.0;
        experiment.steps = 24;
        let (c, data) = treeverse_data(&experiment);

        let full = crate::adjoint::adjoint(&experiment, &c, &data, None).image;
        let tree = crate::treeverse::adjoint_treeverse(&experiment, &c, &data, 2).image;
        let error: f64 = full
            .iter()
            .zip(tree.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
            / full.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(error < 1e-9, "treeverse vs full-history image error {error}");
    }

    #[test]
    fn treeverse_validation_rejects_a_missing_reverse_step() {
        let mut experiment = tiny_experiment();
        experiment.perturbation = tiny_perturbation(experiment.nx, experiment.nz);
        experiment.source_peak_time = 1.0;
        experiment.steps = 6;
        let (c, data) = treeverse_data(&experiment);
        let result = crate::treeverse::adjoint_treeverse(&experiment, &c, &data, 2);
        let mut actions = result.actions[0].clone();
        let position = actions
            .iter()
            .position(|a| a.action == "grad" && a.step == 3)
            .expect("grad at step 3");
        actions.remove(position);
        assert!(crate::treeverse::validate_schedule(&actions, 6, 2).is_err());
    }
}
