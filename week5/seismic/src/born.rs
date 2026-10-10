//! Born forward modeling: `d = J m` at the background velocity `c0`.
//!
//! The state is `(u_previous, u_current)`. The Enzyme JVP of the numerical
//! timestep propagates the tangent state
//!
//! ```text
//! delta_s_(n+1) = A_n delta_s_n + B_n m
//! ```
//!
//! while the primal state runs at `c0`. The initial state is zero, so its
//! tangent is zero; the source and the sponge are held fixed. The tangent
//! wavefield is sampled at the experiment receivers after every step, giving
//! `born_data[shot, step, receiver]`.

use crate::{sample_receivers, ricker, Experiment};

unsafe extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn enzyme_timestep(
        nx: usize,
        nz: usize,
        inv_dx2: f64,
        dt: f64,
        c: *const f64,
        dc: *const f64,
        sigma: *const f64,
        den: *const f64,
        src: *const f64,
        prev: *const f64,
        dprev: *const f64,
        cur: *const f64,
        dcur: *const f64,
        next: *mut f64,
        dnext: *mut f64,
    );
}

/// Safe wrapper around one Enzyme timestep JVP. Every slice must have
/// `nx * nz` elements; the lengths are checked before the C call.
#[allow(clippy::too_many_arguments)]
fn timestep_jvp(
    nx: usize,
    nz: usize,
    dx: f64,
    dt: f64,
    c: &[f64],
    dc: &[f64],
    sigma: &[f64],
    den: &[f64],
    src: &[f64],
    prev: &[f64],
    dprev: &[f64],
    cur: &[f64],
    dcur: &[f64],
    next: &mut [f64],
    dnext: &mut [f64],
) {
    let n = nx * nz;
    assert_eq!(c.len(), n, "c");
    assert_eq!(dc.len(), n, "dc");
    assert_eq!(sigma.len(), n, "sigma");
    assert_eq!(den.len(), n, "den");
    assert_eq!(src.len(), n, "src");
    assert_eq!(prev.len(), n, "prev");
    assert_eq!(dprev.len(), n, "dprev");
    assert_eq!(cur.len(), n, "cur");
    assert_eq!(dcur.len(), n, "dcur");
    assert_eq!(next.len(), n, "next");
    assert_eq!(dnext.len(), n, "dnext");
    unsafe {
        enzyme_timestep(
            nx,
            nz,
            1.0 / (dx * dx),
            dt,
            c.as_ptr(),
            dc.as_ptr(),
            sigma.as_ptr(),
            den.as_ptr(),
            src.as_ptr(),
            prev.as_ptr(),
            dprev.as_ptr(),
            cur.as_ptr(),
            dcur.as_ptr(),
            next.as_mut_ptr(),
            dnext.as_mut_ptr(),
        );
    }
}

/// Primal and tangent receiver recordings of every shot, both laid out as
/// `[shot, step, receiver]`.
struct KernelRun {
    #[allow(dead_code)]
    traces: Vec<f64>,
    born: Vec<f64>,
}

/// Run the primal state and the tangent state together through the Enzyme JVP.
///
/// `c` is the velocity field the primal is evaluated at; `m` is the tangent
/// direction (`dc`). Sampled tangents after each step are the Born data.
fn run_kernel(experiment: &Experiment, c: &[f64], m: &[f64]) -> KernelRun {
    let nx = experiment.nx;
    let nz = experiment.nz;
    let n = nx * nz;
    let dt = experiment.dt;
    let steps = experiment.steps;
    let n_receivers = experiment.receivers.len();
    let n_shots = experiment.shots.len();
    let shot_size = steps * n_receivers;

    assert_eq!(c.len(), n, "velocity field");
    assert_eq!(m.len(), n, "velocity perturbation");

    let sigma = experiment.sponge();
    let denominator: Vec<f64> = sigma.iter().map(|s| 1.0 + dt * s).collect();

    let mut traces = vec![0.0; n_shots * shot_size];
    let mut born = vec![0.0; n_shots * shot_size];

    for (shot_index, shot) in experiment.shots.iter().enumerate() {
        let footprint = experiment.footprint(*shot);

        let mut previous = vec![0.0; n];
        let mut current = vec![0.0; n];
        let mut next = vec![0.0; n];
        let mut d_previous = vec![0.0; n];
        let mut d_current = vec![0.0; n];
        let mut d_next = vec![0.0; n];
        let mut source = vec![0.0; n];

        let base = shot_index * shot_size;
        for step in 0..steps {
            let pulse = ricker(
                step as f64 * dt,
                experiment.source_frequency,
                experiment.source_peak_time,
                experiment.source_amplitude,
            );
            for (value, weight) in source.iter_mut().zip(footprint.iter()) {
                *value = pulse * weight;
            }
            // The output tangents are fully overwritten by Enzyme; zeroing keeps
            // the boundary contribution exactly zero regardless.
            d_next.fill(0.0);

            timestep_jvp(
                nx,
                nz,
                experiment.dx,
                dt,
                c,
                m,
                &sigma,
                &denominator,
                &source,
                &previous,
                &d_previous,
                &current,
                &d_current,
                &mut next,
                &mut d_next,
            );

            let sampled = sample_receivers(&next, nx, &experiment.receivers);
            let sampled_tangent = sample_receivers(&d_next, nx, &experiment.receivers);
            let row = base + step * n_receivers;
            traces[row..row + n_receivers].copy_from_slice(&sampled);
            born[row..row + n_receivers].copy_from_slice(&sampled_tangent);

            std::mem::swap(&mut previous, &mut current);
            std::mem::swap(&mut current, &mut next);
            std::mem::swap(&mut d_previous, &mut d_current);
            std::mem::swap(&mut d_current, &mut d_next);
        }
    }

    KernelRun { traces, born }
}

/// Born scattered data `d = J m`, shape `[shot, step, receiver]`.
pub fn born(experiment: &Experiment, c: &[f64], m: &[f64]) -> Vec<f64> {
    run_kernel(experiment, c, m).born
}

/// Primal receiver traces produced by the same kernel, for verification against
/// the validated [`crate::forward`] solver.
#[allow(dead_code)]
pub(crate) fn kernel_traces(experiment: &Experiment, c: &[f64]) -> Vec<f64> {
    let zero = vec![0.0; c.len()];
    run_kernel(experiment, c, &zero).traces
}
