//! Adjoint (reverse-mode) propagation: `m_img = J^T w`.
//!
//! The velocity sensitivity is built by running the primal simulation once per
//! shot with full-history storage, then reversing the saved states. At step `k`
//! the receiver weights for that step seed the adjoint of the newly computed
//! pressure `u^(k+1)`; Enzyme's reverse-mode VJP of the numerical timestep maps
//! that seed back to the adjoints of `u^(k-1)`, `u^k` and the velocity `c`. The
//! velocity adjoint is the image contribution `B_k^T a`, and the pressure
//! adjoints are propagated back as `A_k^T a`. The code only composes Enzyme's
//! local VJPs through time; it never hand-differentiates the timestep.

use crate::{ricker, Experiment};

unsafe extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn enzyme_timestep_primal(
        nx: usize,
        nz: usize,
        inv_dx2: f64,
        dt: f64,
        c: *const f64,
        sigma: *const f64,
        den: *const f64,
        src: *const f64,
        prev: *const f64,
        cur: *const f64,
        next: *mut f64,
    );

    #[allow(clippy::too_many_arguments)]
    fn enzyme_timestep_vjp(
        nx: usize,
        nz: usize,
        inv_dx2: f64,
        dt: f64,
        c: *const f64,
        dc: *mut f64,
        sigma: *const f64,
        den: *const f64,
        src: *const f64,
        prev: *const f64,
        dprev: *mut f64,
        cur: *const f64,
        dcur: *mut f64,
        next: *mut f64,
        dnext: *mut f64,
    );
}

/// Safe wrapper for the primal timestep. All slices have `nx * nz` elements.
#[allow(clippy::too_many_arguments)]
pub(crate) fn primal_timestep(
    nx: usize,
    nz: usize,
    dx: f64,
    dt: f64,
    c: &[f64],
    sigma: &[f64],
    den: &[f64],
    src: &[f64],
    prev: &[f64],
    cur: &[f64],
    next: &mut [f64],
) {
    let n = nx * nz;
    assert_eq!(c.len(), n, "c");
    assert_eq!(sigma.len(), n, "sigma");
    assert_eq!(den.len(), n, "den");
    assert_eq!(src.len(), n, "src");
    assert_eq!(prev.len(), n, "prev");
    assert_eq!(cur.len(), n, "cur");
    assert_eq!(next.len(), n, "next");
    unsafe {
        enzyme_timestep_primal(
            nx,
            nz,
            1.0 / (dx * dx),
            dt,
            c.as_ptr(),
            sigma.as_ptr(),
            den.as_ptr(),
            src.as_ptr(),
            prev.as_ptr(),
            cur.as_ptr(),
            next.as_mut_ptr(),
        );
    }
}

/// Safe wrapper for one Enzyme timestep VJP.
///
/// `dnext` is the output adjoint seed; `dc`, `dprev` and `dcur` are
/// accumulators and must be zeroed before the call. The primal `next` is only
/// passed for completeness; the VJP does not depend on its value.
#[allow(clippy::too_many_arguments)]
pub(crate) fn vjp_timestep(
    nx: usize,
    nz: usize,
    dx: f64,
    dt: f64,
    c: &[f64],
    dc: &mut [f64],
    sigma: &[f64],
    den: &[f64],
    src: &[f64],
    prev: &[f64],
    dprev: &mut [f64],
    cur: &[f64],
    dcur: &mut [f64],
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
        enzyme_timestep_vjp(
            nx,
            nz,
            1.0 / (dx * dx),
            dt,
            c.as_ptr(),
            dc.as_mut_ptr(),
            sigma.as_ptr(),
            den.as_ptr(),
            src.as_ptr(),
            prev.as_ptr(),
            dprev.as_mut_ptr(),
            cur.as_ptr(),
            dcur.as_mut_ptr(),
            next.as_mut_ptr(),
            dnext.as_mut_ptr(),
        );
    }
}

/// Which adjoint storage strategy produced a result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Storage {
    Full,
    Treeverse,
}

impl Storage {
    pub fn as_str(self) -> &'static str {
        match self {
            Storage::Full => "full",
            Storage::Treeverse => "treeverse",
        }
    }
}

/// One audited entry of a Treeverse schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeverseAction {
    /// `store`, `restore`, `call`, `grad` or `fetch`.
    pub action: &'static str,
    /// Timestep the action applies to.
    pub step: usize,
    pub tau: usize,
    pub delta: usize,
    pub depth: usize,
    /// Saved states after the action, including the never-freed `s_0`.
    pub saved_states: usize,
}

/// Full-history counters for one shot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerShotStats {
    pub reverse_calls: usize,
    pub scheduler_forward_calls: usize,
    pub peak_saved_states: usize,
}

/// Aggregated statistics, as required by `seismic.design.toml`.
#[derive(Debug, Clone)]
pub struct AdjointStats {
    pub storage: Storage,
    pub checkpoints: Option<usize>,
    pub reverse_calls: usize,
    pub scheduler_forward_calls: usize,
    pub peak_saved_states: usize,
    pub peak_saved_bytes: usize,
    pub per_shot: Vec<PerShotStats>,
}

/// Adjoint image plus the first shot's reverse-time wavefield recording.
pub struct AdjointResult {
    /// Velocity sensitivity `J^T w`, `[z, x]`.
    pub image: Vec<f64>,
    /// L2 norm of each shot's partial image, in shot order.
    pub shot_norms: Vec<f64>,
    /// Adjoint pressure frames of the first shot, `[frame, z, x]`, float32.
    pub frames: Vec<f32>,
    /// Step index of every frame, in reverse (descending) order.
    pub frame_steps: Vec<usize>,
    /// Per-shot Treeverse schedules; empty for full-history storage.
    pub actions: Vec<Vec<TreeverseAction>>,
    pub stats: AdjointStats,
}

/// Reverse-propagate `weights` through the full saved history of every shot.
///
/// `c` is the background velocity. `weights` is laid out `[shot, step,
/// receiver]` and is normally the Born data. `every` records the first shot's
/// adjoint pressure at steps divisible by `every`.
pub fn adjoint(
    experiment: &Experiment,
    c: &[f64],
    weights: &[f64],
    every: Option<usize>,
) -> AdjointResult {
    let nx = experiment.nx;
    let nz = experiment.nz;
    let n = nx * nz;
    let dt = experiment.dt;
    let steps = experiment.steps;
    let n_receivers = experiment.receivers.len();
    let n_shots = experiment.shots.len();
    let shot_size = steps * n_receivers;

    assert_eq!(c.len(), n, "velocity field");
    assert_eq!(
        weights.len(),
        n_shots * shot_size,
        "weights must be [shot, step, receiver]"
    );

    let sigma = experiment.sponge();
    let denominator: Vec<f64> = sigma.iter().map(|s| 1.0 + dt * s).collect();

    let mut image = vec![0.0; n];
    let mut shot_norms = Vec::with_capacity(n_shots);
    let mut frames = Vec::new();
    let mut frame_steps = Vec::new();
    let mut source = vec![0.0; n];
    let mut per_shot = Vec::with_capacity(n_shots);

    for (shot_index, shot) in experiment.shots.iter().enumerate() {
        let footprint = experiment.footprint(*shot);

        // Full-history forward pass: history[k] is the state before step k,
        // history[steps] is the final state. N + 1 complete states per shot.
        let mut history_prev: Vec<Vec<f64>> = Vec::with_capacity(steps + 1);
        let mut history_cur: Vec<Vec<f64>> = Vec::with_capacity(steps + 1);
        let mut prev = vec![0.0; n];
        let mut cur = vec![0.0; n];
        let mut next = vec![0.0; n];
        history_prev.push(prev.clone());
        history_cur.push(cur.clone());
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
            primal_timestep(
                nx,
                nz,
                experiment.dx,
                dt,
                c,
                &sigma,
                &denominator,
                &source,
                &prev,
                &cur,
                &mut next,
            );
            std::mem::swap(&mut prev, &mut cur);
            std::mem::swap(&mut cur, &mut next);
            history_prev.push(prev.clone());
            history_cur.push(cur.clone());
        }

        // Reverse pass. (ap, au) is the adjoint of the state s_{k+1}.
        let mut ap = vec![0.0; n];
        let mut au = vec![0.0; n];
        let mut dprev = vec![0.0; n];
        let mut dcur = vec![0.0; n];
        let mut dc = vec![0.0; n];
        let mut primal_next = vec![0.0; n];
        let mut shot_image = vec![0.0; n];
        let base = shot_index * shot_size;

        for step in (0..steps).rev() {
            // Receiver pressure is sampled after the update, so the weights seed
            // the adjoint of the newly computed pressure component. Accumulate:
            // several receivers may share one grid cell.
            for (receiver, [rx, rz]) in experiment.receivers.iter().enumerate() {
                au[*rz as usize * nx + *rx as usize] +=
                    weights[base + step * n_receivers + receiver];
            }

            let pulse = ricker(
                step as f64 * dt,
                experiment.source_frequency,
                experiment.source_peak_time,
                experiment.source_amplitude,
            );
            for (value, weight) in source.iter_mut().zip(footprint.iter()) {
                *value = pulse * weight;
            }

            dprev.fill(0.0);
            dcur.fill(0.0);
            dc.fill(0.0);
            primal_next.copy_from_slice(&history_cur[step + 1]);
            vjp_timestep(
                nx,
                nz,
                experiment.dx,
                dt,
                c,
                &mut dc,
                &sigma,
                &denominator,
                &source,
                &history_prev[step],
                &mut dprev,
                &history_cur[step],
                &mut dcur,
                &mut primal_next,
                &mut au,
            );

            // B_k^T a: the velocity contribution for this step.
            for (value, contribution) in shot_image.iter_mut().zip(dc.iter()) {
                *value += contribution;
            }
            // A_k^T a: the pressure adjoint becomes the state for step k-1.
            // `cur` is carried into the next state, so its adjoint adds the
            // incoming adjoint `ap` on top of the VJP's contribution `dcur`.
            for index in 0..n {
                au[index] = ap[index] + dcur[index];
                ap[index] = dprev[index];
            }

            if shot_index == 0
                && let Some(every) = every
                && step % every == 0
            {
                frames.extend(au.iter().map(|value| *value as f32));
                frame_steps.push(step);
            }
        }

        per_shot.push(PerShotStats {
            reverse_calls: steps,
            scheduler_forward_calls: steps,
            peak_saved_states: steps + 1,
        });
        shot_norms.push(shot_image.iter().map(|value| value * value).sum::<f64>().sqrt());
        for (value, partial) in image.iter_mut().zip(shot_image.iter()) {
            *value += partial;
        }
    }

    let reverse_calls = per_shot.iter().map(|shot| shot.reverse_calls).sum();
    let scheduler_forward_calls = per_shot
        .iter()
        .map(|shot| shot.scheduler_forward_calls)
        .sum();
    let peak_saved_states = per_shot
        .iter()
        .map(|shot| shot.peak_saved_states)
        .max()
        .unwrap_or(0);
    let state_bytes = 2 * nx * nz * 8;
    let stats = AdjointStats {
        storage: Storage::Full,
        checkpoints: None,
        reverse_calls,
        scheduler_forward_calls,
        peak_saved_states,
        peak_saved_bytes: peak_saved_states * state_bytes,
        per_shot,
    };

    AdjointResult {
        image,
        shot_norms,
        frames,
        frame_steps,
        actions: Vec::new(),
        stats,
    }
}
