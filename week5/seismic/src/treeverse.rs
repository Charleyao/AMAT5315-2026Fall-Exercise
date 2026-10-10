//! Treeverse checkpointing for the adjoint, ported from
//! `GiggleLiu/TreeverseAlgorithm.jl` (`src/treeverse.jl`) and its NumPy
//! reference in `week5/sim.py`.
//!
//! The scheduler keeps at most `delta + 1` complete states `s_n =
//! (u_previous, u_current)` (the initial `s_0` is never freed). The repetition
//! number `tau` is the smallest integer with `binomial(tau + delta, tau) >= N`.
//! `visit` caches the state at `start` by replaying from the nearest saved
//! state `beta`, recurses into sub-intervals with the Treeverse split rule,
//! runs exactly one Enzyme VJP at `start`, then (unless `start == beta`) frees
//! that slot. Replayed primal steps never contribute to the image; only the
//! Enzyme VJPs do.

use std::collections::{BTreeMap, BTreeSet};

use crate::adjoint::{
    primal_timestep, vjp_timestep, AdjointResult, AdjointStats, PerShotStats, Storage,
    TreeverseAction,
};
use crate::{ricker, Experiment};

/// `C(n, k)` as an exact integer; only used for the small Treeverse capacity.
fn binomial(n: u128, k: u128) -> u128 {
    let k = k.min(n - k);
    let mut result: u128 = 1;
    let mut i: u128 = 0;
    while i < k {
        result = result * (n - i) / (i + 1);
        i += 1;
    }
    result
}

/// Smallest positive `tau` with `binomial(tau + delta, tau) >= n_steps`.
fn binomial_fit(n_steps: usize, delta: usize) -> usize {
    let mut tau = 1usize;
    while binomial((tau + delta) as u128, tau as u128) < n_steps as u128 {
        tau += 1;
    }
    tau
}

/// Treeverse split rule `mid(delta, tau, sigma, phi)` with integer ceiling.
fn middle(delta: usize, tau: usize, start: usize, end: usize) -> usize {
    let k = (delta * start + tau * end + tau + delta - 1) / (tau + delta);
    if k >= end && delta > 0 {
        (start + 1).max(end.saturating_sub(1))
    } else {
        k
    }
}

struct TreeverseContext<'a> {
    experiment: &'a Experiment,
    c: &'a [f64],
    sigma: &'a [f64],
    denominator: &'a [f64],
    footprint: &'a [f64],
    weights: &'a [f64],
    nx: usize,
    nz: usize,
    n: usize,
    dt: f64,
    n_receivers: usize,
    saved: BTreeMap<usize, (Vec<f64>, Vec<f64>)>,
    actions: Vec<TreeverseAction>,
    work_prev: Vec<f64>,
    work_cur: Vec<f64>,
    work_next: Vec<f64>,
    ap: Vec<f64>,
    au: Vec<f64>,
    dprev: Vec<f64>,
    dcur: Vec<f64>,
    dc: Vec<f64>,
    primal_next: Vec<f64>,
    source: Vec<f64>,
    shot_image: Vec<f64>,
    peak_saved_states: usize,
    forward_calls: usize,
    reverse_calls: usize,
}

impl<'a> TreeverseContext<'a> {
    fn log(&mut self, action: &'static str, step: usize, tau: usize, delta: usize, depth: usize) {
        self.actions.push(TreeverseAction {
            action,
            step,
            tau,
            delta,
            depth,
            saved_states: self.saved.len(),
        });
    }

    fn set_source(&mut self, step: usize) {
        let pulse = ricker(
            step as f64 * self.dt,
            self.experiment.source_frequency,
            self.experiment.source_peak_time,
            self.experiment.source_amplitude,
        );
        for (value, weight) in self.source.iter_mut().zip(self.footprint.iter()) {
            *value = pulse * weight;
        }
    }

    /// One replay primal step in the working state; never touches the image.
    fn replay_step(&mut self, step: usize) {
        self.set_source(step);
        primal_timestep(
            self.nx,
            self.nz,
            self.experiment.dx,
            self.dt,
            self.c,
            self.sigma,
            self.denominator,
            &self.source,
            &self.work_prev,
            &self.work_cur,
            &mut self.work_next,
        );
        std::mem::swap(&mut self.work_prev, &mut self.work_cur);
        std::mem::swap(&mut self.work_cur, &mut self.work_next);
        self.forward_calls += 1;
    }

    /// One Enzyme VJP at `step`, using the saved primal state `s_step`.
    fn grad(&mut self, step: usize) {
        let base = step * self.n_receivers;
        for (receiver, [rx, rz]) in self.experiment.receivers.iter().enumerate() {
            self.au[*rz as usize * self.nx + *rx as usize] += self.weights[base + receiver];
        }
        self.set_source(step);

        self.dprev.fill(0.0);
        self.dcur.fill(0.0);
        self.dc.fill(0.0);
        self.primal_next.fill(0.0);
        let (prev, cur) = self
            .saved
            .get(&step)
            .expect("grad state must be saved")
            .clone();
        vjp_timestep(
            self.nx,
            self.nz,
            self.experiment.dx,
            self.dt,
            self.c,
            &mut self.dc,
            self.sigma,
            self.denominator,
            &self.source,
            &prev,
            &mut self.dprev,
            &cur,
            &mut self.dcur,
            &mut self.primal_next,
            &mut self.au,
        );

        // B_k^T a: only the enzyme VJP contributes to the image.
        for (value, contribution) in self.shot_image.iter_mut().zip(self.dc.iter()) {
            *value += contribution;
        }
        // A_k^T a: add the incoming carried adjoint `ap` to the VJP's `dcur`.
        for index in 0..self.n {
            self.au[index] = self.ap[index] + self.dcur[index];
            self.ap[index] = self.dprev[index];
        }
        self.reverse_calls += 1;
    }

    fn visit(&mut self, mut delta: usize, mut tau: usize, beta: usize, start: usize, end: usize, depth: usize) {
        if start > beta {
            delta -= 1;
            let (prev, cur) = self
                .saved
                .get(&beta)
                .expect("restore state must be saved")
                .clone();
            self.work_prev.copy_from_slice(&prev);
            self.work_cur.copy_from_slice(&cur);
            self.log("restore", beta, tau, delta, depth);
            for step in beta..start {
                self.replay_step(step);
                self.log("call", step, tau, delta, depth);
            }
            self.saved
                .insert(start, (self.work_prev.clone(), self.work_cur.clone()));
            self.peak_saved_states = self.peak_saved_states.max(self.saved.len());
            self.log("store", start, tau, delta, depth);
        }

        let mut split = middle(delta, tau, start, end);
        let mut end = end;
        while tau > 0 && split < end {
            self.visit(delta, tau, start, split, end, depth + 1);
            tau -= 1;
            end = split;
            split = middle(delta, tau, start, end);
        }

        self.grad(start);
        self.log("grad", start, tau, delta, depth);
        if start > beta {
            self.saved.remove(&start);
            self.log("fetch", start, tau, delta, depth);
        }
    }
}

/// Treeverse adjoint with `checkpoints` extra saved-state slots.
///
/// `weights` is `[shot, step, receiver]`. The image is bit-for-bit the same
/// mathematical operation as the full-history adjoint; only the state storage
/// strategy differs.
pub fn adjoint_treeverse(
    experiment: &Experiment,
    c: &[f64],
    weights: &[f64],
    checkpoints: usize,
) -> AdjointResult {
    assert!(checkpoints >= 1, "checkpoints must be at least 1");
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
    let tau = binomial_fit(steps, checkpoints);

    let mut image = vec![0.0; n];
    let mut shot_norms = Vec::with_capacity(n_shots);
    let mut actions_per_shot = Vec::with_capacity(n_shots);
    let mut per_shot = Vec::with_capacity(n_shots);

    for (shot_index, shot) in experiment.shots.iter().enumerate() {
        let footprint = experiment.footprint(*shot);
        let shot_weights = &weights[shot_index * shot_size..(shot_index + 1) * shot_size];

        let mut context = TreeverseContext {
            experiment,
            c,
            sigma: &sigma,
            denominator: &denominator,
            footprint: &footprint,
            weights: shot_weights,
            nx,
            nz,
            n,
            dt,
            n_receivers,
            saved: BTreeMap::new(),
            actions: Vec::new(),
            work_prev: vec![0.0; n],
            work_cur: vec![0.0; n],
            work_next: vec![0.0; n],
            ap: vec![0.0; n],
            au: vec![0.0; n],
            dprev: vec![0.0; n],
            dcur: vec![0.0; n],
            dc: vec![0.0; n],
            primal_next: vec![0.0; n],
            source: vec![0.0; n],
            shot_image: vec![0.0; n],
            peak_saved_states: 1,
            forward_calls: 0,
            reverse_calls: 0,
        };
        // s_0 is saved from the start and never freed.
        context.saved.insert(0, (vec![0.0; n], vec![0.0; n]));
        context.visit(checkpoints, tau, 0, 0, steps, 1);

        debug_assert!(context.saved.len() == 1 && context.saved.contains_key(&0));
        debug_assert_eq!(context.reverse_calls, steps);

        shot_norms.push(
            context
                .shot_image
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt(),
        );
        for (value, partial) in image.iter_mut().zip(context.shot_image.iter()) {
            *value += partial;
        }
        per_shot.push(PerShotStats {
            reverse_calls: context.reverse_calls,
            scheduler_forward_calls: context.forward_calls,
            peak_saved_states: context.peak_saved_states,
        });
        actions_per_shot.push(context.actions);
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
        storage: Storage::Treeverse,
        checkpoints: Some(checkpoints),
        reverse_calls,
        scheduler_forward_calls,
        peak_saved_states,
        peak_saved_bytes: peak_saved_states * state_bytes,
        per_shot,
    };

    AdjointResult {
        image,
        shot_norms,
        frames: Vec::new(),
        frame_steps: Vec::new(),
        actions: actions_per_shot,
        stats,
    }
}

/// Replay a schedule exactly as `checker/check` does.
///
/// Returns `(forward_calls, peak_saved_states)` or a description of the first
/// violation. Audits availability, time indices, the budget, the stored-state
/// counter, the descending-once reverse order and the final `{s_0}` state.
pub fn validate_schedule(
    actions: &[TreeverseAction],
    steps: usize,
    delta: usize,
) -> Result<(usize, usize), String> {
    let mut available: BTreeSet<usize> = BTreeSet::from([0]);
    let mut working: Option<usize> = None;
    let mut reverse: Vec<usize> = Vec::new();
    let mut peak = 1usize;
    let mut calls = 0usize;

    for event in actions {
        let step = event.step;
        if step > steps {
            return Err(format!("step {step} is outside 0..={steps}"));
        }
        match event.action {
            "restore" => {
                if !available.contains(&step) {
                    return Err(format!("restore of unsaved state s_{step}"));
                }
                working = Some(step);
            }
            "call" => {
                if working != Some(step) || step >= steps {
                    return Err(format!("call does not advance from s_{step}"));
                }
                working = Some(step + 1);
                calls += 1;
            }
            "store" => {
                if working != Some(step) || available.contains(&step) {
                    return Err(format!("invalid store of s_{step}"));
                }
                available.insert(step);
                peak = peak.max(available.len());
                if available.len() > delta + 1 {
                    return Err(format!("checkpoint budget {} exceeded", delta + 1));
                }
            }
            "grad" => {
                if !available.contains(&step) {
                    return Err(format!("grad reads unsaved s_{step}"));
                }
                reverse.push(step);
            }
            "fetch" => {
                if !available.contains(&step) || step == 0 {
                    return Err(format!("invalid fetch of s_{step}"));
                }
                available.remove(&step);
            }
            other => return Err(format!("unknown action {other}")),
        }
        if event.saved_states != available.len() {
            return Err(format!(
                "stored-state counter {} != {} at s_{step}",
                event.saved_states,
                available.len()
            ));
        }
    }

    let expected: Vec<usize> = (0..steps).rev().collect();
    if reverse != expected {
        return Err("grads are not each reversed once in descending order".into());
    }
    if available != BTreeSet::from([0]) {
        return Err("the schedule leaks checkpoints".into());
    }
    Ok((calls, peak))
}
