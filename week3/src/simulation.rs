//! Temperature-ramp driver and the records it produces.

use crate::lattice::Lattice;
use crate::{metropolis, wolff};
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::Serialize;
use std::path::PathBuf;

/// Which Monte Carlo update defines one step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Update {
    Metropolis,
    Wolff,
}

impl Update {
    pub fn parse(text: &str) -> Result<Self, String> {
        match text {
            "metropolis" => Ok(Update::Metropolis),
            "wolff" => Ok(Update::Wolff),
            other => Err(format!(
                "invalid --update: {other} (expected metropolis or wolff)"
            )),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Update::Metropolis => "metropolis",
            Update::Wolff => "wolff",
        }
    }

    /// Name of one step, as written to `run.json`.
    pub fn time_unit(self) -> &'static str {
        match self {
            Update::Metropolis => "sweep",
            Update::Wolff => "cluster_flip",
        }
    }
}

/// Fully validated run parameters.
#[derive(Clone, Debug)]
pub struct Config {
    pub update: Update,
    pub l: usize,
    pub t_from: f64,
    pub t_to: f64,
    pub t_step: f64,
    pub discard: usize,
    pub measure: usize,
    pub every: usize,
    pub seed: u64,
    pub out: PathBuf,
}

/// Contents of `<out>/run.json`.
#[derive(Clone, Debug, Serialize)]
pub struct RunMeta {
    #[serde(rename = "L")]
    pub l: usize,
    pub update: String,
    pub t_grid: Vec<f64>,
    pub discard: usize,
    pub measure: usize,
    pub seed: u64,
    /// Fixed at 1 by the design; the frame interval is a separate `--every`
    /// argument and is not recorded here.
    pub sample_every: usize,
    pub time_unit: String,
}

/// One line of `<out>/series.jsonl`.
#[derive(Clone, Debug, Serialize)]
pub struct SeriesLine {
    #[serde(rename = "L")]
    pub l: usize,
    #[serde(rename = "T")]
    pub t: f64,
    pub sweep: usize,
    #[serde(rename = "M")]
    pub m: f64,
    #[serde(rename = "E")]
    pub e: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cluster_size: Option<usize>,
}

/// One line of `<out>/spins.jsonl`.
#[derive(Clone, Debug, Serialize)]
pub struct SpinLine {
    #[serde(rename = "L")]
    pub l: usize,
    #[serde(rename = "T")]
    pub t: f64,
    pub sweep: usize,
    pub m: f64,
    pub spins: Vec<i8>,
}

/// Per-temperature summary printed to stdout.
#[derive(Clone, Debug)]
pub struct TempSummary {
    pub t: f64,
    pub mean_abs_m: f64,
    pub acceptance_rate: Option<f64>,
    pub mean_cluster_size: Option<f64>,
}

/// Everything a run produces.
pub struct Report {
    pub meta: RunMeta,
    pub summaries: Vec<TempSummary>,
    pub series: Vec<SeriesLine>,
    pub frames: Vec<SpinLine>,
    /// Recording interval for `frames`, i.e. the `--every` argument.
    pub frame_every: usize,
}

/// Round to six decimal places, the precision used in the output files.
pub fn round6(x: f64) -> f64 {
    let r = (x * 1e6).round() / 1e6;
    if r == 0.0 { 0.0 } else { r }
}

/// Ascending temperature grid from `t_from`, step `t_step`, including `t_to`
/// only when the step lands on it.
pub fn temperature_grid(t_from: f64, t_to: f64, t_step: f64) -> Vec<f64> {
    let scale = 1.0 + t_from.abs().max(t_to.abs());
    let eps = 1e-9 * scale;
    let mut grid = Vec::new();
    let mut k = 0usize;
    loop {
        let t = t_from + k as f64 * t_step;
        if t > t_to + eps {
            break;
        }
        grid.push(round6(t));
        k += 1;
    }
    grid
}

/// Run the ramp. One random stream is created from `seed` and carried through
/// every temperature; each temperature continues from the previous lattice.
pub fn simulate(cfg: &Config) -> Report {
    let t_grid = temperature_grid(cfg.t_from, cfg.t_to, cfg.t_step);
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut lattice = Lattice::all_up(cfg.l);
    let n = lattice.n();

    let mut global_sweep = 0usize;
    let mut summaries = Vec::with_capacity(t_grid.len());
    let mut series = Vec::new();
    let mut frames = Vec::new();

    for &t in &t_grid {
        // Equilibration: no records, but the stream and the global counter advance.
        let mut accepted_total = 0usize;
        for _ in 0..cfg.discard {
            global_sweep += 1;
            if cfg.update == Update::Metropolis {
                accepted_total += metropolis::sweep(&mut lattice, t, &mut rng);
            } else {
                wolff::cluster_flip(&mut lattice, t, &mut rng);
            }
        }

        // Measurement.
        let mut abs_m_sum = 0.0;
        let mut cluster_sum = 0usize;
        for step in 1..=cfg.measure {
            global_sweep += 1;
            let cluster_size = match cfg.update {
                Update::Metropolis => {
                    accepted_total += metropolis::sweep(&mut lattice, t, &mut rng);
                    None
                }
                Update::Wolff => {
                    let size = wolff::cluster_flip(&mut lattice, t, &mut rng);
                    cluster_sum += size;
                    Some(size)
                }
            };

            let m = lattice.magnetization();
            let e = lattice.energy_per_site();
            abs_m_sum += m.abs();

            series.push(SeriesLine {
                l: cfg.l,
                t,
                sweep: step,
                m: round6(m),
                e: round6(e),
                cluster_size,
            });

            if cfg.every > 0 && step % cfg.every == 0 {
                frames.push(SpinLine {
                    l: cfg.l,
                    t,
                    sweep: global_sweep,
                    m: round6(m),
                    spins: lattice.spins.clone(),
                });
            }
        }

        let (acceptance_rate, mean_cluster_size) = match cfg.update {
            Update::Metropolis => {
                let proposals = n * (cfg.discard + cfg.measure);
                let rate = accepted_total as f64 / proposals as f64;
                (Some(round6(rate)), None)
            }
            Update::Wolff => (None, Some(round6(cluster_sum as f64 / cfg.measure as f64))),
        };

        summaries.push(TempSummary {
            t,
            mean_abs_m: round6(abs_m_sum / cfg.measure as f64),
            acceptance_rate,
            mean_cluster_size,
        });
    }

    let meta = RunMeta {
        l: cfg.l,
        update: cfg.update.as_str().to_string(),
        t_grid,
        discard: cfg.discard,
        measure: cfg.measure,
        seed: cfg.seed,
        sample_every: 1,
        time_unit: cfg.update.time_unit().to_string(),
    };

    Report {
        meta,
        summaries,
        series,
        frames,
        frame_every: cfg.every,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(update: Update) -> Config {
        Config {
            update,
            l: 4,
            t_from: 1.5,
            t_to: 1.7,
            t_step: 0.1,
            discard: 2,
            measure: 4,
            every: 2,
            seed: 2026,
            out: PathBuf::from("unused"),
        }
    }

    #[test]
    fn grid_includes_the_upper_bound_only_when_reached() {
        assert_eq!(temperature_grid(1.5, 1.7, 0.1), vec![1.5, 1.6, 1.7]);
        assert_eq!(temperature_grid(1.5, 1.75, 0.1), vec![1.5, 1.6, 1.7]);
        assert_eq!(temperature_grid(2.0, 2.0, 0.1), vec![2.0]);
    }

    #[test]
    fn sample_every_is_fixed_at_one_independently_of_every() {
        let report = simulate(&config(Update::Metropolis)); // every = 2
        assert_eq!(report.meta.sample_every, 1);
        assert_eq!(report.frame_every, 2);
    }

    #[test]
    fn measured_sweep_numbers_restart_at_one_at_each_temperature() {
        let report = simulate(&config(Update::Metropolis));
        assert_eq!(report.series.len(), 3 * 4);
        for chunk in report.series.chunks(4) {
            assert_eq!(
                chunk.iter().map(|r| r.sweep).collect::<Vec<_>>(),
                vec![1, 2, 3, 4]
            );
        }
    }

    #[test]
    fn global_frame_sweeps_include_discard_and_carry_across_the_ramp() {
        let report = simulate(&config(Update::Metropolis));
        // Per temperature: 2 discard + 4 measured = 6 steps; frames at local
        // measured steps 2 and 4, i.e. global steps 4 and 6 for the first
        // temperature, then 10 and 12, then 16 and 18.
        let sweeps: Vec<usize> = report.frames.iter().map(|f| f.sweep).collect();
        assert_eq!(sweeps, vec![4, 6, 10, 12, 16, 18]);
    }

    #[test]
    fn metropolis_reports_an_acceptance_rate_and_wolff_a_cluster_size() {
        let metro = simulate(&config(Update::Metropolis));
        assert!(metro.summaries.iter().all(|s| s.acceptance_rate.is_some()));
        assert!(
            metro
                .summaries
                .iter()
                .all(|s| s.mean_cluster_size.is_none())
        );
        assert!(metro.series.iter().all(|r| r.cluster_size.is_none()));

        let wolf = simulate(&config(Update::Wolff));
        assert!(wolf.summaries.iter().all(|s| s.mean_cluster_size.is_some()));
        assert!(wolf.summaries.iter().all(|s| s.acceptance_rate.is_none()));
        assert!(wolf.series.iter().all(|r| r.cluster_size.is_some()));
    }

    #[test]
    fn same_seed_gives_identical_series() {
        let a = simulate(&config(Update::Metropolis));
        let b = simulate(&config(Update::Metropolis));
        let values = |r: &Report| r.series.iter().map(|s| (s.m, s.e)).collect::<Vec<_>>();
        assert_eq!(values(&a), values(&b));
    }
}
