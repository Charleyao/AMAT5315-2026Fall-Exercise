//! `seismic` command-line program: forward, Born and adjoint acoustic modes.
//!
//! Usage:
//!   seismic --experiment <JSON_PATH> --mode forward [--every N] --out <DIR>
//!   seismic --experiment <JSON_PATH> --mode born --out <DIR>
//!   seismic --experiment <JSON_PATH> --mode adjoint --data <BORN_NPY> [--every N] --out <DIR>

use serde_json::{json, Map, Value};
use std::path::PathBuf;

use seismic::adjoint::adjoint;
use seismic::born::born;
use seismic::{forward, read_npy_f64, write_npy_f32, write_npy_f64, Experiment};

struct Options {
    experiment: PathBuf,
    mode: String,
    data: Option<PathBuf>,
    every: Option<usize>,
    out: PathBuf,
}

fn next_value(args: &[String], index: usize, flag: &str) -> Result<String, String> {
    args.get(index + 1)
        .cloned()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_options() -> Result<Options, String> {
    let args: Vec<String> = std::env::args().collect();
    let mut experiment: Option<PathBuf> = None;
    let mut mode: Option<String> = None;
    let mut data: Option<PathBuf> = None;
    let mut every: Option<usize> = None;
    let mut out: Option<PathBuf> = None;

    let mut index = 1;
    while index < args.len() {
        let flag = args[index].as_str();
        match flag {
            "--experiment" => {
                experiment = Some(PathBuf::from(next_value(&args, index, flag)?))
            }
            "--mode" => mode = Some(next_value(&args, index, flag)?),
            "--data" => data = Some(PathBuf::from(next_value(&args, index, flag)?)),
            "--out" => out = Some(PathBuf::from(next_value(&args, index, flag)?)),
            "--every" => {
                let text = next_value(&args, index, flag)?;
                let parsed: usize = text
                    .parse()
                    .map_err(|_| format!("--every expects a positive integer, got {text}"))?;
                if parsed == 0 {
                    return Err("--every must be a positive integer".into());
                }
                every = Some(parsed);
            }
            other => return Err(format!("unknown argument {other}")),
        }
        index += 2;
    }

    Ok(Options {
        experiment: experiment.ok_or("--experiment is required")?,
        mode: mode.ok_or("--mode is required")?,
        data,
        every,
        out: out.ok_or("--out is required")?,
    })
}

/// The experiment fields that are copied into `run.json`, excluding the
/// physical fields `background` and `perturbation`.
fn compact_experiment(experiment: &Experiment) -> Value {
    let mut map = Map::new();
    if let Some(schema) = &experiment.schema {
        map.insert("schema".into(), json!(schema));
    }
    if let Some(name) = &experiment.name {
        map.insert("name".into(), json!(name));
    }
    map.insert("nx".into(), json!(experiment.nx));
    map.insert("nz".into(), json!(experiment.nz));
    map.insert("dx".into(), json!(experiment.dx));
    map.insert("dt".into(), json!(experiment.dt));
    map.insert("steps".into(), json!(experiment.steps));
    map.insert("source_frequency".into(), json!(experiment.source_frequency));
    map.insert("source_peak_time".into(), json!(experiment.source_peak_time));
    map.insert("source_amplitude".into(), json!(experiment.source_amplitude));
    map.insert("sponge_width".into(), json!(experiment.sponge_width));
    map.insert("sponge_strength".into(), json!(experiment.sponge_strength));
    map.insert("shots".into(), json!(experiment.shots));
    map.insert("receivers".into(), json!(experiment.receivers));
    if let Some(length) = experiment.length_unit_m {
        map.insert("length_unit_m".into(), json!(length));
    }
    if let Some(time) = experiment.time_unit_s {
        map.insert("time_unit_s".into(), json!(time));
    }
    if let Some(provenance) = &experiment.provenance {
        map.insert("provenance".into(), provenance.clone());
    }
    Value::Object(map)
}

fn run_forward(options: &Options) -> Result<(), String> {
    let experiment = Experiment::load(&options.experiment)?;
    std::fs::create_dir_all(&options.out)
        .map_err(|error| format!("{}: {error}", options.out.display()))?;

    let background = experiment.background_flat();
    let n_shots = experiment.shots.len();
    let steps = experiment.steps;
    let n_receivers = experiment.receivers.len();
    let shot_size = steps * n_receivers;

    let mut traces = vec![0.0; n_shots * shot_size];
    let mut first_frames = Vec::new();
    let mut frame_steps = Vec::new();

    for (shot_index, shot) in experiment.shots.iter().enumerate() {
        let record = shot_index == 0 && options.every.is_some();
        let result = forward(&experiment, &background, *shot, options.every, record);
        traces[shot_index * shot_size..(shot_index + 1) * shot_size]
            .copy_from_slice(&result.traces);
        if shot_index == 0 {
            first_frames = result.frames;
            frame_steps = result.frame_steps;
        }
    }

    let traces_path = options.out.join("traces.npy");
    write_npy_f64(&traces_path, &[n_shots, steps, n_receivers], &traces)
        .map_err(|error| format!("{}: {error}", traces_path.display()))?;

    if let Some(_every) = options.every {
        let wavefield: Vec<f32> = first_frames.iter().map(|value| *value as f32).collect();
        let wavefield_path = options.out.join("wavefield.npy");
        write_npy_f32(
            &wavefield_path,
            &[frame_steps.len(), experiment.nz, experiment.nx],
            &wavefield,
        )
        .map_err(|error| format!("{}: {error}", wavefield_path.display()))?;

        // Echo = wavefield in background+perturbation minus background only.
        let mut perturbed_speed = background.clone();
        for (speed, perturbation) in perturbed_speed
            .iter_mut()
            .zip(experiment.perturbation_flat().iter())
        {
            *speed += perturbation;
        }
        let perturbed = forward(
            &experiment,
            &perturbed_speed,
            experiment.shots[0],
            options.every,
            true,
        );
        let echo: Vec<f32> = first_frames
            .iter()
            .zip(perturbed.frames.iter())
            .map(|(background, perturbed)| (*perturbed - *background) as f32)
            .collect();
        let echo_path = options.out.join("echo.npy");
        write_npy_f32(
            &echo_path,
            &[frame_steps.len(), experiment.nz, experiment.nx],
            &echo,
        )
        .map_err(|error| format!("{}: {error}", echo_path.display()))?;
    }

    let mut run = Map::new();
    run.insert("experiment".into(), compact_experiment(&experiment));
    run.insert(
        "experiment_file".into(),
        json!(options.experiment.to_string_lossy()),
    );
    if let Some(every) = options.every {
        let times: Vec<f64> = frame_steps
            .iter()
            .map(|step| *step as f64 * experiment.dt)
            .collect();
        run.insert(
            "recording".into(),
            json!({"every": every, "steps": frame_steps, "times": times}),
        );
    }
    let run_path = options.out.join("run.json");
    let run_text = serde_json::to_string_pretty(&Value::Object(run))
        .map_err(|error| error.to_string())?;
    std::fs::write(&run_path, format!("{run_text}\n"))
        .map_err(|error| format!("{}: {error}", run_path.display()))?;

    let result = json!({
        "mode": "forward",
        "nx": experiment.nx,
        "nz": experiment.nz,
        "steps": experiment.steps,
        "dt": experiment.dt,
        "dx": experiment.dx,
        "shots": experiment.shots,
        "receivers": experiment.receivers,
    });
    let result_path = options.out.join("result.json");
    let result_text = serde_json::to_string(&result).map_err(|error| error.to_string())?;
    std::fs::write(&result_path, format!("{result_text}\n"))
        .map_err(|error| format!("{}: {error}", result_path.display()))?;

    // Official contract: a header, then one line per shot.
    println!("shot\tmode\tdata_l2_norm");
    for shot_index in 0..n_shots {
        let slice = &traces[shot_index * shot_size..(shot_index + 1) * shot_size];
        let norm = slice.iter().map(|value| value * value).sum::<f64>().sqrt();
        println!("{shot_index}\tforward\t{norm:.9}");
    }

    Ok(())
}

fn run_born(options: &Options) -> Result<(), String> {
    if options.every.is_some() {
        return Err("--every applies to forward or adjoint, not born".into());
    }
    let experiment = Experiment::load(&options.experiment)?;
    std::fs::create_dir_all(&options.out)
        .map_err(|error| format!("{}: {error}", options.out.display()))?;

    let background = experiment.background_flat();
    let perturbation = experiment.perturbation_flat();
    let n_shots = experiment.shots.len();
    let steps = experiment.steps;
    let n_receivers = experiment.receivers.len();
    let shot_size = steps * n_receivers;

    let data = born(&experiment, &background, &perturbation);
    let data_path = options.out.join("born_data.npy");
    write_npy_f64(&data_path, &[n_shots, steps, n_receivers], &data)
        .map_err(|error| format!("{}: {error}", data_path.display()))?;

    let mut run = Map::new();
    run.insert("experiment".into(), compact_experiment(&experiment));
    run.insert(
        "experiment_file".into(),
        json!(options.experiment.to_string_lossy()),
    );
    let run_path = options.out.join("run.json");
    let run_text = serde_json::to_string_pretty(&Value::Object(run))
        .map_err(|error| error.to_string())?;
    std::fs::write(&run_path, format!("{run_text}\n"))
        .map_err(|error| format!("{}: {error}", run_path.display()))?;

    let result = json!({
        "mode": "born",
        "nx": experiment.nx,
        "nz": experiment.nz,
        "steps": experiment.steps,
        "dt": experiment.dt,
        "dx": experiment.dx,
        "shots": experiment.shots,
        "receivers": experiment.receivers,
    });
    let result_path = options.out.join("result.json");
    let result_text = serde_json::to_string(&result).map_err(|error| error.to_string())?;
    std::fs::write(&result_path, format!("{result_text}\n"))
        .map_err(|error| format!("{}: {error}", result_path.display()))?;

    // Official contract: a header, then one line per shot.
    println!("shot\tmode\tdata_l2_norm");
    for shot_index in 0..n_shots {
        let slice = &data[shot_index * shot_size..(shot_index + 1) * shot_size];
        let norm = slice.iter().map(|value| value * value).sum::<f64>().sqrt();
        println!("{shot_index}\tborn\t{norm:.9}");
    }

    Ok(())
}

fn run_adjoint(options: &Options) -> Result<(), String> {
    let experiment = Experiment::load(&options.experiment)?;
    std::fs::create_dir_all(&options.out)
        .map_err(|error| format!("{}: {error}", options.out.display()))?;

    let data_path = options
        .data
        .as_ref()
        .ok_or("--data is required for adjoint: pass the born_data.npy receiver weights")?;
    let (shape, weights) = read_npy_f64(data_path)?;
    let expected = [
        experiment.shots.len(),
        experiment.steps,
        experiment.receivers.len(),
    ];
    if shape[..] != expected[..] {
        return Err(format!(
            "{}: expected float64 shape {expected:?}, got {shape:?}",
            data_path.display()
        ));
    }

    let background = experiment.background_flat();
    let result = adjoint(&experiment, &background, &weights, options.every);

    let image_path = options.out.join("image.npy");
    write_npy_f64(&image_path, &[experiment.nz, experiment.nx], &result.image)
        .map_err(|error| format!("{}: {error}", image_path.display()))?;

    let mut run = Map::new();
    run.insert("experiment".into(), compact_experiment(&experiment));
    run.insert(
        "experiment_file".into(),
        json!(options.experiment.to_string_lossy()),
    );
    if let Some(every) = options.every {
        let wavefield_path = options.out.join("wavefield.npy");
        write_npy_f32(
            &wavefield_path,
            &[result.frame_steps.len(), experiment.nz, experiment.nx],
            &result.frames,
        )
        .map_err(|error| format!("{}: {error}", wavefield_path.display()))?;
        let times: Vec<f64> = result
            .frame_steps
            .iter()
            .map(|step| *step as f64 * experiment.dt)
            .collect();
        run.insert(
            "recording".into(),
            json!({"every": every, "steps": result.frame_steps, "times": times}),
        );
    }
    let run_path = options.out.join("run.json");
    let run_text = serde_json::to_string_pretty(&Value::Object(run))
        .map_err(|error| error.to_string())?;
    std::fs::write(&run_path, format!("{run_text}\n"))
        .map_err(|error| format!("{}: {error}", run_path.display()))?;

    let per_shot: Vec<Value> = result
        .stats
        .per_shot
        .iter()
        .map(|shot| {
            json!({
                "reverse_calls": shot.reverse_calls,
                "scheduler_forward_calls": shot.scheduler_forward_calls,
                "peak_saved_states": shot.peak_saved_states,
            })
        })
        .collect();
    let statistics = json!({
        "storage": "full",
        "checkpoints": Value::Null,
        "reverse_calls": result.stats.reverse_calls,
        "scheduler_forward_calls": result.stats.scheduler_forward_calls,
        "peak_saved_states": result.stats.peak_saved_states,
        "peak_saved_bytes": result.stats.peak_saved_bytes,
        "per_shot": per_shot,
    });
    let result_json = json!({
        "mode": "adjoint",
        "nx": experiment.nx,
        "nz": experiment.nz,
        "steps": experiment.steps,
        "dt": experiment.dt,
        "dx": experiment.dx,
        "shots": experiment.shots,
        "receivers": experiment.receivers,
        "statistics": statistics,
    });
    let result_path = options.out.join("result.json");
    let result_text = serde_json::to_string(&result_json).map_err(|error| error.to_string())?;
    std::fs::write(&result_path, format!("{result_text}\n"))
        .map_err(|error| format!("{}: {error}", result_path.display()))?;

    // Official contract: a header, then one line per shot.
    println!("shot\tmode\tdata_l2_norm");
    for (shot_index, norm) in result.shot_norms.iter().enumerate() {
        println!("{shot_index}\tadjoint\t{norm:.9}");
    }

    Ok(())
}

fn main() {
    let result = parse_options().and_then(|options| match options.mode.as_str() {
        "forward" => run_forward(&options),
        "born" => run_born(&options),
        "adjoint" => run_adjoint(&options),
        other => Err(format!(
            "unsupported mode '{other}': only 'forward', 'born' and 'adjoint' are implemented"
        )),
    });
    if let Err(error) = result {
        eprintln!("seismic: {error}");
        std::process::exit(2);
    }
}
