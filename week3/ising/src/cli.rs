//! Manual command-line parsing for the `ising` binary.

use crate::simulation::{Config, Update};
use std::path::PathBuf;

pub fn usage() -> String {
    "usage:
  ising --update metropolis|wolff --l <L> --t-from <T> --t-to <T> --t-step <dT> \\
        --discard <n> --measure <n> --seed <n> [--every <n>] --out <folder>

options:
  --update    metropolis (L*L single-site proposals per step) or wolff (one cluster flip per step)
  --l         integer lattice side, at least 2
  --t-from    lowest temperature; the ramp ascends from here, from an all-up lattice
  --t-to      temperature upper bound; included only if reached by the step
  --t-step    temperature step; each temperature starts from the previous lattice
  --discard   equilibration steps discarded at each temperature
  --measure   measured steps at each temperature
  --seed      random seed; one stream per run, carried through the ramp
  --every     record frames at measured steps every, 2*every, ... (default 0 = none)
  --out       output folder; may exist, same-named files are overwritten"
        .to_string()
}

/// Parse `args` (including the program name) into a validated [`Config`].
pub fn parse(args: &[String]) -> Result<Config, String> {
    let mut update: Option<Update> = None;
    let mut l: Option<usize> = None;
    let mut t_from: Option<f64> = None;
    let mut t_to: Option<f64> = None;
    let mut t_step: Option<f64> = None;
    let mut discard: Option<usize> = None;
    let mut measure: Option<usize> = None;
    let mut every: usize = 0;
    let mut seed: Option<u64> = None;
    let mut out: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        let key = args[i].as_str();
        let value = args
            .get(i + 1)
            .ok_or_else(|| format!("missing value for {key}\n{}", usage()))?;
        match key {
            "--update" => update = Some(Update::parse(value)?),
            "--l" => l = Some(parse_usize("--l", value)?),
            "--t-from" => t_from = Some(parse_f64("--t-from", value)?),
            "--t-to" => t_to = Some(parse_f64("--t-to", value)?),
            "--t-step" => t_step = Some(parse_f64("--t-step", value)?),
            "--discard" => discard = Some(parse_usize("--discard", value)?),
            "--measure" => measure = Some(parse_usize("--measure", value)?),
            "--every" => every = parse_usize("--every", value)?,
            "--seed" => seed = Some(parse_u64("--seed", value)?),
            "--out" => out = Some(PathBuf::from(value)),
            other => return Err(format!("unknown argument: {other}\n{}", usage())),
        }
        i += 2;
    }

    let mut missing = Vec::new();
    if update.is_none() {
        missing.push("--update");
    }
    if l.is_none() {
        missing.push("--l");
    }
    if t_from.is_none() {
        missing.push("--t-from");
    }
    if t_to.is_none() {
        missing.push("--t-to");
    }
    if t_step.is_none() {
        missing.push("--t-step");
    }
    if discard.is_none() {
        missing.push("--discard");
    }
    if measure.is_none() {
        missing.push("--measure");
    }
    if seed.is_none() {
        missing.push("--seed");
    }
    if out.is_none() {
        missing.push("--out");
    }
    if !missing.is_empty() {
        return Err(format!(
            "missing required argument(s): {}\n{}",
            missing.join(", "),
            usage()
        ));
    }

    let l = l.unwrap();
    let t_from = t_from.unwrap();
    let t_to = t_to.unwrap();
    let t_step = t_step.unwrap();
    let measure = measure.unwrap();

    if l < 2 {
        return Err(format!("--l must be at least 2 (got {l})"));
    }
    if t_from <= 0.0 {
        return Err(format!("--t-from must be positive (got {t_from})"));
    }
    if t_to < t_from {
        return Err(format!(
            "--t-to ({t_to}) must not be below --t-from ({t_from})"
        ));
    }
    if t_step <= 0.0 {
        return Err(format!("--t-step must be positive (got {t_step})"));
    }
    if measure < 1 {
        return Err(format!("--measure must be at least 1 (got {measure})"));
    }

    Ok(Config {
        update: update.unwrap(),
        l,
        t_from,
        t_to,
        t_step,
        discard: discard.unwrap(),
        measure,
        every,
        seed: seed.unwrap(),
        out: out.unwrap(),
    })
}

fn parse_usize(flag: &str, value: &str) -> Result<usize, String> {
    value
        .parse()
        .map_err(|_| format!("invalid {flag}: {value} (expected a non-negative integer)"))
}

fn parse_u64(flag: &str, value: &str) -> Result<u64, String> {
    value
        .parse()
        .map_err(|_| format!("invalid {flag}: {value} (expected a non-negative integer)"))
}

fn parse_f64(flag: &str, value: &str) -> Result<f64, String> {
    value
        .parse()
        .map_err(|_| format!("invalid {flag}: {value} (expected a number)"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        let mut v = vec!["ising".to_string()];
        v.extend(list.iter().map(|s| s.to_string()));
        v
    }

    const BASE: &[&str] = &[
        "--update",
        "metropolis",
        "--l",
        "8",
        "--t-from",
        "1.5",
        "--t-to",
        "3.5",
        "--t-step",
        "0.1",
        "--discard",
        "10",
        "--measure",
        "20",
        "--seed",
        "2026",
        "--out",
        "runs/ramp",
    ];

    #[test]
    fn parses_a_full_command_line() {
        let cfg = parse(&args(BASE)).unwrap();
        assert_eq!(cfg.update, Update::Metropolis);
        assert_eq!(cfg.l, 8);
        assert_eq!(cfg.discard, 10);
        assert_eq!(cfg.measure, 20);
        assert_eq!(cfg.every, 0, "every defaults to zero");
        assert_eq!(cfg.seed, 2026);
        assert_eq!(cfg.out, PathBuf::from("runs/ramp"));
    }

    #[test]
    fn missing_required_argument_is_an_error() {
        let without_seed: Vec<&str> = BASE
            .to_vec()
            .chunks(2)
            .filter(|pair| pair[0] != "--seed")
            .flatten()
            .copied()
            .collect();
        let err = parse(&args(&without_seed)).unwrap_err();
        assert!(err.contains("--seed"), "error was: {err}");
    }

    #[test]
    fn rejects_bad_values() {
        let mut bad_l = BASE.to_vec();
        let pos = bad_l.iter().position(|s| *s == "--l").unwrap();
        bad_l[pos + 1] = "1";
        assert!(
            parse(&args(&bad_l))
                .unwrap_err()
                .contains("--l must be at least 2")
        );

        let mut bad_step = BASE.to_vec();
        let pos = bad_step.iter().position(|s| *s == "--t-step").unwrap();
        bad_step[pos + 1] = "0";
        assert!(
            parse(&args(&bad_step))
                .unwrap_err()
                .contains("--t-step must be positive")
        );

        let mut bad_range = BASE.to_vec();
        let pos = bad_range.iter().position(|s| *s == "--t-to").unwrap();
        bad_range[pos + 1] = "1.0";
        assert!(parse(&args(&bad_range)).unwrap_err().contains("--t-to"));
    }

    #[test]
    fn rejects_unknown_update_and_flags() {
        let mut bad = BASE.to_vec();
        let pos = bad.iter().position(|s| *s == "--update").unwrap();
        bad[pos + 1] = "glauber";
        assert!(parse(&args(&bad)).unwrap_err().contains("--update"));
        assert!(
            parse(&args(&["--nope", "1"]))
                .unwrap_err()
                .contains("unknown argument")
        );
    }
}
