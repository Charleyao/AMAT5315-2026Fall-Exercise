//! Manual command-line parsing for md run|check|video.

use crate::ForceMethod;
use std::path::PathBuf;

pub enum Command {
    Run(RunArgs),
    Check { dir: PathBuf },
    Video { dir: PathBuf, out: PathBuf },
}

pub struct RunArgs {
    pub n: usize,
    pub rho: f64,
    pub temperature: f64,
    pub dt: f64,
    pub eq_steps: usize,
    pub steps: usize,
    pub sample_every: usize,
    pub seed: u64,
    pub force: ForceMethod,
    pub out: PathBuf,
}

impl Default for RunArgs {
    fn default() -> Self {
        Self {
            n: 100,
            rho: 0.8,
            temperature: 0.5,
            dt: 0.01,
            eq_steps: 2000,
            steps: 10000,
            sample_every: 50,
            seed: 2026,
            force: ForceMethod::Cells,
            out: PathBuf::from("artifacts"),
        }
    }
}

fn usage() -> String {
    "usage:\n  md run [--n 100] [--rho 0.8] [--temperature 0.5] [--dt 0.01] [--eq-steps 2000] [--steps 10000] [--sample-every 50] [--seed 2026] [--force naive|cells] [--out artifacts]\n  md check [artifacts]\n  md video [artifacts] [--out artifacts/run.mp4]".to_string()
}

pub fn parse(args: &[String]) -> Result<Command, String> {
    let rest = &args[1..];
    if rest.is_empty() {
        return Err(usage());
    }
    match rest[0].as_str() {
        "run" => parse_run(&rest[1..]),
        "check" => parse_check(&rest[1..]),
        "video" => parse_video(&rest[1..]),
        "help" | "--help" | "-h" => Err(usage()),
        other => Err(format!("unknown subcommand: {other}\n{}", usage())),
    }
}

fn parse_run(args: &[String]) -> Result<Command, String> {
    let mut run = RunArgs::default();
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        let value = args
            .get(i + 1)
            .ok_or_else(|| format!("missing value for {key}"))?;
        match key {
            "--n" => run.n = value.parse().map_err(|_| format!("invalid --n: {value}"))?,
            "--rho" => run.rho = value.parse().map_err(|_| format!("invalid --rho: {value}"))?,
            "--temperature" => run.temperature = value.parse().map_err(|_| format!("invalid --temperature: {value}"))?,
            "--dt" => run.dt = value.parse().map_err(|_| format!("invalid --dt: {value}"))?,
            "--eq-steps" => run.eq_steps = value.parse().map_err(|_| format!("invalid --eq-steps: {value}"))?,
            "--steps" => run.steps = value.parse().map_err(|_| format!("invalid --steps: {value}"))?,
            "--sample-every" => run.sample_every = value.parse().map_err(|_| format!("invalid --sample-every: {value}"))?,
            "--seed" => run.seed = value.parse().map_err(|_| format!("invalid --seed: {value}"))?,
            "--force" => run.force = ForceMethod::parse(value)
                .map_err(|_| format!("invalid --force: {value} (expected naive or cells)"))?,
            "--out" => run.out = PathBuf::from(value),
            other => return Err(format!("unknown flag: {other}\n{}", usage())),
        }
        i += 2;
    }
    Ok(Command::Run(run))
}

fn parse_check(args: &[String]) -> Result<Command, String> {
    let mut dir = PathBuf::from("artifacts");
    for a in args {
        if a.starts_with("--") {
            return Err(format!("unknown flag: {a}\n{}", usage()));
        }
        dir = PathBuf::from(a);
    }
    Ok(Command::Check { dir })
}

fn parse_video(args: &[String]) -> Result<Command, String> {
    let mut dir = PathBuf::from("artifacts");
    let mut out = PathBuf::from("artifacts/run.mp4");
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                let value = args.get(i + 1).ok_or("missing value for --out")?;
                out = PathBuf::from(value);
                i += 2;
            }
            other => {
                if other.starts_with("--") {
                    return Err(format!("unknown flag: {other}\n{}", usage()));
                }
                dir = PathBuf::from(other);
                i += 1;
            }
        }
    }
    Ok(Command::Video { dir, out })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn parse_run_defaults() {
        match parse(&args(&["md", "run"])).unwrap() {
            Command::Run(r) => {
                assert_eq!(r.n, 100);
                assert_eq!(r.rho, 0.8);
                assert_eq!(r.temperature, 0.5);
                assert_eq!(r.dt, 0.01);
                assert_eq!(r.eq_steps, 2000);
                assert_eq!(r.steps, 10000);
                assert_eq!(r.sample_every, 50);
                assert_eq!(r.seed, 2026);
                assert_eq!(r.out, PathBuf::from("artifacts"));
            }
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn parse_run_overrides() {
        match parse(&args(&["md", "run", "--n", "36", "--out", "x"])).unwrap() {
            Command::Run(r) => {
                assert_eq!(r.n, 36);
                assert_eq!(r.out, PathBuf::from("x"));
            }
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn parse_check_and_video() {
        match parse(&args(&["md", "check", "dir"])).unwrap() {
            Command::Check { dir } => assert_eq!(dir, PathBuf::from("dir")),
            _ => panic!("expected Check"),
        }
        match parse(&args(&["md", "video", "dir", "--out", "o.mp4"])).unwrap() {
            Command::Video { dir, out } => {
                assert_eq!(dir, PathBuf::from("dir"));
                assert_eq!(out, PathBuf::from("o.mp4"));
            }
            _ => panic!("expected Video"),
        }
    }

    #[test]
    fn parse_run_force_defaults_to_cells() {
        match parse(&args(&["md", "run"])).unwrap() {
            Command::Run(r) => assert_eq!(r.force, ForceMethod::Cells),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn parse_run_force_naive() {
        match parse(&args(&["md", "run", "--force", "naive"])).unwrap() {
            Command::Run(r) => assert_eq!(r.force, ForceMethod::Naive),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn parse_run_force_cells() {
        match parse(&args(&["md", "run", "--force", "cells"])).unwrap() {
            Command::Run(r) => assert_eq!(r.force, ForceMethod::Cells),
            _ => panic!("expected Run"),
        }
    }

    #[test]
    fn parse_run_force_rejects_invalid() {
        match parse(&args(&["md", "run", "--force", "bogus"])) {
            Err(msg) => assert!(msg.contains("invalid --force"), "error was: {msg}"),
            Ok(_) => panic!("expected an error for an invalid --force value"),
        }
    }
}
