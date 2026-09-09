use md::cli::{parse, Command, RunArgs};
use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    match parse(&args) {
        Ok(Command::Run(args)) => run_cmd(&args),
        Ok(Command::Check { dir }) => check_cmd(&dir),
        Ok(Command::Video { dir, out }) => video_cmd(&dir, &out),
        Err(msg) => {
            eprintln!("{msg}");
            ExitCode::from(2)
        }
    }
}

fn run_cmd(args: &RunArgs) -> ExitCode {
    let config = md::RunConfig {
        n: args.n,
        rho: args.rho,
        temperature: args.temperature,
        dt: args.dt,
        eq_steps: args.eq_steps,
        steps: args.steps,
        sample_every: args.sample_every,
        seed: args.seed,
    };
    match md::run_simulation(&config) {
        Ok(output) => {
            if let Err(e) = md::write_output(&args.out, &output) {
                eprintln!("error writing output: {e}");
                return ExitCode::from(1);
            }
            println!("wrote {}/run.json", args.out.display());
            println!(
                "wrote {}/traj.jsonl ({} frames)",
                args.out.display(),
                output.frames.len()
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}

fn check_cmd(dir: &Path) -> ExitCode {
    match md::read_output(dir) {
        Ok(output) => match md::check(&output) {
            Ok(report) => {
                println!(
                    "secular_drift={:.6} (<2e-3) {}",
                    report.secular_drift,
                    pass(report.secular_drift < 2e-3)
                );
                println!(
                    "T_speed={:.6} (|T_speed-0.5|={:.6} <0.05) {}",
                    report.t_speed,
                    (report.t_speed - 0.5).abs(),
                    pass((report.t_speed - 0.5).abs() < 0.05)
                );
                println!(
                    "chi2_over_22={:.6} (<2) {}",
                    report.chi2_over_22,
                    pass(report.chi2_over_22 < 2.0)
                );
                if report.passed {
                    println!("md check: PASS");
                    ExitCode::SUCCESS
                } else {
                    println!("md check: FAIL");
                    ExitCode::from(1)
                }
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::from(1)
            }
        },
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}

fn video_cmd(dir: &Path, out: &Path) -> ExitCode {
    match md::render_video(dir, out) {
        Ok(report) => {
            println!(
                "wrote {} ({} frames, {} bytes)",
                report.out_path, report.frames, report.size_bytes
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}

fn pass(ok: bool) -> &'static str {
    if ok {
        "PASS"
    } else {
        "FAIL"
    }
}
