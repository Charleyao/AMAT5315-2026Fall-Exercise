use ising::cli::{parse, usage};
use ising::simulation::{Report, Update, simulate};
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().skip(1).any(|a| a == "--help" || a == "-h") {
        print!("{}", usage());
        return ExitCode::SUCCESS;
    }

    let cfg = match parse(&args) {
        Ok(cfg) => cfg,
        Err(msg) => {
            eprintln!("{msg}");
            return ExitCode::from(2);
        }
    };

    let report = simulate(&cfg);

    if let Err(e) = ising::io::write_output(&cfg.out, &report) {
        eprintln!("error writing output: {e}");
        return ExitCode::from(1);
    }

    print_summary(&report, cfg.update);
    ExitCode::SUCCESS
}

/// Header plus one tab-separated line per temperature.
fn print_summary(report: &Report, update: Update) {
    match update {
        Update::Metropolis => println!("T\tmean_abs_M\tacceptance_rate"),
        Update::Wolff => println!("T\tmean_abs_M\tmean_cluster_size"),
    }
    for s in &report.summaries {
        let extra = match update {
            Update::Metropolis => s.acceptance_rate.unwrap_or(f64::NAN),
            Update::Wolff => s.mean_cluster_size.unwrap_or(f64::NAN),
        };
        println!("{:.6}\t{:.6}\t{:.6}", s.t, s.mean_abs_m, extra);
    }
}
