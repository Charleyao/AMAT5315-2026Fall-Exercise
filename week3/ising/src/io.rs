//! File output: `run.json`, `series.jsonl` and (optionally) `spins.jsonl`.

use crate::simulation::Report;
use serde::Serialize;
use std::path::Path;

fn to_io_error(e: serde_json::Error) -> std::io::Error {
    std::io::Error::other(e)
}

fn jsonl<T: Serialize>(items: &[T]) -> Result<String, std::io::Error> {
    let mut buf = String::new();
    for item in items {
        buf.push_str(&serde_json::to_string(item).map_err(to_io_error)?);
        buf.push('\n');
    }
    Ok(buf)
}

/// Write the run metadata and records into `out`, creating the folder if
/// needed and overwriting files of the same names.
pub fn write_output(out: &Path, report: &Report) -> std::io::Result<()> {
    std::fs::create_dir_all(out)?;

    let run_json = serde_json::to_string_pretty(&report.meta).map_err(to_io_error)?;
    std::fs::write(out.join("run.json"), run_json + "\n")?;
    std::fs::write(out.join("series.jsonl"), jsonl(&report.series)?)?;

    if report.meta.sample_every > 0 {
        std::fs::write(out.join("spins.jsonl"), jsonl(&report.frames)?)?;
    } else {
        // Do not leave a stale frame file from an earlier run beside fresh data.
        let stale = out.join("spins.jsonl");
        if stale.exists() {
            std::fs::remove_file(stale)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::{Config, Update, simulate};

    fn run(dir: &Path, every: usize) -> Report {
        let cfg = Config {
            update: Update::Metropolis,
            l: 4,
            t_from: 1.5,
            t_to: 1.5,
            t_step: 1.0,
            discard: 0,
            measure: 4,
            every,
            seed: 1,
            out: dir.to_path_buf(),
        };
        let report = simulate(&cfg);
        write_output(dir, &report).unwrap();
        report
    }

    #[test]
    fn writes_all_three_files_and_round_trips() {
        let dir = std::env::temp_dir().join("ising_io_test");
        let _ = std::fs::remove_dir_all(&dir);
        run(&dir, 2);

        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("run.json")).unwrap()).unwrap();
        assert_eq!(meta["L"].as_u64(), Some(4));
        assert_eq!(meta["update"].as_str(), Some("metropolis"));
        assert_eq!(meta["sample_every"].as_u64(), Some(2));
        assert_eq!(meta["time_unit"].as_str(), Some("sweep"));
        assert_eq!(meta["t_grid"].as_array().unwrap().len(), 1);

        let series = std::fs::read_to_string(dir.join("series.jsonl")).unwrap();
        assert_eq!(series.lines().count(), 4);
        let row: serde_json::Value = serde_json::from_str(series.lines().next().unwrap()).unwrap();
        assert!(row["M"].is_number() && row["E"].is_number());
        assert!(row["cluster_size"].is_null());

        let spins = std::fs::read_to_string(dir.join("spins.jsonl")).unwrap();
        assert_eq!(spins.lines().count(), 2);
        let frame: serde_json::Value = serde_json::from_str(spins.lines().next().unwrap()).unwrap();
        assert_eq!(frame["spins"].as_array().unwrap().len(), 16);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn every_zero_writes_no_frames_and_removes_a_stale_file() {
        let dir = std::env::temp_dir().join("ising_io_no_frames_test");
        let _ = std::fs::remove_dir_all(&dir);
        run(&dir, 2);
        assert!(dir.join("spins.jsonl").exists());

        run(&dir, 0);
        assert!(!dir.join("spins.jsonl").exists());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
