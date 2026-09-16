//! Integration tests that exercise the compiled `ising` binary.

use std::process::Command;

fn run(dir: &std::path::Path, extra: &[&str]) -> std::process::Output {
    let mut args = vec![
        "--update",
        "metropolis",
        "--l",
        "4",
        "--t-from",
        "1.5",
        "--t-to",
        "1.7",
        "--t-step",
        "0.1",
        "--discard",
        "3",
        "--measure",
        "4",
        "--seed",
        "2026",
        "--out",
    ];
    let dir_str = dir.to_str().unwrap();
    args.push(dir_str);
    args.extend_from_slice(extra);
    Command::new(env!("CARGO_BIN_EXE_ising"))
        .args(args)
        .output()
        .expect("failed to run ising binary")
}

#[test]
fn run_writes_expected_files_and_stdout() {
    let dir = std::env::temp_dir().join("ising_cli_run_test");
    let _ = std::fs::remove_dir_all(&dir);

    let output = run(&dir, &["--every", "2"]);
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut lines = stdout.lines();
    assert_eq!(lines.next(), Some("T\tmean_abs_M\tacceptance_rate"));
    let rows: Vec<&str> = lines.collect();
    assert_eq!(rows.len(), 3, "one row per temperature");
    assert_eq!(rows[0].split('\t').count(), 3);

    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("run.json")).unwrap()).unwrap();
    assert_eq!(meta["L"].as_u64(), Some(4));
    assert_eq!(meta["t_grid"].as_array().unwrap().len(), 3);
    assert_eq!(meta["time_unit"].as_str(), Some("sweep"));
    assert_eq!(meta["sample_every"].as_u64(), Some(2));

    let series = std::fs::read_to_string(dir.join("series.jsonl")).unwrap();
    assert_eq!(
        series.lines().count(),
        12,
        "3 temperatures x 4 measured steps"
    );
    let first: serde_json::Value = serde_json::from_str(series.lines().next().unwrap()).unwrap();
    assert_eq!(first["sweep"].as_u64(), Some(1));
    assert_eq!(first["L"].as_u64(), Some(4));

    let spins = std::fs::read_to_string(dir.join("spins.jsonl")).unwrap();
    assert_eq!(spins.lines().count(), 6, "2 frames per temperature");
    let frame: serde_json::Value = serde_json::from_str(spins.lines().next().unwrap()).unwrap();
    assert_eq!(frame["sweep"].as_u64(), Some(5), "3 discard + 2 measured");
    let values = frame["spins"].as_array().unwrap();
    assert_eq!(values.len(), 16);
    assert!(
        values
            .iter()
            .all(|v| v.as_i64() == Some(1) || v.as_i64() == Some(-1))
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn every_zero_writes_no_spins_file() {
    let dir = std::env::temp_dir().join("ising_cli_no_spins_test");
    let _ = std::fs::remove_dir_all(&dir);

    let output = run(&dir, &[]);
    assert!(output.status.success());
    assert!(dir.join("run.json").exists());
    assert!(dir.join("series.jsonl").exists());
    assert!(!dir.join("spins.jsonl").exists());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn wolff_time_unit_and_cluster_sizes() {
    let dir = std::env::temp_dir().join("ising_cli_wolff_test");
    let _ = std::fs::remove_dir_all(&dir);

    let output = Command::new(env!("CARGO_BIN_EXE_ising"))
        .args([
            "--update",
            "wolff",
            "--l",
            "4",
            "--t-from",
            "3.0",
            "--t-to",
            "3.0",
            "--t-step",
            "1.0",
            "--discard",
            "1",
            "--measure",
            "3",
            "--every",
            "1",
            "--seed",
            "5",
            "--out",
            dir.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run ising binary");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.starts_with("T\tmean_abs_M\tmean_cluster_size"));

    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("run.json")).unwrap()).unwrap();
    assert_eq!(meta["time_unit"].as_str(), Some("cluster_flip"));

    let series = std::fs::read_to_string(dir.join("series.jsonl")).unwrap();
    let row: serde_json::Value = serde_json::from_str(series.lines().next().unwrap()).unwrap();
    assert!(row["cluster_size"].is_u64());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn invalid_arguments_exit_with_code_two() {
    let output = Command::new(env!("CARGO_BIN_EXE_ising"))
        .args(["--l", "1"])
        .output()
        .expect("failed to run ising binary");
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("--l"));
}
