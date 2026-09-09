//! Integration tests that exercise the compiled `md` binary.

use std::process::Command;

#[test]
fn cli_run_writes_readable_json() {
    let dir = std::env::temp_dir().join("md_cli_run_test");
    let _ = std::fs::remove_dir_all(&dir);

    let status = Command::new(env!("CARGO_BIN_EXE_md"))
        .args([
            "run", "--n", "36", "--eq-steps", "0", "--steps", "50",
            "--sample-every", "50", "--out", dir.to_str().unwrap(),
        ])
        .status()
        .expect("failed to run md binary");
    assert!(status.success());

    let run_json = std::fs::read_to_string(dir.join("run.json")).unwrap();
    let meta: serde_json::Value = serde_json::from_str(&run_json).unwrap();
    assert_eq!(meta["n"].as_u64(), Some(36));
    assert_eq!(meta["integrator"].as_str(), Some("velocity-verlet"));
    assert_eq!(meta["box"].as_array().unwrap().len(), 2);
    assert_eq!(meta["rho"].as_f64(), Some(0.8));

    let traj = std::fs::read_to_string(dir.join("traj.jsonl")).unwrap();
    let lines: Vec<&str> = traj.lines().collect();
    assert_eq!(lines.len(), 1);
    let frame: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(frame["step"].as_u64(), Some(50));
    assert!(frame["t"].is_number());
    assert_eq!(frame["pos"].as_array().unwrap().len(), 36);
    assert_eq!(frame["vel"].as_array().unwrap().len(), 36);
    assert!(frame["E_pot"].is_number());
    assert!(frame["E_kin"].is_number());

    let _ = std::fs::remove_dir_all(&dir);
}
