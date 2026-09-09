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

#[test]
fn default_run_and_check_pass_physics_limits() {
    // The contract: default run -> md check exits 0 (all physics limits pass).
    let dir = std::env::temp_dir().join("md_contract_test");
    let _ = std::fs::remove_dir_all(&dir);

    let run = Command::new(env!("CARGO_BIN_EXE_md"))
        .args(["run", "--out", dir.to_str().unwrap()])
        .status()
        .expect("failed to run md run");
    assert!(run.success());

    let check = Command::new(env!("CARGO_BIN_EXE_md"))
        .args(["check", dir.to_str().unwrap()])
        .output()
        .expect("failed to run md check");
    assert!(
        check.status.success(),
        "md check failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&check.stdout),
        String::from_utf8_lossy(&check.stderr)
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn check_missing_dir_fails() {
    let out = Command::new(env!("CARGO_BIN_EXE_md"))
        .args(["check", "/nonexistent/md_dir"])
        .output()
        .expect("failed to run md check");
    assert!(!out.status.success());
}

#[test]
fn video_renders_mp4_when_ffmpeg_present() {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("ffmpeg not installed; skipping video encode test");
        return;
    }
    let dir = std::env::temp_dir().join("md_video_test");
    let _ = std::fs::remove_dir_all(&dir);

    let run = Command::new(env!("CARGO_BIN_EXE_md"))
        .args([
            "run", "--n", "36", "--eq-steps", "0", "--steps", "50",
            "--sample-every", "50", "--out", dir.to_str().unwrap(),
        ])
        .status()
        .expect("failed to run md run");
    assert!(run.success());

    let out = dir.join("run.mp4");
    let video = Command::new(env!("CARGO_BIN_EXE_md"))
        .args(["video", dir.to_str().unwrap(), "--out", out.to_str().unwrap()])
        .status()
        .expect("failed to run md video");
    assert!(video.success());

    let size = std::fs::metadata(&out).unwrap().len();
    assert!(size < 2_000_000, "video too large: {size} bytes");
    let _ = std::fs::remove_dir_all(&dir);
}
