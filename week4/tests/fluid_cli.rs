//! CLI contract for `fluid`: reads `field`'s JSON on stdin and writes the
//! artifacts promised by `week4/fluid.design.toml`.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

fn run_field(args: &[&str]) -> Vec<u8> {
    let out = Command::new(env!("CARGO_BIN_EXE_field"))
        .args(args)
        .output()
        .expect("run field");
    assert!(out.status.success(), "field failed: {}", String::from_utf8_lossy(&out.stderr));
    out.stdout
}

fn run_fluid(stdin: &[u8], args: &[&str]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_fluid"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn fluid");
    child.stdin.as_mut().unwrap().write_all(stdin).unwrap();
    child.wait_with_output().expect("wait fluid")
}

fn temp_out(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("week4_fluid_{}_{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn read_jsonl(path: &Path) -> Vec<serde_json::Value> {
    std::fs::read_to_string(path)
        .expect("fields.jsonl")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("fields.jsonl line parses"))
        .collect()
}

fn taylor_green_stdin(n: usize) -> Vec<u8> {
    run_field(&["taylor-green", "--n", &n.to_string()])
}

#[test]
fn fluid_reads_field_json_and_writes_the_contract() {
    let out = temp_out("contract");
    let stdout = run_fluid(
        &taylor_green_stdin(8),
        &[
            "--method", "rk4", "--nu", "0.1", "--dt", "0.05", "--t-end", "0.2",
            "--every", "0.1", "--out", out.to_str().unwrap(),
        ],
    );
    assert!(stdout.status.success(), "{}", String::from_utf8_lossy(&stdout.stderr));

    // stdout: header then one line per snapshot (stride = round(0.1/0.05) = 2).
    let text = String::from_utf8(stdout.stdout).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines[0], "t\tE\tZ");
    assert_eq!(lines.len(), 1 + 3, "steps 0, 2, 4: {lines:?}");

    // run.json metadata copied from the pipe plus the solver settings.
    let run: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("run.json")).unwrap()).unwrap();
    assert_eq!(run["case"], "taylor-green");
    assert_eq!(run["n"], 8);
    assert_eq!(run["seed"], serde_json::Value::Null);
    assert_eq!(run["k_band"], serde_json::Value::Null);
    assert_eq!(run["method"], "rk4");
    assert_eq!(run["nu"], 0.1);
    assert_eq!(run["dt"], 0.05);
    assert_eq!(run["t_end"], 0.2);
    assert_eq!(run["snapshot_every"], 0.1);

    // fields.jsonl: step 0 exists, arrays are n*n, steps follow round(every/dt).
    let frames = read_jsonl(&out.join("fields.jsonl"));
    assert_eq!(frames.len(), 3);
    assert_eq!(frames[0]["step"], 0);
    assert_eq!(frames[0]["t"].as_f64().unwrap(), 0.0);
    for (k, fr) in frames.iter().enumerate() {
        assert_eq!(fr["step"], (k * 2) as i64);
        assert_eq!(fr["u"].as_array().unwrap().len(), 64);
        assert_eq!(fr["v"].as_array().unwrap().len(), 64);
        assert_eq!(fr["omega"].as_array().unwrap().len(), 64);
    }
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn row_major_ordering_of_the_first_frame() {
    // n = 8 grid, known Taylor-Green values at step 0.
    let out = temp_out("rowmajor");
    let stdout = run_fluid(
        &taylor_green_stdin(8),
        &[
            "--method", "euler", "--nu", "0.1", "--dt", "0.05", "--t-end", "0.0",
            "--every", "0.1", "--out", out.to_str().unwrap(),
        ],
    );
    assert!(stdout.status.success(), "{}", String::from_utf8_lossy(&stdout.stderr));
    let frames = read_jsonl(&out.join("fields.jsonl"));
    assert_eq!(frames.len(), 1); // only step 0
    let u = frames[0]["u"].as_array().unwrap();
    let v = frames[0]["v"].as_array().unwrap();
    // (ix, iy) = (0, 2): idx = 2*8 + 0 = 16, u = 1, v = 0.
    assert!((u[16].as_f64().unwrap() - 1.0).abs() < 1e-5);
    assert!(v[16].as_f64().unwrap().abs() < 1e-5);
    // (ix, iy) = (2, 0): idx = 0*8 + 2 = 2, u = 0, v = -1.
    assert!(u[2].as_f64().unwrap().abs() < 1e-5);
    assert!((v[2].as_f64().unwrap() + 1.0).abs() < 1e-5);
    // omega(0,0) = -2, omega(2,0) = 0 -> row-major index confirms ordering.
    let om = frames[0]["omega"].as_array().unwrap();
    assert!((om[0].as_f64().unwrap() + 2.0).abs() < 1e-4);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn all_three_methods_are_accepted() {
    for method in ["euler", "rk2", "rk4"] {
        let out = temp_out(method);
        let stdout = run_fluid(
            &taylor_green_stdin(8),
            &[
                "--method", method, "--nu", "0.1", "--dt", "0.01", "--t-end", "0.02",
                "--every", "0.01", "--out", out.to_str().unwrap(),
            ],
        );
        assert!(stdout.status.success(), "{method}: {}", String::from_utf8_lossy(&stdout.stderr));
        let _ = std::fs::remove_dir_all(&out);
    }
}

#[test]
fn invalid_method_is_rejected() {
    let out = temp_out("invalid");
    let stdout = run_fluid(
        &taylor_green_stdin(8),
        &[
            "--method", "leapfrog", "--nu", "0.1", "--dt", "0.01", "--t-end", "0.02",
            "--every", "0.01", "--out", out.to_str().unwrap(),
        ],
    );
    assert_eq!(stdout.status.code(), Some(2));
    assert!(stdout.stdout.is_empty(), "error output leaked to stdout");
    assert!(!stdout.stderr.is_empty());
    let _ = std::fs::remove_dir_all(&out);
}
