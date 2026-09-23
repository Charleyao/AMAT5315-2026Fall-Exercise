//! CLI contract for `field`, taken straight from `week4/field.design.toml`.
//!
//! These run the actual binary, so they check that stdout is *only* the JSON
//! object the design promises and that diagnostics stay on stderr.

use std::process::Command;

fn field(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_field"))
        .args(args)
        .output()
        .expect("run field")
}

#[test]
fn taylor_green_example_emits_only_json() {
    let out = field(&["taylor-green", "--n", "64", "--nu", "0.1", "--t", "1"]);
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8(out.stdout).unwrap();
    let trimmed = stdout.trim();
    assert!(trimmed.starts_with('{') && trimmed.ends_with('}'), "{trimmed:.80}");
    assert!(stdout.contains("\"case\":\"taylor-green\""));
    assert!(stdout.contains("\"n\":64"));
    assert!(stdout.contains("\"seed\":null"));
    assert!(stdout.contains("\"k_band\":null"));
    assert!(out.stderr.is_empty(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn random_example_emits_only_json() {
    let out = field(&["random", "--n", "128", "--seed", "2026", "--k-min", "2", "--k-max", "6"]);
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8(out.stdout).unwrap();
    let trimmed = stdout.trim();
    assert!(trimmed.starts_with('{') && trimmed.ends_with('}'), "{trimmed:.80}");
    assert!(stdout.contains("\"case\":\"random\""));
    assert!(stdout.contains("\"n\":128"));
    assert!(stdout.contains("\"seed\":2026"));
    assert!(stdout.contains("\"k_band\":[2,6]"));
    assert!(out.stderr.is_empty(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn taylor_green_requires_nu_when_t_is_positive() {
    let out = field(&["taylor-green", "--n", "8", "--t", "1"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(out.stdout.is_empty(), "error output leaked to stdout");
    assert!(!out.stderr.is_empty());
}
