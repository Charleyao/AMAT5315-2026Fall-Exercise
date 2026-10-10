//! Compile the isolated Enzyme kernel with `-Zautodiff=Enable`.
//!
//! The whole crate must not see the autodiff pass: argument parsing, JSON and
//! allocation live in the normal `std` crate, while only this `no_std` static
//! library is handed to Enzyme. It is linked back through a small C ABI.

use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=src/kernel.rs");
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let status = Command::new(env::var_os("RUSTC").expect("RUSTC"))
        .args([
            "--edition=2024",
            "--crate-name",
            "enzyme_kernel",
            "--crate-type",
            "staticlib",
            "-C",
            "opt-level=3",
            "-C",
            "lto=fat",
            "-C",
            "panic=abort",
            "-Zautodiff=Enable",
            "src/kernel.rs",
            "-o",
        ])
        .arg(out.join("libenzyme_kernel.a"))
        .status()
        .expect("compile the Enzyme kernel with the pinned rustc");
    assert!(status.success(), "Enzyme kernel compilation failed");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=enzyme_kernel");
}
