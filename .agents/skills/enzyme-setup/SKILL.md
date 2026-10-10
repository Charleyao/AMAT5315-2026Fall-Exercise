---
name: enzyme-setup
description: Use when a Rust crate must differentiate numerical code with Enzyme (rustc's autodiff), when a build with -Zautodiff fails with a TypeAnalysis or Enzyme assertion, or when a sheet asks to set up the Enzyme toolchain for a simulation kernel.
---

# Enzyme setup

Set up a Rust crate so that Enzyme differentiates a numerical kernel while the rest of
the crate stays ordinary Rust. Validated on x86_64 Linux with `nightly-2026-09-05`.

**The constraint.** This nightly's Enzyme pass fails when it sees a whole program
(CLI parsing, JSON, allocation). The kernel that is differentiated lives in its own
`no_std` static library, compiled by `build.rs` with `-Zautodiff=Enable`, and the
main crate calls it through a C ABI. Never enable `-Zautodiff=Enable` globally in
`.cargo/config.toml`.

## Steps

1. **Pin the compiler.** Run
   `rustup toolchain install nightly-2026-09-05 --profile minimal --component enzyme`
   and write `rust-toolchain.toml` at the crate root:
   ```toml
   [toolchain]
   channel = "nightly-2026-09-05"
   components = ["enzyme"]
   profile = "minimal"
   ```
2. **`Cargo.toml`.** `edition = "2024"` for the whole crate (the `unsafe extern` and
   `#[unsafe(no_mangle)]` syntax below need it); release profile `lto = "fat"`,
   `codegen-units = 1`, `opt-level = 3`; add a `[[bin]]` section per binary with
   `name`, `path`, and `test = false` (integration tests run the built executable).
3. **The kernel** `src/kernel.rs` is an input to `build.rs` only: never `mod kernel;`
   it from the main crate. Its header and footer are fixed:
   ```rust
   #![no_std]
   #![feature(autodiff)]
   use core::autodiff::{autodiff_forward, autodiff_reverse};
   // ... plain functions over f64 and slices, and their C ABI wrappers ...
   unsafe extern "C" {
       fn exp(x: f64) -> f64;   // transcendentals come from the C math library
       fn abort() -> !;
   }
   #[panic_handler]
   fn panic(_: &core::panic::PanicInfo) -> ! { unsafe { abort() } }
   ```
   Integer powers are written as multiplications. Activity annotations: `Const` for
   fixed inputs, `Dual` for active scalars and slices in forward mode, `Active`
   (scalars) or `Duplicated` (slices) in reverse mode. Export one
   `#[unsafe(no_mangle)] pub unsafe extern "C"` wrapper per primal and derivative
   that takes raw pointers and lengths.
4. **`build.rs`** compiles the kernel with Cargo's own `RUSTC` and links it:
   ```rust
   use std::{env, path::PathBuf, process::Command};
   fn main() {
       println!("cargo:rerun-if-changed=src/kernel.rs");
       let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
       let status = Command::new(env::var_os("RUSTC").unwrap())
           .args(["--edition=2024", "--crate-name", "enzyme_kernel", "--crate-type", "staticlib",
                  "-C", "opt-level=3", "-C", "lto=fat", "-C", "panic=abort",
                  "-Zautodiff=Enable", "src/kernel.rs", "-o"])
           .arg(out.join("libenzyme_kernel.a"))
           .status().expect("compile the Enzyme kernel with the pinned rustc");
       assert!(status.success(), "Enzyme kernel compilation failed");
       println!("cargo:rustc-link-search=native={}", out.display());
       println!("cargo:rustc-link-lib=static=enzyme_kernel");
   }
   ```
5. **The main crate** declares the wrappers in an `unsafe extern "C"` block and calls
   them only through safe functions that own every buffer and check every slice
   length before the call. No pointer escapes a call.
6. **Smoke test before real work.** Put this in the kernel, build with
   `cargo build --release`, and call it from a test or a `main` guarded by a flag:
   ```rust
   #[autodiff_forward(cube_forward, Dual, Dual)]
   #[autodiff_reverse(cube_reverse, Active, Active)]
   fn cube(x: f64) -> f64 { x * x * x }
   #[unsafe(no_mangle)]
   pub unsafe extern "C" fn enzyme_cube(x: f64, out: *mut f64) {
       let (y, dy) = cube_forward(x, 1.0);
       let (_, adj) = cube_reverse(x, 1.0);
       unsafe { *out = y; *out.add(1) = dy; *out.add(2) = adj; }
   }
   ```
   At `x = 2` the three values are `8`, `12`, `12`. Anything else, or a linker error
   about `exp`, means the kernel is not isolated or `-Zautodiff` is not on. Negative
   control: rerun the `build.rs` command by hand without `-Zautodiff=Enable`; it must
   fail with "using the autodiff feature requires -Z autodiff=Enable", which proves
   the derivatives above came from Enzyme and not from a stub.

## Rules

- Enzyme itself must differentiate the physics (the timestep, the energy). A
  handwritten tangent or adjoint belongs only in an independent check.
- One local derivative call per step: a reverse call receives the output adjoint and
  returns the input adjoints; the surrounding Rust loop composes steps through time.
- Keep the kernel free of `std`, allocation, and formatting. Serialization, argument
  parsing, and tests live in the main crate.
- Use the course Linux machine if the platform cannot run this nightly.

## Common mistakes

| Mistake | Fix |
|---|---|
| `-Zautodiff=Enable` in `.cargo/config.toml` | Remove it; only `build.rs` passes the flag, only to the kernel. |
| `f64::exp` in the kernel | Bind `exp` from the C library in an `extern "C"` block. |
| `powi` or `powf` in the kernel | Write the product out. |
| A `Vec` or `format!` in the kernel | Move it to the main crate; the kernel is `no_std`. |
| Wrapper called with a short slice | Check lengths in the safe wrapper before the FFI call. |
| Both crates define a `panic_handler` | Only the kernel does; the main crate keeps `std`. |
