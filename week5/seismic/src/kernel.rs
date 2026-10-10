//! Enzyme kernel: the numerical timestep with its forward-mode JVP and
//! reverse-mode VJP.
//!
//! This file is never `mod`-included by the crate. `build.rs` compiles it on
//! its own with `-Zautodiff=Enable` and links the resulting static archive, so
//! the autodiff pass only ever sees this small `no_std` translation unit.

#![no_std]
#![feature(autodiff)]
use core::autodiff::{autodiff_forward, autodiff_reverse};

/// One damped acoustic timestep on a C-order `[z, x]` grid:
///
/// ```text
/// next = ( 2 cur - (1 - dt sigma) prev + dt^2 (c^2 lap(cur) + src) ) / den
/// ```
///
/// `lap` is the five-point Laplacian with a zero outer Dirichlet boundary, so
/// the boundary cells of `next` stay zero as well. Only `c`, `prev` and `cur`
/// are active. The sponge coefficients `sigma`/`den`, the source `src`, the
/// timestep and the grid spacing are `Const`, and the initial state is supplied
/// by the caller, so its tangent is zero.
///
/// The generated `timestep_jvp` receives the tangent for each active argument
/// directly after the primal argument, in this order:
///
/// ```text
/// timestep_jvp(nx, nz, inv_dx2, dt,
///              c, dc, sigma, den, src,
///              prev, dprev, cur, dcur, next, dnext)
/// ```
#[autodiff_forward(
    timestep_jvp,
    Const, Const, Const, Const,
    Dual, Const, Const, Const,
    Dual, Dual, Dual
)]
#[autodiff_reverse(
    timestep_vjp,
    Const, Const, Const, Const,
    Duplicated, Const, Const, Const,
    Duplicated, Duplicated, Duplicated
)]
fn timestep(
    nx: usize,
    nz: usize,
    inv_dx2: f64,
    dt: f64,
    c: &[f64],
    sigma: &[f64],
    den: &[f64],
    src: &[f64],
    prev: &[f64],
    cur: &[f64],
    next: &mut [f64],
) {
    // Start from a zero field so the outer Dirichlet boundary has zero tangent.
    let n = nx * nz;
    let mut index = 0;
    while index < n {
        next[index] = 0.0;
        index += 1;
    }

    let mut z = 1;
    while z + 1 < nz {
        let mut x = 1;
        while x + 1 < nx {
            let index = z * nx + x;
            let lap = (cur[index + 1] + cur[index - 1] + cur[index + nx] + cur[index - nx]
                - 4.0 * cur[index])
                * inv_dx2;
            next[index] = (2.0 * cur[index]
                - (1.0 - dt * sigma[index]) * prev[index]
                + dt * dt * (c[index] * c[index] * lap + src[index]))
                / den[index];
            x += 1;
        }
        z += 1;
    }
}

/// C ABI wrapper for the primal timestep only.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn enzyme_timestep_primal(
    nx: usize,
    nz: usize,
    inv_dx2: f64,
    dt: f64,
    c: *const f64,
    sigma: *const f64,
    den: *const f64,
    src: *const f64,
    prev: *const f64,
    cur: *const f64,
    next: *mut f64,
) {
    let n = nx * nz;
    unsafe {
        let c = core::slice::from_raw_parts(c, n);
        let sigma = core::slice::from_raw_parts(sigma, n);
        let den = core::slice::from_raw_parts(den, n);
        let src = core::slice::from_raw_parts(src, n);
        let prev = core::slice::from_raw_parts(prev, n);
        let cur = core::slice::from_raw_parts(cur, n);
        let next = core::slice::from_raw_parts_mut(next, n);
        timestep(nx, nz, inv_dx2, dt, c, sigma, den, src, prev, cur, next);
    }
}

/// C ABI wrapper for the primal timestep and its forward-mode JVP.
///
/// Every buffer is caller-owned and has `nx * nz` elements. The safe Rust
/// wrapper checks the slice lengths before this is reached.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn enzyme_timestep(
    nx: usize,
    nz: usize,
    inv_dx2: f64,
    dt: f64,
    c: *const f64,
    dc: *const f64,
    sigma: *const f64,
    den: *const f64,
    src: *const f64,
    prev: *const f64,
    dprev: *const f64,
    cur: *const f64,
    dcur: *const f64,
    next: *mut f64,
    dnext: *mut f64,
) {
    let n = nx * nz;
    unsafe {
        let c = core::slice::from_raw_parts(c, n);
        let dc = core::slice::from_raw_parts(dc, n);
        let sigma = core::slice::from_raw_parts(sigma, n);
        let den = core::slice::from_raw_parts(den, n);
        let src = core::slice::from_raw_parts(src, n);
        let prev = core::slice::from_raw_parts(prev, n);
        let dprev = core::slice::from_raw_parts(dprev, n);
        let cur = core::slice::from_raw_parts(cur, n);
        let dcur = core::slice::from_raw_parts(dcur, n);
        let next = core::slice::from_raw_parts_mut(next, n);
        let dnext = core::slice::from_raw_parts_mut(dnext, n);
        timestep_jvp(
            nx, nz, inv_dx2, dt, c, dc, sigma, den, src, prev, dprev, cur, dcur, next, dnext,
        );
    }
}

/// C ABI wrapper for the reverse-mode VJP of the timestep.
///
/// The caller seeds `dnext` with the output adjoint; `dc`, `dprev` and `dcur`
/// are accumulators and must be zeroed first. Every buffer has `nx * nz`
/// elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn enzyme_timestep_vjp(
    nx: usize,
    nz: usize,
    inv_dx2: f64,
    dt: f64,
    c: *const f64,
    dc: *mut f64,
    sigma: *const f64,
    den: *const f64,
    src: *const f64,
    prev: *const f64,
    dprev: *mut f64,
    cur: *const f64,
    dcur: *mut f64,
    next: *mut f64,
    dnext: *mut f64,
) {
    let n = nx * nz;
    unsafe {
        let c = core::slice::from_raw_parts(c, n);
        let dc = core::slice::from_raw_parts_mut(dc, n);
        let sigma = core::slice::from_raw_parts(sigma, n);
        let den = core::slice::from_raw_parts(den, n);
        let src = core::slice::from_raw_parts(src, n);
        let prev = core::slice::from_raw_parts(prev, n);
        let dprev = core::slice::from_raw_parts_mut(dprev, n);
        let cur = core::slice::from_raw_parts(cur, n);
        let dcur = core::slice::from_raw_parts_mut(dcur, n);
        let next = core::slice::from_raw_parts_mut(next, n);
        let dnext = core::slice::from_raw_parts_mut(dnext, n);
        timestep_vjp(
            nx, nz, inv_dx2, dt, c, dc, sigma, den, src, prev, dprev, cur, dcur, next, dnext,
        );
    }
}

unsafe extern "C" {
    fn abort() -> !;
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    unsafe { abort() }
}
