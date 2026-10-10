#!/usr/bin/env python3
"""Week 5 Part 1 VERIFY (1): compare forward/reverse AD and centered differences.

For the Lennard-Jones energy U(r) = 4*(r**-12 - r**-6), this compares four
derivative estimates on 601 equally spaced separations from r = 0.95 to 2.5:

  * the analytical derivative  dU/dr = 24*(r**-7 - 2*r**-13),
  * the hand-written forward-mode implementation in ``lj_ad.forward_mode``,
  * the hand-written reverse-mode implementation in ``lj_ad.reverse_mode``,
  * a centered finite difference with h = 1e-6.

The hand-written scalar routines are reused unchanged and mapped over the
sample with ``jax.vmap``. All arithmetic uses JAX 64-bit floats. A two-panel
figure is written to ``week5/artifacts/ad/modes.png`` and the maximum absolute
errors are printed and checked.

Run from anywhere with the week-5 JAX environment:

    week5/.venv/bin/python week5/scripts/compare_ad_modes.py
"""

# 64-bit floats must be enabled before any JAX array is created.
import jax

jax.config.update("jax_enable_x64", True)

import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
WEEK5 = HERE.parent
sys.path.insert(0, str(HERE))

import lj_ad  # noqa: E402  (path is set up above)

OUT = WEEK5 / "artifacts" / "ad" / "modes.png"

N_SAMPLES = 601
R_MIN, R_MAX = 0.95, 2.5
FD_STEP = 1e-6
FD_REFERENCE = 5e-9          # approximate expected finite-difference error
FD_REFERENCE_RTOL = 0.10     # accept within 10% of the reference
AD_LIMIT = 1e-12
ZERO_CROSSING = 2.0 ** (1.0 / 6.0)   # 1.122462048309373
ZERO_TOL = 1e-3


def lj_energy(r):
    """Ellipsis-free scalar energy in the a, b, c form used for Task 1."""
    return lj_ad.energy_nodes(r)["U"]


def analytical(r):
    """dU/dr = 24*(r**-7 - 2*r**-13)."""
    return 24.0 * (r**-7 - 2.0 * r**-13)


def forward_derivative(r):
    """Reuse the scalar forward-mode chain, mapped over the samples."""
    _, tangents = jax.vmap(lj_ad.forward_mode)(r)
    return tangents["U"]


def reverse_derivative(r):
    """Reuse the scalar reverse-mode chain, mapped over the samples."""
    _, adjoints = jax.vmap(lj_ad.reverse_mode)(r)
    return adjoints["r"]


def finite_difference(r, h=FD_STEP):
    """Centered difference of the energy with step h."""
    return (lj_energy(r + h) - lj_energy(r - h)) / (2.0 * h)


def zero_crossing(r, derivative):
    """Linear interpolation of the first sign change of ``derivative``."""
    signs = np.sign(np.asarray(derivative))
    indices = np.where(np.diff(signs) != 0)[0]
    if indices.size == 0:
        raise ValueError("derivative never crosses zero on the sampled range")
    i = int(indices[0])
    r0, r1 = float(r[i]), float(r[i + 1])
    d0, d1 = float(derivative[i]), float(derivative[i + 1])
    return r0 - d0 * (r1 - r0) / (d1 - d0)


def finite_errors(values):
    """Return errors unchanged plus a log-safe copy with exact zeros masked."""
    errors = np.abs(np.asarray(values, dtype=np.float64))
    log_safe = np.where(errors > 0.0, errors, np.nan)
    return errors, log_safe


def main():
    r = jnp.linspace(R_MIN, R_MAX, N_SAMPLES, dtype=jnp.float64)
    assert r.shape == (N_SAMPLES,), r.shape
    assert float(r[0]) == R_MIN and float(r[-1]) == R_MAX

    analytic = analytical(r)
    forward = forward_derivative(r)
    reverse = reverse_derivative(r)
    difference = finite_difference(r)

    forward_error, forward_log = finite_errors(forward - analytic)
    reverse_error, reverse_log = finite_errors(reverse - analytic)
    difference_error, difference_log = finite_errors(difference - analytic)

    forward_max = float(forward_error.max())
    reverse_max = float(reverse_error.max())
    difference_max = float(difference_error.max())

    root = zero_crossing(r, analytic)
    root_error = abs(root - ZERO_CROSSING)

    print("Week 5 Part 1 VERIFY (1): LJ derivative comparison")
    print(f"  samples          : {N_SAMPLES} equally spaced in "
          f"[{R_MIN}, {R_MAX}]")
    print(f"  finite-difference: centered, h = {FD_STEP:g}")
    print("\nmaximum absolute error vs analytical derivative:")
    print(f"  forward-mode AD  : {forward_max:.6e}")
    print(f"  reverse-mode AD  : {reverse_max:.6e}")
    print(f"  finite difference: {difference_max:.6e}")
    print(f"\nanalytical zero crossing r0 = {root:.9f}  "
          f"(2^(1/6) = {ZERO_CROSSING:.9f}, |r0 - 2^(1/6)| = {root_error:.3e})")
    for name, log_series in (
        ("forward", forward_log),
        ("reverse", reverse_log),
        ("finite difference", difference_log),
    ):
        masked = int(np.isnan(log_series).sum())
        if masked:
            print(f"  note: {masked} exact zero error(s) masked in the "
                  f"{name} log panel (raw errors unchanged)")

    checks = [
        ("forward-mode max error < 1e-12", forward_max < AD_LIMIT),
        ("reverse-mode max error < 1e-12", reverse_max < AD_LIMIT),
        ("forward-mode max error < finite-difference max",
         forward_max < difference_max),
        ("reverse-mode max error < finite-difference max",
         reverse_max < difference_max),
        (f"finite-difference max error near {FD_REFERENCE:g}",
         abs(difference_max - FD_REFERENCE) <= FD_REFERENCE_RTOL * FD_REFERENCE),
        (f"analytical zero crossing near 2^(1/6) (tol {ZERO_TOL:g})",
         root_error < ZERO_TOL),
    ]

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 5.2),
                                      layout="constrained")

    # Left: the four derivative estimates.
    left.axhline(0.0, color="0.7", lw=0.8, zorder=0)
    left.plot(np.asarray(r), np.asarray(analytic), color="black", lw=2.6,
              label="Analytical", zorder=3)
    left.plot(np.asarray(r), np.asarray(forward), color="#1f77b4", lw=1.8,
              ls="--", label="Forward-mode AD", zorder=4)
    left.plot(np.asarray(r), np.asarray(reverse), color="#2ca02c", lw=1.8,
              ls=":", label="Reverse-mode AD", zorder=5)
    left.plot(np.asarray(r)[::20], np.asarray(difference)[::20], "o",
              color="#d62728", ms=6, mfc="none", mew=1.3,
              label="Centered finite difference ($h=10^{-6}$)")

    left.axvline(ZERO_CROSSING, color="0.5", lw=0.9, ls="-.")
    left.annotate(r"$r = 2^{1/6} = 1.12246$",
                  xy=(ZERO_CROSSING, 0.0), xytext=(1.5, -18.0),
                  arrowprops=dict(arrowstyle="->", color="0.4"),
                  color="0.25", fontsize=9)
    left.set_xlabel(r"Separation $r$")
    left.set_ylabel(r"$dU/dr$")
    left.set_title("Derivative estimates")
    left.legend(loc="upper left", frameon=False)
    left.set_xlim(R_MIN, R_MAX)

    # Right: absolute error, log scale.
    right.semilogy(np.asarray(r), forward_log, color="#1f77b4", lw=1.6,
                   label="Forward-mode AD")
    right.semilogy(np.asarray(r), reverse_log, color="#2ca02c", lw=1.6,
                   ls=":", label="Reverse-mode AD")
    right.semilogy(np.asarray(r), difference_log, color="#d62728", lw=1.6,
                   label="Centered finite difference")
    right.axhline(AD_LIMIT, color="0.6", lw=0.9, ls="--")
    right.annotate(r"$10^{-12}$", xy=(R_MAX, AD_LIMIT), xytext=(-2, 4),
                   textcoords="offset points", ha="right", color="0.4",
                   fontsize=9)
    right.set_xlabel(r"Separation $r$")
    right.set_ylabel(r"$|\,dU/dr - (dU/dr)_{\rm analytical}\,|$")
    right.set_title("Absolute error against the analytical derivative")
    right.legend(loc="lower right", frameon=False)
    right.set_xlim(R_MIN, R_MAX)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {OUT}")

    print("\nverification:")
    failed = False
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        failed = failed or not passed
    if failed:
        print("\nsome checks FAILED", file=sys.stderr)
        return 1
    print("\nall checks PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
