#!/usr/bin/env python3
"""Part 2 peak analysis of the Ising susceptibility.

The Part 2 figures in :mod:`plot_thermo` show ``chi`` for ``L = 32`` and
``L = 64`` on the fine ``2.0 .. 2.6`` window ramps.  This script extracts the
peak of each curve:

* the temperature and height of the largest measured point, and
* a weighted parabola fit through the three points around that maximum,

    ``chi(T) = a + b T + c T^2``,   vertex ``T* = -b / 2c``,

which interpolates the peak to below the ``dT = 0.05`` grid step.  The fit
weights are the delete-one-block jackknife errors from
:func:`plot_thermo.susceptibility`.

Only the statistical errors of the measured points are propagated.  The peak
positions and heights are also affected by the finite ``dT`` grid and by
finite-size rounding, which the error bars do not cover; those are left for
Part 3.

The two sizes are then compared through the standard finite-size scaling
ansatz ``chi_max ~ L^{gamma/nu}`` and ``T*(L) = T_c + B L^{-1/nu}``.  With
only two sizes we fix ``nu = 1`` to turn the measured peak shift into an
estimate of ``T_c``.

Writes ``evidence/susceptibility_peak.png`` and
``evidence/peak-analysis.txt``.  Run with the course environment:

    cd week3 && python plot_thermo.py && python peak_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_thermo as pt

WEEK3 = Path(__file__).resolve().parent
EVIDENCE = WEEK3 / "evidence"

# Exact 2D Ising critical exponents, for reference only.
GAMMA_OVER_NU = 7.0 / 4.0


def parabola_vertex(temperature, chi, error):
    """Weighted parabola fit around the grid maximum.

    ``chi(T) = a + b T + c T^2`` is fit by weighted least squares with
    weights ``1 / error^2``.  Returns the vertex ``(T*, sigma_T, chi*,
    sigma_chi, coefficients, covariance)``.  The vertex errors come from the
    linear propagation of the fit covariance through the vertex formulas.
    """
    temperature = np.asarray(temperature, dtype=float)
    chi = np.asarray(chi, dtype=float)
    error = np.asarray(error, dtype=float)

    design = np.vstack(
        [np.ones_like(temperature), temperature, temperature ** 2]
    ).T
    weights = np.diag(1.0 / error ** 2)
    covariance = np.linalg.inv(design.T @ weights @ design)
    a, b, c = covariance @ (design.T @ weights @ chi)

    vertex_t = -b / (2.0 * c)
    vertex_chi = a - b * b / (4.0 * c)

    # d(vertex)/d(a, b, c)
    jac_t = np.array([0.0, -1.0 / (2.0 * c), b / (2.0 * c * c)])
    jac_chi = np.array([1.0, -b / (2.0 * c), b * b / (4.0 * c * c)])
    vertex_t_err = float(np.sqrt(jac_t @ covariance @ jac_t))
    vertex_chi_err = float(np.sqrt(jac_chi @ covariance @ jac_chi))

    return (
        float(vertex_t),
        vertex_t_err,
        float(vertex_chi),
        vertex_chi_err,
        np.array([a, b, c]),
        covariance,
    )


def local_peak(temperature, chi, error):
    """Grid peak plus a parabola fit through the neighbouring points.

    Returns a dictionary with both the raw grid maximum and the interpolated
    vertex, so the two can be compared in the report.
    """
    temperature = np.asarray(temperature, dtype=float)
    chi = np.asarray(chi, dtype=float)
    error = np.asarray(error, dtype=float)

    grid_index = int(np.argmax(chi))
    lo = max(0, grid_index - 1)
    hi = min(chi.size, grid_index + 2)

    vertex = parabola_vertex(
        temperature[lo:hi], chi[lo:hi], error[lo:hi]
    )
    return {
        "grid_index": grid_index,
        "grid_temperature": float(temperature[grid_index]),
        "grid_chi": float(chi[grid_index]),
        "fit_temperature": vertex[0],
        "fit_temperature_error": vertex[1],
        "fit_chi": vertex[2],
        "fit_chi_error": vertex[3],
        "coefficients": vertex[4],
        "fit_slice": (lo, hi),
    }


def measure_chi(run, lattice):
    """chi(T) and its jackknife error at every temperature of one window run."""
    temperatures, magnetizations = pt.load_series(run)
    grid = np.unique(temperatures)
    chi = np.empty(grid.size)
    error = np.empty(grid.size)
    for i, temperature in enumerate(grid):
        value, err, *_ = pt.susceptibility(
            magnetizations[temperatures == temperature], lattice, temperature
        )
        chi[i] = value
        error[i] = err
    return grid, chi, error


def analyse():
    """Run the peak analysis for both lattice sizes."""
    results = {}
    for run, lattice in (("window-l32", 32), ("window-l64", 64)):
        grid, chi, error = measure_chi(run, lattice)
        results[lattice] = {"grid": grid, "chi": chi, "error": error,
                            "peak": local_peak(grid, chi, error)}
    return results


def finite_size_summary(results):
    """Finite-size scaling numbers derived from the two fitted peaks."""
    low = results[32]["peak"]
    high = results[64]["peak"]

    ratio = high["fit_chi"] / low["fit_chi"]
    ratio_error = ratio * np.sqrt(
        (high["fit_chi_error"] / high["fit_chi"]) ** 2
        + (low["fit_chi_error"] / low["fit_chi"]) ** 2
    )
    exponent = np.log(ratio) / np.log(2.0)
    exponent_error = ratio_error / (ratio * np.log(2.0))

    shift = low["fit_temperature"] - high["fit_temperature"]
    shift_error = np.sqrt(
        low["fit_temperature_error"] ** 2
        + high["fit_temperature_error"] ** 2
    )
    # With nu = 1 and T*(L) = Tc + B/L, extrapolating the two points gives
    # Tc = 2 T*(64) - T*(32).
    tc = 2.0 * high["fit_temperature"] - low["fit_temperature"]
    tc_error = np.sqrt(
        (2.0 * high["fit_temperature_error"]) ** 2
        + low["fit_temperature_error"] ** 2
    )
    return {
        "ratio": ratio,
        "ratio_error": ratio_error,
        "exponent": exponent,
        "exponent_error": exponent_error,
        "shift": shift,
        "shift_error": shift_error,
        "tc": tc,
        "tc_error": tc_error,
    }


def format_report(results, summary):
    """Human-readable report, also written next to the figure."""
    lines = [
        "Part 2 peak analysis of the Ising susceptibility",
        "=" * 47,
        "Window ramps 2.0 <= T <= 2.6, dT = 0.05, 100000 sweeps per T.",
        "Peak = weighted parabola through the three points around the",
        "largest measured chi; errors are jackknife statistical errors.",
        "",
    ]
    for lattice in (32, 64):
        peak = results[lattice]["peak"]
        lo, hi = peak["fit_slice"]
        grid = results[lattice]["grid"]
        lines += [
            f"L = {lattice}",
            f"  grid peak    chi = {peak['grid_chi']:.2f} "
            f"at T = {peak['grid_temperature']:.2f}",
            f"  parabola     chi* = {peak['fit_chi']:.2f} "
            f"+- {peak['fit_chi_error']:.2f} "
            f"at T* = {peak['fit_temperature']:.4f} "
            f"+- {peak['fit_temperature_error']:.4f}",
            f"  fit uses     T = "
            + ", ".join(f"{t:.2f}" for t in grid[lo:hi]),
            "",
        ]

    lines += [
        "Finite-size scaling (two sizes, nu fixed to 1 for Tc)",
        "-" * 47,
        f"  chi*(64)/chi*(32) = {summary['ratio']:.3f} "
        f"+- {summary['ratio_error']:.3f}",
        f"  implied gamma/nu  = {summary['exponent']:.3f} "
        f"+- {summary['exponent_error']:.3f}   "
        f"(exact 7/4 = {GAMMA_OVER_NU:.3f})",
        f"  T*(32) - T*(64)   = {summary['shift']:.4f} "
        f"+- {summary['shift_error']:.4f}",
        f"  T_c (nu = 1)      = {summary['tc']:.4f} "
        f"+- {summary['tc_error']:.4f}",
        f"  Onsager T_c       = {pt.TC:.4f}",
        "",
        "Statistical errors only; the dT = 0.05 grid and finite-size",
        "rounding are systematic and are not included.",
    ]
    return "\n".join(lines)


def plot_peaks(results, summary):
    """Zoom on the critical region with the two parabola fits."""
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    for lattice, color in ((32, "C0"), (64, "C1")):
        grid = results[lattice]["grid"]
        chi = results[lattice]["chi"]
        error = results[lattice]["error"]
        peak = results[lattice]["peak"]

        ax.errorbar(
            grid,
            chi,
            yerr=error,
            fmt="o",
            markersize=4,
            color=color,
            capsize=2,
            elinewidth=0.8,
            label=rf"$L = {lattice}$ data",
        )

        lo, hi = peak["fit_slice"]
        a, b, c = peak["coefficients"]
        t_fit = np.linspace(grid[lo] - 0.02, grid[hi - 1] + 0.02, 100)
        ax.plot(
            t_fit,
            a + b * t_fit + c * t_fit ** 2,
            color=color,
            linestyle="--",
            linewidth=1.2,
            label=rf"$L = {lattice}$ parabola fit",
        )
        ax.plot(
            peak["fit_temperature"],
            peak["fit_chi"],
            marker="*",
            markersize=12,
            color=color,
            markeredgecolor="black",
            markeredgewidth=0.5,
        )

    ax.axvline(pt.TC, color="gray", linestyle=":", linewidth=1.0)
    ax.text(
        pt.TC + 0.002,
        0.95,
        rf"$T_c = {pt.TC:.4f}$",
        transform=ax.get_xaxis_transform(),
        fontsize=8,
        va="top",
    )
    ax.annotate(
        rf"$T_c$ estimate $= {summary['tc']:.4f}$ (peak shift, $\nu=1$)",
        xy=(0.03, 0.05),
        xycoords="axes fraction",
        fontsize=8,
    )
    ax.set_xlabel("temperature $T$")
    ax.set_ylabel(r"susceptibility $\chi$")
    ax.set_title(r"Susceptibility peak near $T_c$")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    results = analyse()
    summary = finite_size_summary(results)

    report = format_report(results, summary)
    print(report)
    report_path = EVIDENCE / "peak-analysis.txt"
    report_path.write_text(report + "\n")
    print(f"\nwrote {report_path}")

    fig = plot_peaks(results, summary)
    figure_path = EVIDENCE / "susceptibility_peak.png"
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    print(f"wrote {figure_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
