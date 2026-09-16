#!/usr/bin/env python3
"""Part 2 peak analysis, following the learning sheet.

For each lattice size:

1. evaluate ``chi(T) = L^2 (<M^2> - <|M|>^2) / T`` on the fine window grid;
2. take the temperature with the largest ``chi``;
3. fit a quadratic to the five points centred on it (two below, the maximum,
   two above);
4. report the vertex of that quadratic as ``T_peak``.

It then reports ``T_c = 2 T_peak(64) - T_peak(32)`` and the mean ``|M|`` at
the lowest temperature for both sizes.

Sampling errors (autocorrelation, jackknife) are deliberately *not* part of
this result; they belong to the Part 3 discussion.  That separate analysis
lives in ``week3/peak_analysis.py``.

Run with the course environment from ``week3/``::

    python scripts/peaks.py

Writes ``week3/evidence/peaks.txt``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

WEEK3 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WEEK3))

import plot_thermo as pt  # noqa: E402  (needs WEEK3 on sys.path)

EVIDENCE = WEEK3 / "evidence"


def susceptibility(temperatures, magnetizations, lattice, temperature):
    """chi(T) = L^2 (<M^2> - <|M|>^2) / T for one temperature."""
    values = magnetizations[temperatures == temperature]
    return (
        lattice ** 2
        * (np.mean(values ** 2) - np.mean(np.abs(values)) ** 2)
        / temperature
    )


def quadratic_peak(temperature, chi):
    """Fit a quadratic to the five points around the largest chi.

    Returns ``(T_peak, coefficients, (low, high))`` where ``coefficients`` is
    ``[a, b, c]`` of ``a T^2 + b T + c`` and ``(low, high)`` is the half-open
    slice of the five fitted points.  The vertex is ``-b / (2 a)``.
    """
    temperature = np.asarray(temperature, dtype=float)
    chi = np.asarray(chi, dtype=float)

    maximum = int(np.argmax(chi))
    low = maximum - 2
    high = maximum + 3
    if low < 0 or high > temperature.size:
        raise ValueError("need two temperature points on each side of the peak")

    coefficients = np.polyfit(temperature[low:high], chi[low:high], 2)
    a, b, _ = coefficients
    return float(-b / (2.0 * a)), coefficients, (low, high)


def analyse_window(run, lattice):
    """Grid, chi(T) and the five-point peak for one window ramp."""
    temperatures, magnetizations = pt.load_series(run)
    grid = np.unique(temperatures)
    chi = np.array(
        [
            susceptibility(temperatures, magnetizations, lattice, temperature)
            for temperature in grid
        ]
    )
    peak, coefficients, (low, high) = quadratic_peak(grid, chi)
    return {
        "grid": grid,
        "chi": chi,
        "peak": peak,
        "coefficients": coefficients,
        "fit_slice": (low, high),
        "maximum_index": int(np.argmax(chi)),
    }


def mean_abs_magnetization(run):
    """Mean |M| as a function of T for one ramp (temperature, value)."""
    temperatures, magnetizations = pt.load_series(run)
    grid = np.unique(temperatures)
    mean_abs = np.array(
        [np.mean(np.abs(magnetizations[temperatures == t])) for t in grid]
    )
    return grid, mean_abs


def format_report(results):
    """The learning-sheet numbers in a plain-text report."""
    l32 = results[32]
    l64 = results[64]
    tc = 2.0 * l64["peak"] - l32["peak"]

    lines = [
        "Part 2 peak analysis: five-point quadratic",
        "=" * 44,
        "Susceptibility chi(T) = L^2 (<M^2> - <|M|>^2) / T",
        "Window ramps 2.00 <= T <= 2.60, dT = 0.05, 100000 sweeps per T.",
        "",
    ]
    for lattice in (32, 64):
        data = results[lattice]
        low, high = data["fit_slice"]
        grid = data["grid"]
        i = data["maximum_index"]
        a, b, c = data["coefficients"]
        lines += [
            f"L = {lattice}",
            f"  grid maximum : chi = {data['chi'][i]:.4f} at T = {grid[i]:.2f}",
            f"  fit points   : "
            + ", ".join(f"{t:.2f}" for t in grid[low:high]),
            f"  quadratic    : chi = {a:.6f} T^2 + {b:.6f} T + {c:.6f}",
            f"  T_peak       : {data['peak']:.6f}",
            "",
        ]

    lines += [
        "Results",
        "-" * 44,
        f"  T_peak(L = 32) = {l32['peak']:.6f}",
        f"  T_peak(L = 64) = {l64['peak']:.6f}",
        f"  T_c = 2*T_peak(64) - T_peak(32) = {tc:.6f}",
        f"  (Onsager T_c = {pt.TC:.6f})",
        "",
        "Mean |M| at the lowest temperature",
        "-" * 44,
    ]
    grid32, m32 = mean_abs_magnetization("coarse-l32")
    grid64, m64 = mean_abs_magnetization("coarse-l64")
    lines.append(
        f"  T = {grid32[0]:.2f} (coarse runs): "
        f"L = 32  {m32[0]:.6f}   L = 64  {m64[0]:.6f}"
    )
    return "\n".join(lines)


def main():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    results = {
        32: analyse_window("window-l32", 32),
        64: analyse_window("window-l64", 64),
    }
    report = format_report(results)
    print(report)
    report_path = EVIDENCE / "peaks.txt"
    report_path.write_text(report + "\n")
    print(f"\nwrote {report_path}")


if __name__ == "__main__":
    main()
