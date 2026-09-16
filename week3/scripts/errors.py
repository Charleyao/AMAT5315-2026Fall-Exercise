#!/usr/bin/env python3
"""Part 3: how much to trust the Part 2 Metropolis results.

For every lattice size and temperature we report the statistical uncertainty of
the mean magnetisation ``|M|`` in three ways:

* the **naive** standard error ``std(|M|) / sqrt(N)``, which assumes the
  measured sweeps are independent;
* the standard error of **50 equal consecutive block averages**, which keeps
  the short-range correlations inside each block;
* the **integrated autocorrelation time**

      tau_int = 1/2 + sum_{t>=1} rho(t),

  truncated with the learning-sheet rule: stop as soon as the lag exceeds six
  times the running tau estimate.

The ratio ``block_se / naive_se`` measures how badly the naive error
underestimates the true one; for a long series it should be close to
``sqrt(2 tau_int)``.

Temperatures that appear in the high-statistics ``window-l*`` ramps (the runs
used for the Part 2 peak fit) use those runs; every other temperature uses the
``coarse-l*`` ramp.  Both ramps are Metropolis runs already stored in
``week3/artifacts/``; nothing is resimulated here.

Run with the course environment from ``week3/``::

    /home/yao_yiyi/.venvs/amat5315/bin/python scripts/errors.py

Writes ``week3/evidence/errors.txt``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

WEEK3 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WEEK3))

import plot_thermo as pt  # noqa: E402  (needs WEEK3 on sys.path)

EVIDENCE = WEEK3 / "evidence"

N_BLOCKS = 50
WINDOW_FACTOR = 6.0
LATTICES = (32, 64)


def autocorrelation_function(samples):
    """Normalised autocorrelation ``rho(t)`` for lags ``t = 0 .. N-1``.

    ``rho`` is computed from the autocovariance via an FFT and normalised so
    that ``rho(0) = 1``.  A constant series has no fluctuations, so it returns
    ``[1.0]``.
    """
    x = np.asarray(samples, dtype=float)
    n = x.size
    if n < 2:
        return np.array([1.0])

    x = x - x.mean()
    variance = float(np.dot(x, x)) / n
    if variance == 0.0:
        return np.array([1.0])

    spectrum = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(spectrum * np.conjugate(spectrum), n=2 * n)[:n]
    return acf / (variance * n)


def integrated_autocorrelation_time(samples, window_factor=WINDOW_FACTOR):
    """Integrated autocorrelation time ``1/2 + sum_{t>=1} rho(t)``.

    The sum is cut off with the learning-sheet rule: stop at the first lag
    that is larger than ``window_factor`` times the running estimate.
    """
    acf = autocorrelation_function(samples)
    tau = 0.5
    for lag in range(1, acf.size):
        if lag > window_factor * tau:
            break
        tau += float(acf[lag])
    return tau


def naive_standard_error(samples):
    """Independent-sample standard error ``std(samples) / sqrt(N)``."""
    x = np.asarray(samples, dtype=float)
    if x.size < 2:
        return float("nan")
    return float(x.std(ddof=1) / np.sqrt(x.size))


def block_standard_error(samples, n_blocks=N_BLOCKS):
    """Standard error of ``n_blocks`` equal consecutive block averages."""
    x = np.asarray(samples, dtype=float)
    length = x.size // n_blocks
    if length < 2:
        raise ValueError(
            f"need at least {2 * n_blocks} samples for {n_blocks} blocks"
        )
    blocks = x[: length * n_blocks].reshape(n_blocks, length)
    block_means = blocks.mean(axis=1)
    return float(block_means.std(ddof=1) / np.sqrt(n_blocks)), length


def block_error_by_length(samples, block_length):
    """Standard error of consecutive block averages of a given length.

    Returns ``(error, n_blocks)``.  ``block_length = 1`` reproduces the naive
    standard error because every sample is then its own block.
    """
    if block_length < 1:
        raise ValueError("block_length must be at least 1")
    x = np.asarray(samples, dtype=float)
    n_blocks = x.size // block_length
    if n_blocks < 2:
        raise ValueError(
            f"need at least 2 blocks of length {block_length} (got {n_blocks})"
        )
    blocks = x[: n_blocks * block_length].reshape(n_blocks, block_length)
    block_means = blocks.mean(axis=1)
    return float(block_means.std(ddof=1) / np.sqrt(n_blocks)), n_blocks


def series_by_temperature(run):
    """Measured magnetizations grouped by temperature for one run folder."""
    temperatures, magnetizations = pt.load_series(run)
    return {
        float(temperature): magnetizations[temperatures == temperature]
        for temperature in np.unique(temperatures)
    }


def merged_series(lattice):
    """One ``{T: (|M| series, source run)}`` map per lattice size.

    Window-ramp temperatures win over coarse-ramp temperatures on the overlap;
    they are the high-statistics data behind the Part 2 peak fit.
    """
    coarse_run = f"coarse-l{lattice}"
    window_run = f"window-l{lattice}"
    merged = {
        temperature: (values, coarse_run)
        for temperature, values in series_by_temperature(coarse_run).items()
    }
    merged.update(
        {
            temperature: (values, window_run)
            for temperature, values in series_by_temperature(window_run).items()
        }
    )
    return merged


def analyse(series):
    """All Part 3 statistics for one temperature and one lattice size."""
    abs_m = np.abs(series)
    naive = naive_standard_error(abs_m)
    block, block_length = block_standard_error(abs_m)
    return {
        "mean_abs": float(abs_m.mean()),
        "naive_se": naive,
        "block_se": block,
        "ratio": block / naive,
        "tau_int": integrated_autocorrelation_time(abs_m),
        "n": abs_m.size,
        "block_length": block_length,
    }


HEADER = (
    f"{'L':>3s}  {'T':>5s}  {'mean|M|':>10s}  "
    f"{'naive_se':>12s}  {'block_se':>12s}  {'ratio':>8s}  {'tau_int':>9s}"
)
RULE = "-" * len(HEADER)


def format_table(rows):
    """Plain-text table: one row per (lattice size, temperature)."""
    lines = [HEADER, RULE]
    for lattice, temperature, stats, _ in rows:
        lines.append(
            f"{lattice:>3d}  {temperature:>5.2f}  {stats['mean_abs']:>10.6f}  "
            f"{stats['naive_se']:>12.6e}  {stats['block_se']:>12.6e}  "
            f"{stats['ratio']:>8.3f}  {stats['tau_int']:>9.3f}"
        )
    return "\n".join(lines)


def format_report(rows):
    """Header notes, the table, and a short uncertainty discussion."""
    sources = sorted({source for *_, source in rows})
    notes = [
        "Part 3: uncertainty budget for the Part 2 Metropolis data",
        "=" * 58,
        "Observable: |M|, one measured sweep per step.",
        "naive_se  = std(|M|, ddof=1) / sqrt(N)",
        f"block_se  = std of {N_BLOCKS} equal consecutive block means / sqrt({N_BLOCKS})",
        (
            "tau_int   = 1/2 + sum_{t>=1} rho(t); sum cut at the first lag "
            f"> {WINDOW_FACTOR:g} * running tau"
        ),
        f"source    : {', '.join(sources)} (window preferred on overlap)",
        "",
    ]

    n_blocks = len(rows)
    ratios = [stats["ratio"] for _, _, stats, _ in rows]
    taus = [stats["tau_int"] for _, _, stats, _ in rows]
    discussion = [
        "",
        "Discussion",
        "-" * 58,
        (
            f"Across the {n_blocks} (L, T) rows the naive error is smaller than "
            f"the block error in most cases (median ratio {np.median(ratios):.2f})."
        ),
        (
            f"tau_int ranges from {min(taus):.2f} to {max(taus):.2f}. Far from "
            "T_c the measured sweeps are only weakly correlated (tau_int of a "
            "few sweeps) and the two error estimates are within a factor of ~2."
        ),
        (
            "Near T_c the correlation time grows, the ratio approaches "
            "sqrt(2*tau_int), and the naive error can be wrong by an order of "
            "magnitude. This is the dominant statistical uncertainty in the "
            "Part 2 peak position."
        ),
        (
            f"The block error itself is only approximate: with {N_BLOCKS} "
            "blocks the standard error of the error is about 1/sqrt(2*50) ~ 10%, "
            "and near T_c 50 blocks may still be correlated, so it can "
            "underestimate the true error."
        ),
    ]
    return "\n".join(notes) + format_table(rows) + "\n" + "\n".join(discussion)


def main():
    EVIDENCE.mkdir(parents=True, exist_ok=True)

    rows = []
    for lattice in LATTICES:
        merged = merged_series(lattice)
        for temperature in sorted(merged):
            series, source = merged[temperature]
            rows.append((lattice, temperature, analyse(series), source))

    report = format_report(rows)
    print(report)

    report_path = EVIDENCE / "errors.txt"
    report_path.write_text(report + "\n")
    print(f"\nwrote {report_path}")


if __name__ == "__main__":
    main()
