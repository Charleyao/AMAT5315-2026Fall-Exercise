#!/usr/bin/env python3
"""Part 3: block bootstrap of the five-point T_c estimate.

Uses only the high-statistics ``window-l32`` / ``window-l64`` Metropolis runs
already stored in ``week3/artifacts/`` (no resimulation).  For each lattice
size and each temperature in the critical window the measured ``|M|`` series
is split into consecutive blocks, and blocks are resampled with replacement
for 500 bootstrap replicates.

For one replicate the full procedure of Part 2 is repeated:

1. recompute ``chi(T) = L^2 (<M^2> - <|M|>^2) / T`` from the resampled data;
2. take the largest ``chi`` point for each lattice size;
3. fit a quadratic to the five temperatures centred on it;
4. use its vertex as ``T_peak``, then ``T_c = 2 T_peak(64) - T_peak(32)``.

A fit fails if its parabola does not bend downward (``a >= 0``) or if the
vertex lies outside the five fitted temperatures.  The three block lengths
2000, 4000 and 8000 sweeps are compared; the sampling error is called stable
only if their bootstrap standard deviations agree within one tenth of their
mean.

Run with the course environment from ``week3/``::

    /home/yao_yiyi/.venvs/amat5315/bin/python scripts/bootstrap.py

Writes ``week3/evidence/chi-bootstrap.png`` and
``week3/evidence/chi-bootstrap.txt``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WEEK3 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WEEK3))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import plot_thermo as pt  # noqa: E402  (needs WEEK3 on sys.path)
from errors import series_by_temperature  # noqa: E402

EVIDENCE = WEEK3 / "evidence"

LATTICES = (32, 64)
BLOCK_LENGTHS = (2000, 4000, 8000)
N_BOOT = 500
SEED = 2026
STABILITY_TOLERANCE = 0.1  # stds must agree within one tenth of their mean
ENVELOPE_BLOCK_LENGTH = 4000  # block length whose 500 fits are shaded
ONSAGER_TC = 2.26919  # exact critical temperature, 2 / ln(1 + sqrt(2))


def bootstrap_chi(block_m2, block_abs, lattice, temperature, n_boot, rng):
    """500 bootstrap ``chi`` values from resampled consecutive blocks."""
    n_blocks = block_m2.size
    indices = rng.integers(0, n_blocks, size=(n_boot, n_blocks))
    mean_m2 = block_m2[indices].mean(axis=1)
    mean_abs = block_abs[indices].mean(axis=1)
    return lattice ** 2 * (mean_m2 - mean_abs ** 2) / temperature


def block_means(series, block_length):
    """Per-block means of ``M^2`` and ``|M|`` for equal consecutive blocks."""
    m = np.asarray(series, dtype=float)
    n_blocks = m.size // block_length
    if n_blocks < 2:
        raise ValueError(
            f"need at least 2 blocks of length {block_length} (got {n_blocks})"
        )
    trimmed = m[: n_blocks * block_length]
    m2 = trimmed ** 2
    abs_m = np.abs(trimmed)
    return (
        m2.reshape(n_blocks, block_length).mean(axis=1),
        abs_m.reshape(n_blocks, block_length).mean(axis=1),
    )


def susceptibility(series, lattice, temperature):
    """chi(T) on the full (unresampled) series."""
    m = np.asarray(series, dtype=float)
    return lattice ** 2 * (np.mean(m ** 2) - np.mean(np.abs(m)) ** 2) / temperature


def fit_five_point(temperatures, chi):
    """Quadratic through the five temperatures centred on max(chi).

    Returns ``None`` when the maximum is too close to an edge or when the
    fitted parabola does not bend downward or peaks outside the five points.
    Otherwise returns a dict with the vertex and coefficients.
    """
    temperatures = np.asarray(temperatures, dtype=float)
    chi = np.asarray(chi, dtype=float)
    order = int(np.argmax(chi))
    if order < 2 or order + 3 > temperatures.size:
        return None

    low, high = order - 2, order + 3
    t5 = temperatures[low:high]
    coefficients = np.polyfit(t5, chi[low:high], 2)
    a, b, _ = coefficients
    if a >= 0.0:
        return None
    vertex = -b / (2.0 * a)
    if vertex < t5[0] or vertex > t5[-1]:
        return None
    return {
        "vertex": float(vertex),
        "coefficients": coefficients,
        "t_low": float(t5[0]),
        "t_high": float(t5[-1]),
        "index": order,
    }


def sampling_error_stable(stds, tolerance=STABILITY_TOLERANCE):
    """True if all standard deviations are within ``tolerance`` of their mean."""
    stds = np.asarray(stds, dtype=float)
    mean = float(stds.mean())
    return bool(mean > 0.0 and np.all(np.abs(stds - mean) <= tolerance * mean))


def fit_bootstrap(bootstrap_chi_by_temperature, temperatures):
    """Fit one replicate per bootstrap column; returns fits for both lattices."""
    fits = {lattice: [] for lattice in LATTICES}
    for lattice in LATTICES:
        chi_table = bootstrap_chi_by_temperature[lattice]  # (n_temps, n_boot)
        for replicate in range(chi_table.shape[1]):
            fits[lattice].append(fit_five_point(temperatures, chi_table[:, replicate]))
    return fits


def run_block_length(series, temperatures, block_length, rng):
    """Run the whole bootstrap for one block length."""
    bootstrap_chi_by_temperature = {}
    for lattice in LATTICES:
        columns = [
            bootstrap_chi(
                *block_means(series[lattice][temperature], block_length),
                lattice,
                temperature,
                N_BOOT,
                rng,
            )
            for temperature in temperatures
        ]
        bootstrap_chi_by_temperature[lattice] = np.array(columns)  # (n_temps, N_BOOT)

    fits = fit_bootstrap(bootstrap_chi_by_temperature, temperatures)
    failures = {
        lattice: sum(fit is None for fit in fits[lattice]) for lattice in LATTICES
    }
    valid = [
        replicate
        for replicate in range(N_BOOT)
        if fits[32][replicate] is not None and fits[64][replicate] is not None
    ]
    tc = np.array(
        [
            2.0 * fits[64][replicate]["vertex"] - fits[32][replicate]["vertex"]
            for replicate in valid
        ]
    )
    return {
        "block_length": block_length,
        "fits": fits,
        "failures": failures,
        "valid": len(valid),
        "mean_tc": float(tc.mean()) if tc.size else float("nan"),
        "std_tc": float(tc.std(ddof=1)) if tc.size > 1 else float("nan"),
        "tc": tc,
    }


def full_data_results(series, temperatures):
    """Full-data chi, five-point fit and T_c, for reference and plotting."""
    chi = {
        lattice: np.array(
            [susceptibility(series[lattice][t], lattice, t) for t in temperatures]
        )
        for lattice in LATTICES
    }
    fits = {lattice: fit_five_point(temperatures, chi[lattice]) for lattice in LATTICES}
    tc = 2.0 * fits[64]["vertex"] - fits[32]["vertex"]
    return chi, fits, tc


def parabola_envelope(fits, grid):
    """Pointwise min/max of a list of fitted parabolas over ``grid``."""
    lower = np.full(grid.size, np.nan)
    upper = np.full(grid.size, np.nan)
    for fit in fits:
        if fit is None:
            continue
        mask = (grid >= fit["t_low"]) & (grid <= fit["t_high"])
        if not mask.any():
            continue
        values = np.polyval(fit["coefficients"], grid[mask])
        lower[mask] = np.fmin(lower[mask], values)
        upper[mask] = np.fmax(upper[mask], values)
    return lower, upper


def make_figure(series, temperatures, chi_full, fits_full, chi_error, results):
    """Two-panel figure: chi(T) with fits and the bootstrap envelope."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), constrained_layout=True)

    for ax, lattice in zip(axes, LATTICES):
        chi = chi_full[lattice]
        error = chi_error[lattice]
        fit = fits_full[lattice]

        ax.errorbar(
            temperatures,
            chi,
            yerr=error,
            fmt="o",
            color="C0",
            markersize=4,
            capsize=2,
            elinewidth=0.8,
            label="data",
        )

        t_fit = np.linspace(fit["t_low"], fit["t_high"], 100)
        ax.plot(
            t_fit,
            np.polyval(fit["coefficients"], t_fit),
            color="C3",
            linewidth=1.6,
            label=rf"five-point fit, $T_{{\rm peak}} = {fit['vertex']:.4f}$",
        )

        envelope_fits = results[ENVELOPE_BLOCK_LENGTH]["fits"][lattice]
        grid = np.linspace(temperatures[0], temperatures[-1], 500)
        lower, upper = parabola_envelope(envelope_fits, grid)
        ax.fill_between(
            grid,
            lower,
            upper,
            color="C0",
            alpha=0.25,
            label=(
                f"bootstrap envelope "
                f"({len(envelope_fits)} fits, block {ENVELOPE_BLOCK_LENGTH})"
            ),
        )

        ax.axvline(
            ONSAGER_TC,
            color="gray",
            linestyle=":",
            linewidth=1.2,
            label=rf"Onsager $T_c = {ONSAGER_TC}$",
        )
        ax.axvline(fit["vertex"], color="C3", linestyle="--", linewidth=0.9)
        ax.set_xlim(temperatures[0] - 0.02, temperatures[-1] + 0.02)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("temperature $T$")
        ax.set_ylabel(r"susceptibility $\chi$")
        ax.set_title(rf"$L = {lattice}$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        r"Susceptibility near $T_c$ with five-point fits and bootstrap envelope"
    )
    return fig


def format_report(temperatures, chi_full, fits_full, tc_full, results):
    """Plain-text table of bootstrap T_c and the stability verdict."""
    lines = [
        "Part 3 block bootstrap of the five-point T_c estimate",
        "=" * 58,
        "Window runs only; chi(T) = L^2 (<M^2> - <|M|>^2) / T",
        f"{N_BOOT} replicates per block length, seed = {SEED}",
        "",
        "Full-data reference (no bootstrap)",
        "-" * 58,
        f"  T_peak(L = 32) = {fits_full[32]['vertex']:.6f}",
        f"  T_peak(L = 64) = {fits_full[64]['vertex']:.6f}",
        f"  T_c            = {tc_full:.6f}",
        "",
        "Bootstrap results",
        "-" * 58,
        f"{'block':>7s}  {'mean T_c':>10s}  {'std T_c':>10s}  "
        f"{'fail L=32':>9s}  {'fail L=64':>9s}  {'fail tot':>8s}  {'used':>5s}",
    ]
    for block_length in BLOCK_LENGTHS:
        result = results[block_length]
        failures = result["failures"]
        lines.append(
            f"{block_length:>7d}  {result['mean_tc']:>10.6f}  "
            f"{result['std_tc']:>10.6f}  {failures[32]:>9d}  "
            f"{failures[64]:>9d}  {failures[32] + failures[64]:>8d}  "
            f"{result['valid']:>5d}"
        )

    stds = [results[b]["std_tc"] for b in BLOCK_LENGTHS]
    stable = sampling_error_stable(stds)
    mean_std = float(np.mean(stds))
    lines += [
        "",
        "Stability of the sampling error",
        "-" * 58,
        "  std T_c: "
        + ", ".join(
            f"{b}: {results[b]['std_tc']:.6f}" for b in BLOCK_LENGTHS
        ),
        f"  mean std      = {mean_std:.6f}",
        f"  spread from mean = "
        f"{max(abs(s - mean_std) for s in stds) / mean_std:.1%}",
        (
            "  verdict: STABLE (all within 10% of their mean)"
            if stable
            else "  verdict: UNRESOLVED (block lengths disagree by more than 10%)"
        ),
    ]
    return "\n".join(lines)


def main():
    EVIDENCE.mkdir(parents=True, exist_ok=True)

    # Load the window runs once and index them by temperature and lattice.
    series = {}
    for lattice in LATTICES:
        by_temperature = series_by_temperature(f"window-l{lattice}")
        series[lattice] = by_temperature
    temperatures = np.array(sorted(series[32]), dtype=float)

    chi_full, fits_full, tc_full = full_data_results(series, temperatures)
    chi_error = {
        lattice: np.array(
            [
                pt.susceptibility(series[lattice][t], lattice, t)[1]
                for t in temperatures
            ]
        )
        for lattice in LATTICES
    }

    rng = np.random.default_rng(SEED)
    results = {
        block_length: run_block_length(series, temperatures, block_length, rng)
        for block_length in BLOCK_LENGTHS
    }

    report = format_report(temperatures, chi_full, fits_full, tc_full, results)
    print(report)
    (EVIDENCE / "chi-bootstrap.txt").write_text(report + "\n")

    fig = make_figure(series, temperatures, chi_full, fits_full, chi_error, results)
    path = EVIDENCE / "chi-bootstrap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {path}")
    print(f"wrote {EVIDENCE / 'chi-bootstrap.txt'}")


if __name__ == "__main__":
    main()
