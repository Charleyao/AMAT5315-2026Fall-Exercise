#!/usr/bin/env python3
"""Magnetization and susceptibility from the week 3 Metropolis runs.

The ``artifacts/`` folder holds two kinds of temperature ramp:

``coarse-l32`` / ``coarse-l64``
    A wide ramp, ``T = 1.5 .. 3.5`` in steps of 0.1, 5000 measured sweeps per
    temperature.  It covers the ordered and disordered phases but is noisy
    near ``T_c`` where the Metropolis autocorrelation time is large.
``window-l32`` / ``window-l64``
    A fine ramp, ``T = 2.0 .. 2.6`` in steps of 0.05, 100000 measured sweeps
    per temperature.  It resolves the critical region where the
    susceptibility peaks.

This script draws two figures:

``evidence/magnetization.png``
    A single mean ``|M|`` curve for ``L = 64``, spliced from the two ramps:
    the coarse ramp outside the critical window and the window ramp inside
    ``2.0 <= T <= 2.6`` (its points replace the coarse ones on the overlap).
    Onsager's exact spontaneous magnetization of the *infinite* lattice is
    drawn alongside it,

        M(T) = (1 - sinh(2/T)^-4)^(1/8)   for T < T_c,  0 otherwise.

``evidence/susceptibility.png``
    ``chi(T) = L^2 (<M^2> - <|M|>^2) / T`` against temperature for ``L = 32``
    and ``L = 64`` from the high-statistics window ramps.

Uncertainties
-------------
Successive Metropolis sweeps are correlated, so a naive ``std/sqrt(N)``
underestimates the error.  We estimate the integrated autocorrelation time
``tau`` with the usual Sokal window [1]_ and use it in two ways:

* mean ``|M|``: ``sem = std(|M|) * sqrt(2 tau / N)``;
* ``chi``: a nonlinear function of ``<M^2>`` and ``<|M|>``, so we use a
  delete-one-block jackknife with blocks several autocorrelation times long.

[1] N. Madras and A. D. Sokal, J. Stat. Phys. 50, 109 (1988).

Exact command used to generate the figures
------------------------------------------
    cd /home/yao_yiyi/AMAT5315-2026Fall-Exercise/week3 && \
        /home/yao_yiyi/.venvs/amat5315/bin/python plot_thermo.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WEEK3 = Path(__file__).resolve().parent
ARTIFACTS = WEEK3 / "artifacts"
EVIDENCE = WEEK3 / "evidence"

# Onsager's critical temperature for the square lattice, 2 / ln(1 + sqrt(2)).
TC = 2.0 / np.log(1.0 + np.sqrt(2.0))


def onsager_magnetization(temperature):
    """Exact spontaneous magnetization of the infinite 2D Ising model."""
    t = np.atleast_1d(np.asarray(temperature, dtype=float))
    result = np.zeros_like(t)
    ordered = t < TC
    sinh = np.sinh(2.0 / t[ordered])
    result[ordered] = (1.0 - sinh ** -4.0) ** 0.125
    return result


def load_series(run):
    """Read one run's series.jsonl and return the temperature and spin arrays."""
    temperatures = []
    magnetizations = []
    with (ARTIFACTS / run / "series.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            temperatures.append(row["T"])
            magnetizations.append(row["M"])
    return np.asarray(temperatures), np.asarray(magnetizations)


def integrated_autocorr_time(samples):
    """Integrated autocorrelation time via the Sokal-window estimator."""
    x = np.asarray(samples, dtype=float)
    n = x.size
    if n < 2:
        return 1.0
    x = x - x.mean()
    variance = float(np.dot(x, x)) / n
    if variance == 0.0:
        return 1.0
    # Autocorrelation through an FFT: cheaper than a direct sum for 1e5 points.
    spectrum = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(spectrum * np.conjugate(spectrum), n=2 * n)[:n]
    acf /= variance * n
    total = 0.0
    for lag in range(1, n):
        total += acf[lag]
        if lag >= 5.0 * (1.0 + 2.0 * total):
            break
    return 1.0 + 2.0 * total


def mean_abs_magnetization(magnetization):
    """Mean |M| with an autocorrelation-corrected standard error."""
    abs_m = np.abs(magnetization)
    tau = integrated_autocorr_time(abs_m)
    sem = float(abs_m.std(ddof=1)) * np.sqrt(2.0 * tau / abs_m.size)
    return float(abs_m.mean()), sem, tau


def block_length(magnetization):
    """Block length a few autocorrelation times long, at least six blocks.

    ``chi`` is dominated by ``<M^2>``, so the longest of the ``|M|`` and
    ``M^2`` autocorrelation times sets the block length.
    """
    tau = max(
        integrated_autocorr_time(np.abs(magnetization)),
        integrated_autocorr_time(magnetization ** 2),
    )
    length = max(4.0 * tau, 20.0)
    return int(min(length, magnetization.size / 6.0)), tau


def susceptibility(magnetization, lattice, temperature):
    """chi and its jackknife error from delete-one-block resampling."""
    length, tau = block_length(magnetization)
    n = (magnetization.size // length) * length
    blocks = magnetization[:n].reshape(-1, length)
    n_blocks = blocks.shape[0]

    def chi_from_sums(square_sum, abs_sum, count):
        mean_square = square_sum / count
        mean_abs = abs_sum / count
        return lattice ** 2 * (mean_square - mean_abs ** 2) / temperature

    square_per_block = (blocks.astype(float) ** 2).sum(axis=1)
    abs_per_block = np.abs(blocks).sum(axis=1)
    chi = chi_from_sums(square_per_block.sum(), abs_per_block.sum(), n)

    leave_one_out = np.empty(n_blocks)
    for b in range(n_blocks):
        leave_one_out[b] = chi_from_sums(
            square_per_block.sum() - square_per_block[b],
            abs_per_block.sum() - abs_per_block[b],
            n - length,
        )
    variance = (n_blocks - 1.0) / n_blocks * np.sum(
        (leave_one_out - leave_one_out.mean()) ** 2
    )
    return chi, float(np.sqrt(variance)), length, n_blocks, tau


def spliced_l64():
    """One L = 64 magnetization curve from the coarse and window ramps.

    The coarse ramp supplies the full range, but the window ramp replaces it
    on ``2.0 <= T <= 2.6`` where its many more sweeps resolve the transition.
    The two grids overlap at the window edges, so those temperatures come
    only from the window run.
    """
    pieces = []
    for run, use in (
        ("coarse-l64", lambda t: (t < 2.0) | (t > 2.6)),
        ("window-l64", lambda t: (t >= 2.0) & (t <= 2.6)),
    ):
        temperatures, magnetizations = load_series(run)
        for temperature in np.unique(temperatures):
            if not use(temperature):
                continue
            value, error, _ = mean_abs_magnetization(
                magnetizations[temperatures == temperature]
            )
            pieces.append((temperature, value, error))
    pieces.sort()
    return (
        np.array([p[0] for p in pieces]),
        np.array([p[1] for p in pieces]),
        np.array([p[2] for p in pieces]),
    )


def plot_magnetization(ax):
    """Spliced mean |M| for L = 64, plus Onsager's infinite-lattice curve."""
    t_line = np.linspace(1.4, 3.6, 400)
    ax.plot(
        t_line,
        onsager_magnetization(t_line),
        color="black",
        linewidth=1.5,
        label="Onsager, infinite lattice",
    )

    temperatures, mean_abs, sem_abs = spliced_l64()
    ax.errorbar(
        temperatures,
        mean_abs,
        yerr=sem_abs,
        fmt="o",
        markersize=4,
        color="C0",
        capsize=2,
        elinewidth=0.8,
        label="Metropolis ramp, L = 64",
    )

    ax.axvline(TC, color="gray", linestyle=":", linewidth=1.0)
    ax.text(TC + 0.02, 1.01, rf"$T_c = {TC:.4f}$", fontsize=8, va="top")
    ax.set_xlim(1.4, 3.6)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("temperature $T$")
    ax.set_ylabel(r"mean $|M|$")
    ax.set_title(r"Magnetization of the 2D Ising model, $L = 64$")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)


def plot_susceptibility(ax):
    """chi for L = 32 and L = 64 from the critical-window ramps."""
    for run, lattice, color in (
        ("window-l32", 32, "C0"),
        ("window-l64", 64, "C1"),
    ):
        temperatures, magnetizations = load_series(run)
        chi, error = [], []
        for temperature in np.unique(temperatures):
            value, err, *_ = susceptibility(
                magnetizations[temperatures == temperature], lattice, temperature
            )
            chi.append(value)
            error.append(err)
        ax.errorbar(
            np.unique(temperatures),
            chi,
            yerr=error,
            fmt="o",
            markersize=4,
            color=color,
            capsize=2,
            elinewidth=0.8,
            label=rf"$L = {lattice}$",
        )

    ax.axvline(TC, color="gray", linestyle=":", linewidth=1.0)
    ax.text(TC + 0.01, 0.95, rf"$T_c = {TC:.4f}$", transform=ax.get_xaxis_transform(),
            fontsize=8, va="top")
    ax.set_xlabel("temperature $T$")
    ax.set_ylabel(r"susceptibility $\chi = L^2(\langle M^2\rangle - \langle|M|\rangle^2)/T$")
    ax.set_title(r"Susceptibility of the 2D Ising model near $T_c$")
    ax.legend()
    ax.grid(alpha=0.3)


def report_uncertainties():
    """Print the autocorrelation times and effective sample sizes used."""
    print("run            L   T      N       tau      N_eff   value     error")
    for run, lattice_guess in (("coarse-l64", 64), ("window-l64", 64),
                               ("window-l32", 32), ("coarse-l32", 32)):
        temperatures, magnetizations = load_series(run)
        for temperature in np.unique(temperatures):
            m = magnetizations[temperatures == temperature]
            mean_abs, sem_abs, tau_abs = mean_abs_magnetization(m)
            n_eff = m.size / (2.0 * tau_abs)
            print(
                f"{run:14s} {lattice_guess:3d} {temperature:5.2f} "
                f"{m.size:7d} {tau_abs:8.1f} {n_eff:7.1f} "
                f"{mean_abs:8.4f} {sem_abs:8.4f}"
            )
        print()


def main():
    EVIDENCE.mkdir(parents=True, exist_ok=True)

    report_uncertainties()

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    plot_magnetization(ax)
    fig.tight_layout()
    magnetization_path = EVIDENCE / "magnetization.png"
    fig.savefig(magnetization_path, dpi=150, bbox_inches="tight")
    print(f"wrote {magnetization_path}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    plot_susceptibility(ax)
    fig.tight_layout()
    susceptibility_path = EVIDENCE / "susceptibility.png"
    fig.savefig(susceptibility_path, dpi=150, bbox_inches="tight")
    print(f"wrote {susceptibility_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
