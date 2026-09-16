#!/usr/bin/env python3
"""Part 3: ``|M|`` autocorrelation and block-error binning, L = 64, T = 2.3.

Uses the Metropolis window run already stored in ``week3/artifacts/`` (no
resimulation).  Two panels:

1. the normalised autocorrelation ``rho(t)`` of ``|M|`` versus lag ``t``, over
   enough lags (0 .. 4000) to show the slow critical decay;
2. the standard error of the mean ``|M|`` estimated from consecutive block
   averages, for block lengths 1, 2, 4, ..., 32768 on a log axis.  Block
   length 1 is the naive error; if the curve flattens, the plateau is the
   correlated error estimate.

The script prints and annotates a plateau verdict.  Near ``T_c`` the run may
be too short for the block curve to stabilise, in which case the sampling
error is unresolved.

Run with the course environment from ``week3/``::

    /home/yao_yiyi/.venvs/amat5315/bin/python scripts/acf_binning.py

Writes ``week3/evidence/acf-binning.png``.
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
from errors import (  # noqa: E402
    autocorrelation_function,
    block_error_by_length,
    integrated_autocorrelation_time,
    naive_standard_error,
)

EVIDENCE = WEEK3 / "evidence"

RUN = "window-l64"
TEMPERATURE = 2.3
LATTICE = 64
MAX_LAG = 4000
BLOCK_LENGTHS = [1 << k for k in range(16)]  # 1, 2, ..., 32768
RELIABLE_MIN_BLOCKS = 6  # error-of-error <= 1/sqrt(2*5) ~ 32%
LARGE_START_FACTOR = 16  # compare the largest block with one 16x shorter
PLATEAU_GROWTH = 0.20  # accept <= 20% growth over that factor as flat


def load_abs_magnetization(run, temperature):
    """``|M|`` at one temperature from a stored Metropolis run."""
    temperatures, magnetizations = pt.load_series(run)
    return np.abs(magnetizations[temperatures == temperature]).astype(float)


def block_error_curve(abs_m, block_lengths):
    """Block-error estimates and block counts for a list of block lengths."""
    errors, counts = [], []
    for length in block_lengths:
        if abs_m.size // length < 2:
            continue  # fewer than two blocks: no meaningful error
        error, n_blocks = block_error_by_length(abs_m, length)
        errors.append(error)
        counts.append(n_blocks)
    return (
        np.array(block_lengths[: len(errors)], dtype=float),
        np.array(errors),
        np.array(counts),
    )


def plateau_verdict(block_lengths, errors, counts):
    """Heuristic plateau test on the large-block end.

    Uses only block lengths that still leave ``RELIABLE_MIN_BLOCKS`` blocks and
    checks how much the error grows between the largest one and a block length
    ``LARGE_START_FACTOR`` times shorter.  A flat (plateaued) curve grows by
    at most ``PLATEAU_GROWTH``.

    Returns ``(reached, lengths, errors, counts, growth, start_index)``.
    """
    reliable = counts >= RELIABLE_MIN_BLOCKS
    lengths = block_lengths[reliable]
    errs = errors[reliable]
    cnts = counts[reliable]
    if lengths.size < 2:
        return False, lengths, errs, cnts, float("nan"), 0

    start = int(np.argmin(np.abs(lengths - lengths[-1] / LARGE_START_FACTOR)))
    growth = float(errs[-1] / errs[start] - 1.0)
    return growth <= PLATEAU_GROWTH, lengths, errs, cnts, growth, start


def make_figure(abs_m):
    """The two-panel figure and its plateau verdict text."""
    rho = autocorrelation_function(abs_m)
    tau = integrated_autocorrelation_time(abs_m)
    naive = naive_standard_error(abs_m)

    lengths, errors, counts = block_error_curve(abs_m, BLOCK_LENGTHS)
    reached, tail_lengths, tail_errors, tail_counts, growth, start = plateau_verdict(
        lengths, errors, counts
    )

    if reached:
        verdict = (
            "Clear plateau: block error "
            f"≈ {tail_errors[-1]:.2e} (growth {growth:.0%})."
        )
    else:
        verdict = (
            "No clear plateau: block error still rising "
            f"(+{growth:.0%} over {LARGE_START_FACTOR}x); sampling error unresolved."
        )

    fig, (ax_acf, ax_block) = plt.subplots(
        1, 2, figsize=(12.0, 4.8), constrained_layout=True
    )

    # Panel 1: autocorrelation.
    lags = np.arange(rho.size)
    ax_acf.plot(lags[: MAX_LAG + 1], rho[: MAX_LAG + 1], color="C0", linewidth=1.0)
    ax_acf.axhline(0.0, color="gray", linewidth=0.8)
    ax_acf.axvline(tau, color="C1", linestyle="--", linewidth=1.0)
    ax_acf.text(
        tau + 60,
        0.85,
        rf"$\tau_{{\rm int}} \approx {tau:.0f}$ sweeps",
        color="C1",
        fontsize=9,
    )
    ax_acf.set_xlim(0, MAX_LAG)
    ax_acf.set_ylim(-0.25, 1.05)
    ax_acf.set_xlabel("lag $t$ (sweeps)")
    ax_acf.set_ylabel(r"$\rho(t)$")
    ax_acf.set_title(rf"ACF of $|M|$, $L = {LATTICE}$, $T = {TEMPERATURE}$")
    ax_acf.grid(alpha=0.3)

    # Panel 2: block-error curve.
    block_sem = errors / np.sqrt(2.0 * (counts - 1.0))  # error of the error
    ax_block.errorbar(
        lengths,
        errors,
        yerr=block_sem,
        fmt="o-",
        color="C3",
        markersize=4,
        capsize=2,
        linewidth=1.0,
        elinewidth=0.8,
    )
    ax_block.set_xscale("log")
    ax_block.set_xlabel("block length")
    ax_block.set_ylabel(r"standard error of mean $|M|$")
    ax_block.set_title(rf"Block-error binning, $L = {LATTICE}$, $T = {TEMPERATURE}$")
    ax_block.grid(alpha=0.3, which="both")

    # Naive point, asymptotic expectation, and the reliable-block region.
    ax_block.axhline(
        naive, color="C0", linestyle=":", linewidth=1.0, label="naive error"
    )
    expected = naive * np.sqrt(2.0 * tau)
    ax_block.axhline(
        expected,
        color="C2",
        linestyle="--",
        linewidth=1.0,
        label=rf"naive $\sqrt{{2\tau_{{\rm int}}}} \approx {expected:.2e}$",
    )
    ax_block.axvspan(
        abs_m.size / RELIABLE_MIN_BLOCKS,
        lengths[-1],
        color="gray",
        alpha=0.08,
        label=f"fewer than {RELIABLE_MIN_BLOCKS} blocks",
    )
    ax_block.legend(fontsize=8, loc="upper left")
    ax_block.text(
        0.98,
        0.04,
        verdict,
        transform=ax_block.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        wrap=True,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    return fig, rho, tau, naive, lengths, errors, counts, (
        reached,
        tail_lengths,
        tail_errors,
        tail_counts,
        growth,
        start,
        verdict,
    )


def main():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    abs_m = load_abs_magnetization(RUN, TEMPERATURE)
    print(
        f"loaded {abs_m.size} |M| sweeps for L = {LATTICE}, "
        f"T = {TEMPERATURE} from {RUN}"
    )

    fig, _, tau, naive, _, _, _, verdict_info = make_figure(abs_m)
    path = EVIDENCE / "acf-binning.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    reached, tail_lengths, tail_errors, tail_counts, growth, start, verdict = (
        verdict_info
    )
    print(f"tau_int  = {tau:.1f} sweeps")
    print(f"naive SE = {naive:.3e}")
    print(
        f"large-block range judged: L={int(tail_lengths[start])} "
        f".. {int(tail_lengths[-1])} "
        f"(n_blocks {int(tail_counts[start])} .. {int(tail_counts[-1])})"
    )
    print(
        f"block error {tail_errors[start]:.3e} -> {tail_errors[-1]:.3e} "
        f"(growth {growth:+.0%})"
    )
    print(verdict)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
