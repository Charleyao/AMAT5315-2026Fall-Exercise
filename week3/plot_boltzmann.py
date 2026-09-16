#!/usr/bin/env python3
"""Boltzmann histogram check for the 2D Ising runs at T = 3.0 and T = 3.1.

For each run every row of ``runs/<T>/series.jsonl`` holds the energy per site
``E``.  The total energy of an ``L x L`` lattice is ``E * L**2``; with
``L = 64`` that factor is 4096.  Boltzmann statistics predicts

    P_T(E) ~ exp(-E_total / T),

so the log ratio of the histograms at two temperatures is linear in the total
energy with slope

    d/dE_total ln(P_3.1 / P_3.0) = 1/3.0 - 1/3.1.

This script bins both runs with bins 40 energy units wide, keeps only the bins
where both histograms hold at least five rows, and plots the log ratio together
with a line of the predicted slope.

Exact command used to generate the figure
-----------------------------------------
    cd /home/yao_yiyi/AMAT5315-2026Fall-Exercise/week3 && \
        /home/yao_yiyi/.venvs/amat5315/bin/python plot_boltzmann.py

The figure is saved to ``week3/evidence/boltzmann.png``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WEEK3 = Path(__file__).resolve().parent
RUNS = WEEK3 / "runs"
OUT = WEEK3 / "evidence" / "boltzmann.png"

# Run folders and temperatures.  T3.0 is the "first" histogram and T3.1 is the
# "second" one, so the ratio plotted is T3.1 / T3.0.
RUNS_T = {"T3.0": 3.0, "T3.1": 3.1}
L = 64
BIN_WIDTH = 40.0
MIN_COUNT = 5


def load_total_energy(run: str) -> np.ndarray:
    """Read a run's series.jsonl and return E_total = E * L**2 for every row."""
    path = RUNS / run / "series.jsonl"
    with path.open() as handle:
        energies = [json.loads(line)["E"] for line in handle]
    return np.asarray(energies, dtype=float) * L * L


def binned_log_ratio(e1: np.ndarray, e2: np.ndarray):
    """Histogram both samples and return the log ratio of well-populated bins.

    Bins are ``BIN_WIDTH`` wide and shared by both samples.  Only bins with at
    least ``MIN_COUNT`` rows in both histograms are returned.
    """
    low = np.floor(min(e1.min(), e2.min()) / BIN_WIDTH) * BIN_WIDTH
    high = np.ceil(max(e1.max(), e2.max()) / BIN_WIDTH) * BIN_WIDTH
    edges = np.arange(low, high + BIN_WIDTH, BIN_WIDTH)

    count1, _ = np.histogram(e1, bins=edges)
    count2, _ = np.histogram(e2, bins=edges)
    keep = (count1 >= MIN_COUNT) & (count2 >= MIN_COUNT)

    centers = 0.5 * (edges[:-1] + edges[1:])
    ratio = np.log(count2[keep] / count1[keep])
    return centers[keep], ratio, count1, count2, edges


def main() -> None:
    e1 = load_total_energy("T3.0")
    e2 = load_total_energy("T3.1")

    x, y, count1, count2, edges = binned_log_ratio(e1, e2)

    # Predicted slope from the Boltzmann distribution: 1/T1 - 1/T2.
    slope = 1.0 / RUNS_T["T3.0"] - 1.0 / RUNS_T["T3.1"]
    # Best-fit line with that fixed slope: choose the intercept that minimises
    # the squared residuals of the ratio points.
    intercept = float(np.mean(y - slope * x))
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept

    fig, (ax_hist, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=(8.0, 7.0),
        height_ratios=[1.0, 1.6],
        sharex=True,
    )

    # --- Top panel: the two total-energy histograms ----------------------
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax_hist.step(centers, count1, where="mid", label="T = 3.0", color="C0")
    ax_hist.step(centers, count2, where="mid", label="T = 3.1", color="C1")
    ax_hist.set_ylabel("rows per bin")
    ax_hist.set_title(
        r"$L = 64$ Ising total energy, bins 40 wide ($E \times 4096$)"
    )
    ax_hist.legend()
    ax_hist.grid(alpha=0.3)

    # --- Bottom panel: log ratio and the predicted slope -----------------
    ax_ratio.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
    ax_ratio.plot(
        x,
        y,
        "o",
        color="black",
        label=r"$\ln(N_{3.1}/N_{3.0})$",
    )
    ax_ratio.plot(
        x_line,
        y_line,
        color="red",
        label=rf"slope $1/3.0 - 1/3.1 = {slope:.5f}$",
    )
    note = f"bins with $N\\geq {MIN_COUNT}$ in both runs: {x.size}"
    ax_ratio.text(0.02, 0.05, note, transform=ax_ratio.transAxes, fontsize=8)
    ax_ratio.set_xlabel(r"total energy $E \times 4096$")
    ax_ratio.set_ylabel(r"$\ln(N_{3.1}/N_{3.0})$")
    ax_ratio.legend()
    ax_ratio.grid(alpha=0.3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
