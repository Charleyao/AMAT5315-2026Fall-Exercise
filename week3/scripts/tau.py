#!/usr/bin/env python3
"""Part 3: integrated autocorrelation time of ``|M|`` versus temperature.

For ``L = 32`` and ``L = 64`` this evaluates the same tau_int definition and
truncation rule as ``scripts/errors.py``:

    tau_int = 1/2 + sum_{t>=1} rho(t),

cutting the sum at the first lag larger than six times the running estimate.
Each temperature uses the ``window-l*`` run where available (2.00 <= T <= 2.60,
the Part 2 peak-fit data) and the ``coarse-l*`` run elsewhere.

Both lattices are drawn on one semilog-y figure so the growth of tau_int near
``T_c`` and the size dependence can be compared.

Run with the course environment from ``week3/``::

    /home/yao_yiyi/.venvs/amat5315/bin/python scripts/tau.py

Writes ``week3/evidence/tau.png``.
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
from errors import integrated_autocorrelation_time, merged_series  # noqa: E402

EVIDENCE = WEEK3 / "evidence"

LATTICES = (32, 64)
COLORS = {32: "C0", 64: "C1"}


def tau_vs_temperature(merged):
    """Sorted temperatures and tau_int of ``|M|`` for a merged series map."""
    temperatures = np.array(sorted(merged), dtype=float)
    taus = np.array(
        [
            integrated_autocorrelation_time(np.abs(merged[temperature][0]))
            for temperature in temperatures
        ]
    )
    return temperatures, taus


def make_figure():
    """One semilog-y figure with tau_int(T) for both lattice sizes."""
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)

    peaks = {}
    for lattice in LATTICES:
        temperatures, taus = tau_vs_temperature(merged_series(lattice))
        peaks[lattice] = (temperatures[int(np.argmax(taus))], taus.max())
        ax.semilogy(
            temperatures,
            taus,
            "o-",
            color=COLORS[lattice],
            markersize=4,
            linewidth=1.0,
            label=rf"$L = {lattice}$",
        )

    ax.axvline(pt.TC, color="gray", linestyle=":", linewidth=1.0)
    ax.text(
        pt.TC + 0.02,
        1.05,
        rf"$T_c = {pt.TC:.3f}$",
        fontsize=8,
        va="bottom",
    )
    ax.set_ylim(0.8, 1500.0)
    for lattice, (temperature, tau) in peaks.items():
        ax.annotate(
            rf"max $\tau \approx {tau:.0f}$",
            xy=(temperature, tau),
            xytext=(temperature + 0.12, tau * 1.25),
            color=COLORS[lattice],
            fontsize=8,
            arrowprops=dict(arrowstyle="-", color=COLORS[lattice], linewidth=0.7),
        )

    ax.set_xlabel("temperature $T$")
    ax.set_ylabel(r"$\tau_{\rm int}$ of $|M|$ (sweeps)")
    ax.set_title(r"Integrated autocorrelation time of $|M|$")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    return fig


def main():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    fig = make_figure()
    path = EVIDENCE / "tau.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
