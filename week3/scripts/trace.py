#!/usr/bin/env python3
"""Part 3: magnetization traces at one critical and one high temperature.

Draws ``|M|`` for ``L = 64`` over the first 2000 recorded sweeps at

* ``T = 2.30``, just below the Onsager temperature ``T_c ~= 2.269``, where the
  Part 3 autocorrelation analysis found ``tau_int`` of several hundred sweeps;
* ``T = 3.00``, well above ``T_c``, where ``tau_int`` is only a few sweeps.

Both series are read from the Metropolis runs already stored in
``week3/artifacts/``; nothing is resimulated.  ``T = 2.30`` comes from the
high-statistics ``window-l64`` ramp (the Part 2 peak-fit data), ``T = 3.00``
from ``coarse-l64``.

Run with the course environment from ``week3/``::

    /home/yao_yiyi/.venvs/amat5315/bin/python scripts/trace.py

Writes ``week3/evidence/trace.png``.
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
from errors import integrated_autocorrelation_time  # noqa: E402

EVIDENCE = WEEK3 / "evidence"

LATTICE = 64
N_SWEEPS = 2000
# (temperature, run folder, colour)
TRACES = (
    (2.30, "window-l64", "C0"),
    (3.00, "coarse-l64", "C3"),
)


def first_sweeps(temperatures, magnetizations, temperature, count):
    """First ``count`` recorded magnetizations at ``temperature``, in order.

    The series files are written in ascending sweep order within each
    temperature, so filtering preserves the measured order and slicing takes
    the earliest sweeps.
    """
    temperatures = np.asarray(temperatures, dtype=float)
    magnetizations = np.asarray(magnetizations, dtype=float)
    selected = magnetizations[temperatures == temperature]
    return selected[:count]


def load_trace(temperature, run, count=N_SWEEPS):
    """``|M|``, sweep numbers, ``tau_int`` and the full-run mean for one trace.

    ``tau_int`` and the full-run mean are measured on the *full* stored series,
    not only the plotted 2000 sweeps, so the panels can be compared with the
    Part 3 table in ``evidence/errors.txt``.
    """
    temperatures, magnetizations = pt.load_series(run)
    full = np.abs(magnetizations[temperatures == temperature])
    abs_m = np.abs(first_sweeps(temperatures, magnetizations, temperature, count))
    return (
        np.arange(1, abs_m.size + 1),
        abs_m,
        integrated_autocorrelation_time(full),
        float(full.mean()),
    )


def make_figure():
    """Two stacked ``|M|`` traces sharing the recorded-sweep axis."""
    fig, axes = plt.subplots(
        2, 1, figsize=(8.0, 6.0), sharex=True, constrained_layout=True
    )
    for ax, (temperature, run, colour) in zip(axes, TRACES):
        sweeps, abs_m, tau, full_mean = load_trace(temperature, run)
        ax.plot(sweeps, abs_m, color=colour, linewidth=0.8)
        ax.set_ylabel(r"$|M|$")
        ax.grid(alpha=0.3)
        ax.set_title(
            rf"$T = {temperature:.2f}$, $L = {LATTICE}$   "
            rf"($\tau_{{\rm int}} = {tau:.1f}$ sweeps; "
            rf"trace mean $= {abs_m.mean():.4f}$, "
            rf"full-run mean $= {full_mean:.4f}$)",
            fontsize=10,
        )
    axes[-1].set_xlabel("recorded sweep")
    fig.suptitle(
        rf"$|M|$ traces for $L = {LATTICE}$: critical (long memory) vs high $T$"
    )
    return fig


def main():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    fig = make_figure()
    path = EVIDENCE / "trace.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
