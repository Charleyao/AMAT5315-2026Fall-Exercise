#!/usr/bin/env python3
"""Part 4: work-normalized autocorrelation time, Metropolis vs Wolff.

Uses the existing L = 64 runs in ``week3/artifacts/`` (no resimulation):

* ``window-l64`` -- Metropolis, one sweep per measurement;
* ``wolff-l64``  -- Wolff, one cluster move per measurement, with the cluster
  size recorded on every row.

For both algorithms we evaluate the same integrated autocorrelation time of
``|M|`` as the rest of the week 3 analysis:

    tau_int = 1/2 + sum_{t>=1} rho(t),

cutting the sum at the first lag larger than six times the running estimate
(``scripts/errors.py``).

* Metropolis already measures work in sweeps, so ``tau_work = tau_int``.
* Wolff measures time in cluster moves, so one move costs ``<c>`` spin
  updates on average and

      tau_work = tau_moves * <c> / L^2,

  which converts cluster moves to sweep-equivalent work for L = 64.

Run with the course environment from ``week3/``::

    /home/yao_yiyi/.venvs/amat5315/bin/python scripts/compare.py

Writes ``week3/evidence/tau-compare.png`` and
``week3/evidence/tau-compare.txt``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WEEK3 = Path(__file__).resolve().parent.parent
ARTIFACTS = WEEK3 / "artifacts"
EVIDENCE = WEEK3 / "evidence"
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(WEEK3))
sys.path.insert(0, str(SCRIPTS))

import plot_thermo as pt  # noqa: E402  (needs WEEK3 on sys.path)
from errors import integrated_autocorrelation_time  # noqa: E402

METRO_RUN = "window-l64"
WOLFF_RUN = "wolff-l64"
LATTICE = 64
REPORT_TEMPERATURE = 2.3
ONSAGER_TC = 2.26919


def group_by_temperature(temperatures, values):
    """Group ``values`` by temperature, preserving within-group order."""
    temperatures = np.asarray(temperatures)
    values = np.asarray(values)
    return {
        float(temperature): values[temperatures == temperature]
        for temperature in np.unique(temperatures)
    }


def wolff_tau_work(tau_moves, mean_cluster_size, lattice):
    """Convert Wolff tau_int in cluster moves to sweep-equivalent work."""
    return float(tau_moves) * float(mean_cluster_size) / float(lattice) ** 2


def load_wolff_series(run):
    """Temperature, magnetization and cluster size for a Wolff run."""
    temperatures = []
    magnetizations = []
    cluster_sizes = []
    with (ARTIFACTS / run / "series.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            temperatures.append(row["T"])
            magnetizations.append(row["M"])
            cluster_sizes.append(row["cluster_size"])
    return (
        np.asarray(temperatures),
        np.asarray(magnetizations),
        np.asarray(cluster_sizes, dtype=float),
    )


def metropolis_curve(run):
    """tau_int(|M|) in sweeps versus temperature for a Metropolis run."""
    temperatures, magnetizations = pt.load_series(run)
    by_temperature = group_by_temperature(temperatures, magnetizations)
    grid = np.array(sorted(by_temperature), dtype=float)
    tau = np.array(
        [
            integrated_autocorrelation_time(np.abs(by_temperature[t]))
            for t in grid
        ]
    )
    return grid, tau, tau  # tau_work is tau_int when time is already sweeps


def wolff_curve(run, lattice):
    """tau_int in cluster moves, mean cluster size, and tau_work for Wolff."""
    temperatures, magnetizations, cluster_sizes = load_wolff_series(run)
    by_magnetization = group_by_temperature(temperatures, magnetizations)
    by_cluster_size = group_by_temperature(temperatures, cluster_sizes)
    grid = np.array(sorted(by_magnetization), dtype=float)

    tau_moves = np.array(
        [
            integrated_autocorrelation_time(np.abs(by_magnetization[t]))
            for t in grid
        ]
    )
    mean_cluster = np.array([np.mean(by_cluster_size[t]) for t in grid])
    tau_work = np.array(
        [
            wolff_tau_work(tau, cluster, lattice)
            for tau, cluster in zip(tau_moves, mean_cluster)
        ]
    )
    return grid, tau_moves, mean_cluster, tau_work


def value_at(temperatures, values, temperature):
    """Return the value at an exact grid temperature."""
    mask = temperatures == temperature
    if not mask.any():
        raise ValueError(f"temperature {temperature} not on the grid")
    return float(values[mask][0])


def format_report(metro, wolff, temperature=REPORT_TEMPERATURE):
    """Report the work-normalized autocorrelation times at one temperature."""
    metro_grid, metro_tau_int, metro_tau_work = metro
    wolff_grid, wolff_tau_moves, wolff_mean_cluster, wolff_tau_work = wolff

    m_tau_int = value_at(metro_grid, metro_tau_int, temperature)
    m_tau_work = value_at(metro_grid, metro_tau_work, temperature)
    w_tau_moves = value_at(wolff_grid, wolff_tau_moves, temperature)
    w_mean_cluster = value_at(wolff_grid, wolff_mean_cluster, temperature)
    w_tau_work = value_at(wolff_grid, wolff_tau_work, temperature)
    ratio = m_tau_work / w_tau_work

    lines = [
        "Part 4: work-normalized autocorrelation time at T = "
        f"{temperature:.2f}",
        "=" * 62,
        f"L = {LATTICE}; Metropolis run {METRO_RUN}, Wolff run {WOLFF_RUN}.",
        (
            "tau_int = 1/2 + sum_{t>=1} rho(t), truncated with the "
            "six-times-running-tau rule."
        ),
        (
            "Work conversion: Metropolis tau_work = tau_int (sweeps); "
            "Wolff tau_work = tau_moves * <c> / L^2."
        ),
        "",
        f"  Metropolis tau_int (sweeps)          = {m_tau_int:.3f}",
        f"  Metropolis tau_work (sweeps)         = {m_tau_work:.3f}",
        f"  Wolff tau_int (cluster moves)        = {w_tau_moves:.3f}",
        f"  Wolff mean cluster size <c>          = {w_mean_cluster:.3f}",
        f"  Wolff tau_work (sweeps)              = {w_tau_work:.4f}",
        f"  ratio tau_work(Metropolis)/tau_work(Wolff) = {ratio:.1f}",
    ]
    return "\n".join(lines)


def make_figure(metro, wolff):
    """Semilog-y figure of work-normalized autocorrelation time vs T."""
    metro_grid, _, metro_tau_work = metro
    wolff_grid, _, _, wolff_tau_work = wolff

    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)

    ax.semilogy(
        metro_grid,
        metro_tau_work,
        "o-",
        color="C0",
        markersize=4,
        linewidth=1.0,
        label="Metropolis ($\\tau_{\\rm work} = \\tau_{\\rm int}$)",
    )
    ax.semilogy(
        wolff_grid,
        wolff_tau_work,
        "s-",
        color="C1",
        markersize=4,
        linewidth=1.0,
        label="Wolff ($\\tau_{\\rm work} = \\tau_{\\rm int}\\,\\langle c\\rangle/L^2$)",
    )

    ax.axvline(ONSAGER_TC, color="gray", linestyle=":", linewidth=1.0)
    ax.text(
        ONSAGER_TC + 0.01,
        1.04,
        rf"Onsager $T_c = {ONSAGER_TC}$",
        transform=ax.get_xaxis_transform(),
        fontsize=8,
        va="bottom",
    )
    ax.set_xlabel("temperature $T$")
    ax.set_ylabel(r"work-normalized autocorrelation time $\tau_{\rm work}$ (sweeps)")
    ax.set_title(rf"Work-normalized autocorrelation time of $|M|$, $L = {LATTICE}$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    return fig


def main():
    EVIDENCE.mkdir(parents=True, exist_ok=True)

    metro = metropolis_curve(METRO_RUN)
    wolff = wolff_curve(WOLFF_RUN, LATTICE)

    report = format_report(metro, wolff)
    print(report)

    report_path = EVIDENCE / "tau-compare.txt"
    report_path.write_text(report + "\n")

    fig = make_figure(metro, wolff)
    figure_path = EVIDENCE / "tau-compare.png"
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nwrote {figure_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
