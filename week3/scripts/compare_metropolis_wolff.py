#!/usr/bin/env python3
"""Part 4: compare Metropolis and Wolff magnetizations, and Wolff chi(T).

Uses only the runs already stored in ``week3/artifacts/`` (no resimulation):

* ``window-l64``  -- Metropolis, L = 64, fine critical-window ramp;
* ``wolff-l64``   -- Wolff,     L = 64, same temperature grid;
* ``wolff-l32``   -- Wolff,     L = 32, same temperature grid.

Panel 1 plots mean ``|M|`` versus temperature for L = 64 for both algorithms.
The plotted error bars are block-bootstrap standard errors with a block length
of 4000 measurements.  The same bootstrap is also run at block lengths 2000
and 8000 purely to check whether the error estimate is block-length sensitive;
if the three standard errors disagree by more than 10% of their mean the
Metropolis-vs-Wolff agreement is reported as provisional.

Panel 2 computes the Wolff susceptibility

    chi(T) = L^2 * (mean(M^2) - mean(|M|)^2) / T

for L = 32 and L = 64, fits a quadratic to the five temperature points centred
on the largest chi, marks the fitted T_peak values, and reports

    T_c = 2 * T_peak(64) - T_peak(32).

The exact Onsager value 2.26919 is marked for comparison.

Run with the course environment from ``week3/``::

    /home/yao_yiyi/.venvs/amat5315/bin/python scripts/compare_metropolis_wolff.py

Writes ``week3/evidence/magnetization-compare.png`` and
``week3/evidence/magnetization-compare.txt``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WEEK3 = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(WEEK3))
sys.path.insert(0, str(SCRIPTS))

import bootstrap  # noqa: E402  (needs WEEK3 and SCRIPTS on sys.path)
from errors import series_by_temperature  # noqa: E402

EVIDENCE = WEEK3 / "evidence"

BLOCK_LENGTHS = (2000, 4000, 8000)
PLOT_BLOCK_LENGTH = 4000
N_BOOT = 500
SEED = 2026
STABILITY_TOLERANCE = 0.1
ONSAGER_TC = 2.26919  # exact critical temperature requested for this figure


def block_bootstrap_error(series, block_length, n_boot, rng):
    """Block-bootstrap standard error of the mean ``|M|``.

    The measured series is split into consecutive blocks of ``block_length``
    measurements, blocks are resampled with replacement, and the standard
    deviation of the bootstrap block means is returned.
    """
    abs_m = np.abs(np.asarray(series, dtype=float))
    n_blocks = abs_m.size // block_length
    if n_blocks < 2:
        raise ValueError(
            f"need at least 2 blocks of length {block_length} (got {n_blocks})"
        )
    blocks = abs_m[: n_blocks * block_length].reshape(n_blocks, block_length)
    block_means = blocks.mean(axis=1)
    indices = rng.integers(0, n_blocks, size=(n_boot, n_blocks))
    bootstrap_means = block_means[indices].mean(axis=1)
    return float(bootstrap_means.std(ddof=1))


def d_statistic(mean1, sigma1, mean2, sigma2):
    """``d = |mean1 - mean2| / sqrt(sigma1^2 + sigma2^2)``."""
    return float(abs(mean1 - mean2) / np.sqrt(sigma1 ** 2 + sigma2 ** 2))


def agreement_verdict(d, stable1, stable2):
    """Classify a Metropolis-vs-Wolff comparison from ``d`` and stability."""
    if d > 3.0:
        return "discrepancy"
    if stable1 and stable2:
        return "agreement"
    return "agreement provisional"


def bootstrap_error_by_length(series, block_lengths, n_boot, rng):
    """Bootstrap std of mean ``|M|`` at each requested block length."""
    return {
        block_length: block_bootstrap_error(series, block_length, n_boot, rng)
        for block_length in block_lengths
    }


def stable_errors(errors, tolerance=STABILITY_TOLERANCE):
    """True unless the block-length stds disagree by more than tolerance."""
    return bootstrap.sampling_error_stable(
        [errors[block_length] for block_length in BLOCK_LENGTHS],
        tolerance=tolerance,
    )


def load_means_and_errors(run, temperatures, rng):
    """Mean ``|M|`` and per-block-length bootstrap errors for one run."""
    series_by_temperature_ = series_by_temperature(run)
    means = np.array(
        [np.mean(np.abs(series_by_temperature_[t])) for t in temperatures]
    )
    errors = [
        bootstrap_error_by_length(
            series_by_temperature_[t], BLOCK_LENGTHS, N_BOOT, rng
        )
        for t in temperatures
    ]
    return means, errors


def panel1(ax, temperatures, metro_means, wolff_means, metro_errors, wolff_errors):
    """Mean |M| versus T for L = 64, Metropolis and Wolff, with 4000-block bars."""
    offset = 0.008
    metro_yerr = [errors[PLOT_BLOCK_LENGTH] for errors in metro_errors]
    wolff_yerr = [errors[PLOT_BLOCK_LENGTH] for errors in wolff_errors]

    ax.errorbar(
        temperatures - offset,
        metro_means,
        yerr=metro_yerr,
        fmt="o-",
        color="C0",
        markersize=4,
        capsize=2,
        elinewidth=0.8,
        label=f"Metropolis (window-l64)",
    )
    ax.errorbar(
        temperatures + offset,
        wolff_means,
        yerr=wolff_yerr,
        fmt="s-",
        color="C1",
        markersize=4,
        capsize=2,
        elinewidth=0.8,
        label="Wolff (wolff-l64)",
    )

    ax.axvline(ONSAGER_TC, color="gray", linestyle=":", linewidth=1.0)
    ax.text(
        ONSAGER_TC + 0.01,
        0.98,
        rf"Onsager $T_c = {ONSAGER_TC}$",
        transform=ax.get_xaxis_transform(),
        fontsize=8,
        va="top",
    )
    ax.set_xlabel("temperature $T$")
    ax.set_ylabel(r"mean $|M|$")
    ax.set_title(r"L = 64: Metropolis vs Wolff")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def panel2(ax, results):
    """Wolff chi(T) with five-point fits and marked peak/critical values."""
    for lattice, color in ((32, "C0"), (64, "C1")):
        data = results[lattice]
        temperatures = data["temperatures"]
        chi = data["chi"]
        fit = data["fit"]

        ax.plot(
            temperatures,
            chi,
            "o-",
            color=color,
            markersize=4,
            label=rf"Wolff $\chi$, $L = {lattice}$",
        )

        t_fit = np.linspace(fit["t_low"], fit["t_high"], 100)
        ax.plot(
            t_fit,
            np.polyval(fit["coefficients"], t_fit),
            "--",
            color=color,
            linewidth=1.4,
            label=(
                rf"five-point fit, $T_{{\rm peak}}({lattice})"
                rf" = {fit['vertex']:.4f}$"
            ),
        )
        ax.axvline(fit["vertex"], color=color, linestyle=":", linewidth=1.0)

    tc = results["tc"]
    ax.axvline(tc, color="C3", linestyle="--", linewidth=1.1)
    ax.text(
        tc + 0.008,
        0.03,
        rf"$T_c = 2T_{{\rm peak}}(64) - T_{{\rm peak}}(32) = {tc:.4f}$",
        transform=ax.get_xaxis_transform(),
        fontsize=8,
        va="bottom",
        color="C3",
    )
    ax.axvline(ONSAGER_TC, color="gray", linestyle=":", linewidth=1.0)
    ax.text(
        ONSAGER_TC - 0.008,
        0.03,
        rf"Onsager $T_c = {ONSAGER_TC}$",
        transform=ax.get_xaxis_transform(),
        fontsize=8,
        va="bottom",
        ha="right",
    )

    ax.set_xlabel("temperature $T$")
    ax.set_ylabel(r"$\chi = L^2(\langle M^2\rangle - \langle|M|\rangle^2)/T$")
    ax.set_title("Wolff susceptibility")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def wolff_panel2_results():
    """chi(T), five-point fit, and T_c from the Wolff artifacts."""
    results = {}
    for lattice in (32, 64):
        run = f"wolff-l{lattice}"
        by_temperature = series_by_temperature(run)
        temperatures = np.array(sorted(by_temperature), dtype=float)
        chi = np.array(
            [
                bootstrap.susceptibility(by_temperature[t], lattice, t)
                for t in temperatures
            ]
        )
        fit = bootstrap.fit_five_point(temperatures, chi)
        if fit is None:
            raise RuntimeError(
                f"five-point fit failed for {run}; check the chi(T) peak"
            )
        results[lattice] = {
            "run": run,
            "temperatures": temperatures,
            "chi": chi,
            "fit": fit,
        }
    results["tc"] = 2.0 * results[64]["fit"]["vertex"] - results[32]["fit"]["vertex"]
    return results


def format_report(temperatures, metro, wolff, panel2_results):
    """Plain-text table of the comparison and the Wolff peak analysis."""
    means_m, errors_m = metro
    means_w, errors_w = wolff

    rows = []
    for i, temperature in enumerate(temperatures):
        sigma_m = errors_m[i][PLOT_BLOCK_LENGTH]
        sigma_w = errors_w[i][PLOT_BLOCK_LENGTH]
        stable_m = stable_errors(errors_m[i])
        stable_w = stable_errors(errors_w[i])
        d = d_statistic(means_m[i], sigma_m, means_w[i], sigma_w)
        verdict = agreement_verdict(d, stable_m, stable_w)
        rows.append(
            (
                temperature,
                means_m[i],
                sigma_m,
                stable_m,
                means_w[i],
                sigma_w,
                stable_w,
                d,
                verdict,
            )
        )

    lines = [
        "Part 4: Metropolis vs Wolff verification",
        "=" * 62,
        (
            "Observable: mean |M| for L = 64 on the window grid "
            "2.00 <= T <= 2.60."
        ),
        (
            f"Plotted error bars: block-bootstrap standard error, block length "
            f"{PLOT_BLOCK_LENGTH}, {N_BOOT} replicates, seed = {SEED}."
        ),
        (
            f"Block-length stability checked at {', '.join(map(str, BLOCK_LENGTHS))} "
            f"(agreement if all within {STABILITY_TOLERANCE:.0%} of their mean)."
        ),
        (
            "Comparison statistic: "
            "d = |mean1 - mean2| / sqrt(sigma1^2 + sigma2^2)."
        ),
        (
            "Verdict: discrepancy if d > 3; agreement if d <= 3 and both "
            "errors stable; agreement provisional if either error is "
            "block-length sensitive."
        ),
        "",
        f"{'T':>5s}  {'mean M':>10s}  {'sigma M':>10s}  {'stable':>6s}  "
        f"{'mean W':>10s}  {'sigma W':>10s}  {'stable':>6s}  {'d':>7s}  "
        f"{'verdict':>20s}",
        "-" * 62,
    ]
    for temperature, m, sm, stable_m, w, sw, stable_w, d, verdict in rows:
        lines.append(
            f"{temperature:>5.2f}  {m:>10.6f}  {sm:>10.6e}  {str(stable_m):>6s}  "
            f"{w:>10.6f}  {sw:>10.6e}  {str(stable_w):>6s}  {d:>7.3f}  "
            f"{verdict:>20s}"
        )

    l32 = panel2_results[32]["fit"]
    l64 = panel2_results[64]["fit"]
    tc = panel2_results["tc"]
    lines += [
        "",
        "Wolff susceptibility peak",
        "-" * 62,
        f"  T_peak(L = 32) = {l32['vertex']:.6f}",
        f"  T_peak(L = 64) = {l64['vertex']:.6f}",
        f"  T_c = 2*T_peak(64) - T_peak(32) = {tc:.6f}",
        f"  Onsager T_c = {ONSAGER_TC:.5f}",
        f"  difference   = {abs(tc - ONSAGER_TC):.6f}",
        "",
        "Uncertainty discussion",
        "-" * 62,
        (
            "The |M| error bars are bootstrap errors of consecutive block means, "
            "so they retain autocorrelation within each block. Near T_c the "
            "Metropolis run has long autocorrelation times and its errors are "
            "several times larger than the Wolff errors, which use a cluster "
            "flip per measurement and decorrelate much faster."
        ),
        (
            "The stability column is 'True' when the 2000/4000/8000 bootstrap "
            "errors agree within 10% of their mean. Where it is 'False' the "
            "error is block-length sensitive, and the agreement verdict is "
            "downgraded to provisional even when d <= 3."
        ),
        (
            "The Wolff T_peak fit inherits the resolution of the 0.05 temperature "
            "grid and the statistical uncertainty of chi(T); no bootstrap error "
            "is quoted for it here."
        ),
    ]
    return "\n".join(lines)


def main():
    EVIDENCE.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    metro_run = "window-l64"
    wolff_run = "wolff-l64"
    temperatures = np.array(
        sorted(series_by_temperature(metro_run)), dtype=float
    )

    metro = load_means_and_errors(metro_run, temperatures, rng)
    wolff = load_means_and_errors(wolff_run, temperatures, rng)
    panel2_results = wolff_panel2_results()

    report = format_report(temperatures, metro, wolff, panel2_results)
    print(report)
    report_path = EVIDENCE / "magnetization-compare.txt"
    report_path.write_text(report + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    panel1(axes[0], temperatures, metro[0], wolff[0], metro[1], wolff[1])
    panel2(axes[1], panel2_results)
    figure_path = EVIDENCE / "magnetization-compare.png"
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nwrote {figure_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
