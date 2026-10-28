"""VERIFY (1): unstable-bit fraction of the Hopfield network vs alpha.

For N in {100, 200, 400, 800} and alpha = P/N from 0.05 to 0.25 (step 0.01),
20 independent random pattern sets are generated per point. The measured
quantity is the fraction of stored-pattern bits with ``xi_i * h_i < 0``,
compared with the Gaussian prediction ``Phi(-1/sqrt(alpha))``.

Usage::

    cd week6
    uv run python hopfield_experiment.py
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")  # headless rendering; must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np

import hopfield

WEEK_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = WEEK_DIR / "artifacts"
EVIDENCE_DIR = WEEK_DIR / "evidence"
RESULTS_PATH = ARTIFACTS_DIR / "hopfield_unstable_fraction.json"
FIGURE_PATH = EVIDENCE_DIR / "hopfield_unstable_fraction.png"

N_VALUES = [100, 200, 400, 800]
ALPHAS = np.round(np.arange(0.05, 0.2501, 0.01), 2)
N_SETS = 20
SEED = 5315  # reproducible pattern sets
COLORS = {100: "#1f77b4", 200: "#2ca02c", 400: "#ff7f0e", 800: "#d62728"}


def run_point(
    n_spins: int, alpha: float, key: jax.Array, n_sets: int = N_SETS
) -> np.ndarray:
    """Unstable fraction for ``n_sets`` pattern sets at one (N, alpha) point."""
    n_patterns = int(round(alpha * n_spins))
    bits = jax.random.bernoulli(key, 0.5, (n_sets, n_patterns, n_spins))
    patterns = jnp.where(bits, 1.0, -1.0)

    # Batched Hebbian couplings J = X^T X / N with zero diagonal.
    couplings = jnp.einsum("spi,spj->sij", patterns, patterns) / n_spins
    couplings = 0.5 * (couplings + jnp.swapaxes(couplings, 1, 2))
    couplings = couplings * (1.0 - jnp.eye(n_spins, dtype=jnp.float32))[None]

    fields = jnp.einsum("sij,spj->spi", couplings, patterns)
    unstable = (patterns * fields < 0).mean(axis=(1, 2))
    return np.asarray(unstable, dtype=np.float64)


def run_experiment() -> dict:
    theory = [hopfield.gaussian_prediction(alpha) for alpha in ALPHAS]
    results = {}
    for n_index, n_spins in enumerate(N_VALUES):
        means, ses, per_set = [], [], []
        for alpha_index, alpha in enumerate(ALPHAS):
            key = jax.random.fold_in(
                jax.random.PRNGKey(SEED), n_index * 1000 + alpha_index
            )
            values = run_point(n_spins, float(alpha), key)
            means.append(float(values.mean()))
            ses.append(float(values.std(ddof=1) / np.sqrt(len(values))))
            per_set.append(values.tolist())
        results[str(n_spins)] = {"mean": means, "se": ses, "per_set": per_set}
    return {
        "settings": {
            "n_spins": N_VALUES,
            "alpha": ALPHAS.tolist(),
            "alpha_range": [float(ALPHAS[0]), float(ALPHAS[-1])],
            "alpha_step": 0.01,
            "pattern_sets_per_point": N_SETS,
            "seed": SEED,
            "patterns": "independent random +/-1 (P = round(alpha * N))",
            "unstable_bit": "xi_i * h_i < 0",
            "theory": "Phi(-1/sqrt(alpha))",
        },
        "theory": theory,
        "results": results,
    }


def build_figure(data: dict) -> plt.Figure:
    """Linear and log-scale comparison of measured curves and the theory."""
    alphas = np.asarray(data["settings"]["alpha"])
    theory = np.asarray(data["theory"])

    fig, (ax_linear, ax_log) = plt.subplots(1, 2, figsize=(12.4, 4.8))
    for n_spins in N_VALUES:
        result = data["results"][str(n_spins)]
        means = np.asarray(result["mean"])
        ses = np.asarray(result["se"])
        ax_linear.errorbar(
            alphas, means, yerr=ses, color=COLORS[n_spins],
            marker="o", markersize=3, linewidth=1.2, capsize=2,
            label=f"N = {n_spins}",
        )
        # Resolution floor for the log panel when a measured point is zero.
        n_patterns = np.maximum(1, np.round(alphas * n_spins))
        floor = 1.0 / (n_spins * n_patterns * N_SETS)
        plotted = np.maximum(means, floor)
        lower = np.maximum(plotted - ses, 0.5 * floor)
        upper = plotted + ses
        ax_log.errorbar(
            alphas, plotted, yerr=[plotted - lower, upper - plotted],
            color=COLORS[n_spins], marker="o", markersize=3, linewidth=1.2,
            capsize=2, label=f"N = {n_spins}",
        )

    for ax in (ax_linear, ax_log):
        ax.plot(
            alphas, theory, color="black", linestyle="--", linewidth=1.5,
            label=r"$\Phi(-1/\sqrt{\alpha})$",
        )
        ax.set_xlabel(r"$\alpha = P/N$")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    ax_linear.set_ylabel("unstable-bit fraction")
    ax_linear.set_title("(a) linear scale")
    ax_log.set_yscale("log")
    ax_log.set_ylabel("unstable-bit fraction (log scale)")
    ax_log.set_title("(b) log scale")
    fig.suptitle(
        "Hopfield network: unstable-bit fraction vs storage load "
        f"({N_SETS} pattern sets per point)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def main() -> None:
    data = run_experiment()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    figure = build_figure(data)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)

    theory = np.asarray(data["theory"])
    print("Hopfield unstable-bit fraction: mean absolute deviation from theory")
    deviations = {}
    for n_spins in N_VALUES:
        means = np.asarray(data["results"][str(n_spins)]["mean"])
        deviations[n_spins] = float(np.mean(np.abs(means - theory)))
        print(f"  N = {n_spins:>3}: mean |measured - theory| = {deviations[n_spins]:.5f}")

    print("\nSelected points (alpha: theory, measured mean +/- SE)")
    for alpha in (0.10, 0.15, 0.20, 0.25):
        index = int(np.where(np.isclose(ALPHAS, alpha))[0][0])
        rows = []
        for n_spins in N_VALUES:
            result = data["results"][str(n_spins)]
            rows.append(
                f"N={n_spins}: {result['mean'][index]:.4f}"
                f"+/-{result['se'][index]:.4f}"
            )
        print(f"  alpha={alpha:.2f} theory={theory[index]:.4f}  " + "  ".join(rows))

    # Finite-size convergence: larger N should be closer to the theory.
    assert deviations[800] < deviations[100], "N=800 is not closer than N=100"
    assert deviations[800] <= deviations[200], "N=800 is not closer than N=200"
    assert all(
        np.isfinite(data["results"][str(n)]["mean"]).all()
        and np.isfinite(data["results"][str(n)]["se"]).all()
        for n in N_VALUES
    )
    # Coupling sanity check on one explicit pattern set.
    key = jax.random.PRNGKey(SEED)
    patterns = hopfield.random_patterns(key, 10, 100)
    couplings = hopfield.build_couplings(patterns)
    symmetric = bool(np.array_equal(np.asarray(couplings), np.asarray(couplings).T))
    zero_diagonal = bool(np.allclose(np.diag(np.asarray(couplings)), 0.0))
    print(f"\n  J symmetric: {symmetric}; zero diagonal: {zero_diagonal}")
    assert symmetric and zero_diagonal
    print(f"\n  data:   {RESULTS_PATH.relative_to(WEEK_DIR)}")
    print(f"  figure: {FIGURE_PATH.relative_to(WEEK_DIR)}")


if __name__ == "__main__":
    main()
