"""VERIFY (2): storage capacity of the Hopfield network from recall.

For each (N, alpha) point the same 20 random pattern sets as VERIFY (1) are
used (same seed 5315). For every set, recall is started from the first
``min(10, P)`` stored patterns and run with zero-temperature asynchronous
updates in random order until a full sweep produces no flips.

Recorded per set: mean final overlap with the original pattern and the recall
success rate (overlap >= 0.9). Threshold crossings are the alpha values where
the mean final overlap falls through 0.9; the sheet references are
0.186, 0.170, 0.156, 0.150 for N = 100, 200, 400, 800.

Usage::

    cd week6
    uv run python hopfield_recall.py
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
RESULTS_PATH = ARTIFACTS_DIR / "hopfield_recall_capacity.json"
FIGURE_PATH = EVIDENCE_DIR / "hopfield_recall_capacity.png"

N_VALUES = [100, 200, 400, 800]
ALPHAS = np.round(np.arange(0.05, 0.2501, 0.01), 2)
N_SETS = 20
SEED = 5315  # same seed/pattern sets as VERIFY (1)
STARTS_PER_SET = 10
SUCCESS_OVERLAP = 0.9
ALPHA_C = 0.138
REFERENCE_CROSSINGS = {100: 0.186, 200: 0.170, 400: 0.156, 800: 0.150}
COLORS = {100: "#1f77b4", 200: "#2ca02c", 400: "#ff7f0e", 800: "#d62728"}


def threshold_crossing(x: np.ndarray, y: np.ndarray, level: float) -> float:
    """First linear-interpolated alpha where the curve ``y`` crosses ``level``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    for i in range(len(y) - 1):
        if y[i] == y[i + 1]:
            continue
        if (y[i] - level) * (y[i + 1] - level) <= 0.0:
            return float(
                x[i] + (level - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
            )
    return float("nan")


def run_point(n_spins: int, alpha: float, key: jax.Array) -> dict:
    """Recall from the first stored patterns for 20 pattern sets."""
    n_patterns = int(round(alpha * n_spins))
    bits = jax.random.bernoulli(key, 0.5, (N_SETS, n_patterns, n_spins))
    patterns = jnp.where(bits, 1.0, -1.0)
    starts = min(STARTS_PER_SET, n_patterns)

    per_set_overlap = []
    per_set_success = []
    all_overlaps = []
    for set_index in range(N_SETS):
        J = hopfield.build_couplings(patterns[set_index])
        initial = patterns[set_index, :starts]
        final, _sweeps, _converged = hopfield.async_recall_batch(
            J, initial, jax.random.fold_in(key, 10_000 + set_index)
        )
        overlaps = np.asarray(jnp.mean(final * initial, axis=1), dtype=np.float64)
        all_overlaps.append(overlaps)
        per_set_overlap.append(float(overlaps.mean()))
        per_set_success.append(float((overlaps >= SUCCESS_OVERLAP).mean()))
    return {
        "overlap": np.asarray(per_set_overlap),
        "success": np.asarray(per_set_success),
        "all_overlaps": np.concatenate(all_overlaps),
        "starts": starts,
    }


def run_experiment() -> dict:
    results = {}
    for n_index, n_spins in enumerate(N_VALUES):
        overlap_mean, overlap_se, success_mean, success_se = [], [], [], []
        per_set_overlap, per_set_success = [], []
        for alpha_index, alpha in enumerate(ALPHAS):
            key = jax.random.fold_in(
                jax.random.PRNGKey(SEED), n_index * 1000 + alpha_index
            )
            point = run_point(n_spins, float(alpha), key)
            overlap_mean.append(float(point["overlap"].mean()))
            overlap_se.append(
                float(point["overlap"].std(ddof=1) / np.sqrt(N_SETS))
            )
            success_mean.append(float(point["success"].mean()))
            success_se.append(
                float(point["success"].std(ddof=1) / np.sqrt(N_SETS))
            )
            per_set_overlap.append(point["overlap"].tolist())
            per_set_success.append(point["success"].tolist())

        results[str(n_spins)] = {
            "mean_overlap": overlap_mean,
            "se_overlap": overlap_se,
            "success": success_mean,  # P(overlap >= 0.9)
            "se_success": success_se,
            "per_set_overlap": per_set_overlap,  # [alpha][set]
            "per_set_success": per_set_success,
            "crossing": threshold_crossing(ALPHAS, overlap_mean, SUCCESS_OVERLAP),
            "reference_crossing": REFERENCE_CROSSINGS[n_spins],
        }

    return {
        "settings": {
            "n_spins": N_VALUES,
            "alpha": ALPHAS.tolist(),
            "pattern_sets_per_point": N_SETS,
            "starts_per_set": f"min({STARTS_PER_SET}, P)",
            "success_overlap_threshold": SUCCESS_OVERLAP,
            "seed": SEED,
            "pattern_sets_reused_from": "VERIFY (1), same seed and keys",
            "threshold_crossing": "alpha where mean final overlap crosses 0.9",
            "alpha_c": ALPHA_C,
            "reference_crossings": REFERENCE_CROSSINGS,
        },
        "results": results,
    }


def build_figure(data: dict) -> plt.Figure:
    """Panel A: mean final overlap; panel B: recall success probability."""
    alphas = np.asarray(data["settings"]["alpha"])
    fig, (ax_overlap, ax_success) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    for n_spins in N_VALUES:
        result = data["results"][str(n_spins)]
        color = COLORS[n_spins]
        ax_overlap.errorbar(
            alphas, result["mean_overlap"], yerr=result["se_overlap"],
            color=color, marker="o", markersize=3, linewidth=1.2, capsize=2,
            label=f"N = {n_spins}",
        )
        ax_success.errorbar(
            alphas, result["success"], yerr=result["se_success"],
            color=color, marker="o", markersize=3, linewidth=1.2, capsize=2,
            label=f"N = {n_spins}",
        )
        crossing = result["crossing"]
        if np.isfinite(crossing):
            ax_overlap.axvline(
                crossing, color=color, linestyle=":", linewidth=1.0, alpha=0.7
            )
            ax_overlap.plot(
                [crossing], [SUCCESS_OVERLAP], marker="x", color=color, markersize=7
            )

    for axis in (ax_overlap, ax_success):
        axis.axvline(
            ALPHA_C, color="black", linestyle="--", linewidth=1.3,
            label=rf"$\alpha_c = {ALPHA_C}$",
        )
        axis.set_xlabel(r"$\alpha = P/N$")
        axis.grid(alpha=0.3)
        axis.legend(frameon=False, fontsize=8)

    ax_overlap.axhline(SUCCESS_OVERLAP, color="#777777", linestyle=":", linewidth=1.0)
    ax_overlap.set_ylabel("mean final overlap with original pattern")
    ax_overlap.set_title("(a) Final overlap (crossing at 0.9 marked)")
    ax_success.set_ylabel("recall success probability (overlap >= 0.9)")
    ax_success.set_title("(b) Recall success probability")
    fig.suptitle(
        "Hopfield recall capacity: 20 pattern sets, first 10 stored patterns "
        "per set",
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

    print("Hopfield recall capacity: mean final overlap and success (overlap >= 0.9)")
    print(f"{'N':>4} {'crossing':>9} {'reference':>10} {'|diff|':>7} {'success@0.05':>13} {'success@0.25':>13}")
    crossings = {}
    for n_spins in N_VALUES:
        result = data["results"][str(n_spins)]
        crossing = result["crossing"]
        reference = result["reference_crossing"]
        crossings[n_spins] = crossing
        print(
            f"{n_spins:>4} {crossing:>9.4f} {reference:>10.3f} "
            f"{abs(crossing - reference):>7.4f} "
            f"{result['success'][0]:>13.3f} {result['success'][-1]:>13.3f}"
        )

    print("\nAt alpha = 0.20 (near/above capacity):")
    index = int(np.where(np.isclose(ALPHAS, 0.20))[0][0])
    for n_spins in N_VALUES:
        result = data["results"][str(n_spins)]
        print(
            f"  N={n_spins:>3}: overlap={result['mean_overlap'][index]:.3f}"
            f"+/-{result['se_overlap'][index]:.3f}  "
            f"success={result['success'][index]:.3f}"
        )

    # --- explicit verification -------------------------------------------
    for n_spins in N_VALUES:
        result = data["results"][str(n_spins)]
        crossing = result["crossing"]
        reference = result["reference_crossing"]
        assert np.isfinite(crossing), f"N={n_spins}: no crossing found"
        assert abs(crossing - reference) < 0.02, (
            f"N={n_spins}: crossing {crossing:.4f} differs from reference "
            f"{reference} by >= 0.02"
        )
        assert result["success"][0] >= 0.95, f"N={n_spins}: low-alpha recall is not near perfect"
        assert result["success"][-1] < result["success"][0] - 0.15, (
            f"N={n_spins}: success did not decrease near capacity"
        )
        assert np.all(np.isfinite(result["mean_overlap"]))
        assert np.all(np.isfinite(result["success"]))

    ordered = [crossings[n] for n in N_VALUES]
    assert all(
        ordered[i + 1] < ordered[i] for i in range(len(ordered) - 1)
    ), f"crossings do not decrease with N: {ordered}"

    # Sharper transition with N: width between the 0.9 and 0.5 success levels.
    widths = {}
    for n_spins in N_VALUES:
        success = data["results"][str(n_spins)]["success"]
        widths[n_spins] = threshold_crossing(ALPHAS, success, 0.5) - threshold_crossing(
            ALPHAS, success, 0.9
        )
    print("\nTransition width (alpha at success 0.5 minus alpha at success 0.9):")
    for n_spins in N_VALUES:
        print(f"  N={n_spins:>3}: {widths[n_spins]:.4f}")
    assert widths[800] < widths[100], "transition did not sharpen with N"
    assert np.isfinite(widths[800]) and np.isfinite(widths[100])

    print("\n  [ok] recall near perfect at low alpha")
    print("  [ok] overlap/success decrease near capacity")
    print("  [ok] threshold crossings decrease with N")
    print("  [ok] each crossing within 0.02 of the reference")
    print("  [ok] transition sharpens for larger N")
    print(f"\n  data:   {RESULTS_PATH.relative_to(WEEK_DIR)}")
    print(f"  figure: {FIGURE_PATH.relative_to(WEEK_DIR)}")


if __name__ == "__main__":
    main()
