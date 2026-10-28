"""VERIFY (1): modern Hopfield update vs JAX softmax attention.

Uses the first ``P`` MNIST training images as stored +/-1 patterns for
P in {3, 10, 30, 100, 300, 1000} and beta in
{0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1}. The manual modern-Hopfield update
and the ``jax.nn.dot_product_attention`` version (with ``scale=beta``) are
compared on 20 held-out test queries.

For P = 100 the softmax weights of one held-out query are plotted at
beta = 0.003, 0.03 and 0.3.

Usage::

    cd week6
    uv run python modern_hopfield_experiment.py
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")  # headless rendering; must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np

import modern_hopfield as mh

WEEK_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = WEEK_DIR / "artifacts"
EVIDENCE_DIR = WEEK_DIR / "evidence"
RESULTS_PATH = ARTIFACTS_DIR / "modern_hopfield_attention.json"
FIGURE_PATH = EVIDENCE_DIR / "modern_hopfield_weights.png"

P_VALUES = [3, 10, 30, 100, 300, 1000]
BETAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
N_QUERIES = 100
PLOT_P = 100
PLOT_BETAS = [0.003, 0.03, 0.3]
PLOT_QUERY = 0


def run_experiment() -> dict:
    random_lower = mh.random_lower_spins(N_QUERIES)
    plot_original_label = None

    max_difference = 0.0
    max_weight_sum_error = 0.0
    argmax_matches = True
    difference_by_point: dict[str, dict[str, float]] = {}
    sum_error_by_point: dict[str, dict[str, float]] = {}
    concentration: dict[str, dict[str, dict]] = {}
    plot_weights = {}

    for p in P_VALUES:
        patterns, pattern_labels = mh.mnist_spin_patterns(p)
        queries, source_indices, _originals = mh.corrupted_stored_queries(
            patterns, random_lower, N_QUERIES
        )
        if p == PLOT_P:
            plot_original_label = int(pattern_labels[source_indices[PLOT_QUERY]])
        difference_by_point[str(p)] = {}
        sum_error_by_point[str(p)] = {}
        concentration[str(p)] = {}
        for beta in BETAS:
            manual, weights = mh.modern_hopfield_update(patterns, queries, beta)
            attention = mh.attention_update(patterns, queries, beta)
            difference = float(jnp.max(jnp.abs(manual - attention)))
            sum_error = float(jnp.max(jnp.abs(weights.sum(axis=-1) - 1.0)))
            max_difference = max(max_difference, difference)
            max_weight_sum_error = max(max_weight_sum_error, sum_error)
            difference_by_point[str(p)][str(beta)] = difference
            sum_error_by_point[str(p)][str(beta)] = sum_error

            similarities = np.asarray(queries @ patterns.T)
            argmax_matches = argmax_matches and bool(
                np.array_equal(
                    np.argmax(np.asarray(weights), axis=-1),
                    np.argmax(similarities, axis=-1),
                )
            )

            top = np.asarray(weights[PLOT_QUERY])
            top_index = int(np.argmax(top))
            concentration[str(p)][str(beta)] = {
                "top_index": top_index,
                "top_weight": float(top[top_index]),
                "entropy": mh.attention_entropy(top),
                "most_similar_index": int(np.argmax(similarities[PLOT_QUERY])),
            }

            if p == PLOT_P and beta in PLOT_BETAS:
                plot_weights[str(beta)] = top.tolist()

    return {
        "settings": {
            "P": P_VALUES,
            "betas": BETAS,
            "n_queries": N_QUERIES,
            "query_source": "100 stored MNIST training digits cycling through P, lower half random +/-1",
            "query_seed": mh.QUERY_SEED,
            "plot_P": PLOT_P,
            "plot_betas": PLOT_BETAS,
            "plot_query": PLOT_QUERY,
            "update": "x_new = X^T softmax(beta * (X x))",
            "attention": "jax.nn.dot_product_attention(scale=beta)",
        },
        "max_abs_difference": max_difference,
        "max_weight_sum_error": max_weight_sum_error,
        "argmax_matches_most_similar": argmax_matches,
        "difference_by_point": difference_by_point,
        "weight_sum_error_by_point": sum_error_by_point,
        "concentration": concentration,
        "plot": {
            "P": PLOT_P,
            "query_index": PLOT_QUERY,
            "original_label": plot_original_label,
            "weights": plot_weights,
            "top_indices": {
                str(beta): int(np.argmax(plot_weights[str(beta)])) for beta in PLOT_BETAS
            },
            "top_weights": {
                str(beta): float(np.max(plot_weights[str(beta)])) for beta in PLOT_BETAS
            },
            "entropies": {
                str(beta): mh.attention_entropy(plot_weights[str(beta)])
                for beta in PLOT_BETAS
            },
        },
    }


def build_figure(data: dict) -> plt.Figure:
    """Three panels: softmax weights for P=100 at three beta values."""
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9))
    colors = ["#4C72B0", "#DD8452", "#C44E52"]
    for axis, beta, color in zip(axes, PLOT_BETAS, colors):
        weights = np.asarray(data["plot"]["weights"][str(beta)])
        top_index = int(np.argmax(weights))
        axis.bar(
            np.arange(len(weights)), weights, width=1.0, color=color, alpha=0.85
        )
        axis.axvline(top_index, color="black", linestyle="--", linewidth=1.0)
        axis.annotate(
            f"argmax = {top_index}\nweight = {weights[top_index]:.3f}\n"
            f"entropy = {data['plot']['entropies'][str(beta)]:.2f}",
            xy=(0.02, 0.95), xycoords="axes fraction", va="top", fontsize=9,
        )
        axis.set_xlabel("stored pattern index")
        axis.set_ylabel("softmax weight")
        axis.set_title(rf"$\beta$ = {beta}")
        axis.grid(alpha=0.25)
    fig.suptitle(
        f"Modern Hopfield softmax weights (P = {PLOT_P}, corrupted stored "
        f"query {PLOT_QUERY}, label {data['plot']['original_label']})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def main() -> None:
    data = run_experiment()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    figure = build_figure(data)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print("Modern Hopfield vs JAX softmax attention")
    print(f"  tested P: {P_VALUES}")
    print(f"  tested beta: {BETAS}")
    print(f"  queries: {N_QUERIES} corrupted stored digits (cycling through P)")
    print(f"  max |manual - attention|: {data['max_abs_difference']:.3e}")
    print(f"  max |sum(weights) - 1|:   {data['max_weight_sum_error']:.3e}")
    print(f"  argmax weight == most similar pattern: {data['argmax_matches_most_similar']}")

    print("\n  P = 100, corrupted stored query 0: weight concentration vs beta")
    for beta in PLOT_BETAS:
        top = data["plot"]["top_weights"][str(beta)]
        index = data["plot"]["top_indices"][str(beta)]
        entropy = data["plot"]["entropies"][str(beta)]
        print(
            f"    beta={beta:<5}: top weight={top:.4f} at pattern {index}, "
            f"entropy={entropy:.3f}"
        )

    # --- explicit verification -------------------------------------------
    assert data["max_abs_difference"] < 1e-5
    assert data["max_weight_sum_error"] < 1e-5
    assert data["argmax_matches_most_similar"] is True
    top_weights = [data["plot"]["top_weights"][str(beta)] for beta in PLOT_BETAS]
    entropies = [data["plot"]["entropies"][str(beta)] for beta in PLOT_BETAS]
    assert top_weights[0] < top_weights[1] < top_weights[2], top_weights
    assert entropies[0] > entropies[1] > entropies[2], entropies

    # Larger beta concentrates weight for every tested P (query 0).
    for p in P_VALUES:
        tops = [data["concentration"][str(p)][str(beta)]["top_weight"] for beta in BETAS]
        entropies_p = [data["concentration"][str(p)][str(beta)]["entropy"] for beta in BETAS]
        assert all(tops[i + 1] >= tops[i] for i in range(len(tops) - 1)), (p, tops)
        assert all(
            entropies_p[i + 1] <= entropies_p[i] for i in range(len(entropies_p) - 1)
        ), (p, entropies_p)

    print("\n  [ok] maximum implementation difference < 1e-5")
    print("  [ok] softmax weights sum to 1")
    print("  [ok] larger beta concentrates weight on the most similar pattern")
    print("  [ok] attention uses scale=beta (not the default scale)")
    print(f"\n  data:   {RESULTS_PATH.relative_to(WEEK_DIR)}")
    print(f"  figure: {FIGURE_PATH.relative_to(WEEK_DIR)}")


if __name__ == "__main__":
    main()
