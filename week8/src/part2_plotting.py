"""Figures for Week 8 Part 2: curse of dimensionality and bigram goodness-of-fit."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REFERENCE_BPC = {3: 1.473, 4: 1.310, 5: 1.46, 6: 1.79}
REFERENCE_UNSEEN = {3: 0.006, 4: 0.042, 5: 0.122, 6: 0.235}


def plot_curse_of_dimensionality(metrics: dict[int, dict], output_path: str | Path) -> Path:
    """Plot held-out bits/char and unseen n-gram fraction versus order."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    orders = sorted(metrics)
    bpc = np.array([metrics[n]["bits_per_char"] for n in orders])
    unseen = 100.0 * np.array([metrics[n]["unseen_fraction"] for n in orders])
    reference_orders = sorted(REFERENCE_BPC)
    reference_bpc = np.array([REFERENCE_BPC[n] for n in reference_orders])
    reference_unseen = 100.0 * np.array(
        [REFERENCE_UNSEEN[n] for n in reference_orders]
    )
    minimum_order = orders[int(np.argmin(bpc))]

    fig, (left, right) = plt.subplots(1, 2, figsize=(13.2, 5.4))

    left.plot(orders, bpc, "o-", color="#1f5fa8", lw=2.0, label="measured")
    left.plot(
        reference_orders,
        reference_bpc,
        "x--",
        color="0.45",
        ms=8,
        lw=1.0,
        label="reference",
    )
    left.axvline(
        minimum_order,
        color="#1f5fa8",
        ls=":",
        lw=1.2,
        alpha=0.7,
    )
    for n, value in zip(orders, bpc):
        left.annotate(
            f"{value:.3f}",
            xy=(n, value),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#1f5fa8",
        )
    left.set_title(f"Held-out loss (minimum at n={minimum_order})")
    left.set_xlabel("n-gram order n")
    left.set_ylabel("bits per character")
    left.set_xticks(orders)
    left.set_ylim(float(bpc.min()) - 0.14, float(bpc.max()) + 0.14)
    left.grid(alpha=0.25)
    left.legend(frameon=False, fontsize=9)

    right.plot(orders, unseen, "s-", color="#b03a2e", lw=2.0, label="measured")
    right.plot(
        reference_orders,
        reference_unseen,
        "x--",
        color="0.45",
        ms=8,
        lw=1.0,
        label="reference",
    )
    for n, value in zip(orders, unseen):
        right.annotate(
            f"{value:.2f}%",
            xy=(n, value),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#b03a2e",
        )
    right.set_title("Held-out n-grams never seen in training")
    right.set_xlabel("n-gram order n")
    right.set_ylabel("unseen fraction (%)")
    right.set_xticks(orders)
    right.set_ylim(-1.0, float(unseen.max()) + 2.0)
    right.grid(alpha=0.25)
    right.legend(frameon=False, fontsize=9)

    fig.suptitle(
        "Figure 7 reproduction: the curse of dimensionality in n-gram models",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_chi_square_diagnostics(
    observed: np.ndarray,
    expected: np.ndarray,
    per_row_statistic: np.ndarray,
    context_pieces: list[str],
    chi_result: dict,
    output_path: str | Path,
) -> Path:
    """Plot pooled observed-vs-expected counts and per-context contributions."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pooled_observed = observed.sum(axis=0)
    pooled_expected = expected.sum(axis=0)
    positive = (pooled_observed > 0) & (pooled_expected > 0)

    fig, (left, right) = plt.subplots(1, 2, figsize=(14.0, 5.6))
    left.scatter(
        pooled_expected[positive],
        pooled_observed[positive],
        s=12,
        alpha=0.55,
        color="#1f5fa8",
        edgecolors="none",
    )
    limit = [
        min(pooled_expected[positive].min(), pooled_observed[positive].min()),
        max(pooled_expected[positive].max(), pooled_observed[positive].max()),
    ]
    left.plot(limit, limit, color="0.3", ls="--", lw=1.0, label="y = x")
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel("expected count (pooled over 20 contexts)")
    left.set_ylabel("sampled count")
    left.set_title("Pooled next-token counts")
    left.legend(frameon=False, fontsize=9)
    left.grid(alpha=0.2, which="both")

    positions = np.arange(len(per_row_statistic))
    right.bar(
        positions,
        per_row_statistic,
        color="#b03a2e",
        alpha=0.8,
    )
    right.set_xticks(positions)
    right.set_xticklabels(context_pieces, rotation=90, fontsize=7)
    right.set_ylabel("per-context chi-squared contribution")
    right.set_title("Fit by conditioning context")
    right.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "VERIFY (3): bigram ancestral sampling vs model probabilities\n"
        f"pooled chi-squared = {chi_result['statistic']:.1f}, "
        f"dof = {chi_result['dof']}, p = {chi_result['p_value']:.4f}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path
