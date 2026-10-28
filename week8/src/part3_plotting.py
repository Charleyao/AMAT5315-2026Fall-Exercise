"""Figures for Week 8 Part 3: memory stores and MLP training."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .memory import (
    MECHANISM_LABELS,
    MECHANISMS,
    EXPECTED_CAPACITIES_D64,
)

COLORS = {
    "outer_product": "#1f5fa8",
    "least_squares": "#2a8f5f",
    "relu_p1": "#d08b1f",
    "relu_p2": "#b03a2e",
    "relu_p3": "#7b3fa0",
}


def plot_memory_capacity_d64(
    m_values: np.ndarray,
    ratios_median_d: np.ndarray,
    capacities: dict[str, float],
    theoretical: dict[str, np.ndarray],
    output_path: str | Path,
    d: int = 64,
) -> Path:
    """Figure 9: R versus M for all five stores at d=64."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m_values = np.asarray(m_values, dtype=float)

    fig, ax = plt.subplots(figsize=(9.4, 6.6))
    for mechanism_index, mechanism in enumerate(MECHANISMS):
        color = COLORS[mechanism]
        measured = ratios_median_d[:, mechanism_index]
        positive = measured > 0
        ax.plot(
            m_values[positive],
            measured[positive],
            "o-",
            color=color,
            lw=1.8,
            ms=5,
            label=f"{MECHANISM_LABELS[mechanism]} "
            f"(capacity {capacities[mechanism]:.1f})",
        )
        theory = theoretical[mechanism]
        theory_positive = theory > 0
        ax.plot(
            m_values[theory_positive],
            theory[theory_positive],
            "--",
            color=color,
            lw=1.0,
            alpha=0.55,
        )
        capacity = capacities[mechanism]
        if np.isfinite(capacity) and m_values[0] <= capacity <= m_values[-1]:
            ax.plot(
                capacity,
                1.0,
                marker="x",
                color=color,
                ms=9,
                mew=2.0,
            )

    ax.axhline(1.0, color="black", lw=1.4, ls="-", label="R = 1")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of stored pairs M")
    ax.set_ylabel("interference-to-signal ratio R")
    ax.set_title(
        f"Figure 9 reproduction: memory capacity at d={d}\n"
        "solid: measured median of 3 draws; dashed: theoretical prediction"
    )
    ax.grid(alpha=0.22, which="both")
    ax.legend(fontsize=8.3, frameon=False, loc="upper left")
    ax.text(
        0.98,
        0.03,
        "R = ||interference|| / ||signal|| = sqrt(energy ratio)\n"
        "crosses use log-log interpolation",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.35",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_capacity_vs_dimension(
    capacities: dict[int, dict[str, float]],
    slopes: dict[str, float],
    output_path: str | Path,
) -> Path:
    """Figure 10: fitted memory capacity scaling with dimension."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dimensions = np.array(sorted(capacities), dtype=float)

    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    for mechanism in MECHANISMS:
        color = COLORS[mechanism]
        values = np.array(
            [capacities[int(d)][mechanism] for d in dimensions],
            dtype=float,
        )
        slope = slopes[mechanism]
        intercept = np.mean(
            np.log(values) - slope * np.log(dimensions)
        )
        fitted = np.exp(intercept) * dimensions**slope
        ax.plot(
            dimensions,
            values,
            "o",
            color=color,
            ms=7,
            label=f"{MECHANISM_LABELS[mechanism]}: slope {slope:.2f}",
        )
        ax.plot(
            dimensions,
            fitted,
            "--",
            color=color,
            lw=1.3,
            alpha=0.8,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(dimensions)
    ax.set_xticklabels([str(int(d)) for d in dimensions])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("dimension d")
    ax.set_ylabel("capacity (M where R crosses 1)")
    ax.set_title(
        "Figure 10 reproduction: memory capacity versus dimension\n"
        "markers: measured capacities; dashed: fitted power law"
    )
    ax.grid(alpha=0.22, which="both")
    ax.legend(fontsize=8.7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_mlp_training(
    history: dict,
    final_validation_loss: float,
    bigram_reference: float,
    output_path: str | Path,
) -> Path:
    """Training/validation loss curve with the Part 2 bigram reference."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    steps = np.asarray(history["step"], dtype=float)
    train_loss = np.asarray(history["train_loss"], dtype=float)
    val_steps = np.asarray(history["validation_step"], dtype=float)
    val_loss = np.asarray(history["validation_loss"], dtype=float)

    window = 50
    if len(train_loss) >= window:
        kernel = np.ones(window) / window
        smooth = np.convolve(train_loss, kernel, mode="valid")
        smooth_steps = steps[window - 1 :]
    else:
        smooth = train_loss
        smooth_steps = steps

    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    ax.plot(steps, train_loss, color="0.65", lw=0.6, alpha=0.55, label="train (raw)")
    ax.plot(
        smooth_steps,
        smooth,
        color="#1f5fa8",
        lw=1.8,
        label=f"train ({window}-step moving average)",
    )
    ax.plot(
        val_steps,
        val_loss,
        "o-",
        color="#b03a2e",
        lw=1.5,
        ms=3.5,
        label="validation",
    )
    ax.axhline(
        bigram_reference,
        color="black",
        ls="--",
        lw=1.2,
        label=f"Part 2 bigram = {bigram_reference:.3f}",
    )
    ax.axhline(
        final_validation_loss,
        color="#2a8f5f",
        ls=":",
        lw=1.2,
        label=f"final MLP = {final_validation_loss:.4f}",
    )
    ax.set_xlim(100, max(steps.max(), 1))
    ax.set_ylim(2.98, 3.16)
    ax.set_xlabel("training step")
    ax.set_ylabel("cross-entropy (nats/token)")
    ax.set_title("Previous-token MLP: training and held-out loss")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8.6, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_expected_capacity_table(
    capacities: dict[str, float],
    output_path: str | Path,
) -> Path:
    """Simple bar chart of d=64 capacities against reference values."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [MECHANISM_LABELS[mechanism] for mechanism in MECHANISMS]
    measured = np.array([capacities[m] for m in MECHANISMS])
    expected = np.array(
        [EXPECTED_CAPACITIES_D64[m] for m in MECHANISMS]
    )
    positions = np.arange(len(MECHANISMS))
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.bar(positions - 0.18, measured, width=0.36, color="#1f5fa8", label="measured")
    ax.bar(positions + 0.18, expected, width=0.36, color="0.7", label="reference")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("capacity")
    ax.set_title("Measured versus reference capacity at d=64")
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path
