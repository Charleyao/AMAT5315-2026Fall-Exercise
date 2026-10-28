"""Figures for Week 8 Part 5: Transformer comparison, context, and sampling."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "Uniform": "0.55",
    "Unigram": "#8a6d3b",
    "Bigram": "#d08b1f",
    "Trigram": "#b03a2e",
    "Transformer": "#1f5fa8",
}


def plot_model_bpc_comparison(
    names: list[str],
    bpc_values: list[float],
    reference_bpc: float,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    positions = np.arange(len(names))
    colors = [COLORS.get(name, "#1f5fa8") for name in names]

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    bars = ax.bar(positions, bpc_values, color=colors, width=0.68)
    for bar, value in zip(bars, bpc_values):
        ax.annotate(
            f"{value:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ax.axhline(
        reference_bpc,
        color="#b03a2e",
        ls="--",
        lw=1.3,
        label=f"acceptance threshold = {reference_bpc:.3f}",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(names)
    ax.set_ylabel("held-out bits per character")
    ax.set_title("VERIFY (1): held-out bits/char by model")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_context_length(
    position_loss: np.ndarray,
    bigram_reference: float,
    trigram_reference: float,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    positions = np.arange(1, len(position_loss) + 1)
    final8 = float(np.mean(position_loss[-8:]))

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.plot(
        positions,
        position_loss,
        "o-",
        color="#1f5fa8",
        lw=2.0,
        ms=4,
        label="Transformer",
    )
    ax.axhline(
        bigram_reference,
        color="#d08b1f",
        ls="--",
        lw=1.3,
        label=f"Bigram = {bigram_reference:.3f}",
    )
    ax.axhline(
        trigram_reference,
        color="#b03a2e",
        ls="--",
        lw=1.3,
        label=f"Trigram = {trigram_reference:.3f}",
    )
    ax.axvline(32, color="0.35", ls=":", lw=1.0)
    ax.scatter(
        positions[-8:],
        position_loss[-8:],
        color="#1a7f37",
        s=22,
        zorder=4,
        label=f"final 8 (mean {final8:.3f})",
    )
    ax.set_xlabel("context length (number of tokens read)")
    ax.set_ylabel("held-out cross-entropy (nats/token)")
    ax.set_title("Figure 17 reproduction: Transformer loss versus context length")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_transformer_training(history: dict, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    steps = np.asarray(history["step"])
    train_loss = np.asarray(history["train_loss"])
    val_steps = np.asarray(history["validation_step"])
    val_loss = np.asarray(history["validation_loss"])

    window = 100
    kernel = np.ones(window) / window
    smooth = np.convolve(train_loss, kernel, mode="valid")
    smooth_steps = steps[window - 1 :]

    fig, (left, right) = plt.subplots(1, 2, figsize=(14.0, 5.5))
    left.plot(steps, train_loss, color="0.7", lw=0.5, alpha=0.55)
    left.plot(smooth_steps, smooth, color="#1f5fa8", lw=1.8, label="train")
    left.plot(
        val_steps,
        val_loss,
        "o-",
        color="#b03a2e",
        lw=1.4,
        ms=4,
        label="validation",
    )
    left.set_xlabel("step")
    left.set_ylabel("cross-entropy (nats/token)")
    left.set_title("Training and validation loss")
    left.grid(alpha=0.22)
    left.legend(frameon=False)

    val_bpc = np.asarray(history["validation_bpc"])
    right.plot(
        val_steps,
        val_bpc,
        "o-",
        color="#1f5fa8",
        lw=1.6,
        ms=4,
    )
    right.axhline(1.310, color="#b03a2e", ls="--", lw=1.2)
    right.set_xlabel("step")
    right.set_ylabel("validation bits/char")
    right.set_title("Held-out bits per character")
    right.grid(alpha=0.22)
    fig.suptitle("Previous-token Transformer training", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_text_samples(
    samples: list[dict],
    output_path: str | Path,
    suptitle: str,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(samples), 1, figsize=(12.0, 1.45 * len(samples) + 1.0)
    )
    if len(samples) == 1:
        axes = [axes]
    for ax, sample in zip(axes, samples):
        ax.axis("off")
        ax.text(
            0.0,
            0.93,
            sample["label"],
            fontsize=10,
            fontweight="bold",
            va="top",
            color="#1f5fa8",
        )
        ax.text(
            0.0,
            0.58,
            textwrap.fill(sample["text"], width=120),
            fontsize=9,
            va="top",
            family="monospace",
            color="0.15",
        )
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def plot_verify4(
    surprise_t1: np.ndarray,
    entropy_t1: np.ndarray,
    surprise_t07: np.ndarray,
    entropy_t07: np.ndarray,
    z_t1: float,
    z_t07: float,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for ax, surprise, entropy, z, title in (
        (
            left,
            surprise_t1,
            entropy_t1,
            z_t1,
            "Generated at T=1",
        ),
        (
            right,
            surprise_t07,
            entropy_t07,
            z_t07,
            "Generated at T=0.7, scored at T=1",
        ),
    ):
        ax.scatter(entropy, surprise, s=18, alpha=0.7, color="#1f5fa8")
        limits = [
            min(entropy.min(), surprise.min()),
            max(entropy.max(), surprise.max()),
        ]
        ax.plot(limits, limits, color="0.3", ls="--", lw=1.0, label="surprise = entropy")
        ax.set_xlabel("mean predicted entropy (nats/token)")
        ax.set_ylabel("mean surprise (nats/token)")
        ax.set_title(f"{title}, z = {z:.2f}")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, fontsize=8.5)
    fig.suptitle(
        "VERIFY (4): per-sequence surprise versus predicted entropy",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path
