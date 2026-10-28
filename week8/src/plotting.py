"""Figure 5 reproduction: relational cosine fans for three word relations."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, reproducible figure generation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .relations import RELATION_TITLES


def _arrow_endpoint(cosine: float, side: str) -> tuple[float, float]:
    """Map a cosine to a unit arrow angle: angle = arccos(cosine).

    Pretrained arrows lie above the horizontal axis, shuffled-control arrows
    below it, so the same angle magnitude is directly comparable.
    """
    cosine = float(np.clip(cosine, -1.0, 1.0))
    y = float(np.sqrt(max(0.0, 1.0 - cosine**2)))
    return cosine, y if side == "above" else -y


def _spread_labels(ys: np.ndarray, side: str) -> np.ndarray:
    """Assign non-overlapping label heights, preserving the vertical order."""
    order = np.argsort(ys)
    if side == "above":
        slots = np.linspace(0.11, 0.96, len(ys))
    else:
        slots = -np.linspace(0.96, 0.11, len(ys))
    positions = np.empty(len(ys), dtype=float)
    positions[order] = slots
    return positions


def _draw_fan(
    ax: plt.Axes,
    result: dict,
    *,
    side: str,
    color: str,
) -> None:
    """Draw nine unit arrows whose angles encode the nine cosines."""
    cosines = np.asarray(result["cosines"], dtype=float)
    endpoints = np.array([_arrow_endpoint(c, side) for c in cosines])
    label_y = _spread_labels(endpoints[:, 1], side)

    for pair, (x, y), text_y, cosine in zip(
        result["pairs"], endpoints, label_y, cosines
    ):
        ax.annotate(
            "",
            xy=(x, y),
            xytext=(0.0, 0.0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=1.25,
                alpha=0.85,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=9,
            ),
            zorder=3,
        )
        label = f"{pair['a']}-{pair['b']}  {cosine:+.2f}"
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(1.12, text_y),
            fontsize=6.6,
            color=color,
            va="center",
            ha="left",
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                lw=0.55,
                alpha=0.45,
                shrinkA=0,
                shrinkB=1,
            ),
            zorder=4,
        )

    # Black dashed arrow: the group mean, included for visual comparison.
    mean = float(result["mean_cosine"])
    mean_x, mean_y = _arrow_endpoint(mean, side)
    ax.annotate(
        "",
        xy=(mean_x, mean_y),
        xytext=(0.0, 0.0),
        arrowprops=dict(
            arrowstyle="-|>",
            color="black",
            lw=2.0,
            ls="--",
            shrinkA=0,
            shrinkB=0,
            mutation_scale=12,
        ),
        zorder=5,
    )


def plot_relation_fans(
    pretrained: dict[str, dict],
    shuffled: dict[str, dict],
    output_path: str | Path,
) -> Path:
    """Create the three-panel Figure 5 reproduction and save it."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 8.4))
    theta = np.linspace(-np.pi / 2, np.pi / 2, 500)

    for ax, relation in zip(axes, pretrained):
        title = RELATION_TITLES[relation]
        ax.plot(np.cos(theta), np.sin(theta), color="0.55", lw=0.7, ls=":")
        ax.axhline(0.0, color="0.15", lw=1.0)
        ax.axvline(0.0, color="0.15", lw=0.8)

        _draw_fan(ax, pretrained[relation], side="above", color="#1f5fa8")
        _draw_fan(ax, shuffled[relation], side="below", color="#b03a2e")

        ax.set_title(
            f"{title}\n"
            f"pretrained mean = {pretrained[relation]['mean_cosine']:.3f}  |  "
            f"shuffled mean = {shuffled[relation]['mean_cosine']:.3f}",
            fontsize=11,
            pad=10,
        )
        ax.set_xlim(-0.06, 1.52)
        ax.set_ylim(-1.08, 1.08)
        ax.set_aspect("equal")
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_xlabel("cos(angle) = leave-one-out cosine", fontsize=9)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.grid(axis="x", color="0.9", lw=0.6)

    fig.suptitle(
        "Figure 5 reproduction: relational offsets in potion-base-8M\n"
        "Unit arrows: angle = arccos(leave-one-out cosine); "
        "above = pretrained table, below = row-shuffled control (seed 1730)",
        fontsize=12.5,
        y=0.985,
    )
    handles = [
        Line2D([0], [0], color="#1f5fa8", lw=1.8, label="pretrained (above axis)"),
        Line2D([0], [0], color="#b03a2e", lw=1.8, label="shuffled control (below axis)"),
        Line2D([0], [0], color="black", lw=2.0, ls="--", label="group mean"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.895))
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path
