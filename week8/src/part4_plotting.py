"""Figure 13 reproduction: recall of a 32x32 picture from attention memories."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .part4_memory import METHOD_LABELS, METHODS, M_VALUES, PICTURE_SIZE

GOOD_COLOR = "#1a7f37"
BAD_COLOR = "#b03a2e"


def plot_attention_recall(
    records: dict,
    output_path: str | Path,
    m_values: tuple[int, ...] = M_VALUES,
) -> Path:
    """Draw the 4x4 grid of recalled pictures with per-panel R values."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    yy, xx = np.mgrid[0:PICTURE_SIZE, 0:PICTURE_SIZE]
    fig, axes = plt.subplots(
        len(m_values),
        len(METHODS),
        figsize=(15.5, 15.8),
        squeeze=False,
    )

    for row, m in enumerate(m_values):
        for column, method in enumerate(METHODS):
            ax = axes[row, column]
            record = records[m][method]
            image = np.clip(record["normalized_image"], -1.0, 1.0)
            ink = np.clip((image + 1.0) / 2.0, 0.0, 1.0)
            sizes = 3.0 + 30.0 * ink**2

            ax.scatter(
                xx.ravel(),
                yy.ravel(),
                s=sizes.ravel(),
                c=image.ravel(),
                cmap="gray_r",
                vmin=-1.0,
                vmax=1.0,
                marker="o",
                linewidths=0.0,
            )
            ax.set_xlim(-1.0, PICTURE_SIZE)
            ax.set_ylim(PICTURE_SIZE, -1.0)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])

            ratio = record["R"]
            good = ratio < 1.0
            color = GOOD_COLOR if good else BAD_COLOR
            status = "R < 1" if good else "R >= 1"
            ax.set_title(
                f"{METHOD_LABELS[method]}\nR = {ratio:.3f}  ({status})",
                fontsize=10.5,
                color=color,
                fontweight="bold",
                pad=6,
            )
            for spine in ax.spines.values():
                spine.set_color(color)
                spine.set_linewidth(2.0)

    for row, m in enumerate(m_values):
        position = axes[row, 0].get_position()
        fig.text(
            0.018,
            0.5 * (position.y0 + position.y1),
            f"M = {m}",
            rotation=90,
            va="center",
            ha="center",
            fontsize=15,
            fontweight="bold",
        )

    fig.suptitle(
        "Figure 13 reproduction: recall of a 32x32 character from M memories\n"
        "dots sized by normalized recalled pixel value; green frame/title = R < 1",
        fontsize=14,
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        "Target memory is index 0; query is its own key.  "
        "Ink = +1, background = -1.  "
        "Weak stores become unrecognizable as M grows; softmax stays clear.",
        ha="center",
        fontsize=8.8,
        color="0.3",
    )
    fig.tight_layout(rect=(0.04, 0.025, 1.0, 0.95))
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path
