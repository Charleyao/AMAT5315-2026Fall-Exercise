"""Shared gallery + class-frequency plotting utilities for Week 6.

Introduced for VERIFY (1) (MNIST) and reused for VERIFY (2) (CIFAR-10).

The figure has two panels, following the reference on page 4 of the Week 6
learning sheet:

* left: a labeled gallery with the first two *training* images of every class,
  shown at native resolution with nearest-neighbour rendering (no smoothing);
* right: grouped bars comparing the per-class fractions of the training and
  test splits.

The function is dataset-agnostic: MNIST passes grayscale images (``cmap="gray"``)
and numeric labels; CIFAR-10 passes RGB images (``cmap=None``) and class names.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless rendering; must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

NUM_CLASSES = 10


def first_two_per_class(
    labels: np.ndarray, num_classes: int = NUM_CLASSES
) -> dict[int, list[int]]:
    """Return the first two stored indices of each class, in file order."""
    result = {}
    for label in range(num_classes):
        indices = np.flatnonzero(labels == label)[:2]
        if len(indices) < 2:
            raise ValueError(f"class {label} has fewer than two training images")
        result[label] = [int(index) for index in indices]
    return result


def class_counts(labels: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Count the images per class."""
    return np.bincount(labels, minlength=num_classes)


def class_fractions(labels: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Empirical class frequencies; the fractions sum to 1."""
    counts = class_counts(labels, num_classes)
    return counts / counts.sum()


def build_distribution_figure(
    images: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    class_labels: list[str],
    *,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    interpolation: str = "nearest",
    gid_key: str = "class",
    suptitle: str = "",
    bar_title: str = "",
    bar_xlabel: str = "class",
    xtick_labels: list[str] | None = None,
    xtick_rotation: float = 0.0,
    gallery_width_ratio: float = 1.0,
    chart_width_ratio: float = 1.35,
    figsize: tuple[float, float] = (11.5, 6.4),
    left: float = 0.05,
    label_fontsize: float = 11,
):
    """Build the two-panel distribution figure and return ``(figure, report)``.

    ``report`` contains the exact gallery arrays (copies of the source pixels)
    and the per-split counts/fractions, so the numbers can be verified without
    reading pixels back from the rendered canvas.
    """
    num_classes = len(class_labels)
    gallery_indices = first_two_per_class(train_labels, num_classes)

    train_counts = class_counts(train_labels, num_classes)
    test_counts = class_counts(test_labels, num_classes)
    train_fractions = train_counts / train_counts.sum()
    test_fractions = test_counts / test_counts.sum()

    fig = plt.figure(figsize=figsize)
    grid = GridSpec(
        1, 2, width_ratios=[gallery_width_ratio, chart_width_ratio], wspace=0.24,
        left=left, right=0.98, top=0.87, bottom=0.13,
    )

    # --- left panel: one row per class, two native-resolution images per row
    gallery_grid = grid[0].subgridspec(num_classes, 2, wspace=0.06, hspace=0.06)
    gallery = {}
    for label in range(num_classes):
        gallery[label] = []
        for slot, index in enumerate(gallery_indices[label]):
            ax = fig.add_subplot(gallery_grid[label, slot])
            ax.imshow(
                images[index],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation=interpolation,  # no smoothing: original pixels
                aspect="equal",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            # Record which training row this axes shows (checked by the tests).
            ax.set_gid(f"{gid_key}={label};index={index}")
            if slot == 0:
                ax.set_ylabel(
                    class_labels[label],
                    rotation=0,
                    labelpad=9,
                    fontsize=label_fontsize,
                    va="center",
                    ha="right",
                )
            gallery[label].append(images[index].copy())

    # --- right panel: grouped bar chart of train vs test class fractions
    ax_bar = fig.add_subplot(grid[1])
    positions = np.arange(num_classes)
    width = 0.4
    ax_bar.bar(
        positions - width / 2, train_fractions, width,
        label="train", color="#4C72B0", edgecolor="white", linewidth=0.5,
    )
    ax_bar.bar(
        positions + width / 2, test_fractions, width,
        label="test", color="#DD8452", edgecolor="white", linewidth=0.5,
    )
    ax_bar.set_xticks(positions)
    ax_bar.set_xticklabels(
        xtick_labels if xtick_labels is not None else [str(i) for i in range(num_classes)],
        rotation=xtick_rotation,
        ha="right" if xtick_rotation else "center",
    )
    ax_bar.set_xlabel(bar_xlabel)
    ax_bar.set_ylabel("fraction of images")
    ax_bar.set_title(bar_title)
    ax_bar.set_ylim(0, max(train_fractions.max(), test_fractions.max()) * 1.28)
    ax_bar.grid(axis="y", alpha=0.3)
    ax_bar.legend(frameon=False, loc="upper right")

    if suptitle:
        fig.suptitle(suptitle, fontsize=13)

    report = {
        "gallery": gallery,
        "gallery_indices": gallery_indices,
        "train_counts": train_counts,
        "test_counts": test_counts,
        "train_fractions": train_fractions,
        "test_fractions": test_fractions,
    }
    return fig, report
