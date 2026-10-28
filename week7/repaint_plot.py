"""Part 5 VERIFY (1) RePaint completion and variance figure."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from cifar_ddpm import display_images
from cifar_verify1_plot import CIFAR10_CLASSES


def save_repaint_figure(
    originals: np.ndarray,
    completions: np.ndarray,
    variance: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    seeds: tuple[int, ...],
    output_path: str,
) -> None:
    """Show original, masked input, four completions, and pixelwise variance."""
    originals = display_images(originals)
    completions = display_images(completions)
    variance = np.asarray(variance, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    hidden = 1.0 - mask
    num_images, num_completions = completions.shape[:2]
    variance_map = variance.mean(axis=-1)
    vmax = max(float(variance_map.max()), 1e-12)

    figure, axes = plt.subplots(
        num_images,
        2 + num_completions + 1,
        figsize=((2 + num_completions + 1) * 1.9, num_images * 2.15),
        layout="constrained",
    )
    axes = np.atleast_2d(axes)
    for row in range(num_images):
        masked_display = originals[row] * mask + 0.5 * hidden
        panel_images = [originals[row], masked_display]
        panel_titles = ["original", "masked"]
        for completion_index in range(num_completions):
            panel_images.append(completions[row, completion_index])
            panel_titles.append(f"seed {seeds[completion_index]}")
        for column, (image, title) in enumerate(zip(panel_images, panel_titles)):
            axis = axes[row, column]
            axis.imshow(image, interpolation="nearest", vmin=0.0, vmax=1.0)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(title, fontsize=8)
        variance_axis = axes[row, -1]
        image = variance_axis.imshow(
            variance_map[row],
            cmap="magma",
            interpolation="nearest",
            vmin=0.0,
            vmax=vmax,
        )
        variance_axis.set_xticks([])
        variance_axis.set_yticks([])
        if row == 0:
            variance_axis.set_title("pixel variance", fontsize=8)
        label = CIFAR10_CLASSES[int(labels[row])]
        axes[row, 0].set_ylabel(f"image {row}\n{label}", fontsize=8)
    figure.colorbar(
        image,
        ax=list(axes[:, -1]),
        shrink=0.8,
        pad=0.02,
        label="mean RGB variance across completions",
    )
    figure.suptitle(
        "Part 5 VERIFY (1): RePaint completions of hidden 20x20 center",
        fontsize=13,
    )
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


__all__ = ["save_repaint_figure"]
