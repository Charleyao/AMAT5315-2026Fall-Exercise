"""Part 4 VERIFY (1) CIFAR-10 galleries and reverse-diffusion snapshots."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from cifar_ddpm import display_images

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def save_reverse_diffusion_gallery(
    trajectory: np.ndarray,
    steps: list[int] | tuple[int, ...],
    output_path: str,
) -> None:
    """Show every sample at selected reverse-diffusion noise levels."""
    trajectory = np.asarray(trajectory)
    if trajectory.ndim != 5:
        raise ValueError("trajectory must have shape (time, samples, 32, 32, 3)")
    num_steps = trajectory.shape[0] - 1
    num_samples = trajectory.shape[1]
    steps = list(steps)

    figure, axes = plt.subplots(
        num_samples,
        len(steps),
        figsize=(len(steps) * 1.9, num_samples * 2.0),
    )
    axes = np.atleast_2d(axes)
    for row in range(num_samples):
        for column, t in enumerate(steps):
            index = num_steps - int(t)
            image = display_images(trajectory[index, row])
            axis = axes[row, column]
            axis.imshow(image, interpolation="nearest", vmin=0.0, vmax=1.0)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(f"t={t}", fontsize=9)
            if column == 0:
                axis.set_ylabel(f"sample {row}", fontsize=9)
    figure.suptitle(
        "Pretrained CIFAR-10 DDPM: reverse diffusion (display map + clip only)",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_final_gallery(images: np.ndarray, output_path: str) -> None:
    """Display all final generated images at native 32x32 resolution."""
    images = display_images(images)
    count = images.shape[0]
    columns = 4 if count >= 4 else count
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 2.2, rows * 2.35))
    axes = np.atleast_1d(axes).ravel()
    for index in range(axes.size):
        axis = axes[index]
        if index >= count:
            axis.axis("off")
            continue
        axis.imshow(images[index], interpolation="nearest", vmin=0.0, vmax=1.0)
        axis.set_title(f"generated {index}", fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("Part 4 VERIFY (1): four final 32x32 DDPM samples", fontsize=12)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_generated_vs_real_gallery(
    generated: np.ndarray,
    real: np.ndarray,
    real_labels: np.ndarray,
    output_path: str,
) -> None:
    """Week 6-style nearest-neighbour gallery: generated vs real CIFAR-10."""
    generated = display_images(generated)
    real = display_images(real)
    real_labels = np.asarray(real_labels)
    count = min(generated.shape[0], real.shape[0], real_labels.shape[0])

    figure, axes = plt.subplots(2, count, figsize=(count * 2.15, 4.8))
    axes = np.atleast_2d(axes)
    for column in range(count):
        top = axes[0, column]
        bottom = axes[1, column]
        top.imshow(generated[column], interpolation="nearest", vmin=0.0, vmax=1.0)
        top.set_title(f"generated {column}", fontsize=9)
        top.set_xticks([])
        top.set_yticks([])
        bottom.imshow(real[column], interpolation="nearest", vmin=0.0, vmax=1.0)
        bottom.set_title(
            f"real: {CIFAR10_CLASSES[int(real_labels[column])]}", fontsize=9
        )
        bottom.set_xticks([])
        bottom.set_yticks([])
    axes[0, 0].set_ylabel("generated", fontsize=9)
    axes[1, 0].set_ylabel("real CIFAR-10", fontsize=9)
    figure.suptitle(
        "Generated samples vs real CIFAR-10 test images "
        "(Week 6 gallery style, nearest-neighbour 32x32)",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


__all__ = [
    "CIFAR10_CLASSES",
    "save_final_gallery",
    "save_generated_vs_real_gallery",
    "save_reverse_diffusion_gallery",
]
