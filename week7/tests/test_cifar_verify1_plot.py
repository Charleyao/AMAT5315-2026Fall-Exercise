"""Tests for Part 4 CIFAR-10 gallery figures."""

import matplotlib

matplotlib.use("Agg")

import numpy as np

from cifar_verify1_plot import (
    save_final_gallery,
    save_generated_vs_real_gallery,
    save_reverse_diffusion_gallery,
)


def test_cifar_gallery_figures_are_written(tmp_path):
    trajectory = np.zeros((1001, 2, 32, 32, 3), dtype=np.float32)
    images = np.zeros((4, 32, 32, 3), dtype=np.float32)
    labels = np.array([3, 8, 8, 0])
    reverse_path = tmp_path / "reverse.png"
    final_path = tmp_path / "final.png"
    comparison_path = tmp_path / "comparison.png"

    save_reverse_diffusion_gallery(
        trajectory, [1000, 500, 0], str(reverse_path)
    )
    save_final_gallery(images, str(final_path))
    save_generated_vs_real_gallery(
        images, images, labels, str(comparison_path)
    )

    assert reverse_path.stat().st_size > 0
    assert final_path.stat().st_size > 0
    assert comparison_path.stat().st_size > 0
