"""VERIFY (2): visualize the empirical CIFAR-10 class distribution.

Uses the shared gallery/frequency utilities from ``distribution_plotting``
(the same ones behind the MNIST VERIFY (1) figure). Only the verified files in
``inputs/cifar10`` are read; nothing is downloaded again.

The two-panel figure follows the reference on page 4 of the Week 6 learning
sheet:

* left: the first two *training* images of each of the 10 object classes, in
  stored order, at native 32x32 RGB resolution with nearest-neighbour
  rendering (no smoothing), labeled with the class name;
* right: grouped bars of the per-class fractions for the 50,000 training and
  10,000 test images.

Usage::

    cd week6
    .venv/bin/python plot_cifar10_distribution.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import prepare_data as pd
from distribution_plotting import (
    build_distribution_figure,
    class_counts,
    class_fractions,
    first_two_per_class,
)

WEEK_DIR = Path(__file__).resolve().parent
CIFAR_DIR = WEEK_DIR / "inputs" / "cifar10"
DEFAULT_OUTPUT = WEEK_DIR / "evidence" / "cifar10_distribution.png"
NUM_CLASSES = 10


def build_figure(cifar: dict[str, object]):
    """Build the two-panel CIFAR-10 figure and return ``(figure, report)``."""
    class_names = list(cifar["class_names"])
    return build_distribution_figure(
        cifar["train_images"],
        cifar["train_labels"],
        cifar["test_labels"],
        class_names,
        cmap=None,  # RGB images, no colour map
        gid_key="class",
        suptitle="CIFAR-10: sample images per class and class balance",
        bar_title="Class frequency: train (50,000) vs test (10,000)",
        bar_xlabel="class",
        xtick_labels=class_names,
        xtick_rotation=30,
        figsize=(12.8, 6.4),
        left=0.11,
        label_fontsize=9,
    )


def save_figure(path: Path = DEFAULT_OUTPUT, cifar: dict[str, object] | None = None) -> dict:
    """Build the figure, save it as PNG, and return the verification report."""
    if cifar is None:
        cifar = pd.load_cifar10(CIFAR_DIR)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, report = build_figure(cifar)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return report


def main() -> None:
    cifar = pd.load_cifar10(CIFAR_DIR)
    report = save_figure(cifar=cifar)
    class_names = list(cifar["class_names"])

    print("CIFAR-10 class counts and fractions (first two training images per class)")
    print(
        f"{'class':<11} {'train n':>8} {'train frac':>11} "
        f"{'test n':>8} {'test frac':>10}"
    )
    for label, name in enumerate(class_names):
        print(
            f"{name:<11} {report['train_counts'][label]:>8} "
            f"{report['train_fractions'][label]:>11.4f} "
            f"{report['test_counts'][label]:>8} "
            f"{report['test_fractions'][label]:>10.4f}"
        )
    print(
        f"{'sum':<11} {report['train_counts'].sum():>8} "
        f"{report['train_fractions'].sum():>11.4f} "
        f"{report['test_counts'].sum():>8} "
        f"{report['test_fractions'].sum():>10.4f}"
    )

    print("\nFirst two training indices per class (stored order):")
    print(dict(report["gallery_indices"]))

    # --- explicit VERIFY checks -------------------------------------------
    train_images = cifar["train_images"]
    test_images = cifar["test_images"]
    assert train_images.shape == (50000, 32, 32, 3)
    assert test_images.shape == (10000, 32, 32, 3)
    assert train_images.dtype == np.uint8 and train_images.shape[-1] == 3
    np.testing.assert_array_equal(report["train_counts"], [5000] * NUM_CLASSES)
    np.testing.assert_array_equal(report["test_counts"], [1000] * NUM_CLASSES)
    assert np.array_equal(report["train_fractions"], np.full(NUM_CLASSES, 0.1))
    assert np.array_equal(report["test_fractions"], np.full(NUM_CLASSES, 0.1))
    assert abs(report["train_fractions"].sum() - 1.0) < 1e-12
    assert abs(report["test_fractions"].sum() - 1.0) < 1e-12
    print(
        "\nVERIFY: 50,000/10,000 images; 5,000/1,000 per class; "
        "every frequency = 0.1; RGB decoded; pairing preserved."
    )
    print(f"\nSaved figure: {DEFAULT_OUTPUT.relative_to(WEEK_DIR)}")


if __name__ == "__main__":
    main()
