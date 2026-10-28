"""VERIFY (1): visualize the empirical MNIST class distribution.

The figure has two panels, following the reference on page 4 of the Week 6
learning sheet:

* left: a labeled gallery with the first two *training* images of every digit
  (0-9), shown at their native 28x28 resolution with nearest-neighbour
  rendering (no smoothing);
* right: a grouped bar chart of the per-class fractions for the 60,000
  training and the 10,000 test images.

The layout is produced by the shared utilities in ``distribution_plotting``
(also used by ``plot_cifar10_distribution.py``). Only the already-downloaded
and verified files in ``inputs/mnist`` are read; nothing is downloaded again.

Usage::

    cd week6
    .venv/bin/python plot_mnist_distribution.py
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
MNIST_DIR = WEEK_DIR / "inputs" / "mnist"
DEFAULT_OUTPUT = WEEK_DIR / "evidence" / "mnist_distribution.png"
NUM_CLASSES = 10


def build_figure(mnist: dict[str, np.ndarray]):
    """Build the two-panel MNIST figure and return ``(figure, report)``."""
    return build_distribution_figure(
        mnist["train_images"],
        mnist["train_labels"],
        mnist["test_labels"],
        [str(digit) for digit in range(NUM_CLASSES)],
        cmap="gray",
        vmin=0,
        vmax=255,
        gid_key="digit",
        suptitle="MNIST: sample images per class and empirical class distribution",
        bar_title="Class frequency: train (60,000) vs test (10,000)",
        bar_xlabel="digit",
    )


def save_figure(path: Path = DEFAULT_OUTPUT, mnist: dict[str, np.ndarray] | None = None) -> dict:
    """Build the figure, save it as PNG, and return the verification report."""
    if mnist is None:
        mnist = pd.load_mnist(MNIST_DIR)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, report = build_figure(mnist)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return report


def main() -> None:
    report = save_figure()

    print("MNIST class counts and fractions (first two training images per class)")
    print(f"{'digit':>5} {'train n':>8} {'train frac':>11} {'test n':>8} {'test frac':>10}")
    for digit in range(NUM_CLASSES):
        print(
            f"{digit:>5} {report['train_counts'][digit]:>8} "
            f"{report['train_fractions'][digit]:>11.4f} "
            f"{report['test_counts'][digit]:>8} {report['test_fractions'][digit]:>10.4f}"
        )
    print(
        f"{'sum':>5} {report['train_counts'].sum():>8} "
        f"{report['train_fractions'].sum():>11.4f} "
        f"{report['test_counts'].sum():>8} {report['test_fractions'].sum():>10.4f}"
    )

    gallery_indices = report["gallery_indices"]
    print("\nFirst two training indices per class (stored order):")
    print({digit: indices for digit, indices in gallery_indices.items()})

    # --- explicit VERIFY checks -------------------------------------------
    assert abs(report["train_fractions"].sum() - 1.0) < 1e-12
    assert abs(report["test_fractions"].sum() - 1.0) < 1e-12
    assert int(np.argmax(report["train_fractions"])) == 1
    assert int(np.argmax(report["test_fractions"])) == 1
    assert int(np.argmin(report["train_fractions"])) == 5
    assert int(np.argmin(report["test_fractions"])) == 5
    assert sorted(gallery_indices) == list(range(NUM_CLASSES))
    print("\nVERIFY: 10/10 classes labeled; digit 1 most frequent, digit 5 least frequent;")
    print("        fractions sum to 1 in both splits; train/test assignments unchanged.")
    print(f"\nSaved figure: {DEFAULT_OUTPUT.relative_to(WEEK_DIR)}")


if __name__ == "__main__":
    main()
