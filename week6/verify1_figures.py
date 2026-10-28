"""VERIFY (1) figures for Week 6 Part 2: MNIST binary data and RBM layout.

Generates three figures under ``week6/evidence/``:

A. ``mnist_binary_gallery.png`` — first 16 MNIST training images in their
   original stored order, binarized (``v = 1`` if pixel >= 128 else ``0``).
B. ``mnist_spin_configuration.png`` — one enlarged 28x28 spin configuration
   (white = +1 ink, black = -1 background).
C. ``rbm_architecture.png`` — 784 visible and 128 hidden binary units with
   connections only between the two layers (no within-layer couplings).

Only the verified MNIST files in ``inputs/mnist`` are read.

Usage::

    cd week6
    uv run python verify1_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering; must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np

import rbm

WEEK_DIR = Path(__file__).resolve().parent
MNIST_DIR = WEEK_DIR / "inputs" / "mnist"
EVIDENCE_DIR = WEEK_DIR / "evidence"

DEFAULT_BINARY_GALLERY = EVIDENCE_DIR / "mnist_binary_gallery.png"
DEFAULT_SPIN = EVIDENCE_DIR / "mnist_spin_configuration.png"
DEFAULT_ARCHITECTURE = EVIDENCE_DIR / "rbm_architecture.png"


def build_binary_gallery(
    binary_images: np.ndarray,
    labels: np.ndarray,
    count: int = 16,
    rows: int = 2,
    cols: int = 8,
):
    """Figure A: first ``count`` training images as binary pixels, in stored order."""
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.35, rows * 1.95))
    axes = np.atleast_2d(axes)

    report = {"images": [], "labels": [], "indices": []}
    for position in range(count):
        row, col = divmod(position, cols)
        ax = axes[row, col]
        image = np.asarray(binary_images[position]).reshape(28, 28)
        ax.imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest",  # no smoothing: show the binary pixels
        )
        ax.set_title(f"{position}: label {int(labels[position])}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        report["images"].append(image.copy())
        report["labels"].append(int(labels[position]))
        report["indices"].append(position)

    for position in range(count, rows * cols):
        row, col = divmod(position, cols)
        axes[row, col].axis("off")

    fig.suptitle(
        "First 16 MNIST training images (stored order), binarized: "
        "v = 1 if pixel >= 128 else 0",
        fontsize=11,
        y=0.99,
    )
    # Explicit spacing keeps the second row's titles clear of the first row.
    fig.subplots_adjust(
        left=0.02, right=0.98, top=0.80, bottom=0.04, wspace=0.08, hspace=0.45
    )
    return fig, report


def build_spin_figure(spins: np.ndarray, index: int = 0, label: int | None = None):
    """Figure B: one enlarged 28x28 spin configuration (+1 white ink, -1 black)."""
    spins = np.asarray(spins)
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    image = ax.imshow(
        spins,
        cmap="gray",
        vmin=-1,
        vmax=1,
        interpolation="nearest",  # native 28x28 pixels, no smoothing
    )
    ax.set_xticks([])
    ax.set_yticks([])
    title = f"MNIST spin configuration (training image {index}"
    title += f", label {label})" if label is not None else ")"
    ax.set_title(title + "\nwhite = +1 ink, black = -1 background", fontsize=11)
    colorbar = fig.colorbar(image, ax=ax, ticks=[-1, 1], fraction=0.046, pad=0.04)
    colorbar.ax.set_yticklabels(["-1 (background)", "+1 (ink)"])
    return fig, {"spins": spins.copy(), "index": index, "label": label}


def build_architecture_figure(
    visible_units: int = rbm.VISIBLE_UNITS,
    hidden_units: int = rbm.HIDDEN_UNITS,
    drawn_visible: int = 10,
    drawn_hidden: int = 7,
):
    """Figure C: bipartite RBM diagram (visible <-> hidden only)."""
    visible_x, hidden_x = 0.22, 0.78
    top, bottom = 0.84, 0.16
    visible_ys = np.linspace(top, bottom, drawn_visible)
    hidden_ys = np.linspace(top, bottom, drawn_hidden)

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    # Connections: every representative visible node to every hidden node,
    # and never within a layer.
    for y_visible in visible_ys:
        for y_hidden in hidden_ys:
            (line,) = ax.plot(
                [visible_x, hidden_x], [y_visible, y_hidden],
                color="#7f7f7f", linewidth=0.5, alpha=0.35, zorder=1,
            )
            line.set_gid("visible-hidden")

    for y_visible in visible_ys:
        node = plt.Circle(
            (visible_x, y_visible), 0.018,
            facecolor="#4C72B0", edgecolor="white", linewidth=0.8, zorder=3,
        )
        node.set_gid("visible-node")
        ax.add_patch(node)

    for y_hidden in hidden_ys:
        node = plt.Circle(
            (hidden_x, y_hidden), 0.022,
            facecolor="#DD8452", edgecolor="white", linewidth=0.8, zorder=3,
        )
        node.set_gid("hidden-node")
        ax.add_patch(node)

    ax.text(
        visible_x, top + 0.05,
        f"Visible layer\n{visible_units} binary units\n(28x28 pixels)",
        ha="center", va="bottom", fontsize=11,
    )
    ax.text(
        hidden_x, top + 0.05,
        f"Hidden layer\n{hidden_units} binary units",
        ha="center", va="bottom", fontsize=11,
    )
    ax.text(
        visible_x, bottom - 0.04, "representative nodes", ha="center",
        va="top", fontsize=8, style="italic", color="#555555",
    )
    ax.text(
        hidden_x, bottom - 0.04, "representative nodes", ha="center",
        va="top", fontsize=8, style="italic", color="#555555",
    )
    ax.text(
        0.5, 0.03,
        "Connections only between visible and hidden layers\n"
        "(no visible-visible or hidden-hidden couplings)",
        ha="center", va="bottom", fontsize=9,
    )
    ax.set_title("RBM architecture: bipartite visible-hidden graph", fontsize=12)

    report = {
        "visible_units": visible_units,
        "hidden_units": hidden_units,
        "drawn_visible": drawn_visible,
        "drawn_hidden": drawn_hidden,
        "visible_x": visible_x,
        "hidden_x": hidden_x,
    }
    return fig, report


def save_all(
    binary_images: np.ndarray, labels: np.ndarray, output_dir: Path = EVIDENCE_DIR
) -> dict[str, Path]:
    """Build and save all three VERIFY (1) figures as PNG files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "binary_gallery": output_dir / DEFAULT_BINARY_GALLERY.name,
        "spin_configuration": output_dir / DEFAULT_SPIN.name,
        "architecture": output_dir / DEFAULT_ARCHITECTURE.name,
    }

    gallery, _ = build_binary_gallery(binary_images, labels)
    gallery.savefig(paths["binary_gallery"], dpi=200, bbox_inches="tight")
    plt.close(gallery)

    spins = rbm.to_spins(np.asarray(binary_images[0]).reshape(28, 28))
    spin_figure, _ = build_spin_figure(spins, index=0, label=int(labels[0]))
    spin_figure.savefig(paths["spin_configuration"], dpi=200, bbox_inches="tight")
    plt.close(spin_figure)

    architecture, _ = build_architecture_figure()
    architecture.savefig(paths["architecture"], dpi=200, bbox_inches="tight")
    plt.close(architecture)
    return paths


def main() -> None:
    images, labels = rbm.load_binary_mnist("train", MNIST_DIR)
    paths = save_all(images, labels)

    print("VERIFY (1) figures")
    print(f"  first 16 labels (stored order): {labels[:16].tolist()}")
    print(f"  binary pixels per image:        {images.shape[1]}")
    spins = rbm.to_spins(np.asarray(images[0]).reshape(28, 28))
    print(f"  spin values:                    {sorted(np.unique(spins).tolist())}")
    print(f"  max visible-visible links:      0")
    print(f"  max hidden-hidden links:        0")
    for name, path in paths.items():
        print(f"  saved {name}: {path.relative_to(WEEK_DIR)}")


if __name__ == "__main__":
    main()
