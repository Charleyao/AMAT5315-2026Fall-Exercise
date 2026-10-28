"""Tests for the Week 6 Part 2 VERIFY (1) figures.

Figure A: first 16 training images (stored order) as binary pixels.
Figure B: one enlarged 28x28 spin configuration (+1 ink / -1 background).
Figure C: RBM architecture (784 visible, 128 hidden, no within-layer couplings).

Run with::

    cd week6
    uv run python -m pytest -q
"""

from pathlib import Path

import numpy as np
import pytest

import rbm
import verify1_figures as vf

WEEK_DIR = Path(__file__).resolve().parent
MNIST_DIR = WEEK_DIR / "inputs" / "mnist"


@pytest.fixture(scope="module")
def data():
    images, labels = rbm.load_binary_mnist("train", MNIST_DIR)
    return images, labels


@pytest.fixture(scope="module")
def gallery(data):
    images, labels = data
    fig, report = vf.build_binary_gallery(images, labels)
    yield fig, report
    vf.plt.close(fig)


@pytest.fixture(scope="module")
def spin(data):
    images, labels = data
    spins = rbm.to_spins(images[0].reshape(28, 28))
    fig, report = vf.build_spin_figure(spins, index=0, label=int(labels[0]))
    yield fig, report
    vf.plt.close(fig)


@pytest.fixture(scope="module")
def architecture():
    fig, report = vf.build_architecture_figure()
    yield fig, report
    vf.plt.close(fig)


def test_gallery_has_first_16_images_in_stored_order(data, gallery):
    images, labels = data
    _fig, report = gallery
    assert report["indices"] == list(range(16))
    np.testing.assert_array_equal(report["labels"], labels[:16])
    for position in range(16):
        np.testing.assert_array_equal(
            report["images"][position], images[position].reshape(28, 28)
        )


def test_gallery_images_are_binary_at_native_size(gallery):
    fig, report = gallery
    axes = [ax for ax in fig.axes if ax.images]
    assert len(axes) == 16
    for ax in axes:
        image = ax.images[0]
        array = np.asarray(image.get_array())
        assert array.shape == (28, 28)
        assert set(np.unique(array).tolist()) <= {0, 1}
        assert image.get_interpolation() == "nearest"


def test_gallery_titles_show_index_and_label(data, gallery):
    _images, labels = data
    fig, _report = gallery
    titles = [ax.get_title() for ax in fig.axes if ax.images]
    assert len(titles) == 16
    for index, title in enumerate(titles):
        assert str(index) in title
        assert str(int(labels[index])) in title


def test_spin_figure_is_strictly_plus_or_minus_one(spin):
    _fig, report = spin
    values = set(np.unique(report["spins"]).tolist())
    assert values == {-1, 1}
    assert report["spins"].shape == (28, 28)
    assert report["index"] == 0


def test_spin_figure_uses_white_for_ink_and_black_background(spin):
    fig, report = spin
    ax = next(ax for ax in fig.axes if ax.images)
    image = ax.images[0]
    assert image.get_clim() == (-1.0, 1.0)
    cmap = image.get_cmap()
    black = np.asarray(cmap(0.0))[:3]  # spin -1
    white = np.asarray(cmap(1.0))[:3]  # spin +1
    np.testing.assert_allclose(black, [0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(white, [1, 1, 1], atol=1e-6)
    # The stored first image has ink (some +1 spins) and background (-1).
    assert report["spins"].max() == 1 and report["spins"].min() == -1


def test_architecture_has_784_visible_and_128_hidden_units(architecture):
    _fig, report = architecture
    assert report["visible_units"] == 784
    assert report["hidden_units"] == 128
    assert report["visible_units"] == rbm.VISIBLE_UNITS
    assert report["hidden_units"] == rbm.HIDDEN_UNITS


def test_architecture_connections_only_between_layers(architecture):
    fig, report = architecture
    ax = fig.axes[0]
    connections = [line for line in ax.lines if line.get_gid() == "visible-hidden"]
    assert len(connections) == report["drawn_visible"] * report["drawn_hidden"]
    left_x = report["visible_x"]
    right_x = report["hidden_x"]
    for line in connections:
        xs = line.get_xdata()
        assert len(xs) == 2
        assert {float(xs[0]), float(xs[1])} == {left_x, right_x}

    visible_nodes = [p for p in ax.patches if p.get_gid() == "visible-node"]
    hidden_nodes = [p for p in ax.patches if p.get_gid() == "hidden-node"]
    assert len(visible_nodes) == report["drawn_visible"]
    assert len(hidden_nodes) == report["drawn_hidden"]


def test_architecture_has_no_within_layer_connection_lines(architecture):
    fig, _report = architecture
    ax = fig.axes[0]
    for line in ax.lines:
        xs = set(float(x) for x in line.get_xdata())
        if line.get_gid() == "visible-hidden":
            assert len(xs) == 2  # one endpoint per layer
        else:
            assert len(xs) == 1  # decorative only, never layer-crossing


def test_architecture_labels_mention_layer_sizes(architecture):
    fig, _report = architecture
    texts = " ".join(text.get_text() for text in fig.axes[0].texts)
    assert "784" in texts
    assert "128" in texts
    assert "visible" in texts.lower()
    assert "hidden" in texts.lower()


def test_all_figures_are_saved_to_evidence(data, tmp_path):
    images, labels = data
    paths = vf.save_all(images, labels, tmp_path)
    assert set(paths) == {"binary_gallery", "spin_configuration", "architecture"}
    for path in paths.values():
        assert path.is_file()
        assert path.stat().st_size > 10_000
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert vf.DEFAULT_BINARY_GALLERY == WEEK_DIR / "evidence" / "mnist_binary_gallery.png"
    assert vf.DEFAULT_SPIN == WEEK_DIR / "evidence" / "mnist_spin_configuration.png"
    assert vf.DEFAULT_ARCHITECTURE == WEEK_DIR / "evidence" / "rbm_architecture.png"
