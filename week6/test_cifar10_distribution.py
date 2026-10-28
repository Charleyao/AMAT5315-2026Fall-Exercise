"""Tests for the Week 6 VERIFY (2) CIFAR-10 distribution figure.

Checked properties (from the exercise):

* 50,000 training and 10,000 test images;
* exactly 5,000 training and 1,000 test images per class;
* every class frequency is exactly 0.1;
* the three RGB channels are decoded in the correct order;
* the original image-label pairing and stored order are preserved;
* the gallery shows the first two training images per class at native 32x32
  RGB with nearest-neighbour rendering.

Run with::

    cd week6
    .venv/bin/python -m pytest -q
"""

from pathlib import Path

import numpy as np
import pytest

import plot_cifar10_distribution as plot

WEEK_DIR = Path(__file__).resolve().parent
CIFAR_DIR = WEEK_DIR / "inputs" / "cifar10"
BATCH_DIR = CIFAR_DIR / "cifar-10-batches-bin"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def raw_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Independent decode of a CIFAR-10 binary batch (labels, RGB images)."""
    records = np.frombuffer(path.read_bytes(), dtype=np.uint8).reshape(-1, 3073)
    labels = records[:, 0].copy()
    # Explicit channel-order decode: R plane, then G plane, then B plane.
    red = records[:, 1:1 + 1024].reshape(-1, 32, 32)
    green = records[:, 1 + 1024:1 + 2048].reshape(-1, 32, 32)
    blue = records[:, 1 + 2048:].reshape(-1, 32, 32)
    images = np.stack([red, green, blue], axis=-1)
    return labels, images


@pytest.fixture(scope="module")
def cifar():
    return plot.pd.load_cifar10(CIFAR_DIR)


@pytest.fixture(scope="module")
def figure(cifar):
    fig, report = plot.build_figure(cifar)
    yield fig, report
    plot.plt.close(fig)


def test_split_sizes(cifar):
    assert cifar["train_images"].shape == (50000, 32, 32, 3)
    assert cifar["test_images"].shape == (10000, 32, 32, 3)
    assert cifar["train_labels"].shape == (50000,)
    assert cifar["test_labels"].shape == (10000,)


def test_each_class_has_exactly_5000_training_images(figure):
    _fig, report = figure
    np.testing.assert_array_equal(report["train_counts"], [5000] * 10)


def test_each_class_has_exactly_1000_test_images(figure):
    _fig, report = figure
    np.testing.assert_array_equal(report["test_counts"], [1000] * 10)


def test_every_class_frequency_is_exactly_one_tenth(figure):
    _fig, report = figure
    assert np.array_equal(report["train_fractions"], np.full(10, 0.1))
    assert np.array_equal(report["test_fractions"], np.full(10, 0.1))
    assert report["train_fractions"].sum() == pytest.approx(1.0, abs=1e-12)
    assert report["test_fractions"].sum() == pytest.approx(1.0, abs=1e-12)


def test_rgb_channels_are_decoded_in_the_correct_order(cifar):
    train_labels, train_images = raw_batch(BATCH_DIR / "data_batch_1.bin")
    np.testing.assert_array_equal(cifar["train_labels"][:10000], train_labels)
    np.testing.assert_array_equal(cifar["train_images"][:10000], train_images)

    test_labels, test_images = raw_batch(BATCH_DIR / "test_batch.bin")
    np.testing.assert_array_equal(cifar["test_labels"], test_labels)
    np.testing.assert_array_equal(cifar["test_images"], test_images)


def test_images_are_uint8_rgb_with_full_pixel_range(cifar):
    for split in ("train", "test"):
        images = cifar[f"{split}_images"]
        assert images.dtype == np.uint8
        assert images.shape[-1] == 3
        for channel in range(3):
            assert images[..., channel].min() == 0
            assert images[..., channel].max() == 255
    # A colour image must not collapse to three identical channels.
    assert not np.array_equal(
        cifar["train_images"][0][..., 0], cifar["train_images"][0][..., 2]
    )


def test_gallery_covers_all_ten_classes(figure):
    _fig, report = figure
    assert sorted(report["gallery_indices"]) == list(range(10))


def test_gallery_picks_first_two_in_stored_order(cifar, figure):
    _fig, report = figure
    raw_labels, _ = raw_batch(BATCH_DIR / "data_batch_1.bin")
    for label, indices in report["gallery_indices"].items():
        expected = np.flatnonzero(cifar["train_labels"] == label)[:2]
        assert indices == expected.tolist(), label
        assert indices[0] < indices[1]
        # The first batch's raw labels are the oracle for stored order.
        if indices[1] < len(raw_labels):
            assert raw_labels[indices[0]] == label
            assert raw_labels[indices[1]] == label


def test_gallery_pixels_are_original_rgb(cifar, figure):
    _fig, report = figure
    for label, indices in report["gallery_indices"].items():
        for slot, index in enumerate(indices):
            gallery_image = report["gallery"][label][slot]
            assert gallery_image.shape == (32, 32, 3)
            assert gallery_image.dtype == np.uint8
            np.testing.assert_array_equal(
                gallery_image, cifar["train_images"][index]
            )


def test_gallery_renders_native_size_nearest_neighbour_with_class_labels(figure):
    fig, _report = figure
    gallery_axes = [ax for ax in fig.axes if ax.images]
    assert len(gallery_axes) == 20  # 10 classes x 2 images
    for ax in gallery_axes:
        image = ax.images[0]
        assert image.get_array().shape == (32, 32, 3)
        assert image.get_interpolation() == "nearest"
    labels = sorted(ax.get_ylabel() for ax in gallery_axes if ax.get_ylabel())
    assert labels == sorted(CLASS_NAMES)


def test_each_gallery_axis_is_labelled_with_its_true_class(cifar, figure):
    fig, _report = figure
    for ax in fig.axes:
        if not ax.images:
            continue
        parts = dict(part.split("=") for part in ax.get_gid().split(";"))
        label, index = int(parts["class"]), int(parts["index"])
        assert cifar["train_labels"][index] == label
        if ax.get_ylabel():
            assert ax.get_ylabel() == CLASS_NAMES[label]


def test_image_label_pairing_and_files_unchanged(cifar):
    """Concatenated batches keep their labels; the archive checksum is intact."""
    labels = np.concatenate(
        [raw_batch(BATCH_DIR / f"data_batch_{i}.bin")[0] for i in range(1, 6)]
    )
    np.testing.assert_array_equal(cifar["train_labels"], labels)
    test_labels, _ = raw_batch(BATCH_DIR / "test_batch.bin")
    np.testing.assert_array_equal(cifar["test_labels"], test_labels)
    assert (
        plot.pd.md5sum(CIFAR_DIR / "cifar-10-binary.tar.gz") == plot.pd.CIFAR_MD5
    )


def test_bar_chart_shows_train_and_test_series(figure):
    fig, _report = figure
    bar_axes = [ax for ax in fig.axes if ax.containers]
    assert len(bar_axes) == 1
    ax = bar_axes[0]
    assert len(ax.containers) == 2
    assert [t.get_text() for t in ax.get_xticklabels()] == CLASS_NAMES
    legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert legend_labels == ["train", "test"]


def test_figure_is_saved_to_evidence(tmp_path):
    out = tmp_path / "cifar10_distribution.png"
    report = plot.save_figure(out)
    assert out.is_file()
    assert out.stat().st_size > 10_000
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert np.array_equal(report["train_fractions"], np.full(10, 0.1))
    assert plot.DEFAULT_OUTPUT == WEEK_DIR / "evidence" / "cifar10_distribution.png"
