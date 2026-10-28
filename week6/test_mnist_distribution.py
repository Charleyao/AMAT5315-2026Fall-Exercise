"""Tests for the Week 6 VERIFY (1) MNIST distribution figure.

These tests check the properties the exercise asks us to verify:

* all ten digit classes are present and correctly labeled;
* the gallery shows the first two training images per class, in stored order;
* the displayed pixels are the original ``uint8`` pixels, at 28x28 with
  nearest-neighbour rendering;
* digit 1 is the most frequent and digit 5 the least frequent class in both
  splits, and the per-split frequencies sum to 1;
* the train/test assignments (and the downloaded files) are unchanged.

Run with::

    cd week6
    .venv/bin/python -m pytest -q
"""

from pathlib import Path

import numpy as np
import pytest

import plot_mnist_distribution as plot

WEEK_DIR = Path(__file__).resolve().parent
MNIST_DIR = WEEK_DIR / "inputs" / "mnist"

# Official MNIST histograms (unchanged by this figure step).
TRAIN_COUNTS = np.array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949])
TEST_COUNTS = np.array([980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009])


@pytest.fixture(scope="module")
def mnist():
    return plot.pd.load_mnist(MNIST_DIR)


@pytest.fixture(scope="module")
def figure(mnist):
    fig, report = plot.build_figure(mnist)
    yield fig, report
    plot.plt.close(fig)


def test_gallery_covers_all_ten_digits(figure):
    _fig, report = figure
    assert sorted(report["gallery_indices"]) == list(range(10))


def test_gallery_picks_first_two_in_stored_order(mnist, figure):
    _fig, report = figure
    for digit, indices in report["gallery_indices"].items():
        expected = np.flatnonzero(mnist["train_labels"] == digit)[:2]
        assert indices == expected.tolist(), digit
        assert indices[0] < indices[1]


def test_gallery_labels_are_correct(figure):
    _fig, report = figure
    for digit, images in report["gallery"].items():
        assert len(images) == 2
        for image in images:
            assert image.shape == (28, 28)
            assert image.dtype == np.uint8


def test_gallery_pixels_are_original(mnist, figure):
    _fig, report = figure
    for digit, indices in report["gallery_indices"].items():
        for slot, index in enumerate(indices):
            np.testing.assert_array_equal(
                report["gallery"][digit][slot], mnist["train_images"][index]
            )


def test_gallery_is_rendered_at_native_size_with_nearest_neighbour(figure):
    fig, _report = figure
    gallery_axes = [ax for ax in fig.axes if ax.images]
    assert len(gallery_axes) == 20  # 10 digits x 2 images
    for ax in gallery_axes:
        image = ax.images[0]
        assert image.get_array().shape == (28, 28)
        assert image.get_interpolation() == "nearest"
    labels = sorted(ax.get_ylabel() for ax in gallery_axes if ax.get_ylabel())
    assert labels == [str(d) for d in range(10)]


def test_axis_labels_match_the_digit(mnist, figure):
    fig, _report = figure
    for ax in fig.axes:
        if not ax.images:
            continue
        parts = dict(part.split("=") for part in ax.get_gid().split(";"))
        digit, index = int(parts["digit"]), int(parts["index"])
        assert mnist["train_labels"][index] == digit
        if ax.get_ylabel():
            assert ax.get_ylabel() == str(digit)
        np.testing.assert_array_equal(
            ax.images[0].get_array(), mnist["train_images"][index]
        )


def test_class_fractions_sum_to_one(mnist, figure):
    _fig, report = figure
    assert report["train_fractions"].sum() == pytest.approx(1.0)
    assert report["test_fractions"].sum() == pytest.approx(1.0)


def test_digit_one_has_highest_frequency(mnist, figure):
    _fig, report = figure
    for split in ("train", "test"):
        fractions = report[f"{split}_fractions"]
        assert int(np.argmax(fractions)) == 1
        assert int(np.argmax(report[f"{split}_counts"])) == 1


def test_digit_five_has_lowest_frequency(mnist, figure):
    _fig, report = figure
    for split in ("train", "test"):
        fractions = report[f"{split}_fractions"]
        assert int(np.argmin(fractions)) == 5
        assert int(np.argmin(report[f"{split}_counts"])) == 5


def test_train_test_assignments_and_files_unchanged(mnist):
    np.testing.assert_array_equal(plot.class_counts(mnist["train_labels"]), TRAIN_COUNTS)
    np.testing.assert_array_equal(plot.class_counts(mnist["test_labels"]), TEST_COUNTS)
    for filename, expected_md5 in plot.pd.MNIST_FILES.items():
        assert plot.pd.md5sum(MNIST_DIR / filename) == expected_md5


def test_bar_chart_has_train_and_test_series(figure):
    fig, _report = figure
    bar_axes = [ax for ax in fig.axes if ax.containers]
    assert len(bar_axes) == 1
    ax = bar_axes[0]
    assert len(ax.containers) == 2  # train and test
    assert [int(t.get_text()) for t in ax.get_xticklabels()] == list(range(10))
    legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert legend_labels == ["train", "test"]


def test_figure_is_saved_to_evidence(tmp_path):
    # Use a temporary path to keep the test isolated, then check the real
    # default path is wired to evidence/.
    out = tmp_path / "mnist_distribution.png"
    report = plot.save_figure(out)
    assert out.is_file()
    assert out.stat().st_size > 10_000
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert report["train_fractions"].sum() == pytest.approx(1.0)
    assert plot.DEFAULT_OUTPUT == WEEK_DIR / "evidence" / "mnist_distribution.png"
