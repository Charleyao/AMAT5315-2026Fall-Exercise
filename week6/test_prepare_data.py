"""Integrity tests for the Week 6 MNIST and CIFAR-10 datasets.

These tests parse the *original* binary formats (MNIST idx files and the
CIFAR-10 binary batches) and check the properties the exercise expects:
dataset sizes, image shapes, pixel ranges, class counts and preserved
train/test splits.

Run with:

    cd week6
    .venv/bin/python -m pytest -q
"""

from pathlib import Path

import numpy as np
import pytest

import prepare_data as pd

MNIST_DIR = Path(__file__).resolve().parent / "inputs" / "mnist"
CIFAR_DIR = Path(__file__).resolve().parent / "inputs" / "cifar10"

# Official MNIST training-set label histogram (LeCun et al.).
MNIST_TRAIN_CLASS_COUNTS = [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]
MNIST_TEST_CLASS_COUNTS = [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]


@pytest.fixture(scope="module")
def mnist():
    return pd.load_mnist(MNIST_DIR)


@pytest.fixture(scope="module")
def cifar10():
    return pd.load_cifar10(CIFAR_DIR)


def test_mnist_files_match_published_checksums():
    for filename, expected_md5 in pd.MNIST_FILES.items():
        path = MNIST_DIR / filename
        assert path.is_file(), f"missing {path}; run the downloader first"
        assert pd.md5sum(path) == expected_md5


def test_cifar10_archive_matches_published_checksum():
    archive = CIFAR_DIR / "cifar-10-binary.tar.gz"
    assert archive.is_file(), f"missing {archive}; run the downloader first"
    assert pd.md5sum(archive) == pd.CIFAR_MD5


def test_mnist_split_sizes(mnist):
    assert mnist["train_images"].shape == (60000, 28, 28)
    assert mnist["test_images"].shape == (10000, 28, 28)
    assert mnist["train_labels"].shape == (60000,)
    assert mnist["test_labels"].shape == (10000,)


def test_mnist_images_are_uint8_in_full_pixel_range(mnist):
    for split in ("train", "test"):
        images = mnist[f"{split}_images"]
        assert images.dtype == np.uint8
        assert images.min() == 0
        assert images.max() == 255


def test_mnist_has_ten_digit_classes_with_known_counts(mnist):
    np.testing.assert_array_equal(
        np.bincount(mnist["train_labels"], minlength=10), MNIST_TRAIN_CLASS_COUNTS
    )
    np.testing.assert_array_equal(
        np.bincount(mnist["test_labels"], minlength=10), MNIST_TEST_CLASS_COUNTS
    )
    assert sorted(np.unique(mnist["train_labels"]).tolist()) == list(range(10))


def test_mnist_image_label_pairing_is_preserved(mnist):
    """Parsed labels must match the raw idx label bytes, in file order."""
    raw = (MNIST_DIR / "train-labels-idx1-ubyte.gz").read_bytes()
    raw_labels = pd.gzip_payload(raw)[8:]
    np.testing.assert_array_equal(mnist["train_labels"], np.frombuffer(raw_labels, np.uint8))


def test_cifar10_split_sizes_and_shapes(cifar10):
    assert cifar10["train_images"].shape == (50000, 32, 32, 3)
    assert cifar10["test_images"].shape == (10000, 32, 32, 3)
    assert cifar10["train_labels"].shape == (50000,)
    assert cifar10["test_labels"].shape == (10000,)


def test_cifar10_images_are_uint8_in_full_pixel_range(cifar10):
    images = np.concatenate([cifar10["train_images"], cifar10["test_images"]])
    assert images.dtype == np.uint8
    assert images.min() == 0
    assert images.max() == 255


def test_cifar10_has_ten_named_classes(cifar10):
    assert cifar10["class_names"] == [
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
    ]


def test_cifar10_class_balance(cifar10):
    np.testing.assert_array_equal(
        np.bincount(cifar10["train_labels"], minlength=10), [5000] * 10
    )
    np.testing.assert_array_equal(
        np.bincount(cifar10["test_labels"], minlength=10), [1000] * 10
    )


def test_cifar10_image_label_pairing_is_preserved(cifar10):
    """The first training record label must match the raw first byte."""
    raw = (CIFAR_DIR / "cifar-10-batches-bin" / "data_batch_1.bin").read_bytes()
    assert raw[0] == cifar10["train_labels"][0]
    assert len(raw) == 10000 * pd.CIFAR_RECORD_BYTES


def test_loaders_are_deterministic(mnist, cifar10):
    """Re-reading the files must give identical arrays (no shuffling)."""
    mnist_again = pd.load_mnist(MNIST_DIR)
    cifar_again = pd.load_cifar10(CIFAR_DIR)
    np.testing.assert_array_equal(mnist["train_images"], mnist_again["train_images"])
    np.testing.assert_array_equal(mnist["train_labels"], mnist_again["train_labels"])
    np.testing.assert_array_equal(
        cifar10["test_images"], cifar_again["test_images"]
    )
    np.testing.assert_array_equal(
        cifar10["test_labels"], cifar_again["test_labels"]
    )
