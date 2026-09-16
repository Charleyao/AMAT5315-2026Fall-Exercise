"""Tests for the Part 3 ACF/block-binning helpers in ``scripts/acf_binning.py``."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from acf_binning import block_error_curve, plateau_verdict  # noqa: E402


def test_block_error_curve_keeps_valid_lengths_and_counts():
    """Lengths that would leave fewer than two blocks are dropped."""
    rng = np.random.default_rng(5)
    abs_m = rng.normal(size=10)

    lengths, errors, counts = block_error_curve(abs_m, [1, 2, 4, 8, 16])

    np.testing.assert_array_equal(lengths, [1.0, 2.0, 4.0])
    np.testing.assert_array_equal(counts, [10, 5, 2])
    assert errors.shape == (3,)


def test_plateau_verdict_accepts_a_flat_tail():
    """A curve that levels off is reported as a plateau."""
    lengths = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    counts = np.array([100, 50, 25, 12, 6, 3])
    errors = np.array([5.0, 4.0, 3.0, 3.0, 3.0, 3.0])

    reached, *_rest, growth, start = plateau_verdict(lengths, errors, counts)

    assert reached
    assert start == 0  # largest reliable length 16 compared with 16 / 16 = 1
    assert growth == pytest.approx(3.0 / 5.0 - 1.0)


def test_plateau_verdict_rejects_a_rising_tail():
    """A curve that is still climbing over 16x is not a plateau."""
    lengths = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    counts = np.array([100, 50, 25, 12, 6, 3])
    errors = np.array([1.0, 2.0, 3.0, 3.5, 4.0, 4.0])

    reached, *_rest, growth, _start = plateau_verdict(lengths, errors, counts)

    assert not reached
    assert growth == pytest.approx(4.0 / 1.0 - 1.0)
