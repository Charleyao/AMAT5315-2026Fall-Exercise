"""Tests for the Part 4 Metropolis-vs-Wolff comparison helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from compare_metropolis_wolff import (  # noqa: E402
    agreement_verdict,
    block_bootstrap_error,
    d_statistic,
)


def test_d_statistic_squares_each_uncertainty_once():
    """d = |mean1 - mean2| / sqrt(sigma1^2 + sigma2^2)."""
    mean1, sigma1 = 1.0, 0.3
    mean2, sigma2 = 1.1, 0.4

    value = d_statistic(mean1, sigma1, mean2, sigma2)

    expected = abs(mean1 - mean2) / np.sqrt(sigma1 ** 2 + sigma2 ** 2)
    assert value == pytest.approx(expected)


def test_agreement_verdict_is_discrepancy_when_d_exceeds_three():
    """d > 3 is a discrepancy even if both errors are stable."""
    assert agreement_verdict(3.01, stable1=True, stable2=True) == "discrepancy"
    assert agreement_verdict(3.0, stable1=True, stable2=True) != "discrepancy"


def test_agreement_verdict_is_agreement_when_close_and_stable():
    """d <= 3 with both errors stable is full agreement."""
    assert agreement_verdict(1.0, stable1=True, stable2=True) == "agreement"


def test_agreement_verdict_is_provisional_when_an_error_is_sensitive():
    """d <= 3 but a block-length-sensitive error makes it provisional."""
    assert (
        agreement_verdict(1.0, stable1=False, stable2=True)
        == "agreement provisional"
    )
    assert (
        agreement_verdict(1.0, stable1=True, stable2=False)
        == "agreement provisional"
    )


def test_block_bootstrap_error_matches_direct_resampling():
    """The bootstrap error is the std of resampled block means."""
    series = np.array(
        [1.0, -1.0, 0.9, -0.9, 1.1, -1.1, 0.8, -0.8, 1.0, -1.0, 0.7, -0.7]
    )
    block_length = 3
    n_boot = 200

    rng = np.random.default_rng(7)
    value = block_bootstrap_error(series, block_length, n_boot, rng)

    abs_m = np.abs(series)
    n_blocks = abs_m.size // block_length
    blocks = abs_m[: n_blocks * block_length].reshape(n_blocks, block_length)
    block_means = blocks.mean(axis=1)
    rng2 = np.random.default_rng(7)
    indices = rng2.integers(0, n_blocks, size=(n_boot, n_blocks))
    expected = block_means[indices].mean(axis=1).std(ddof=1)

    assert value == pytest.approx(expected)


def test_block_bootstrap_error_requires_at_least_two_blocks():
    """Fewer than two blocks cannot give a bootstrap error."""
    series = np.ones(10)
    rng = np.random.default_rng(0)

    with pytest.raises(ValueError):
        block_bootstrap_error(series, block_length=8, n_boot=50, rng=rng)
