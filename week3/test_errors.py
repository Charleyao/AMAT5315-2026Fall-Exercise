"""Tests for the Part 3 uncertainty helpers in ``scripts/errors.py``."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from errors import (  # noqa: E402
    block_standard_error,
    integrated_autocorrelation_time,
    naive_standard_error,
)


def test_naive_standard_error_matches_direct_formula():
    """The naive error is the sample std divided by sqrt(N)."""
    samples = np.array([1.0, 2.0, 3.0, 4.0])

    expected = samples.std(ddof=1) / np.sqrt(samples.size)

    assert naive_standard_error(samples) == pytest.approx(expected)


def test_block_standard_error_is_error_of_block_means():
    """The block error is the standard error of the 50 block averages."""
    rng = np.random.default_rng(7)
    samples = rng.normal(size=1000)

    value, length = block_standard_error(samples, n_blocks=50)

    assert length == 20
    block_means = samples.reshape(50, 20).mean(axis=1)
    expected = block_means.std(ddof=1) / np.sqrt(50)
    assert value == pytest.approx(expected)


def test_tau_int_of_white_noise_is_near_one_half():
    """Independent samples have rho(t) = 0, so tau_int = 1/2."""
    rng = np.random.default_rng(123)
    samples = rng.normal(size=100_000)

    assert integrated_autocorrelation_time(samples) == pytest.approx(0.5, abs=0.1)


def test_tau_int_of_ar1_matches_theory():
    """AR(1) noise has tau_int = 1/2 + phi / (1 - phi)."""
    phi = 0.8
    rng = np.random.default_rng(2026)
    innovations = rng.normal(size=200_000)
    samples = np.empty_like(innovations)
    samples[0] = innovations[0]
    for i in range(1, samples.size):
        samples[i] = phi * samples[i - 1] + innovations[i]

    expected = 0.5 + phi / (1.0 - phi)  # 4.5

    assert integrated_autocorrelation_time(samples) == pytest.approx(
        expected, rel=0.15
    )
