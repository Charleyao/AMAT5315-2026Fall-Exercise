"""Tests for the Part 3 bootstrap helpers in ``scripts/bootstrap.py``."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from bootstrap import (  # noqa: E402
    block_means,
    fit_five_point,
    parabola_envelope,
    sampling_error_stable,
    susceptibility,
)


def test_fit_five_point_recovers_a_known_vertex():
    """A noiseless five-point parabola returns its exact vertex."""
    temperature = np.arange(5, dtype=float)
    chi = 5.0 - 3.0 * (temperature - 2.4) ** 2

    result = fit_five_point(temperature, chi)

    assert result is not None
    assert result["vertex"] == pytest.approx(2.4)


def test_fit_five_point_rejects_an_edge_maximum():
    """A maximum at the first or last point has no five-point window."""
    temperature = np.arange(5, dtype=float)
    chi = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

    assert fit_five_point(temperature, chi) is None


def test_fit_five_point_rejects_a_vertex_outside_the_points():
    """A downward parabola whose vertex is outside the window is a failure."""
    temperature = np.arange(5, dtype=float)
    chi = np.array([0.395, 0.430, 0.696, -1.184, -0.662])

    assert fit_five_point(temperature, chi) is None


def test_block_means_match_a_direct_reshape():
    """Per-block means of M^2 and |M| equal the direct calculation."""
    series = np.arange(10, dtype=float)

    mean_m2, mean_abs = block_means(series, block_length=5)

    np.testing.assert_allclose(mean_m2, [6.0, 51.0])
    np.testing.assert_allclose(mean_abs, [2.0, 7.0])


def test_sampling_error_stable_needs_agreement_within_tolerance():
    """Three statistically equal errors are stable; a spread is not."""
    assert sampling_error_stable([1.0, 1.0, 1.0])
    assert not sampling_error_stable([1.0, 1.0, 2.0])
    assert not sampling_error_stable([0.0, 0.0, 0.0])


def test_susceptibility_matches_the_definition():
    """chi = L^2 (<M^2> - <|M|>^2) / T."""
    series = np.array([1.0, -1.0, 0.5, -0.5])

    value = susceptibility(series, lattice=4, temperature=2.0)

    expected = 4 ** 2 * (np.mean(series ** 2) - np.mean(np.abs(series)) ** 2) / 2.0
    assert value == pytest.approx(expected)


def test_parabola_envelope_is_pointwise_min_and_max():
    """Two constant fits give a flat envelope between their values."""
    fits = [
        {"coefficients": (0.0, 0.0, 1.0), "t_low": 0.0, "t_high": 1.0},
        {"coefficients": (0.0, 0.0, 3.0), "t_low": 0.0, "t_high": 1.0},
    ]
    grid = np.linspace(0.0, 1.0, 5)

    lower, upper = parabola_envelope(fits, grid)

    np.testing.assert_allclose(lower, 1.0)
    np.testing.assert_allclose(upper, 3.0)
