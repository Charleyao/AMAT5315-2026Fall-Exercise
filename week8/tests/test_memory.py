"""Tests for associative-memory stores and capacity interpolation."""

from __future__ import annotations

import numpy as np
import pytest

from src.memory import (
    MECHANISMS,
    capacity_from_curve,
    fit_capacity_slopes,
    local_growth_slopes,
    measure_interference_signal,
    random_unit_arrays,
    run_memory_experiment,
    theoretical_ratios,
)


def test_random_unit_arrays_have_unit_rows():
    keys, values = random_unit_arrays(17, 9, seed=1730)
    assert keys.shape == (17, 9)
    assert values.shape == (17, 9)
    np.testing.assert_allclose(np.linalg.norm(keys, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-12)


def test_measure_returns_five_nonnegative_ratios():
    result = measure_interference_signal(8, 8, seed=42, n_queries=8)
    assert set(result) == set(MECHANISMS)
    assert all(np.isfinite(value) and value >= 0 for value in result.values())
    # Exact interpolation below the dimension: no least-squares interference.
    assert result["least_squares"] == 0.0


def test_outer_product_ratio_near_theory():
    m, d = 64, 16
    result = measure_interference_signal(m, d, seed=7, n_queries=64)
    theoretical = np.sqrt((m - 1) / d)
    assert result["outer_product"] == pytest.approx(theoretical, rel=0.25)


def test_capacity_log_log_interpolation():
    ratios = np.array([0.5, 2.0])
    m_values = np.array([64.0, 128.0])
    expected = 64.0 * np.sqrt(2.0)
    assert capacity_from_curve(ratios, m_values) == pytest.approx(expected)


def test_capacity_zero_below_crossing():
    ratios = np.array([0.0, 1.5])
    m_values = np.array([64.0, 128.0])
    assert capacity_from_curve(ratios, m_values) == 128.0


def test_theoretical_finite_dimensional_relu_moments():
    ratios = theoretical_ratios(np.array([2048.0]), 64)
    # ReLU^2 uses E[z^4] = 3 / (d(d+2)), not the asymptotic 3/d^2.
    expected = np.sqrt(3.0 * 2047.0 / (2.0 * 64.0 * 66.0))
    assert ratios["relu_p2"][0] == pytest.approx(expected)
    assert ratios["least_squares"][0] > 0
    assert theoretical_ratios(np.array([64.0]), 64)["least_squares"][0] == 0.0


def test_fit_capacity_slopes_for_quadratic_capacity():
    capacities = {
        d: {mechanism: float(d**2) for mechanism in MECHANISMS}
        for d in (16, 32, 64)
    }
    slopes = fit_capacity_slopes(capacities)
    assert all(slope == pytest.approx(2.0) for slope in slopes.values())


def test_local_growth_slope_recovers_square_root():
    m_values = np.array([8.0, 16.0, 32.0, 64.0])
    ratios = np.sqrt(m_values)
    assert local_growth_slopes(
        ratios, m_values, lower=1.0, upper=10.0
    ) == pytest.approx(0.5)


def test_small_memory_experiment():
    result = run_memory_experiment(
        dimensions=(8,),
        m_values=np.array([4, 8, 16, 32, 64, 128, 256, 512]),
        draws=2,
        n_queries=16,
        seed=11,
        log=lambda message: None,
    )
    assert result["ratios_median"].shape == (1, 8, len(MECHANISMS))
    assert result["ratios_draws"].shape == (1, 8, 2, len(MECHANISMS))
    assert all(np.isfinite(value) for value in result["capacities"][8].values())
