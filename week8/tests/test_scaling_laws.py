"""Tests for the Part 6 Chinchilla scaling-law fit and bootstrap."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.scaling_laws import (
    CHINCHILLA_COMPUTE,
    PARAMETER_NAMES,
    PUBLISHED_PARAMS,
    REFERENCE_PARAMS,
    bootstrap_fits,
    compute_optimal,
    fit_scaling_law,
    load_chinchilla_data,
    log_residuals,
    predict_loss,
    ratio_at_compute,
    summarize_bootstrap,
)

WEEK_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = WEEK_DIR / "data" / "chinchilla-runs.csv"


def test_load_data_removes_five_highest_losses():
    data = load_chinchilla_data(DATA_FILE)
    assert data["n_total"] == 245
    assert data["n_used"] == 240
    assert len(data["removed"]) == 5
    assert all(
        row["loss"] >= data["L"].max() for row in data["removed"]
    )
    np.testing.assert_allclose(
        data["D"], data["C"] / (6.0 * data["N"])
    )


def test_predict_loss_formula():
    params = (1.8, 400.0, 2000.0, 0.34, 0.28)
    N = np.array([1e9])
    D = np.array([2e10])
    expected = 1.8 + 400.0 / 1e9**0.34 + 2000.0 / 2e10**0.28
    assert predict_loss(params, N, D)[0] == pytest.approx(expected)


@pytest.fixture(scope="module")
def fitted():
    data = load_chinchilla_data(DATA_FILE)
    fit = fit_scaling_law(data["N"], data["D"], data["L"])
    return data, fit


def test_fit_matches_reference_and_acceptance(fitted):
    _, fit = fitted
    params = fit["params"]
    assert fit["success"]
    assert 0.32 <= params["alpha"] <= 0.38
    assert 0.32 <= params["beta"] <= 0.42
    assert 1.75 <= params["E"] <= 1.90
    for name in PARAMETER_NAMES:
        assert params[name] == pytest.approx(
            REFERENCE_PARAMS[name], rel=0.05
        )


def test_residual_means_match_verification(fitted):
    data, fit = fitted
    our = log_residuals(
        tuple(fit["params"][name] for name in PARAMETER_NAMES),
        data["N"],
        data["D"],
        data["L"],
    )
    published = log_residuals(
        tuple(PUBLISHED_PARAMS[name] for name in PARAMETER_NAMES),
        data["N"],
        data["D"],
        data["L"],
    )
    assert abs(our.mean()) <= 0.005
    assert published.mean() > 0.01


def test_compute_optimal_ratio(fitted):
    _, fit = fitted
    params = tuple(fit["params"][name] for name in PARAMETER_NAMES)
    n_optimal, d_optimal = compute_optimal(
        params, np.array([CHINCHILLA_COMPUTE])
    )
    ratio = d_optimal[0] / n_optimal[0]
    assert 10.0 <= ratio <= 30.0
    assert ratio == pytest.approx(17.9, rel=0.05)
    result = ratio_at_compute(params, CHINCHILLA_COMPUTE)
    assert result["D_over_N"] == pytest.approx(ratio)


def test_small_bootstrap_summary(fitted):
    data, fit = fitted
    bootstrap = bootstrap_fits(
        data["N"],
        data["D"],
        data["L"],
        fit["vector"],
        n_bootstrap=5,
        seed=7,
        n_starts=1,
    )
    assert bootstrap["draws"].shape == (5, 5)
    assert bootstrap["n_bootstrap"] == 5
    summary = summarize_bootstrap(bootstrap["draws"])
    for name in PARAMETER_NAMES:
        assert summary[name]["ci95_lower"] <= summary[name]["ci95_upper"]
        assert summary[name]["std_error"] >= 0.0
