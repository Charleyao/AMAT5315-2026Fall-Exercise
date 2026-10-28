"""Saved-output checks for Week 8 Part 6."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

WEEK_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = WEEK_DIR / "artifacts"
EVIDENCE_DIR = WEEK_DIR / "evidence"
RESULTS = ARTIFACT_DIR / "part6_results.json"

pytestmark = pytest.mark.skipif(
    not RESULTS.exists(), reason="run `python -m src.part6` first"
)


@pytest.fixture(scope="module")
def results():
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def test_all_verifications_passed(results):
    assert results["acceptance"]["verify1"]
    assert results["acceptance"]["verify2"]
    assert results["acceptance"]["verify3"]
    assert results["acceptance"]["all_passed"]
    assert results["dataset"]["n_total"] == 245
    assert results["dataset"]["n_used"] == 240


def test_fit_and_residuals(results):
    params = results["fit"]["params"]
    assert 0.32 <= params["alpha"] <= 0.38
    assert 0.32 <= params["beta"] <= 0.42
    assert 1.75 <= params["E"] <= 1.90
    assert abs(results["verification"]["2"]["our_mean_residual"]) <= 0.005
    assert results["verification"]["2"]["published_mean_residual"] > 0.01


def test_compute_optimal_ratio_and_bootstrap(results):
    verify3 = results["verification"]["3"]
    assert 10.0 <= verify3["our"]["D_over_N"] <= 30.0
    lower, upper = verify3["bootstrap_ratio_ci95"]
    assert lower < verify3["our"]["D_over_N"] < upper
    assert np.isfinite([lower, upper]).all()

    with np.load(ARTIFACT_DIR / "part6_bootstrap.npz") as data:
        assert data["draws"].shape == (400, 5)
        assert data["ratios"].shape == (400,)


def test_figures_and_tables_exist(results):
    for key in ("figure19", "figure20", "figure21"):
        figure = Path(results["figures"][key])
        assert figure.exists() and figure.stat().st_size > 0
    assert (ARTIFACT_DIR / "part6_compute_optimal.csv").exists()
    assert (ARTIFACT_DIR / "part6_settings.json").exists()
    assert (EVIDENCE_DIR / "part6_verify.txt").exists()
