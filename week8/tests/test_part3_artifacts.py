"""Integration checks on saved Week 8 Part 3 artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

WEEK_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = WEEK_DIR / "artifacts"
EVIDENCE_DIR = WEEK_DIR / "evidence"
RESULTS = ARTIFACT_DIR / "part3_results.json"

pytestmark = pytest.mark.skipif(
    not RESULTS.exists(), reason="run `python -m src.part3` first"
)


@pytest.fixture(scope="module")
def results():
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def test_all_verifications_passed(results):
    assert results["acceptance"]["verify1"]
    assert results["acceptance"]["verify2"]
    assert results["acceptance"]["verify3"]
    assert results["acceptance"]["all_passed"]


def test_memory_capacities_and_slopes(results):
    capacities = {
        int(d): values
        for d, values in results["memory"]["capacities"].items()
    }
    d64 = capacities[64]
    assert 50 <= d64["outer_product"] <= 85
    assert 100 <= d64["least_squares"] <= 170
    assert results["verification"]["1"]["least_squares_max_r_for_m_le_64"] < 1e-10
    assert abs(capacities[64]["relu_p2"] - 2871) / 2871 <= 0.10
    assert abs(capacities[64]["relu_p3"] - 38124) / 38124 <= 0.10

    slopes = results["memory"]["slopes"]
    for mechanism in ("outer_product", "least_squares", "relu_p1"):
        assert abs(slopes[mechanism] - 1.0) <= 0.10
    assert abs(slopes["relu_p2"] - 2.0) <= 0.15
    assert slopes["relu_p3"] > 2.5


def test_mlp_loss_and_checkpoint(results):
    mlp = results["mlp"]
    assert abs(mlp["final_validation_loss"] - 3.038) <= 0.01
    assert mlp["config"]["steps"] == 4000
    assert mlp["config"]["batch_size"] == 16384
    assert mlp["config"]["peak_lr"] == 0.006
    assert mlp["config"]["embed_dim"] == 64
    assert mlp["config"]["hidden_dim"] == 512

    checkpoint_path = ARTIFACT_DIR / "part3_mlp_checkpoint.npz"
    assert checkpoint_path.exists()
    with np.load(checkpoint_path) as checkpoint:
        assert checkpoint["E"].shape == (512, 64)
        assert checkpoint["W1"].shape == (64, 512)
        assert checkpoint["b1"].shape == (512,)
        assert checkpoint["W2"].shape == (512, 512)
        assert checkpoint["b2"].shape == (512,)
        assert all(
            np.all(np.isfinite(checkpoint[name]))
            for name in ("E", "W1", "b1", "W2", "b2")
        )


def test_memory_npz_shapes(results):
    path = ARTIFACT_DIR / "part3_memory_curves.npz"
    assert path.exists()
    with np.load(path) as data:
        n_dims = len(results["memory"]["dimensions"])
        n_m = len(results["memory"]["m_values"])
        n_mech = len(results["memory"]["mechanisms"])
        assert data["ratios_median"].shape == (n_dims, n_m, n_mech)
        assert data["ratios_draws"].shape == (n_dims, n_m, 3, n_mech)
        assert data["capacities"].shape == (n_dims, n_mech)


def test_required_figures_exist():
    for name in (
        "figure9_memory_capacity_d64.png",
        "figure10_capacity_vs_dimension.png",
        "figure_mlp_training.png",
        "part3_verify.txt",
    ):
        path = EVIDENCE_DIR / name
        assert path.exists() and path.stat().st_size > 0
