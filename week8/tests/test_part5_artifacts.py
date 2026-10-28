"""Saved-output checks for Week 8 Part 5."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.transformer import load_checkpoint, parameter_count

WEEK_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = WEEK_DIR / "artifacts"
EVIDENCE_DIR = WEEK_DIR / "evidence"
RESULTS = ARTIFACT_DIR / "part5_results.json"

pytestmark = pytest.mark.skipif(
    not RESULTS.exists(), reason="run `python -m src.part5` first"
)


@pytest.fixture(scope="module")
def results():
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def test_all_part5_verifications_passed(results):
    assert results["acceptance"]["verify1"]
    assert results["acceptance"]["verify2"]
    assert results["acceptance"]["verify3"]
    assert results["acceptance"]["verify4"]
    assert results["acceptance"]["all_passed"]


def test_transformer_metrics(results):
    model = results["model"]
    assert model["parameter_count"] == 328_832
    assert model["final_bits_per_char"] < 1.310
    assert model["final8_mean_loss"] == pytest.approx(1.65, abs=0.15)
    assert results["verification"]["2"]["after32_mean"] < 2.09


def test_sampling_and_verify4_statistics(results):
    chi = results["verification"]["3"]["chi_square"]
    assert len(chi) == 4
    assert all(item["p_value"] > 0.001 for item in chi)
    tiny = results["verification"]["3"]["tiny_temperature"]
    assert tiny["all_samples_argmax"]

    verify4 = results["verification"]["4"]
    assert -4.0 < verify4["temperature_1"]["z_score"] < 4.0
    assert abs(verify4["temperature_0_7"]["z_score"]) > 40.0


def test_samples_and_sampling_examples(results):
    samples = results["verification"]["1"]["samples"]
    assert [item["model"] for item in samples] == [
        "Uniform",
        "Unigram",
        "Bigram",
        "Trigram",
        "Transformer",
    ]
    assert all(len(item["tokens"]) == 60 for item in samples)

    path = ARTIFACT_DIR / "part5_sampling_method_examples.json"
    examples = json.loads(path.read_text(encoding="utf-8"))
    assert len(examples) == 4
    assert all(len(item["tokens"]) == 60 for item in examples)


def test_checkpoint_and_figures_exist(results):
    checkpoint = ARTIFACT_DIR / results["model"]["checkpoint"]
    assert checkpoint.exists()
    params, config = load_checkpoint(checkpoint)
    assert parameter_count(params) == 328_832
    assert config.context_length == 64

    for key in (
        "bpc_comparison",
        "model_samples",
        "context_length",
        "training",
        "verify4",
    ):
        figure = Path(results["figures"][key])
        assert figure.exists() and figure.stat().st_size > 0

    assert (EVIDENCE_DIR / "part5_verify.txt").exists()
    assert (ARTIFACT_DIR / "part5_settings.json").exists()
    assert (ARTIFACT_DIR / "part5_transformer_results.json").exists()
