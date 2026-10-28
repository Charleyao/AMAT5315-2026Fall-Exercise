"""Checks on saved Week 8 Part 4 outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

WEEK_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = WEEK_DIR / "artifacts"
EVIDENCE_DIR = WEEK_DIR / "evidence"
RESULTS = ARTIFACT_DIR / "part4_results.json"

pytestmark = pytest.mark.skipif(
    not RESULTS.exists(), reason="run `python -m src.part4` first"
)


@pytest.fixture(scope="module")
def results():
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def test_part4_acceptance(results):
    assert results["acceptance"]["verify1"]
    assert results["acceptance"]["all_passed"]


def test_saved_recall_table_has_16_finite_values(results):
    table = results["R_table"]
    assert sorted(int(m) for m in table) == [64, 256, 1024, 4096]
    values = [
        value
        for row in table.values()
        for value in row.values()
    ]
    assert len(values) == 16
    assert all(np.isfinite(value) and value >= 0 for value in values)

    assert results["R_table"]["4096"]["relu_squared"] > 1.0
    assert all(
        row["softmax"] < 0.1 for row in table.values()
    )


def test_saved_data_and_figure_exist():
    for name in (
        "part4_recall_data.npz",
        "part4_results.json",
        "part4_settings.json",
    ):
        assert (ARTIFACT_DIR / name).exists()

    with np.load(ARTIFACT_DIR / "part4_recall_data.npz") as data:
        assert data["character"].shape == (32, 32)
        assert data["R"].shape == (4, 4)
        assert data["normalized_images"].shape == (4, 4, 32, 32)

    figure = EVIDENCE_DIR / "figure13_attention_recall.png"
    report = EVIDENCE_DIR / "part4_verify.txt"
    assert figure.exists() and figure.stat().st_size > 0
    assert report.exists() and report.stat().st_size > 0
