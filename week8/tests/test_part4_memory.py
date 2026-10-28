"""Tests for Part 4 picture memory and attention-style recall."""

from __future__ import annotations

import numpy as np
import pytest

from src.part4_memory import (
    METHODS,
    build_memory,
    make_character,
    recall_from_memory,
    run_picture_experiment,
    stable_softmax,
)


def test_character_bitmap_is_binary_and_recognizable():
    character = make_character()
    assert character.shape == (32, 32)
    assert set(np.unique(character)) == {-1.0, 1.0}
    ink = np.count_nonzero(character > 0)
    assert 150 < ink < 450
    # The apex, crossbar, and two feet must contain ink.
    assert character[4, 15] > 0
    assert character[20, 15] > 0
    assert character[27, 7] > 0 and character[27, 24] > 0


def test_stable_softmax_handles_large_scores():
    scores = np.array([1000.0, 1001.0, 1002.0, -1000.0])
    probabilities = stable_softmax(scores)
    assert probabilities.sum() == pytest.approx(1.0)
    assert np.all(probabilities >= 0)
    np.testing.assert_allclose(
        probabilities,
        stable_softmax(scores - scores.max()),
    )


def test_build_memory_shapes_and_target_picture():
    character = make_character()
    keys, values = build_memory(9, character, seed=1730)
    assert keys.shape == (9, 64)
    np.testing.assert_allclose(np.linalg.norm(keys, axis=1), 1.0)
    np.testing.assert_array_equal(values[0], character.ravel())
    assert set(np.unique(values[1:])) == {-1.0, 1.0}
    assert values.shape == (9, 1024)


def test_single_memory_recalls_exactly():
    character = make_character()
    keys, values = build_memory(1, character, seed=42)
    records = recall_from_memory(keys, values)
    for method in METHODS:
        assert records[method]["R"] == pytest.approx(0.0, abs=1e-14)
        assert records[method]["cosine_similarity"] == pytest.approx(1.0)
        np.testing.assert_allclose(
            records[method]["normalized_image"], character
        )


def test_m64_ratio_ordering():
    character = make_character()
    keys, values = build_memory(64, character, seed=1730 + 64)
    records = recall_from_memory(keys, values)
    assert records["outer_product"]["R"] > records["relu"]["R"]
    assert records["relu"]["R"] > records["relu_squared"]["R"]
    assert records["relu_squared"]["R"] > records["softmax"]["R"]
    assert 0.0 < records["softmax"]["target_weight"] < 1.0
    assert all(
        np.isfinite(records[method]["R"]) for method in METHODS
    )


def test_full_experiment_matches_verification_values():
    experiment = run_picture_experiment()
    records = experiment["records"]
    expected = {
        64: [0.9766, 0.6229, 0.1325, 0.0063],
        256: [2.2451, 1.6727, 0.3895, 0.0183],
        1024: [4.1837, 3.0111, 0.6846, 0.0329],
        4096: [8.0405, 5.7568, 1.2240, 0.0612],
    }
    for m, values in expected.items():
        measured = [records[m][method]["R"] for method in METHODS]
        np.testing.assert_allclose(measured, values, atol=5e-4)
    assert records[1024]["outer_product"]["R"] == pytest.approx(
        4.0, rel=0.15
    )
    assert records[1024]["relu"]["R"] == pytest.approx(2.83, rel=0.15)
    assert records[4096]["relu_squared"]["R"] > 1.0
    assert max(records[m]["softmax"]["R"] for m in expected) < 0.1
    assert min(
        records[m]["softmax"]["cosine_similarity"] for m in expected
    ) >= 0.95
