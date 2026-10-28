"""Unit tests for count-based n-gram models and chi-squared pooling."""

from __future__ import annotations

import numpy as np
import pytest

from src.chi_square import pooled_chi_square
from src.ngram import (
    NGramModel,
    context_and_joint_keys,
    verify_bigram_normalization,
)


def test_context_and_joint_keys_order():
    tokens = np.array([1, 2, 3, 4], dtype=np.int64)
    codes, keys = context_and_joint_keys(tokens, context_length=2, vocab_size=4)
    np.testing.assert_array_equal(codes, [1 * 4 + 2, 2 * 4 + 3])
    np.testing.assert_array_equal(keys, [(1 * 4 + 2) * 4 + 3, (2 * 4 + 3) * 4 + 4])


def test_fit_and_manual_bigram_loss():
    tokens = np.array([0, 0, 0, 0], dtype=np.int64)
    model = NGramModel(order=2, vocab_size=2).fit(tokens)
    metrics = model.evaluate(tokens, smoothing=0.01)
    expected_probability = (3 + 0.01) / (3 + 0.01 * 2)
    assert metrics["nats_per_token"] == pytest.approx(-np.log(expected_probability))
    assert metrics["scored_tokens"] == 3


def test_conditional_matrix_rows_normalized_and_unseen_row_uniform():
    model = NGramModel(order=2, vocab_size=4).fit(
        np.array([0, 0, 1, 2, 1], dtype=np.int64)
    )
    conditional = model.conditional_matrix(0.1)
    np.testing.assert_allclose(conditional.sum(axis=1), 1.0, atol=1e-14)
    # Context 3 never occurs in this tiny training stream.
    np.testing.assert_allclose(conditional[3], np.full(4, 0.25), atol=1e-14)


def test_verify_bigram_normalization_sums_to_one():
    model = NGramModel(order=2, vocab_size=5).fit(
        np.array([0, 1, 2, 3, 4, 0, 2, 4], dtype=np.int64)
    )
    check = verify_bigram_normalization(model.conditional_matrix(0.01))
    for value in check["sums"].values():
        assert value == pytest.approx(1.0, abs=1e-12)
    assert check["max_error"] <= 1e-12


def test_sequence_normalization_for_known_two_state_matrix():
    conditional = np.array([[0.8, 0.2], [0.3, 0.7]])
    check = verify_bigram_normalization(conditional)
    assert check["sums"]["length_1"] == pytest.approx(1.0)
    assert check["sums"]["length_2"] == pytest.approx(1.0)
    assert check["sums"]["length_3"] == pytest.approx(1.0)


def test_unseen_fraction_on_unseen_transitions():
    model = NGramModel(order=2, vocab_size=3).fit(
        np.array([0, 1, 0, 1], dtype=np.int64)
    )
    metrics = model.evaluate(np.array([1, 1, 1], dtype=np.int64), smoothing=0.1)
    # The validation target 1 is preceded by context 1, which has no 1->1 count.
    assert metrics["unseen_fraction"] == pytest.approx(1.0)


def test_sample_bigram_reproducible():
    model = NGramModel(order=2, vocab_size=3).fit(
        np.array([0, 0, 1, 1, 2], dtype=np.int64)
    )
    first = model.sample_bigram(500, smoothing=0.1, seed=1730)
    second = model.sample_bigram(500, smoothing=0.1, seed=1730)
    different = model.sample_bigram(500, smoothing=0.1, seed=1731)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)
    assert first.shape == (500,)


def test_pooled_chi_square_perfect_fit():
    expected = np.array([[50.0, 30.0, 20.0], [10.0, 40.0, 50.0]])
    result = pooled_chi_square(expected.copy(), expected, min_expected=5.0)
    assert result["statistic"] == pytest.approx(0.0, abs=1e-12)
    assert result["p_value"] == pytest.approx(1.0)
    assert result["dof"] == 4


def test_pooled_chi_square_groups_rare_categories():
    expected = np.array([[100.0, 1.0, 1.0], [100.0, 2.0, 1.0]])
    observed = np.array([[100, 0, 2], [100, 0, 3]])
    result = pooled_chi_square(observed, expected, min_expected=5.0)
    assert result["all_individual_expected_ge_min"]
    assert all(detail["pooled_other"] for detail in result["row_details"])
    assert result["total_observations"] == int(observed.sum())
