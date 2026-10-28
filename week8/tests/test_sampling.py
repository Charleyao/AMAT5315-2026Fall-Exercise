"""Tests for Part 5 sampling distributions and diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from src.ngram import NGramModel
from src.part5_generation import generate_scored_sequences, z_score_from_sequences
from src.sampling import (
    SAMPLING_SETTINGS,
    apply_truncation,
    grouped_chi_square,
    ngram_next_distribution,
    sample_indices,
    temperature_probs,
)
from src.transformer import TransformerConfig, initialize_params


def test_temperature_probs_sum_to_one_and_handle_batches():
    logits = np.random.default_rng(0).normal(size=(3, 512))
    probabilities = temperature_probs(logits, 0.8)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert np.all(probabilities >= 0)


def test_top_k_keeps_exactly_twenty_tokens():
    logits = np.random.default_rng(1).normal(size=512)
    probs = apply_truncation(logits, SAMPLING_SETTINGS["top_k"])
    assert np.count_nonzero(probs) == 20
    assert probs.sum() == pytest.approx(1.0)
    assert np.all(probs <= 1.0)


def test_top_p_is_minimal_set_above_threshold():
    logits = np.random.default_rng(2).normal(size=512)
    probs = apply_truncation(logits, SAMPLING_SETTINGS["top_p"])
    sorted_probs = np.sort(probs)[::-1]
    assert np.count_nonzero(probs) < 512
    assert np.cumsum(sorted_probs)[
        np.count_nonzero(probs) - 1
    ] == pytest.approx(1.0)
    if np.count_nonzero(probs) > 1:
        before = np.cumsum(sorted_probs)[
            np.count_nonzero(probs) - 2
        ]
        assert before < 1.0


def test_min_p_keeps_only_tokens_above_fraction_of_max():
    logits = np.random.default_rng(3).normal(size=512)
    setting = SAMPLING_SETTINGS["min_p"]
    base = temperature_probs(logits, setting["temperature"])
    probs = apply_truncation(logits, setting)
    kept = probs > 0
    assert np.all(base[kept] >= setting["min_p"] * base.max())


def test_tiny_temperature_returns_argmax():
    logits = np.random.default_rng(4).normal(size=512)
    probs = temperature_probs(logits, 0.0001)
    assert int(np.argmax(probs)) == int(np.argmax(logits))
    assert probs.max() > 1.0 - 1e-9
    samples = sample_indices(
        probs, 1000, np.random.default_rng(1848)
    )
    assert np.all(samples == np.argmax(logits))


def test_grouped_chi_square_handles_zero_expected_categories():
    observed = np.zeros(512, dtype=np.int64)
    expected = np.zeros(512, dtype=np.float64)
    observed[[1, 2, 3]] = [50, 30, 20]
    expected[[1, 2, 3]] = [50, 30, 20]
    result = grouped_chi_square(observed, expected)
    assert result["statistic"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)
    assert np.isfinite(result["p_value"])


def test_ngram_distribution_is_normalized():
    tokens = np.array([0, 1, 0, 2, 0, 1, 3], dtype=np.int64)
    model = NGramModel(order=2, vocab_size=4).fit(tokens)
    probabilities = ngram_next_distribution(model, [0], 0.1)
    assert probabilities.shape == (4,)
    assert probabilities.sum() == pytest.approx(1.0)
    assert np.all(probabilities > 0)


def test_scored_generation_shapes_and_z_score():
    import jax

    params = initialize_params(
        jax.random.PRNGKey(5), TransformerConfig()
    )
    prefixes = np.random.default_rng(0).integers(
        0, 512, size=(3, 64), dtype=np.int32
    )
    result = generate_scored_sequences(
        params, prefixes, 4, 1.0, seed=1848
    )
    assert result["tokens"].shape == (3, 4)
    assert result["surprise"].shape == (3, 4)
    assert result["entropy"].shape == (3, 4)
    stats = z_score_from_sequences(result)
    assert np.isfinite(stats["z_score"])
    assert stats["n_sequences"] == 3
