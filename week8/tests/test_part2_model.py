"""Integration tests for Week 8 Part 2 using the pinned TinyStories data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.stats import chi2

from src.chi_square import pooled_chi_square
from src.ngram import NGramModel, verify_bigram_normalization
from src.part2_data import (
    DATA_DIR,
    TINYSTORIES_FILE,
    TOKENIZER_FILE,
    encode_stories,
    load_stories,
    load_tokenizer,
    split_stories,
    tokenizer_statistics,
)

pytestmark = pytest.mark.skipif(
    not (
        (DATA_DIR / TINYSTORIES_FILE).exists()
        and (DATA_DIR / TOKENIZER_FILE).exists()
    ),
    reason="TinyStories/tokenizer not downloaded",
)

SMOOTHING_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0]
REFERENCE_BPC = {3: 1.473, 4: 1.310, 5: 1.46, 6: 1.79}
REFERENCE_UNSEEN = {3: 0.006, 4: 0.042, 5: 0.122, 6: 0.235}


@pytest.fixture(scope="module")
def trained():
    stories = load_stories(DATA_DIR / TINYSTORIES_FILE)
    train_stories, valid_stories = split_stories(stories)
    tokenizer = load_tokenizer(DATA_DIR / TOKENIZER_FILE)
    train = encode_stories(tokenizer, train_stories)
    valid = encode_stories(tokenizer, valid_stories)
    stats = tokenizer_statistics(tokenizer, train, valid)

    cut = int(len(train.tokens) * 0.9)
    records = {}
    bigram_model = None
    for order in range(1, 7):
        selection_model = NGramModel(order, vocab_size=512).fit(
            train.tokens[:cut]
        )
        counts, denominators, seen = selection_model.score_counts(
            train.tokens, target_start=cut
        )
        losses = [
            NGramModel.metrics_from_counts(
                counts, denominators, seen, k, vocab_size=512
            )["nats_per_token"]
            for k in SMOOTHING_GRID
        ]
        chosen_k = SMOOTHING_GRID[int(np.argmin(losses))]
        model = NGramModel(order, vocab_size=512).fit(train.tokens)
        metrics = model.evaluate(valid.tokens, chosen_k, char_count=valid.chars)
        records[order] = (chosen_k, metrics)
        if order == 2:
            bigram_model = model

    return {
        "stories": stories,
        "train": train,
        "valid": valid,
        "stats": stats,
        "records": records,
        "bigram_model": bigram_model,
        "tokenizer": tokenizer,
    }


def test_dataset_split_and_tokenizer_statistics(trained):
    train = trained["train"]
    valid = trained["valid"]
    stats = trained["stats"]
    assert train.stories == 24867
    assert valid.stories == 2763
    assert stats["vocab_size"] == 512
    assert stats["train_unk"] == 0 and stats["valid_unk"] == 0
    assert stats["normalized_roundtrip"] == stats["stories"]
    assert stats["overall_chars_per_token"] == pytest.approx(2.05, abs=0.01)
    assert stats["valid_chars_per_token"] == pytest.approx(2.0518, abs=1e-3)


def test_selected_smoothing_values(trained):
    records = trained["records"]
    assert records[2][0] == pytest.approx(0.01)
    assert records[3][0] == pytest.approx(0.01)
    assert records[4][0] == pytest.approx(0.01)
    assert records[5][0] == pytest.approx(0.01)
    assert records[6][0] == pytest.approx(0.001)


def test_held_out_metrics_match_reference(trained):
    records = trained["records"]
    for order, reference in REFERENCE_BPC.items():
        _, metrics = records[order]
        assert metrics["bits_per_char"] == pytest.approx(reference, abs=0.03)
        assert metrics["unseen_fraction"] == pytest.approx(
            REFERENCE_UNSEEN[order], abs=0.02
        )
    bpc = {order: records[order][1]["bits_per_char"] for order in records}
    assert min(bpc, key=bpc.get) == 4


def test_bigram_normalization_integration(trained):
    model = trained["bigram_model"]
    check = verify_bigram_normalization(model.conditional_matrix(0.01))
    assert check["max_error"] <= 1e-12
    for value in check["sums"].values():
        assert value == pytest.approx(1.0, abs=1e-12)


def test_bigram_sampling_chi_square_integration(trained):
    model = trained["bigram_model"]
    conditional = model.conditional_matrix(0.01)
    train_tokens = trained["train"].tokens
    frequencies = np.bincount(train_tokens, minlength=512)
    top = np.argsort(-frequencies, kind="stable")[:20]
    samples = model.sample_bigram(250_000, 0.01, seed=1730)

    previous = samples[:-1]
    following = samples[1:]
    row_lookup = np.full(512, -1, dtype=np.int64)
    row_lookup[top] = np.arange(20)
    rows = row_lookup[previous]
    selected = rows >= 0
    observed = np.zeros((20, 512), dtype=np.int64)
    np.add.at(observed, (rows[selected], following[selected]), 1)
    expected = observed.sum(axis=1)[:, None] * conditional[top]
    result = pooled_chi_square(observed, expected, min_expected=5.0)
    assert result["p_value"] > 0.001
    assert result["dof"] > 0
    assert result["statistic"] > 0
    # The tail probability is a valid chi-squared survival probability.
    assert chi2.sf(result["statistic"], result["dof"]) == pytest.approx(
        result["p_value"]
    )
