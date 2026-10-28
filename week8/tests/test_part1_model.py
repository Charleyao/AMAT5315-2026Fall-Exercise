"""Integration test: pinned model reproduces the Part 1 acceptance criteria."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.embeddings import (
    SPECIAL_TOKENS,
    analyze_relation,
    load_embedding_table,
    load_tokenizer,
    shuffled_table,
)
from src.relations import RELATIONS

WEEK_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = WEEK_DIR / "data"
MODEL_PATH = DATA_DIR / "model.safetensors"
TOKENIZER_PATH = DATA_DIR / "tokenizer.json"

pytestmark = pytest.mark.skipif(
    not (MODEL_PATH.exists() and TOKENIZER_PATH.exists()),
    reason="potion-base-8M files are not downloaded (run src.download_model)",
)

EXPECTED_MEANS = {
    "country_capital": 0.466,
    "male_female": 0.549,
    "adjective_comparative": 0.433,
}


@pytest.fixture(scope="module")
def model():
    return load_embedding_table(MODEL_PATH), load_tokenizer(TOKENIZER_PATH)


def test_embedding_shape_and_token_ids(model):
    table, tokenizer = model
    assert table.shape == (29528, 256)

    results = {
        relation: analyze_relation(table, tokenizer, pairs)
        for relation, pairs in RELATIONS.items()
    }
    pairs = [pair for result in results.values() for pair in result["pairs"]]
    assert len(pairs) == 27
    assert all(0 <= pair["token_id_a"] < table.shape[0] for pair in pairs)
    assert all(0 <= pair["token_id_b"] < table.shape[0] for pair in pairs)
    # Every one of the 54 word occurrences is a single non-special token.
    for pair in pairs:
        for word in (pair["a"], pair["b"]):
            encoding = tokenizer.encode(word)
            word_tokens = [
                token for token in encoding.tokens if token not in SPECIAL_TOKENS
            ]
            assert word_tokens == [word]


def test_pretrained_means_pass_acceptance(model):
    table, tokenizer = model
    for relation, pairs in RELATIONS.items():
        result = analyze_relation(table, tokenizer, pairs)
        assert result["mean_cosine"] >= 0.30
        assert result["mean_cosine"] == pytest.approx(
            EXPECTED_MEANS[relation], abs=0.01
        )


def test_shuffled_control_means_are_near_zero(model):
    table, tokenizer = model
    control = shuffled_table(table, seed=1730)
    for pairs in RELATIONS.values():
        result = analyze_relation(control, tokenizer, pairs)
        assert -0.15 <= result["mean_cosine"] <= 0.15
