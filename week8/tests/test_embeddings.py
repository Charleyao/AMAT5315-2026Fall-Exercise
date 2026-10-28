"""Tests for Week 8 Part 1 embedding utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from src.embeddings import (
    analyze_relation,
    cosine_similarity,
    leave_one_out_cosines,
    offset_vectors,
    shuffled_table,
    unit_rows,
    unit_vector,
    word_token_id,
)
from src.relations import RELATIONS


@dataclass
class FakeEncoding:
    ids: list[int]
    tokens: list[str]


class FakeTokenizer:
    """Small stand-in for tokenizers.Tokenizer in unit tests."""

    def __init__(self, mapping: dict[str, list[str]]):
        self.mapping = mapping
        self.ids = {token: index for index, token in enumerate(
            dict.fromkeys(token for tokens in mapping.values() for token in tokens)
        )}

    def encode(self, word: str) -> FakeEncoding:
        tokens = self.mapping[word]
        return FakeEncoding(ids=[self.ids[token] for token in tokens], tokens=tokens)


def test_unit_vector_has_unit_norm():
    result = unit_vector(np.array([3.0, 4.0]))
    np.testing.assert_allclose(result, [0.6, 0.8])
    assert np.linalg.norm(result) == pytest.approx(1.0)


def test_unit_vector_rejects_zero():
    with pytest.raises(ValueError, match="zero vector"):
        unit_vector(np.zeros(4))


def test_unit_rows_normalizes_each_row():
    result = unit_rows(np.array([[3.0, 4.0], [0.0, 2.0]]))
    np.testing.assert_allclose(result, [[0.6, 0.8], [0.0, 1.0]])


def test_offset_vectors_normalize_before_subtraction():
    table = np.array([[3.0, 0.0], [0.0, 2.0]])
    offsets = offset_vectors(table, [(0, 1)])
    np.testing.assert_allclose(offsets, [[-1.0, 1.0]])


def test_offset_vectors_validate_token_ids():
    table = np.ones((2, 2))
    with pytest.raises(IndexError):
        offset_vectors(table, [(0, 2)])


def test_cosine_similarity_known_values():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_leave_one_out_uses_other_offsets_only():
    offsets = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    cosines = leave_one_out_cosines(offsets)
    expected_first = cosine_similarity(offsets[0], offsets[1:].mean(axis=0))
    assert cosines[0] == pytest.approx(expected_first)
    assert cosines.shape == (3,)


def test_leave_one_out_identical_offsets_give_one():
    offsets = np.tile(np.array([1.0, 2.0, 3.0]), (9, 1))
    np.testing.assert_allclose(leave_one_out_cosines(offsets), np.ones(9))


def test_shuffled_table_is_reproducible_and_a_permutation():
    table = np.arange(30, dtype=float).reshape(10, 3)
    first = shuffled_table(table, seed=1730)
    second = shuffled_table(table, seed=1730)
    different = shuffled_table(table, seed=1731)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)
    np.testing.assert_array_equal(
        np.sort(first, axis=0), np.sort(table, axis=0)
    )


def test_word_token_id_ignores_special_tokens():
    tokenizer = FakeTokenizer({"paris": ["[CLS]", "paris", "[SEP]"]})
    token_id = word_token_id(tokenizer, "paris")
    assert token_id == tokenizer.ids["paris"]


def test_word_token_id_rejects_multi_token_words():
    tokenizer = FakeTokenizer({"unused": ["[CLS]", "un", "##used", "[SEP]"]})
    with pytest.raises(ValueError, match="exactly one word token"):
        word_token_id(tokenizer, "unused")


def test_relations_have_exactly_nine_pairs_each():
    assert set(RELATIONS) == {
        "country_capital",
        "male_female",
        "adjective_comparative",
    }
    for pairs in RELATIONS.values():
        assert len(pairs) == 9
        assert all(len(pair) == 2 for pair in pairs)
        assert len({pair[0] for pair in pairs}) == 9
    assert sum(len(pairs) for pairs in RELATIONS.values()) == 27


def test_analyze_relation_returns_records_and_valid_cosines():
    tokenizer = FakeTokenizer(
        {
            "a": ["[CLS]", "a", "[SEP]"],
            "b": ["[CLS]", "b", "[SEP]"],
            "c": ["[CLS]", "c", "[SEP]"],
            "d": ["[CLS]", "d", "[SEP]"],
        }
    )
    # The special tokens occupy rows 0 and 1; words map to rows 2..5, so use
    # a table with six rows to keep all ids valid.
    table = np.eye(6)
    result = analyze_relation(table, tokenizer, [("a", "b"), ("c", "d")])
    assert len(result["pairs"]) == 2
    assert len(result["cosines"]) == 2
    assert all(-1.0 <= value <= 1.0 for value in result["cosines"])
    assert result["mean_cosine"] == pytest.approx(np.mean(result["cosines"]))
