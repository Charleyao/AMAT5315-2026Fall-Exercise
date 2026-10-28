"""Static word embeddings and relational offsets (NumPy only, no PyTorch).

The model is ``minishlab/potion-base-8M``: a single ``embeddings`` matrix of
shape ``(vocab_size, dim)`` stored in a safetensors file.  Token ids come from
the WordPiece tokenizer shipped with the model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from safetensors.numpy import load_file
from tokenizers import Tokenizer

# Special WordPiece tokens that must not be used as word embeddings.
SPECIAL_TOKENS: tuple[str, ...] = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")


def load_embedding_table(path: str | Path) -> np.ndarray:
    """Load the embedding matrix from a safetensors file without PyTorch."""
    tensors = load_file(str(path))
    if "embeddings" not in tensors:
        raise KeyError(f"expected an 'embeddings' tensor in {path}, got {list(tensors)}")
    table = np.asarray(tensors["embeddings"])
    if table.ndim != 2:
        raise ValueError(f"expected a 2-D embedding table, got shape {table.shape}")
    return table


def load_tokenizer(path: str | Path) -> Tokenizer:
    """Load the WordPiece tokenizer from ``tokenizer.json``."""
    return Tokenizer.from_file(str(path))


def word_token_id(tokenizer: Tokenizer, word: str) -> int:
    """Return the single non-special token id used for ``word``.

    Raises ``ValueError`` if the word is split into several tokens or consists
    only of special tokens, so that the analysis never silently averages
    subword embeddings.
    """
    encoding = tokenizer.encode(word)
    word_ids = [
        token_id
        for token_id, token in zip(encoding.ids, encoding.tokens)
        if token not in SPECIAL_TOKENS
    ]
    if len(word_ids) != 1:
        raise ValueError(
            f"{word!r} must be exactly one word token, got ids={encoding.ids} "
            f"tokens={encoding.tokens}"
        )
    return int(word_ids[0])


def unit_vector(vector: np.ndarray) -> np.ndarray:
    """Return ``vector`` scaled to unit Euclidean length."""
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return vector / norm


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    """Normalize every row of ``matrix`` to unit length."""
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("cannot normalize a matrix containing zero rows")
    return matrix / norms


def offset_vectors(
    table: np.ndarray,
    token_ids: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Build ``unit(B) - unit(A)`` offsets for ordered token-id pairs."""
    pair_array = np.asarray(token_ids, dtype=np.int64)
    if pair_array.ndim != 2 or pair_array.shape[1] != 2:
        raise ValueError("token_ids must be a sequence of (id_a, id_b) pairs")
    if pair_array.size and (
        pair_array.min() < 0 or pair_array.max() >= table.shape[0]
    ):
        raise IndexError(
            f"token id outside embedding table with {table.shape[0]} rows"
        )
    unit_table = unit_rows(table)
    return unit_table[pair_array[:, 1]] - unit_table[pair_array[:, 0]]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors, clipped for float safety."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0.0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def leave_one_out_cosines(offsets: np.ndarray) -> np.ndarray:
    """Cosine of each offset against the mean of the *other* offsets.

    For pair ``i`` the reference is ``mean(offsets[j] for j != i)``.  This
    leave-one-out construction avoids trivially comparing a vector with a mean
    that contains itself.
    """
    offsets = np.asarray(offsets, dtype=np.float64)
    if offsets.ndim != 2:
        raise ValueError(f"offsets must be 2-D, got shape {offsets.shape}")
    if offsets.shape[0] < 2:
        raise ValueError("need at least two offsets for leave-one-out means")

    cosines = np.empty(offsets.shape[0], dtype=np.float64)
    for index in range(offsets.shape[0]):
        others = np.delete(offsets, index, axis=0)
        cosines[index] = cosine_similarity(offsets[index], others.mean(axis=0))
    return cosines


def shuffled_table(table: np.ndarray, seed: int = 1730) -> np.ndarray:
    """Return a copy of ``table`` with its rows randomly permuted."""
    rng = np.random.default_rng(seed)
    return np.asarray(table)[rng.permutation(table.shape[0])]


def analyze_relation(
    table: np.ndarray,
    tokenizer: Tokenizer,
    pairs: Iterable[tuple[str, str]],
) -> dict:
    """Tokenize ``pairs``, build offsets, and compute leave-one-out cosines."""
    pairs = list(pairs)
    token_ids = [
        (word_token_id(tokenizer, a), word_token_id(tokenizer, b))
        for a, b in pairs
    ]
    for (id_a, id_b), (word_a, word_b) in zip(token_ids, pairs):
        for token_id, word in ((id_a, word_a), (id_b, word_b)):
            if token_id >= table.shape[0]:
                raise IndexError(
                    f"token id {token_id} for {word!r} exceeds table rows "
                    f"({table.shape[0]})"
                )

    offsets = offset_vectors(table, token_ids)
    cosines = leave_one_out_cosines(offsets)

    pair_records = []
    for (word_a, word_b), (id_a, id_b), cosine in zip(pairs, token_ids, cosines):
        pair_records.append(
            {
                "a": word_a,
                "b": word_b,
                "token_id_a": id_a,
                "token_id_b": id_b,
                "cosine": float(cosine),
            }
        )

    return {
        "pairs": pair_records,
        "cosines": [float(value) for value in cosines],
        "mean_cosine": float(np.mean(cosines)),
        "std_cosine": float(np.std(cosines)),
    }
