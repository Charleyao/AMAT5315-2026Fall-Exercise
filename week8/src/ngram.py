"""Count-based n-gram language models with add-k smoothing."""

from __future__ import annotations

import numpy as np

DEFAULT_VOCAB_SIZE = 512
DEFAULT_SMOOTHING_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0)


def context_and_joint_keys(
    tokens: np.ndarray,
    context_length: int,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode contexts and (context, next-token) pairs as base-V integers.

    A context of length ``m`` is packed as ``x0*V^(m-1) + ... + x_{m-1}``;
    the joint key appends the next token as the least significant digit.
    For ``m=5`` the largest joint key is below ``512**6 = 2**54`` and safely
    fits in int64.
    """
    tokens = np.asarray(tokens, dtype=np.int64)
    if tokens.ndim != 1:
        raise ValueError("tokens must be a 1-D sequence")
    n_targets = len(tokens) - context_length
    if n_targets <= 0:
        raise ValueError(
            f"need more than {context_length} tokens, got {len(tokens)}"
        )

    codes = np.zeros(n_targets, dtype=np.int64)
    for offset in range(context_length):
        codes = codes * vocab_size + tokens[offset : offset + n_targets]
    keys = codes * vocab_size + tokens[context_length:]
    return codes, keys


def _lookup(sorted_values: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find query positions in a sorted array without assuming all are present."""
    if len(sorted_values) == 0:
        return np.zeros(len(query), dtype=np.int64), np.zeros(len(query), dtype=bool)
    positions = np.searchsorted(sorted_values, query)
    found = np.zeros(len(query), dtype=bool)
    valid = positions < len(sorted_values)
    found[valid] = sorted_values[positions[valid]] == query[valid]
    positions = np.minimum(positions, len(sorted_values) - 1)
    return positions, found


class NGramModel:
    """Sparse count-based n-gram model.

    Training stores only observed ``(context, next_token)`` keys and context
    keys, so the 6-gram model never materializes a ``512**6`` table.
    """

    def __init__(
        self,
        order: int,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
    ):
        if order < 1:
            raise ValueError("order must be at least 1")
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        self.order = order
        self.vocab_size = vocab_size
        self.context_length = order - 1
        self.joint_keys: np.ndarray | None = None
        self.joint_counts: np.ndarray | None = None
        self.context_keys: np.ndarray | None = None
        self.context_counts: np.ndarray | None = None
        self.train_token_count = 0

    def fit(self, tokens: np.ndarray) -> "NGramModel":
        """Count all observed contexts and context-next-token pairs."""
        tokens = np.asarray(tokens, dtype=np.int64)
        codes, keys = context_and_joint_keys(
            tokens, self.context_length, self.vocab_size
        )
        self.joint_keys, self.joint_counts = np.unique(keys, return_counts=True)
        self.context_keys, self.context_counts = np.unique(codes, return_counts=True)
        self.train_token_count = len(tokens)
        return self

    def score_counts(
        self,
        tokens: np.ndarray,
        target_start: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (joint count, context count, seen) for scored targets.

        ``target_start`` selects the first target index to score; contexts may
        come from earlier tokens.  By default the first ``order-1`` tokens are
        skipped because they have no full context.
        """
        if self.joint_keys is None:
            raise RuntimeError("call fit() before scoring")
        tokens = np.asarray(tokens, dtype=np.int64)
        codes, keys = context_and_joint_keys(
            tokens, self.context_length, self.vocab_size
        )
        target_indices = np.arange(self.context_length, len(tokens))
        start = self.context_length if target_start is None else target_start
        start = max(start, self.context_length)
        selected = target_indices >= start
        codes = codes[selected]
        keys = keys[selected]

        joint_positions, joint_seen = _lookup(self.joint_keys, keys)
        context_positions, context_seen = _lookup(self.context_keys, codes)
        counts = np.where(joint_seen, self.joint_counts[joint_positions], 0)
        denominators = np.where(
            context_seen, self.context_counts[context_positions], 0
        )
        return counts, denominators, joint_seen

    @staticmethod
    def metrics_from_counts(
        counts: np.ndarray,
        denominators: np.ndarray,
        seen: np.ndarray,
        smoothing: float,
        *,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        char_count: int | None = None,
    ) -> dict:
        """Compute nats/token, perplexity, bits/char, and unseen fraction."""
        if smoothing <= 0.0:
            raise ValueError("add-k smoothing requires k > 0")
        probabilities = (counts + smoothing) / (
            denominators + smoothing * vocab_size
        )
        log_probabilities = np.log(probabilities)
        nats_per_token = float(-log_probabilities.mean())
        total_nats = float(-log_probabilities.sum())
        result = {
            "nats_per_token": nats_per_token,
            "perplexity": float(np.exp(nats_per_token)),
            "scored_tokens": int(len(counts)),
            "unseen_fraction": float(1.0 - seen.mean()),
            "smoothing": float(smoothing),
        }
        if char_count is not None:
            result["bits_per_char"] = float(
                total_nats / (np.log(2.0) * char_count)
            )
        else:
            result["bits_per_char"] = None
        return result

    def evaluate(
        self,
        tokens: np.ndarray,
        smoothing: float,
        *,
        char_count: int | None = None,
        target_start: int | None = None,
    ) -> dict:
        counts, denominators, seen = self.score_counts(tokens, target_start)
        return self.metrics_from_counts(
            counts,
            denominators,
            seen,
            smoothing,
            vocab_size=self.vocab_size,
            char_count=char_count,
        )

    def conditional_matrix(self, smoothing: float) -> np.ndarray:
        """Dense ``V x V`` conditional table for the bigram model."""
        if self.order != 2:
            raise ValueError("conditional_matrix is only defined for order=2")
        if self.joint_keys is None:
            raise RuntimeError("call fit() before building the matrix")

        joint_matrix = np.zeros(
            (self.vocab_size, self.vocab_size), dtype=np.int64
        )
        joint_matrix.reshape(-1)[self.joint_keys] = self.joint_counts
        context_counts = np.zeros(self.vocab_size, dtype=np.int64)
        context_counts[self.context_keys] = self.context_counts
        probabilities = (joint_matrix + smoothing) / (
            context_counts[:, None] + smoothing * self.vocab_size
        )
        return probabilities

    def sample_bigram(
        self,
        n_tokens: int,
        smoothing: float,
        seed: int,
        initial_token: int | None = None,
    ) -> np.ndarray:
        """Ancestral sampling from the bigram model.

        The first token is uniform unless ``initial_token`` is supplied; every
        later token is drawn from ``P(. | previous token)``.
        """
        if self.order != 2:
            raise ValueError("sample_bigram is only defined for order=2")
        if n_tokens < 1:
            raise ValueError("n_tokens must be positive")
        probabilities = self.conditional_matrix(smoothing)
        cumulative = np.cumsum(probabilities, axis=1)
        cumulative[:, -1] = 1.0

        rng = np.random.default_rng(seed)
        samples = np.empty(n_tokens, dtype=np.int32)
        samples[0] = (
            int(rng.integers(self.vocab_size))
            if initial_token is None
            else int(initial_token)
        )
        for index in range(1, n_tokens):
            draw = rng.random()
            next_token = int(
                np.searchsorted(cumulative[int(samples[index - 1])], draw)
            )
            samples[index] = min(next_token, self.vocab_size - 1)
        return samples


def verify_bigram_normalization(
    conditional: np.ndarray,
) -> dict:
    """Enumerate lengths 1-3 and sum their model probabilities.

    Length-1 tokens use a uniform initial distribution.  Length-3 probability
    mass is contracted over the third token, so only a ``V x V`` array is
    allocated rather than a ``V x V x V`` array.
    """
    vocab_size = conditional.shape[0]
    initial = np.full(vocab_size, 1.0 / vocab_size, dtype=np.float64)
    joint2 = initial[:, None] * conditional
    third_token_sums = conditional.sum(axis=1)
    joint3_projected = joint2 * third_token_sums[None, :]

    sums = {
        "length_1": float(initial.sum()),
        "length_2": float(joint2.sum()),
        "length_3": float(joint3_projected.sum()),
    }
    errors = {name: abs(value - 1.0) for name, value in sums.items()}
    return {
        "sums": sums,
        "errors": errors,
        "max_error": max(errors.values()),
        "max_conditional_row_sum_error": float(
            np.abs(conditional.sum(axis=1) - 1.0).max()
        ),
        "enumerated_length_1": vocab_size,
        "enumerated_length_2": vocab_size**2,
        "contracted_length_3_over": "third token",
        "initial_distribution": "uniform over the 512 tokens",
        "allocated_cells": int(joint2.size),
    }
