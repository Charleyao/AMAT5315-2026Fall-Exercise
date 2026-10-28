"""Sampling distributions, truncation, diagnostics, and n-gram sampling."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

VOCAB_SIZE = 512

SAMPLING_SETTINGS = {
    "temperature": {
        "label": "Temperature T=0.8",
        "kind": "temperature",
        "temperature": 0.8,
    },
    "top_k": {
        "label": "Top-k (k=20, T=0.8)",
        "kind": "top_k",
        "temperature": 0.8,
        "top_k": 20,
    },
    "top_p": {
        "label": "Top-p (p=0.9, T=0.8)",
        "kind": "top_p",
        "temperature": 0.8,
        "top_p": 0.9,
    },
    "min_p": {
        "label": "Min-p (p=0.1, T=0.7)",
        "kind": "min_p",
        "temperature": 0.7,
        "min_p": 0.1,
    },
}


def log_softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    return shifted - np.log(
        np.exp(shifted).sum(axis=-1, keepdims=True)
    )


def temperature_probs(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Numerically stable softmax(logits / T)."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    return np.exp(log_softmax(np.asarray(logits) / temperature))


def apply_truncation(probabilities: np.ndarray, setting: dict) -> np.ndarray:
    """Apply temperature + top-k/top-p/min-p and renormalize."""
    probs = temperature_probs(probabilities, float(setting["temperature"]))
    kind = setting["kind"]

    if kind == "temperature":
        return probs

    if kind == "top_k":
        k = int(setting["top_k"])
        keep = np.argpartition(-probs, k - 1)[:k]
        truncated = np.zeros_like(probs)
        truncated[keep] = probs[keep]
        return truncated / truncated.sum()

    if kind == "top_p":
        threshold = float(setting["top_p"])
        order = np.argsort(-probs)
        sorted_probs = probs[order]
        cumulative = np.cumsum(sorted_probs)
        n_keep = int(np.searchsorted(cumulative, threshold, side="left")) + 1
        n_keep = min(max(n_keep, 1), len(probs))
        truncated = np.zeros_like(probs)
        truncated[order[:n_keep]] = sorted_probs[:n_keep]
        return truncated / truncated.sum()

    if kind == "min_p":
        threshold = float(setting["min_p"]) * float(probs.max())
        truncated = np.where(probs >= threshold, probs, 0.0)
        return truncated / truncated.sum()

    raise ValueError(f"unknown sampling kind: {kind}")


def sample_indices(
    probabilities: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw tokens from a (possibly sparse) probability vector."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0
    uniforms = rng.random(n_samples)
    return np.searchsorted(cumulative, uniforms, side="right")


def grouped_chi_square(
    observed: np.ndarray,
    expected: np.ndarray,
    min_expected: float = 5.0,
) -> dict:
    """One-dimensional Pearson test after pooling rare categories."""
    observed = np.asarray(observed, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if observed.shape != expected.shape or observed.ndim != 1:
        raise ValueError("observed and expected must be equal 1-D arrays")
    keep = expected >= min_expected
    grouped_expected = list(expected[keep])
    grouped_observed = list(observed[keep])
    other_expected = float(expected[~keep].sum())
    if other_expected > 0.0:
        grouped_expected.append(other_expected)
        grouped_observed.append(float(observed[~keep].sum()))
    grouped_expected = np.asarray(grouped_expected)
    grouped_observed = np.asarray(grouped_observed)
    statistic = float(
        np.sum((grouped_observed - grouped_expected) ** 2 / grouped_expected)
    )
    dof = int(len(grouped_expected) - 1)
    return {
        "statistic": statistic,
        "dof": dof,
        "p_value": float(chi2.sf(statistic, dof)),
        "categories_kept": int(keep.sum()),
        "rare_categories_pooled": int((~keep).sum()),
        "min_expected": float(min_expected),
        "total_samples": int(observed.sum()),
    }


def ngram_next_distribution(model, context: list[int], smoothing: float) -> np.ndarray:
    """Next-token distribution for a fitted Part 2 NGramModel."""
    vocab = model.vocab_size
    m = model.context_length
    if m > 0:
        recent = list(context[-m:])
        if len(recent) < m:
            recent = [2] * (m - len(recent)) + recent
        code = 0
        for token in recent:
            code = code * vocab + int(token)
    else:
        code = 0

    low = code * vocab
    high = low + vocab
    left = int(np.searchsorted(model.joint_keys, low, side="left"))
    right = int(np.searchsorted(model.joint_keys, high, side="left"))
    counts = np.zeros(vocab, dtype=np.float64)
    if right > left:
        indices = model.joint_keys[left:right] - low
        counts[indices] = model.joint_counts[left:right]

    position = int(np.searchsorted(model.context_keys, code, side="left"))
    denominator = 0.0
    if position < len(model.context_keys) and model.context_keys[position] == code:
        denominator = float(model.context_counts[position])

    probabilities = (counts + smoothing) / (
        denominator + smoothing * vocab
    )
    return probabilities / probabilities.sum()


def decode_token_stream(tokenizer, token_ids: list[int]) -> str:
    """Decode tokens, rendering EOS boundaries as spaces."""
    segments = []
    current = []
    for token in token_ids:
        if int(token) == tokenizer.eos_id():
            if current:
                segments.append(tokenizer.decode(current))
                current = []
        else:
            current.append(int(token))
    if current:
        segments.append(tokenizer.decode(current))
    return " ".join(segment.strip() for segment in segments if segment.strip())
