"""Generation helpers for the Part 5 Transformer."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from .sampling import (
    apply_truncation,
    sample_indices,
    temperature_probs,
)
from .transformer import CONTEXT_LENGTH, forward

T1_SETTING = {"kind": "temperature", "temperature": 1.0, "label": "T=1"}
_FORWARD_JIT = jax.jit(forward)


def _forward_jit(params, tokens, active_length=None):
    return _FORWARD_JIT(params, tokens, active_length)


def get_context_logits(
    params,
    context: list[int],
) -> np.ndarray:
    """Logits for the next token after a (possibly short) context."""
    if len(context) == 0:
        raise ValueError("context must contain at least one token")
    active = min(len(context), CONTEXT_LENGTH)
    recent = list(map(int, context[-active:]))
    if active < CONTEXT_LENGTH:
        padded = recent + [0] * (CONTEXT_LENGTH - active)
        inputs = jnp.asarray([padded], dtype=jnp.int32)
        active_length = active
    else:
        inputs = jnp.asarray([recent], dtype=jnp.int32)
        active_length = CONTEXT_LENGTH
    logits = _forward_jit(params, inputs, active_length)
    return np.asarray(logits[0, active - 1], dtype=np.float64)


def generate_transformer_tokens(
    params,
    prefix: list[int],
    n_tokens: int,
    setting: dict,
    seed: int,
) -> list[int]:
    """Autoregressively generate tokens with one sampling setting."""
    rng = np.random.default_rng(seed)
    context = list(map(int, prefix))
    generated = []
    for _ in range(n_tokens):
        logits = get_context_logits(params, context)
        probabilities = apply_truncation(logits, setting)
        token = int(sample_indices(probabilities, 1, rng)[0])
        generated.append(token)
        context.append(token)
    return generated


def get_context_distribution(
    params,
    context: list[int] | np.ndarray,
    setting: dict,
) -> np.ndarray:
    logits = get_context_logits(params, list(map(int, context)))
    return apply_truncation(logits, setting)


def generate_scored_sequences(
    params,
    prefixes: np.ndarray,
    n_tokens: int,
    temperature: float,
    seed: int,
) -> dict:
    """Generate a batch and score every token under the temperature-1 model.

    ``prefixes`` must have shape (batch, 64).  Sampling uses ``temperature``;
    surprise and entropy are always computed from the original T=1 softmax.
    """
    prefixes = np.asarray(prefixes, dtype=np.int32)
    if prefixes.ndim != 2 or prefixes.shape[1] != CONTEXT_LENGTH:
        raise ValueError("prefixes must have shape (batch, 64)")
    contexts = prefixes.copy()
    batch = contexts.shape[0]
    rng = np.random.default_rng(seed)
    sampled = np.empty((batch, n_tokens), dtype=np.int32)
    surprise = np.empty((batch, n_tokens), dtype=np.float64)
    entropy = np.empty((batch, n_tokens), dtype=np.float64)
    row_index = np.arange(batch)

    forward_jit = _FORWARD_JIT
    for step in range(n_tokens):
        logits = np.asarray(
            forward_jit(params, jnp.asarray(contexts)), dtype=np.float64
        )[:, -1, :]
        base_probs = temperature_probs(logits, 1.0)
        sampling_probs = temperature_probs(logits, temperature)
        cumulative = np.cumsum(sampling_probs, axis=1)
        cumulative[:, -1] = 1.0
        uniforms = rng.random(batch)
        tokens = (cumulative < uniforms[:, None]).sum(axis=1)
        tokens = np.minimum(tokens, sampling_probs.shape[1] - 1).astype(
            np.int32
        )

        chosen = base_probs[row_index, tokens]
        sampled[:, step] = tokens
        surprise[:, step] = -np.log(chosen)
        entropy[:, step] = -np.sum(
            base_probs * np.log(base_probs), axis=1
        )

        contexts[:, :-1] = contexts[:, 1:]
        contexts[:, -1] = tokens

    return {
        "tokens": sampled,
        "surprise": surprise,
        "entropy": entropy,
    }


def z_score_from_sequences(result: dict) -> dict:
    """Compare per-sequence mean surprise and mean entropy."""
    surprise_mean = result["surprise"].mean(axis=1)
    entropy_mean = result["entropy"].mean(axis=1)
    differences = surprise_mean - entropy_mean
    n_sequences = len(differences)
    mean_difference = float(differences.mean())
    standard_error = float(
        differences.std(ddof=1) / np.sqrt(n_sequences)
    )
    z_score = mean_difference / standard_error
    return {
        "n_sequences": int(n_sequences),
        "tokens_per_sequence": int(result["surprise"].shape[1]),
        "mean_surprise": float(surprise_mean.mean()),
        "mean_entropy": float(entropy_mean.mean()),
        "mean_difference": mean_difference,
        "sequence_std": float(differences.std(ddof=1)),
        "standard_error": standard_error,
        "z_score": float(z_score),
    }
