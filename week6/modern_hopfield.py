"""Modern Hopfield network and its softmax-attention form (Week 6 Part 4).

Patterns are the first ``P`` MNIST training images in original order,
binarized and converted to +/-1 spins, shape ``(P, 784)``. With ``X`` holding
the stored patterns as rows, the modern Hopfield update is

    x_new = X^T @ softmax(beta * (X @ x))

which is exactly softmax attention with query ``x``, keys and values equal to
the stored patterns, and attention scale ``beta`` (not the default
``1/sqrt(d)``).
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import rbm

WEEK_DIR = Path(__file__).resolve().parent
MNIST_DIR = WEEK_DIR / "inputs" / "mnist"


def mnist_spin_patterns(p: int, directory: Path = MNIST_DIR):
    """First ``p`` training images as +/-1 spins, in original stored order."""
    images, labels = rbm.load_binary_mnist("train", directory)
    spins = rbm.to_spins(images[:p]).astype(np.float32)
    return jnp.asarray(spins), np.asarray(labels[:p])


def held_out_spins(n: int = 20, directory: Path = MNIST_DIR):
    """First ``n`` test images as +/-1 spins (held-out queries)."""
    images, labels = rbm.load_binary_mnist("test", directory)
    spins = rbm.to_spins(images[:n]).astype(np.float32)
    return jnp.asarray(spins), np.asarray(labels[:n])


def softmax_weights(patterns: jax.Array, queries: jax.Array, beta: float) -> jax.Array:
    """Attention weights ``softmax(beta * Q K^T)``, shape ``(Q, P)``."""
    patterns = jnp.asarray(patterns, dtype=jnp.float32)
    queries = jnp.atleast_2d(jnp.asarray(queries, dtype=jnp.float32))
    scores = beta * (queries @ patterns.T)
    return jax.nn.softmax(scores, axis=-1)


def modern_hopfield_update(patterns: jax.Array, queries: jax.Array, beta: float):
    """Modern Hopfield update ``X^T softmax(beta X x)``.

    Returns ``(updated_queries, weights)``; 1-D input is treated as one query.
    """
    single = jnp.asarray(queries).ndim == 1
    weights = softmax_weights(patterns, queries, beta)
    updated = weights @ jnp.asarray(patterns, dtype=jnp.float32)
    if single:
        return updated[0], weights[0]
    return updated, weights


def attention_update(patterns: jax.Array, queries: jax.Array, beta: float) -> jax.Array:
    """Same operation via ``jax.nn.dot_product_attention`` with ``scale=beta``.

    Query = held-out state, keys = values = stored patterns. The explicit
    ``scale=beta`` replaces the default ``1/sqrt(head_dim)`` attention scale.
    """
    patterns = jnp.asarray(patterns, dtype=jnp.float32)
    queries = jnp.atleast_2d(jnp.asarray(queries, dtype=jnp.float32))
    # Single head: query (T, 1, H); key/value (S, 1, H).
    output = jax.nn.dot_product_attention(
        queries[:, None, :],
        patterns[:, None, :],
        patterns[:, None, :],
        scale=beta,
    )
    return output[:, 0, :]


def attention_entropy(weights: np.ndarray) -> float:
    """Shannon entropy of the attention weights (nats)."""
    w = np.asarray(weights, dtype=np.float64)
    w = np.clip(w, 1e-30, None)
    return float(-(w * np.log(w)).sum())
