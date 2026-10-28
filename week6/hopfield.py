"""Classical Hopfield network in JAX (Week 6 Part 3).

``N`` Ising spins take values +/-1 and ``P`` independent random +/-1 patterns
are stored with Hebbian couplings

    J_ij = (1/N) sum_mu xi_mu_i xi_mu_j,   J_ii = 0.

The unstable-bit fraction of the stored patterns is the fraction of bits with
``xi_i * h_i < 0``, where ``h_i = sum_j J_ij xi_j``. For random patterns this
approaches the Gaussian prediction ``Phi(-1/sqrt(alpha))`` with
``alpha = P/N`` as ``N`` grows.

Async zero-temperature recall updates one randomly chosen spin at a time and
stops when a full sweep produces no flips. If ``h_i == 0`` the spin is left
unchanged.

Usage::

    cd week6
    uv run python hopfield_experiment.py   # full alpha sweep + figure
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


# --- Patterns and couplings ----------------------------------------------
def random_patterns(key: jax.Array, n_patterns: int, n_spins: int) -> jax.Array:
    """``P`` independent random +/-1 patterns, shape ``(P, N)``."""
    bits = jax.random.bernoulli(key, 0.5, (n_patterns, n_spins))
    return jnp.where(bits, 1.0, -1.0)


def build_couplings(patterns: jax.Array) -> jax.Array:
    """Hebbian couplings ``J = X^T X / N`` with zero diagonal, shape ``(N, N)``.

    ``patterns`` has shape ``(P, N)``. The result is symmetrized explicitly so
    the symmetric matrix is exact up to float32 round-off.
    """
    patterns = jnp.asarray(patterns, dtype=jnp.float32)
    n_spins = patterns.shape[1]
    J = patterns.T @ patterns / n_spins
    J = 0.5 * (J + J.T)
    return J * (1.0 - jnp.eye(n_spins, dtype=jnp.float32))


def local_fields(J: jax.Array, patterns: jax.Array) -> jax.Array:
    """Local fields ``h_i = sum_j J_ij xi_j`` for each pattern, shape ``(P, N)``."""
    patterns = jnp.asarray(patterns, dtype=jnp.float32)
    return patterns @ J.T


def unstable_mask(patterns: jax.Array, fields: jax.Array) -> jax.Array:
    """Boolean mask where ``xi_i * h_i < 0``."""
    return jnp.asarray(patterns) * jnp.asarray(fields) < 0


def unstable_fraction(patterns: jax.Array, J: jax.Array | None = None) -> float:
    """Fraction of unstable bits over all stored patterns."""
    if J is None:
        raise ValueError("J must be provided (or call local_fields first)")
    fields = local_fields(J, patterns)
    return float(jnp.mean(unstable_mask(patterns, fields)))


def gaussian_prediction(alpha: float) -> float:
    """Theoretical unstable fraction ``Phi(-1/sqrt(alpha))``."""
    return 0.5 * math.erfc(1.0 / math.sqrt(2.0 * alpha))


# --- Asynchronous zero-temperature recall ---------------------------------
def async_recall(
    J: jax.Array, spins: jax.Array, key: jax.Array, max_sweeps: int = 1000
):
    """Randomized single-spin recall until a full sweep causes no flips.

    Returns ``(final_spins, sweeps, converged)``. Spins with ``h_i == 0`` are
    left unchanged.
    """
    J = jnp.asarray(J, dtype=jnp.float32)
    spins = jnp.asarray(spins, dtype=jnp.float32)
    n_spins = J.shape[0]

    def sweep(state):
        s, subkey, sweep_count, _changed = state
        subkey, permutation_key = jax.random.split(subkey)
        order = jax.random.permutation(permutation_key, n_spins)

        def update(carry, index):
            s, changed = carry
            field = J[index] @ s
            updated = jnp.where(field == 0.0, s[index], jnp.sign(field))
            flipped = updated != s[index]
            return (s.at[index].set(updated), jnp.logical_or(changed, flipped)), None

        (s, changed), _ = jax.lax.scan(
            update, (s, jnp.array(False)), order
        )
        return s, subkey, sweep_count + 1, changed

    def condition(state):
        _s, _key, sweep_count, changed = state
        return jnp.logical_and(
            sweep_count < max_sweeps,
            jnp.logical_or(sweep_count == 0, changed),
        )

    initial = (spins, key, jnp.array(0), jnp.array(True))
    final_spins, _key, sweeps, changed = jax.lax.while_loop(condition, sweep, initial)
    return final_spins, int(sweeps), not bool(changed)


def pattern_overlap(spins: jax.Array, pattern: jax.Array) -> float:
    """Normalized overlap ``(1/N) sum_i s_i xi_i``."""
    spins = np.asarray(spins)
    pattern = np.asarray(pattern)
    return float(np.mean(spins * pattern))


@partial(jax.jit, static_argnames=("max_sweeps",))
def async_recall_batch(
    J: jax.Array, spins: jax.Array, key: jax.Array, max_sweeps: int = 1000
):
    """Run independent async recalls in parallel (one per row of ``spins``).

    ``spins`` has shape ``(B, N)``; each row is evolved with its own random
    single-spin order until a full sweep makes no flips (or ``max_sweeps``).
    Returns ``(final_spins, sweeps, converged)``.
    """
    J = jnp.asarray(J, dtype=jnp.float32)
    S = jnp.asarray(spins, dtype=jnp.float32)
    n_chains, n_spins = S.shape
    rows = jnp.arange(n_chains)
    base_keys = jax.random.split(key, n_chains)

    def sweep(state):
        S, sweep_count, _changed = state
        permutation_keys = jax.vmap(
            lambda chain_key: jax.random.fold_in(chain_key, sweep_count)
        )(base_keys)
        orders = jax.vmap(
            lambda permutation_key: jax.random.permutation(permutation_key, n_spins)
        )(permutation_keys)

        def update(carry, step):
            S, changed = carry
            index = orders[:, step]
            fields = jnp.sum(J[index] * S, axis=1)
            previous = S[rows, index]
            updated = jnp.where(fields == 0.0, previous, jnp.sign(fields))
            S = S.at[rows, index].set(updated)
            return (S, jnp.logical_or(changed, updated != previous)), None

        (S, changed), _ = jax.lax.scan(
            update, (S, jnp.zeros(n_chains, dtype=jnp.bool_)), jnp.arange(n_spins)
        )
        return S, sweep_count + 1, changed

    def condition(state):
        _S, sweep_count, changed = state
        return jnp.logical_and(
            sweep_count < max_sweeps,
            jnp.logical_or(sweep_count == 0, changed.any()),
        )

    initial = (S, jnp.array(0), jnp.ones(n_chains, dtype=jnp.bool_))
    final, sweeps, changed = jax.lax.while_loop(condition, sweep, initial)
    return final, sweeps, jnp.logical_not(changed.any())
