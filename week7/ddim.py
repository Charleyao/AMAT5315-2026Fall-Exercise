"""Deterministic DDIM sampling and the exact Gaussian-mixture predictor."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp

from diffusion_model import model_apply
from forward_diffusion import (
    CENTERS,
    COMPONENT_STD,
    NUM_DIFFUSION_STEPS,
)

NoisePredictor = Callable[[jax.Array, jax.Array], jax.Array]
FULL_DDIM_LEVELS = tuple(range(NUM_DIFFUSION_STEPS, 0, -1))


def normalize_ddim_levels(levels: Sequence[int]) -> tuple[int, ...]:
    """Validate a selectable decreasing subset of noise levels.

    Levels are positive diffusion times in ``1, ..., T``. The final transition
    to t=0 is implicit. For example, ``(1000, 500, 100)`` performs three model
    calls and then returns the predicted x_0.
    """
    normalized = tuple(int(level) for level in levels)
    if not normalized:
        raise ValueError("levels must not be empty")
    if any(
        level < 1 or level > NUM_DIFFUSION_STEPS for level in normalized
    ):
        raise ValueError("levels must lie in 1, ..., T")
    if any(
        current <= following
        for current, following in zip(normalized, normalized[1:])
    ):
        raise ValueError("levels must be strictly decreasing")
    return normalized


def ddim_reverse_step(
    x_t: jax.Array,
    t: int | jax.Array,
    previous_t: int | jax.Array,
    predicted_noise: jax.Array,
    alpha_bars: jax.Array,
) -> jax.Array:
    """One deterministic DDIM update from level t to previous_t."""
    alpha_bar_t = alpha_bars[t]
    alpha_bar_previous = alpha_bars[previous_t]
    predicted_x0 = (
        x_t - jnp.sqrt(1.0 - alpha_bar_t) * predicted_noise
    ) / jnp.sqrt(alpha_bar_t)
    return (
        jnp.sqrt(alpha_bar_previous) * predicted_x0
        + jnp.sqrt(1.0 - alpha_bar_previous) * predicted_noise
    )


def make_ddim_sampler(
    predictor: NoisePredictor,
    alpha_bars: jax.Array,
    levels: Sequence[int],
):
    """Compile a deterministic DDIM sampler for a decreasing level subset."""
    levels = normalize_ddim_levels(levels)
    current_levels = jnp.asarray(levels, dtype=jnp.int32)
    previous_levels = jnp.asarray(levels[1:] + (0,), dtype=jnp.int32)

    @jax.jit
    def sample(x_T: jax.Array) -> jax.Array:
        def body(x_t, level_pair):
            t, previous_t = level_pair
            predicted_noise = predictor(x_t, t)
            x_previous = ddim_reverse_step(
                x_t, t, previous_t, predicted_noise, alpha_bars
            )
            return x_previous, x_previous

        _, sampled_steps = jax.lax.scan(
            body, x_T, (current_levels, previous_levels)
        )
        return jnp.concatenate([x_T[None, :, :], sampled_steps], axis=0)

    return sample


def exact_conditional_mean_noise(
    x_t: jax.Array,
    t: int | jax.Array,
    alpha_bars: jax.Array,
    centers: jax.Array = CENTERS,
    component_std: float = COMPONENT_STD,
) -> jax.Array:
    """Exact E[epsilon | x_t] for the equal-weight Gaussian mixture.

    The posterior component responsibilities are computed analytically from
    the forward marginal, then the component-conditional noise means are
    combined with those responsibilities.
    """
    x_t = jnp.asarray(x_t, dtype=jnp.float32)
    centers = jnp.asarray(centers, dtype=jnp.float32)
    alpha_bar_t = alpha_bars[t]
    variance = alpha_bar_t * component_std**2 + (1.0 - alpha_bar_t)
    scaled_centers = jnp.reshape(
        jnp.sqrt(alpha_bar_t), (-1, 1, 1)
    ) * centers
    differences = x_t[:, None, :] - scaled_centers

    # Equal mixture weights cancel in the softmax.
    log_responsibilities = -jnp.sum(
        differences**2, axis=-1
    ) / (2.0 * jnp.reshape(variance, (-1, 1)))
    responsibilities = jax.nn.softmax(log_responsibilities, axis=-1)
    component_noise_means = (
        jnp.reshape(jnp.sqrt(1.0 - alpha_bar_t), (-1, 1, 1))
        / jnp.reshape(variance, (-1, 1, 1))
    ) * differences
    return jnp.sum(
        responsibilities[:, :, None] * component_noise_means, axis=1
    )


def make_learned_noise_predictor(params, config):
    """Wrap the Part 1 trained MLP as an ``(x_t, t) -> epsilon`` predictor."""

    def predict(x_t: jax.Array, t: int | jax.Array) -> jax.Array:
        t_batch = jnp.full((x_t.shape[0],), t, dtype=jnp.int32)
        return model_apply(params, x_t, t_batch, config)

    return predict


__all__ = [
    "FULL_DDIM_LEVELS",
    "NoisePredictor",
    "ddim_reverse_step",
    "exact_conditional_mean_noise",
    "make_ddim_sampler",
    "make_learned_noise_predictor",
    "normalize_ddim_levels",
]
