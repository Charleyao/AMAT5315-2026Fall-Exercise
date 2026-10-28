"""Tests for deterministic DDIM sampling and the exact GMM predictor."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ddim import (
    ddim_reverse_step,
    exact_conditional_mean_noise,
    make_ddim_sampler,
    normalize_ddim_levels,
)
from forward_diffusion import (
    COMPONENT_STD,
    alpha_bars_from_betas,
    linear_beta_schedule,
)

ALPHA_BARS = alpha_bars_from_betas(linear_beta_schedule())


def test_normalize_ddim_levels_accepts_a_decreasing_subset():
    levels = normalize_ddim_levels([1000, 750, 500, 100])

    assert levels == (1000, 750, 500, 100)


def test_normalize_ddim_levels_rejects_invalid_subsets():
    with pytest.raises(ValueError):
        normalize_ddim_levels([])
    with pytest.raises(ValueError):
        normalize_ddim_levels([1000, 500, 750])
    with pytest.raises(ValueError):
        normalize_ddim_levels([1001, 500])
    with pytest.raises(ValueError):
        normalize_ddim_levels([500, 0])


def test_ddim_reverse_step_matches_the_deterministic_formula():
    x_t = jnp.array([[1.5, -0.5], [2.0, 0.25]], dtype=jnp.float32)
    predicted_noise = jnp.array(
        [[0.3, -0.2], [0.1, 0.4]], dtype=jnp.float32
    )
    t = 500
    previous_t = 250
    alpha_bar_t = ALPHA_BARS[t]
    alpha_bar_previous = ALPHA_BARS[previous_t]

    expected_x0 = (
        x_t - jnp.sqrt(1.0 - alpha_bar_t) * predicted_noise
    ) / jnp.sqrt(alpha_bar_t)
    expected = (
        jnp.sqrt(alpha_bar_previous) * expected_x0
        + jnp.sqrt(1.0 - alpha_bar_previous) * predicted_noise
    )

    result = ddim_reverse_step(
        x_t, t, previous_t, predicted_noise, ALPHA_BARS
    )

    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_ddim_final_step_returns_predicted_x0_without_noise():
    x_t = jnp.array([[1.5, -0.5]], dtype=jnp.float32)
    predicted_noise = jnp.array([[0.3, -0.2]], dtype=jnp.float32)
    alpha_bar_t = ALPHA_BARS[1]

    expected_x0 = (
        x_t - jnp.sqrt(1.0 - alpha_bar_t) * predicted_noise
    ) / jnp.sqrt(alpha_bar_t)

    result = ddim_reverse_step(
        x_t, 1, 0, predicted_noise, ALPHA_BARS
    )

    np.testing.assert_allclose(result, expected_x0, rtol=1e-6)


def test_ddim_sampler_returns_a_deterministic_trajectory_for_a_subset():
    levels = (5, 3, 1)
    x_T = jax.random.normal(jax.random.PRNGKey(0), (2, 2))

    def zero_predictor(x_t, t):
        return jnp.zeros_like(x_t)

    sampler = make_ddim_sampler(zero_predictor, ALPHA_BARS, levels)
    first = sampler(x_T)
    second = sampler(x_T)

    assert first.shape == (len(levels) + 1, 2, 2)
    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(first[0], x_T)


def test_exact_predictor_is_zero_at_t0():
    x_t = jax.random.normal(jax.random.PRNGKey(1), (5, 2))

    prediction = exact_conditional_mean_noise(x_t, 0, ALPHA_BARS)

    np.testing.assert_allclose(prediction, 0.0, atol=1e-7)


def test_exact_predictor_matches_single_component_formula():
    center = jnp.array([[1.0, -0.5]], dtype=jnp.float32)
    x_t = jnp.array([[1.4, 0.2], [0.7, -0.9]], dtype=jnp.float32)
    t = 500
    alpha_bar_t = ALPHA_BARS[t]
    variance = alpha_bar_t * COMPONENT_STD**2 + (1.0 - alpha_bar_t)
    expected = (
        jnp.sqrt(1.0 - alpha_bar_t)
        / variance
        * (x_t - jnp.sqrt(alpha_bar_t) * center[0])
    )

    prediction = exact_conditional_mean_noise(
        x_t, t, ALPHA_BARS, centers=center
    )

    np.testing.assert_allclose(prediction, expected, rtol=1e-6)


def test_exact_predictor_supports_batched_times():
    x_t = jax.random.normal(jax.random.PRNGKey(2), (5, 2))
    times = jnp.array([1, 100, 400, 750, 1000], dtype=jnp.int32)

    batched = exact_conditional_mean_noise(x_t, times, ALPHA_BARS)
    per_sample = jnp.concatenate(
        [
            exact_conditional_mean_noise(
                x_t[index : index + 1], times[index], ALPHA_BARS
            )
            for index in range(x_t.shape[0])
        ],
        axis=0,
    )

    assert batched.shape == (5, 2)
    np.testing.assert_allclose(batched, per_sample, rtol=1e-6)
