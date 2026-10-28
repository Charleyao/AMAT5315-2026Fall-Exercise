"""Tests for Part 3 VERIFY (2) sampling and metrics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from forward_diffusion import (
    CENTERS,
    alpha_bars_from_betas,
    linear_beta_schedule,
)
from part3_verify2 import (
    coarse_ddpm_step,
    energy_distance,
    evenly_spaced_levels,
    make_coarse_ddpm_sampler,
    sample_metrics,
)

ALPHA_BARS = alpha_bars_from_betas(linear_beta_schedule())


def test_evenly_spaced_levels_start_at_T_and_are_strictly_decreasing():
    assert evenly_spaced_levels(1) == (1000,)
    assert evenly_spaced_levels(2) == (1000, 500)
    assert evenly_spaced_levels(5) == (1000, 800, 600, 400, 200)
    levels = evenly_spaced_levels(1000)
    assert levels[0] == 1000
    assert levels[-1] == 1
    assert len(levels) == 1000
    assert all(
        current > following
        for current, following in zip(levels, levels[1:])
    )


def test_coarse_ddpm_step_matches_adjacent_step_posterior():
    x_t = jnp.array([[1.5, -0.5], [2.0, 0.25]], dtype=jnp.float32)
    predicted_noise = jnp.array(
        [[0.3, -0.2], [0.1, 0.4]], dtype=jnp.float32
    )
    noise = jnp.array([[0.7, -0.6], [0.2, 0.9]], dtype=jnp.float32)
    t = 500
    previous_t = 499
    alpha_bar_t = ALPHA_BARS[t]
    alpha_bar_previous = ALPHA_BARS[previous_t]
    predicted_x0 = (
        x_t - jnp.sqrt(1.0 - alpha_bar_t) * predicted_noise
    ) / jnp.sqrt(alpha_bar_t)
    beta_t = 1.0 - (alpha_bar_t / alpha_bar_previous)
    mean = (
        jnp.sqrt(alpha_bar_previous) * beta_t / (1.0 - alpha_bar_t)
    ) * predicted_x0 + (
        jnp.sqrt(alpha_bar_t / alpha_bar_previous)
        * (1.0 - alpha_bar_previous)
        / (1.0 - alpha_bar_t)
    ) * x_t
    variance = (
        (1.0 - alpha_bar_previous)
        / (1.0 - alpha_bar_t)
        * beta_t
    )

    result = coarse_ddpm_step(
        x_t,
        t,
        previous_t,
        predicted_noise,
        noise,
        ALPHA_BARS,
    )

    np.testing.assert_allclose(
        result, mean + jnp.sqrt(variance) * noise, rtol=1e-6
    )


def test_coarse_ddpm_full_jump_to_zero_has_zero_variance():
    x_t = jnp.array([[1.5, -0.5]], dtype=jnp.float32)
    predicted_noise = jnp.array([[0.3, -0.2]], dtype=jnp.float32)
    noise = jnp.array([[100.0, -100.0]], dtype=jnp.float32)
    alpha_bar_t = ALPHA_BARS[1000]
    predicted_x0 = (
        x_t - jnp.sqrt(1.0 - alpha_bar_t) * predicted_noise
    ) / jnp.sqrt(alpha_bar_t)

    result = coarse_ddpm_step(
        x_t, 1000, 0, predicted_noise, noise, ALPHA_BARS
    )

    np.testing.assert_allclose(result, predicted_x0, rtol=1e-6)


def test_coarse_ddpm_sampler_returns_final_samples():
    levels = (4, 2)
    x_T = jax.random.normal(jax.random.PRNGKey(0), (3, 2))
    noise = jnp.zeros((len(levels), 3, 2), dtype=jnp.float32)

    def zero_predictor(x_t, t):
        return jnp.zeros_like(x_t)

    sampler = make_coarse_ddpm_sampler(
        zero_predictor, ALPHA_BARS, levels
    )
    samples = sampler(x_T, noise)

    assert samples.shape == (3, 2)
    assert bool(jnp.all(jnp.isfinite(samples)))


def test_energy_distance_is_zero_for_identical_sets_and_positive_otherwise():
    key_first, key_second = jax.random.split(jax.random.PRNGKey(1))
    first = jax.random.normal(key_first, (200, 2))
    second = jax.random.normal(key_second, (200, 2))

    assert energy_distance(first, first) == pytest.approx(0.0, abs=1e-7)
    assert energy_distance(first, second) > 0.0


def test_sample_metrics_pass_for_an_exact_target_distribution():
    offsets = np.array(
        [[0.3, 0.3], [0.3, -0.3], [-0.3, 0.3], [-0.3, -0.3]],
        dtype=np.float32,
    )
    centers = np.asarray(CENTERS, dtype=np.float32)
    points = centers[:, None, :] + offsets[None, :, :]
    samples = np.repeat(points.reshape(-1, 2), 125, axis=0)

    metrics = sample_metrics(samples)

    assert metrics["max_population_error"] == pytest.approx(0.0)
    assert metrics["max_width_error"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["max_center_error"] == pytest.approx(0.0, abs=1e-6)
    for cluster in range(4):
        result = metrics["clusters"][str(cluster)]
        assert result["population"] == 500
        assert result["width"] == pytest.approx(0.30, abs=1e-6)
