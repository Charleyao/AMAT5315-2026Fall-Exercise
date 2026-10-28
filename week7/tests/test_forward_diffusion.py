"""Tests for the Part 1 Gaussian mixture and forward diffusion process."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from forward_diffusion import (
    AXIS_LIMIT,
    BETA_END,
    BETA_START,
    CENTERS,
    COMPONENT_COVARIANCE,
    COMPONENT_STD,
    NUM_DIFFUSION_STEPS,
    alpha_bars_from_betas,
    coordinate_stats,
    forward_trajectory,
    linear_beta_schedule,
    q_sample,
    sample_gaussian_mixture,
    verify_terminal_distribution,
)


def test_linear_beta_schedule_has_1000_steps_and_expected_endpoints():
    betas = linear_beta_schedule()

    assert betas.shape == (NUM_DIFFUSION_STEPS,)
    assert float(betas[0]) == pytest.approx(BETA_START, abs=1e-9)
    assert float(betas[-1]) == pytest.approx(BETA_END, abs=1e-9)
    assert bool(jnp.all(jnp.diff(betas) > 0.0))


def test_alpha_bars_start_at_one_and_are_cumulative_products():
    betas = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32)

    alpha_bars = alpha_bars_from_betas(betas)
    expected = jnp.concatenate(
        [jnp.ones((1,), dtype=jnp.float32), jnp.cumprod(1.0 - betas)]
    )

    assert alpha_bars.shape == (4,)
    assert float(alpha_bars[0]) == pytest.approx(1.0)
    np.testing.assert_allclose(alpha_bars, expected, rtol=1e-6)
    assert bool(jnp.all(jnp.diff(alpha_bars) < 0.0))


def test_gaussian_mixture_has_equal_weights_and_covariance_0_09_i():
    num_samples = 40_000
    samples, components = sample_gaussian_mixture(
        jax.random.PRNGKey(5315), num_samples
    )

    assert samples.shape == (num_samples, 2)
    assert components.shape == (num_samples,)
    assert COMPONENT_STD == pytest.approx(math.sqrt(COMPONENT_COVARIANCE))

    counts = np.bincount(np.asarray(components), minlength=4)
    np.testing.assert_allclose(counts / num_samples, 0.25, atol=0.01)

    samples_np = np.asarray(samples)
    for component, center in enumerate(np.asarray(CENTERS)):
        points = samples_np[np.asarray(components) == component]
        np.testing.assert_allclose(points.mean(axis=0), center, atol=0.03)
        np.testing.assert_allclose(
            points.var(axis=0), COMPONENT_COVARIANCE, atol=0.01
        )


def test_q_sample_returns_x0_at_alpha_bar_one_and_noise_at_zero():
    x0 = jnp.array([[1.0, -2.0], [0.5, 3.0]], dtype=jnp.float32)
    noise = jnp.array([[-0.5, 0.25], [1.5, -1.0]], dtype=jnp.float32)

    np.testing.assert_allclose(q_sample(x0, 1.0, noise), x0)
    np.testing.assert_allclose(q_sample(x0, 0.0, noise), noise)


def test_forward_trajectory_returns_each_requested_frame():
    x0 = jnp.array([[1.0, -2.0], [0.5, 3.0]], dtype=jnp.float32)
    noise = jnp.array([[-0.5, 0.25], [1.5, -1.0]], dtype=jnp.float32)
    alpha_bars = jnp.array([1.0, 0.5, 0.0], dtype=jnp.float32)

    frames = forward_trajectory(x0, noise, alpha_bars, jnp.array([0, 1, 2]))

    assert frames.shape == (3, 2, 2)
    np.testing.assert_allclose(frames[0], x0)
    np.testing.assert_allclose(frames[2], noise)
    np.testing.assert_allclose(frames[1], q_sample(x0, 0.5, noise))


def test_coordinate_stats_returns_mean_and_variance_for_both_coordinates():
    samples = jnp.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=jnp.float32
    )

    means, variances = coordinate_stats(samples)

    np.testing.assert_allclose(means, jnp.array([3.0, 4.0]))
    np.testing.assert_allclose(variances, jnp.array([8.0 / 3.0, 8.0 / 3.0]))


def test_verify_terminal_distribution_detects_a_nonstandard_distribution():
    collapsed = jnp.zeros((2000, 2), dtype=jnp.float32)

    result = verify_terminal_distribution(collapsed)

    assert result["means_within_tolerance"] is True
    assert result["variances_within_tolerance"] is False
    assert result["passed"] is False


def test_t1000_forward_samples_are_within_verify1_tolerances():
    num_samples = 2000
    key = jax.random.PRNGKey(5315)
    key_x0, key_noise = jax.random.split(key)

    x0, _ = sample_gaussian_mixture(key_x0, num_samples)
    noise = jax.random.normal(key_noise, x0.shape)
    alpha_bars = alpha_bars_from_betas(linear_beta_schedule())
    assert alpha_bars.shape == (NUM_DIFFUSION_STEPS + 1,)
    x_t1000 = q_sample(x0, alpha_bars[-1], noise)

    result = verify_terminal_distribution(x_t1000)

    assert abs(result["mean_x"]) <= 0.1
    assert abs(result["mean_y"]) <= 0.1
    assert abs(result["variance_x"] - 1.0) <= 0.1
    assert abs(result["variance_y"] - 1.0) <= 0.1
    assert result["passed"] is True


def test_axis_limit_contains_the_initial_mixture_clusters():
    assert AXIS_LIMIT > float(jnp.max(jnp.abs(CENTERS))) + 4.0 * COMPONENT_STD
