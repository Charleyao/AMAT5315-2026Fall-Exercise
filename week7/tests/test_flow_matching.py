"""Tests for Flow Matching: model, exact velocity, Euler step, metrics."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import flow_matching as fm
from forward_diffusion import CENTERS

WEEK_DIR = Path(__file__).resolve().parents[1]
CONFIG = fm.load_flow_config(WEEK_DIR / "flow.toml")


def test_flow_toml_matches_the_page_11_settings():
    config = CONFIG

    assert config.hidden_layers == (128, 128, 128)
    assert config.activation == "silu"
    assert config.time_features == 32
    assert config.time_scale == 1000
    assert config.optimizer == "adam"
    assert config.learning_rate == pytest.approx(0.001)
    assert config.lr_schedule == "cosine"
    assert config.lr_final == pytest.approx(0.0)
    assert config.batch_size == 512
    assert config.updates == 25_000
    assert config.seed == 2029
    assert config.dtype == "float32"
    assert config.checkpoints[-1] == config.updates


def test_time_features_have_32_values_and_use_the_time_scale():
    s = jnp.array([0.0, 0.1, 0.5, 0.9, 1.0], dtype=jnp.float32)

    features = fm.sinusoidal_time_features(s, 32, 1000)

    assert features.shape == (5, 32)
    assert bool(jnp.all(jnp.isfinite(features)))
    np.testing.assert_allclose(features[0, :16], 0.0, atol=1e-6)
    np.testing.assert_allclose(features[0, 16:], 1.0, atol=1e-6)
    assert not np.allclose(features[1], features[2])


def test_velocity_model_shapes_and_initialization():
    params = fm.init_flow_params(jax.random.PRNGKey(0), CONFIG)
    layer_sizes = [2 + CONFIG.time_features, *CONFIG.hidden_layers, 2]

    assert set(params) == {
        name
        for layer in range(len(layer_sizes) - 1)
        for name in (f"W{layer}", f"b{layer}")
    }
    for layer, (fan_in, fan_out) in enumerate(
        zip(layer_sizes[:-1], layer_sizes[1:])
    ):
        assert params[f"W{layer}"].shape == (fan_in, fan_out)
        assert params[f"b{layer}"].shape == (fan_out,)
        np.testing.assert_allclose(params[f"b{layer}"], 0.0)

    x_s = jax.random.normal(jax.random.PRNGKey(1), (4, 2))
    s = jnp.array([0.1, 0.3, 0.7, 0.9], dtype=jnp.float32)
    velocity = fm.velocity_apply(params, x_s, s, CONFIG)

    assert velocity.shape == (4, 2)
    assert bool(jnp.all(jnp.isfinite(velocity)))


def test_training_batch_follows_the_linear_interpolation_and_target():
    x0, z, s, x_s, target = fm.sample_flow_batch(
        jax.random.PRNGKey(2), 64
    )

    assert x0.shape == (64, 2)
    assert z.shape == (64, 2)
    assert s.shape == (64,)
    assert x_s.shape == (64, 2)
    assert target.shape == (64, 2)
    assert float(s.min()) >= 0.0
    assert float(s.max()) <= 1.0
    np.testing.assert_allclose(
        x_s,
        (1.0 - s[:, None]) * x0 + s[:, None] * z,
        rtol=1e-6,
    )
    np.testing.assert_allclose(target, z - x0, rtol=1e-6)


def test_exact_marginal_velocity_endpoints_are_minus_x_and_x():
    points = jax.random.normal(jax.random.PRNGKey(3), (7, 2))

    at_s_zero = fm.exact_marginal_velocity(points, 0.0)
    at_s_one = fm.exact_marginal_velocity(points, 1.0)

    np.testing.assert_allclose(at_s_zero, -points, rtol=1e-6)
    np.testing.assert_allclose(at_s_one, points, rtol=1e-6)


def test_exact_marginal_velocity_supports_batched_times():
    points = jax.random.normal(jax.random.PRNGKey(4), (5, 2))
    times = jnp.array([0.05, 0.2, 0.5, 0.8, 0.95], dtype=jnp.float32)

    batched = fm.exact_marginal_velocity(points, times)
    per_sample = jnp.concatenate(
        [
            fm.exact_marginal_velocity(points[index : index + 1], times[index])
            for index in range(points.shape[0])
        ],
        axis=0,
    )

    assert batched.shape == (5, 2)
    np.testing.assert_allclose(batched, per_sample, rtol=1e-6)


def test_euler_step_matches_the_forward_euler_update():
    x_s = jnp.array([[1.0, -2.0], [0.5, 3.0]], dtype=jnp.float32)
    velocity = jnp.array([[0.2, 0.4], [-0.1, 0.3]], dtype=jnp.float32)

    result = fm.euler_step(x_s, velocity, step_size=0.01)

    np.testing.assert_allclose(result, x_s - 0.01 * velocity, rtol=1e-6)


def test_zero_velocity_model_leaves_euler_samples_unchanged():
    config = replace(CONFIG, hidden_layers=(8, 8))
    params = {
        name: jnp.zeros_like(value)
        for name, value in fm.init_flow_params(
            jax.random.PRNGKey(5), config
        ).items()
    }
    z = jax.random.normal(jax.random.PRNGKey(6), (3, 2))
    sampler = fm.make_euler_sampler(params, config, num_steps=10)

    samples = sampler(z)

    np.testing.assert_allclose(samples, z, atol=1e-7)


def test_relative_rms_error_is_zero_for_identical_fields():
    field = jax.random.normal(jax.random.PRNGKey(7), (10, 2))

    error = fm.relative_rms_error(field, field)

    assert error == pytest.approx(0.0)


def test_grid_and_relative_rms_metrics_pass_for_identical_fields():
    result = fm.evaluate_velocity_fields(
        lambda points, s: fm.exact_marginal_velocity(points, s),
        [0.1, 0.5, 0.9],
    )

    assert result["passed"] is True
    for s in ("0.1", "0.5", "0.9"):
        assert result["relative_rms"][s] == pytest.approx(0.0)
        assert result["pass"][s] is True


def test_flow_verification_table_lists_each_s_value():
    result = fm.evaluate_velocity_fields(
        lambda points, s: fm.exact_marginal_velocity(points, s),
        [0.1, 0.5, 0.9],
    )

    table = fm.format_verification_table(result)

    assert "s=0.1" in table
    assert "s=0.5" in table
    assert "s=0.9" in table
    assert "Overall Part 3 VERIFY (1): PASS" in table


def test_small_flow_training_returns_every_requested_checkpoint():
    small = replace(
        CONFIG,
        hidden_layers=(16, 16),
        updates=4,
        checkpoints=(0, 2, 4),
        batch_size=32,
    )

    checkpoints, losses = fm.train_flow_checkpointed(
        small,
        jax.random.PRNGKey(8),
        chunk_updates=2,
        verbose=False,
    )

    assert set(checkpoints) == {0, 2, 4}
    assert losses.shape == (4,)
    assert bool(jnp.all(jnp.isfinite(losses)))
