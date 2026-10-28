"""Tests for the Week 7 Part 1 time-conditioned noise predictor and DDPM."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import diffusion_model as dm
from forward_diffusion import (
    NUM_DIFFUSION_STEPS,
    alpha_bars_from_betas,
    linear_beta_schedule,
)

WEEK_DIR = Path(__file__).resolve().parents[1]
CONFIG = dm.load_config(WEEK_DIR / "diffusion.toml")
ALPHA_BARS = alpha_bars_from_betas(linear_beta_schedule())


def test_diffusion_toml_matches_the_learning_sheet_settings():
    config = CONFIG

    assert config.hidden_layers == (128, 128, 128)
    assert config.activation == "silu"
    assert config.time_features == 32
    assert config.time_offset == -1.0
    assert config.weight_init == "normal"
    assert config.weight_init_std == "1/sqrt(input_width)"
    assert config.optimizer == "adam"
    assert config.learning_rate == pytest.approx(0.001)
    assert config.lr_schedule == "cosine"
    assert config.lr_final == pytest.approx(0.0)
    assert config.batch_size == 512
    assert config.updates == 25_000
    assert config.seed == 2026
    assert config.dtype == "float32"
    assert config.checkpoints == (
        0,
        50,
        100,
        200,
        500,
        1000,
        2000,
        5000,
        10_000,
        25_000,
    )


def test_sinusoidal_time_features_have_32_values_and_respect_offset():
    times = jnp.array([1, 501, 1000], dtype=jnp.float32)

    features = dm.sinusoidal_time_features(times, 32, -1.0)

    assert features.shape == (3, 32)
    assert bool(jnp.all(jnp.isfinite(features)))
    # t=1 with offset -1 is normalized time zero.
    np.testing.assert_allclose(features[0, :16], 0.0, atol=1e-6)
    np.testing.assert_allclose(features[0, 16:], 1.0, atol=1e-6)
    assert not np.allclose(features[0], features[1])


def test_model_initialization_uses_normal_std_one_over_sqrt_fan_in():
    params = dm.init_model_params(jax.random.PRNGKey(0), CONFIG)
    layer_sizes = [2 + CONFIG.time_features, *CONFIG.hidden_layers, 2]

    assert set(params) == {
        name
        for layer in range(len(layer_sizes) - 1)
        for name in (f"W{layer}", f"b{layer}")
    }
    for layer, (fan_in, fan_out) in enumerate(
        zip(layer_sizes[:-1], layer_sizes[1:])
    ):
        weights = params[f"W{layer}"]
        bias = params[f"b{layer}"]
        assert weights.shape == (fan_in, fan_out)
        assert bias.shape == (fan_out,)
        assert float(jnp.std(weights)) == pytest.approx(
            1.0 / np.sqrt(fan_in),
            rel=max(0.05, 4.0 / np.sqrt(2.0 * fan_in * fan_out)),
        )
        np.testing.assert_allclose(bias, 0.0)


def test_model_accepts_batched_points_and_times():
    params = dm.init_model_params(jax.random.PRNGKey(1), CONFIG)
    x_t = jax.random.normal(jax.random.PRNGKey(2), (4, 2))
    t = jnp.array([1, 250, 500, 1000], dtype=jnp.int32)

    prediction = dm.model_apply(params, x_t, t, CONFIG)

    assert prediction.shape == (4, 2)
    assert bool(jnp.all(jnp.isfinite(prediction)))


def test_training_batch_has_uniform_diffusion_times():
    x_t, t, noise = dm.sample_training_batch(
        jax.random.PRNGKey(3), 256, ALPHA_BARS
    )

    assert x_t.shape == (256, 2)
    assert noise.shape == (256, 2)
    assert int(t.min()) >= 1
    assert int(t.max()) <= NUM_DIFFUSION_STEPS
    assert len(np.unique(np.asarray(t))) > 50


def test_cosine_learning_rate_decays_from_base_to_zero():
    learning_rates = [
        float(dm.cosine_learning_rate(update, 0.001, 1000, 0.0))
        for update in (0, 250, 500, 999, 1000)
    ]

    assert learning_rates[0] == pytest.approx(0.001)
    assert learning_rates[-1] == pytest.approx(0.0)
    assert learning_rates[2] == pytest.approx(0.0005)
    assert all(
        later <= earlier
        for earlier, later in zip(learning_rates, learning_rates[1:])
    )


def test_train_step_updates_parameters_and_tracks_adam_count():
    params = dm.init_model_params(jax.random.PRNGKey(4), CONFIG)
    optimizer_state = dm.init_adam(params)
    _, t, _ = dm.sample_training_batch(
        jax.random.PRNGKey(5), 32, ALPHA_BARS
    )
    x_t, _, noise = dm.sample_training_batch(
        jax.random.PRNGKey(5), 32, ALPHA_BARS
    )
    loss_before = dm.mse_loss(params, x_t, t, noise, CONFIG)

    updated, new_state, loss = dm.train_step(
        params,
        optimizer_state,
        jax.random.PRNGKey(6),
        CONFIG,
        ALPHA_BARS,
    )

    assert bool(jnp.isfinite(loss_before))
    assert bool(jnp.isfinite(loss))
    assert int(new_state["count"]) == 1
    assert any(
        not np.allclose(np.asarray(updated[name]), np.asarray(params[name]))
        for name in params
    )


def test_train_checkpointed_returns_every_requested_checkpoint():
    small = replace(
        CONFIG,
        hidden_layers=(16, 16),
        updates=4,
        checkpoints=(0, 2, 4),
        batch_size=32,
    )

    checkpoints, losses = dm.train_checkpointed(
        small,
        ALPHA_BARS,
        jax.random.PRNGKey(7),
        chunk_updates=2,
        verbose=False,
    )

    assert set(checkpoints) == {0, 2, 4}
    assert losses.shape == (4,)
    assert bool(jnp.all(jnp.isfinite(losses)))


def test_ddpm_reverse_step_uses_beta_variance_and_skips_final_noise():
    betas = jnp.full((NUM_DIFFUSION_STEPS,), 0.02)
    alpha_bars = jnp.full((NUM_DIFFUSION_STEPS + 1,), 0.5)
    alpha_bars = alpha_bars.at[0].set(1.0)
    x_t = jnp.array([[1.5, -0.5], [2.0, 0.25]], dtype=jnp.float32)
    predicted_noise = jnp.array([[0.3, -0.2], [0.1, 0.4]], dtype=jnp.float32)
    noise = jnp.array([[0.7, -0.6], [0.2, 0.9]], dtype=jnp.float32)

    mean = (x_t - 0.02 / jnp.sqrt(0.5) * predicted_noise) / jnp.sqrt(0.98)
    step_before_final = dm.ddpm_reverse_step(
        x_t, 2, predicted_noise, betas, alpha_bars, noise
    )
    final_step = dm.ddpm_reverse_step(
        x_t, 1, predicted_noise, betas, alpha_bars, noise * 1000.0
    )

    np.testing.assert_allclose(
        step_before_final, mean + jnp.sqrt(0.02) * noise, rtol=1e-6
    )
    np.testing.assert_allclose(final_step, mean, rtol=1e-6)


def test_ddpm_reverse_step_does_not_clip_samples():
    betas = jnp.full((NUM_DIFFUSION_STEPS,), 0.02)
    alpha_bars = jnp.full((NUM_DIFFUSION_STEPS + 1,), 0.5)
    alpha_bars = alpha_bars.at[0].set(1.0)
    x_t = jnp.full((1, 2), 100.0, dtype=jnp.float32)
    predicted_noise = jnp.zeros((1, 2), dtype=jnp.float32)
    noise = jnp.zeros((1, 2), dtype=jnp.float32)

    result = dm.ddpm_reverse_step(
        x_t, 2, predicted_noise, betas, alpha_bars, noise
    )

    assert float(result[0, 0]) > 100.0


def test_ddpm_reverse_sampler_is_deterministic_for_fixed_noise():
    small = replace(CONFIG, hidden_layers=(8, 8))
    params = dm.init_model_params(jax.random.PRNGKey(8), small)
    x_T = jax.random.normal(jax.random.PRNGKey(9), (2, 2))
    reverse_noise = jax.random.normal(
        jax.random.PRNGKey(10), (NUM_DIFFUSION_STEPS, 2, 2)
    )
    betas = linear_beta_schedule()

    sampler = dm.make_reverse_sampler(small, betas, ALPHA_BARS)
    trajectory_a = sampler(params, x_T, reverse_noise)
    trajectory_b = sampler(params, x_T, reverse_noise)

    assert trajectory_a.shape == (NUM_DIFFUSION_STEPS + 1, 2, 2)
    assert bool(jnp.all(jnp.isfinite(trajectory_a)))
    np.testing.assert_allclose(trajectory_a, trajectory_b)


def test_checkpoint_save_and_load_round_trip(tmp_path):
    params = dm.init_model_params(jax.random.PRNGKey(11), CONFIG)
    path = tmp_path / "params-000000.npz"

    dm.save_checkpoint(params, path)
    loaded = dm.load_checkpoint(path)

    assert set(loaded) == set(params)
    for name in params:
        np.testing.assert_allclose(loaded[name], np.asarray(params[name]))
