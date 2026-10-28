"""Tests for Part 5 RePaint masks, schedule, and verification metrics."""

import numpy as np
import pytest

import repaint


def test_mask_hides_rows_and_columns_6_to_25():
    mask = repaint.make_repaint_mask()

    assert mask.shape == (32, 32, 1)
    assert mask[0, 0, 0] == 1.0
    assert mask[5, 15, 0] == 1.0
    assert mask[6, 6, 0] == 0.0
    assert mask[25, 25, 0] == 0.0
    assert mask[26, 26, 0] == 1.0
    assert int(mask.sum()) == 32 * 32 - 20 * 20


def test_repaint_schedule_has_50_jumps_and_99_evaluations():
    levels = repaint.REPAINT_LEVELS

    assert levels[0] == 1000
    assert levels[-2:] == (20, 0)
    assert len(levels) == 51
    assert repaint.repaint_jumps()[0] == (1000, 980)
    assert repaint.repaint_jumps()[-1] == (20, 0)
    assert len(repaint.repaint_jumps()) == 50
    assert repaint.repaint_network_evaluations() == 99


def test_test_images_load_with_expected_shape_and_labels():
    images, labels = repaint.load_test_images()

    assert images.shape == (4, 32, 32, 3)
    assert images.dtype == np.float32
    assert labels.shape == (4,)
    assert np.isfinite(images).all()


def test_repaint_metrics_detects_observed_change_and_hidden_variance():
    originals = np.zeros((2, 32, 32, 3), dtype=np.float32)
    completions = np.zeros((2, 4, 32, 32, 3), dtype=np.float32)
    mask = repaint.make_repaint_mask()
    hidden = np.broadcast_to((1.0 - mask) > 0.5, (32, 32, 3))
    for seed in range(4):
        completions[0, seed][hidden] = (seed + 1) * 0.1
        completions[1, seed][hidden] = -(seed + 1) * 0.1

    metrics = repaint.repaint_metrics(completions, originals, mask)

    assert metrics["observed_max_change"] == pytest.approx(0.0)
    assert metrics["max_hidden_std"] > 0.0
    assert metrics["max_pairwise_hidden_difference"] > 0.0
    assert metrics["cross_image_hidden_mean_abs_difference"] > 0.0
    assert metrics["completions_in_range"] is True
    assert len(metrics["hidden_mse"]) == 2
    assert len(metrics["hidden_mse"][0]) == 4


def test_model_timestep_is_zero_indexed_t_minus_one():
    assert repaint.model_timestep(1000) == 999
    assert repaint.model_timestep(20) == 19
    assert repaint.model_timestep(1) == 0


def test_predict_clean_image_formula_and_clip():
    import jax.numpy as jnp

    x_t = jnp.array([[1.0, -2.0]], dtype=jnp.float32)
    predicted_noise = jnp.array([[0.5, -0.25]], dtype=jnp.float32)
    alpha_bar_t = 0.5
    expected = (
        x_t - jnp.sqrt(1.0 - alpha_bar_t) * predicted_noise
    ) / jnp.sqrt(alpha_bar_t)
    clipped = jnp.clip(expected, -1.0, 1.0)

    result = repaint.predict_clean_image(
        x_t, alpha_bar_t, predicted_noise
    )

    import numpy as np
    np.testing.assert_allclose(result, clipped, rtol=1e-6)


def test_jump_posterior_variance_and_noise_coefficient_use_square_roots():
    import numpy as np

    alpha_bar_t = 0.5
    alpha_bar_previous = 0.8
    jump_beta = 1.0 - alpha_bar_t / alpha_bar_previous
    expected_variance = (
        (1.0 - alpha_bar_previous)
        / (1.0 - alpha_bar_t)
        * jump_beta
    )
    expected_coefficient = np.sqrt(
        1.0 - alpha_bar_previous - expected_variance
    )

    variance, coefficient = repaint.jump_posterior_terms(
        alpha_bar_t, alpha_bar_previous
    )

    np.testing.assert_allclose(float(variance), expected_variance, rtol=1e-6)
    np.testing.assert_allclose(
        float(coefficient), expected_coefficient, rtol=1e-6
    )


def test_posterior_jump_matches_manual_mean_and_variance():
    import jax.numpy as jnp
    import numpy as np

    x_t = jnp.array([[0.4, -0.7]], dtype=jnp.float32)
    predicted_noise = jnp.array([[0.2, -0.3]], dtype=jnp.float32)
    reverse_noise = jnp.array([[0.5, 0.1]], dtype=jnp.float32)
    alpha_bar_t = 0.5
    alpha_bar_previous = 0.8
    predicted_x0 = repaint.predict_clean_image(
        x_t, alpha_bar_t, predicted_noise
    )
    variance, coefficient = repaint.jump_posterior_terms(
        alpha_bar_t, alpha_bar_previous
    )
    expected = (
        jnp.sqrt(alpha_bar_previous) * predicted_x0
        + coefficient * predicted_noise
        + jnp.sqrt(variance) * reverse_noise
    )

    result, returned_x0 = repaint.posterior_jump(
        x_t,
        alpha_bar_t,
        alpha_bar_previous,
        predicted_noise,
        reverse_noise,
    )

    np.testing.assert_allclose(result, expected, rtol=1e-6)
    np.testing.assert_allclose(returned_x0, predicted_x0, rtol=1e-6)


def test_final_posterior_jump_has_zero_variance_and_returns_x0():
    import jax.numpy as jnp
    import numpy as np

    x_t = jnp.array([[0.4, -0.7]], dtype=jnp.float32)
    predicted_noise = jnp.array([[0.2, -0.3]], dtype=jnp.float32)
    reverse_noise = jnp.array([[100.0, -100.0]], dtype=jnp.float32)
    predicted_x0 = repaint.predict_clean_image(
        x_t, 0.5, predicted_noise
    )

    variance, _ = repaint.jump_posterior_terms(0.5, 1.0)
    result, _ = repaint.posterior_jump(
        x_t, 0.5, 1.0, predicted_noise, reverse_noise
    )

    assert float(variance) == pytest.approx(0.0)
    np.testing.assert_allclose(result, predicted_x0, rtol=1e-6)


def test_forward_renoise_step_matches_forward_diffusion_formula():
    import jax.numpy as jnp
    import numpy as np

    x = jnp.array([[0.3, -0.5]], dtype=jnp.float32)
    noise = jnp.array([[1.2, -0.4]], dtype=jnp.float32)
    beta = 0.02
    expected = jnp.sqrt(1.0 - beta) * x + jnp.sqrt(beta) * noise

    result = repaint.forward_renoise_step(x, beta, noise)

    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_known_observation_and_mask_merge_at_the_right_pixels():
    import jax.numpy as jnp
    import numpy as np

    observation = jnp.full((32, 32, 3), 0.5, dtype=jnp.float32)
    unknown = jnp.full((32, 32, 3), -0.5, dtype=jnp.float32)
    noise = jnp.zeros((32, 32, 3), dtype=jnp.float32)
    mask = repaint.make_repaint_mask()
    known = repaint.known_observation(observation, 0.5, noise)
    merged = repaint.merge_mask(known, unknown, jnp.asarray(mask))
    hidden = np.broadcast_to((1.0 - mask) > 0.5, (32, 32, 3))
    observed = ~hidden

    np.testing.assert_allclose(
        merged[hidden], -0.5, atol=1e-6
    )
    np.testing.assert_allclose(
        merged[observed], float(0.5 * np.sqrt(0.5)), atol=1e-6
    )


def test_hidden_mse_uses_only_hidden_pixels():
    import numpy as np

    originals = np.zeros((1, 32, 32, 3), dtype=np.float32)
    completions = np.zeros((1, 4, 32, 32, 3), dtype=np.float32)
    mask = repaint.make_repaint_mask()
    hidden = np.broadcast_to((1.0 - mask) > 0.5, (32, 32, 3))
    completions[:, :, hidden] = 1.0

    metrics = repaint.repaint_metrics(completions, originals, mask)

    np.testing.assert_allclose(metrics["hidden_mse"], 1.0, atol=1e-6)


def test_load_test_images_are_nhwc_and_in_minus_one_to_one():
    import numpy as np
    from pathlib import Path

    images, _ = repaint.load_test_images()
    raw = np.load(Path(repaint.__file__).parent / "inputs" / "images.npz")
    expected = np.transpose(raw["test"], (0, 2, 3, 1))

    np.testing.assert_allclose(images, expected)
    assert images.min() >= -1.0 and images.max() <= 1.0
