"""Tests for VERIFY (3) animation and comparison figures."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from verify3 import cluster_metrics
from verify3_plot import (
    REVERSE_AXIS_LIMIT,
    create_reverse_diffusion_animation,
    render_final_comparison,
    render_reverse_keyframes,
)


def _synthetic_exact_samples() -> np.ndarray:
    offsets = np.array(
        [[0.3, 0.3], [0.3, -0.3], [-0.3, 0.3], [-0.3, -0.3]],
        dtype=np.float32,
    )
    centers = np.array(
        [[3.0, 0.0], [0.0, 3.0], [-3.0, 0.0], [0.0, -3.0]],
        dtype=np.float32,
    )
    points = centers[:, None, :] + offsets[None, :, :]
    return np.repeat(points.reshape(-1, 2), 125, axis=0)


def test_reverse_animation_uses_fixed_limits_and_step_labels():
    trajectory = np.zeros((5, 20, 2), dtype=np.float32)
    frame_steps = [0, 1, 2, 3, 4]

    animation, figure, axis = create_reverse_diffusion_animation(
        trajectory, frame_steps, interval_ms=100
    )

    try:
        animation._func(4)
        np.testing.assert_allclose(
            axis.get_xlim(), (-REVERSE_AXIS_LIMIT, REVERSE_AXIS_LIMIT)
        )
        np.testing.assert_allclose(
            axis.get_ylim(), (-REVERSE_AXIS_LIMIT, REVERSE_AXIS_LIMIT)
        )
        assert "step = 4" in axis.get_title()
        animation._init_draw()
    finally:
        plt.close(figure)


def test_final_comparison_and_keyframes_are_written(tmp_path):
    samples = _synthetic_exact_samples()
    labels = np.repeat(np.arange(4), 500)
    metrics = cluster_metrics(jnp.asarray(samples))
    trajectory = np.repeat(samples[None, :, :], 5, axis=0)
    comparison_path = tmp_path / "comparison.png"
    keyframes_path = tmp_path / "keyframes.png"

    render_final_comparison(samples, samples, labels, metrics, str(comparison_path))
    render_reverse_keyframes(
        trajectory, [0, 1, 2, 3, 4], str(keyframes_path)
    )

    assert comparison_path.stat().st_size > 0
    assert keyframes_path.stat().st_size > 0
