"""Tests for Part 2 trajectory and comparison figures."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from part2_plot import (
    TRAJECTORY_AXIS_LIMIT,
    render_exact_predictor_comparison,
    render_sample_comparison,
    render_trajectory_comparison,
)
from part2_verify import part2_metrics


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


def test_trajectory_comparison_uses_fixed_axes(tmp_path):
    trajectory = np.zeros((5, 10, 2), dtype=np.float32)
    output_path = tmp_path / "trajectory.png"

    figure, axis = render_trajectory_comparison(
        trajectory, trajectory, trajectory, sample_index=0, output_path=str(output_path)
    )

    try:
        np.testing.assert_allclose(
            axis.get_xlim(), (-TRAJECTORY_AXIS_LIMIT, TRAJECTORY_AXIS_LIMIT)
        )
        np.testing.assert_allclose(
            axis.get_ylim(), (-TRAJECTORY_AXIS_LIMIT, TRAJECTORY_AXIS_LIMIT)
        )
        assert "sample 0" in axis.get_title()
    finally:
        plt.close(figure)
    assert output_path.stat().st_size > 0


def test_sample_and_exact_predictor_figures_are_written(tmp_path):
    samples = _synthetic_exact_samples()
    metrics = part2_metrics(jnp.asarray(samples))
    sample_path = tmp_path / "samples.png"
    exact_path = tmp_path / "exact.png"

    render_sample_comparison(
        samples,
        samples,
        samples,
        metrics,
        metrics,
        str(sample_path),
    )
    render_exact_predictor_comparison(
        samples, samples, samples, metrics, str(exact_path)
    )

    assert sample_path.stat().st_size > 0
    assert exact_path.stat().st_size > 0
