"""Tests for VERIFY (2) checkpoint animation and figures."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from diffusion_verify2_plot import (
    VERIFY2_AXIS_LIMIT,
    create_endpoint_evolution_animation,
    plot_training_loss,
    render_checkpoint_grid,
)


def _fake_data():
    checkpoints = [0, 50, 100]
    endpoints = np.zeros((3, 4, 2), dtype=np.float32)
    trajectories = np.zeros((3, 6, 4, 2), dtype=np.float32)
    losses = np.linspace(1.0, 0.1, 120, dtype=np.float32)
    return checkpoints, endpoints, trajectories, losses


def test_endpoint_animation_has_fixed_limits_and_checkpoint_label():
    checkpoints, endpoints, _, losses = _fake_data()

    animation, figure, (scatter_axis, loss_axis) = (
        create_endpoint_evolution_animation(
            checkpoints, endpoints, losses, interval_ms=100
        )
    )

    try:
        animation._func(1)
        np.testing.assert_allclose(
            scatter_axis.get_xlim(), (-VERIFY2_AXIS_LIMIT, VERIFY2_AXIS_LIMIT)
        )
        np.testing.assert_allclose(
            scatter_axis.get_ylim(), (-VERIFY2_AXIS_LIMIT, VERIFY2_AXIS_LIMIT)
        )
        assert "update = 50" in scatter_axis.get_title()
        assert len(loss_axis.lines) >= 1
        animation._init_draw()
    finally:
        plt.close(figure)


def test_animation_reports_endpoints_outside_the_fixed_limits():
    checkpoints = [0, 50]
    endpoints = np.array(
        [
            [[100.0, -100.0], [-200.0, 50.0]],
            [[0.0, 0.0], [1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    losses = np.linspace(1.0, 0.1, 120, dtype=np.float32)

    animation, figure, (scatter_axis, _) = create_endpoint_evolution_animation(
        checkpoints, endpoints, losses, interval_ms=100
    )

    try:
        animation._func(0)
        texts = [text.get_text() for text in scatter_axis.texts]
        assert any("2/2" in text and "outside" in text for text in texts)
        animation._init_draw()
    finally:
        plt.close(figure)


def test_checkpoint_grid_and_loss_figure_are_written(tmp_path):
    checkpoints, _, trajectories, losses = _fake_data()
    grid_path = tmp_path / "checkpoints.png"
    loss_path = tmp_path / "loss.png"

    render_checkpoint_grid(checkpoints, trajectories, str(grid_path))
    plot_training_loss(losses, checkpoints, str(loss_path))

    assert grid_path.stat().st_size > 0
    assert loss_path.stat().st_size > 0
