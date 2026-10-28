"""Tests for Flow Matching training-loss and velocity-field figures."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import flow_matching as fm
from flow_plot import plot_flow_training_loss, plot_velocity_fields


def test_velocity_field_figure_contains_all_three_s_values(tmp_path):
    result = fm.evaluate_velocity_fields(
        lambda points, s: fm.exact_marginal_velocity(points, s),
        [0.1, 0.5, 0.9],
        points=15,
    )
    output_path = tmp_path / "velocity_fields.png"

    figure, axes = plot_velocity_fields(result, str(output_path))

    try:
        assert axes.shape == (3, 2)
        for row, s in enumerate(("0.1", "0.5", "0.9")):
            assert f"s={s}" in axes[row, 0].get_title()
            assert f"s={s}" in axes[row, 1].get_title()
        assert output_path.stat().st_size > 0
    finally:
        plt.close(figure)


def test_training_loss_figure_is_written(tmp_path):
    losses = np.linspace(2.0, 0.5, 500, dtype=np.float32)
    output_path = tmp_path / "flow_loss.png"

    plot_flow_training_loss(
        losses, [0, 100, 250, 500], str(output_path)
    )

    assert output_path.stat().st_size > 0
