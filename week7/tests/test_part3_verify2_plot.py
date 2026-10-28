"""Tests for Part 3 VERIFY (2) metric figure."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from part3_verify2_plot import plot_metrics_vs_steps


def test_metrics_plot_contains_all_methods_and_null_shade(tmp_path):
    results = {
        "DDPM": {
            1: {
                "energy_distance": 0.02,
                "max_population_error": 0.04,
                "max_width_error": 0.08,
            },
            10: {
                "energy_distance": 0.01,
                "max_population_error": 0.02,
                "max_width_error": 0.04,
            },
        },
        "DDIM": {
            1: {
                "energy_distance": 0.015,
                "max_population_error": 0.03,
                "max_width_error": 0.06,
            },
            10: {
                "energy_distance": 0.008,
                "max_population_error": 0.015,
                "max_width_error": 0.02,
            },
        },
        "Flow": {
            1: {
                "energy_distance": 0.012,
                "max_population_error": 0.02,
                "max_width_error": 0.05,
            },
            10: {
                "energy_distance": 0.004,
                "max_population_error": 0.01,
                "max_width_error": 0.015,
            },
        },
        "DDIM_exact": {
            1: {
                "energy_distance": 0.018,
                "max_population_error": 0.035,
                "max_width_error": 0.07,
            },
            2: {
                "energy_distance": 0.013,
                "max_population_error": 0.025,
                "max_width_error": 0.05,
            },
        },
    }
    output_path = tmp_path / "metrics.png"

    figure, axes = plot_metrics_vs_steps(
        results, null_q95=0.001, output_path=str(output_path)
    )

    try:
        assert axes.shape == (3,)
        assert len(axes[0].patches) >= 1
        assert len(axes[0].lines) >= 3
        assert output_path.stat().st_size > 0
    finally:
        plt.close(figure)
