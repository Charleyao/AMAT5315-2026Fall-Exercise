"""Tests for Part 2 VERIFY (1) metrics."""

import numpy as np
import jax.numpy as jnp

from forward_diffusion import CENTERS
from part2_verify import (
    POPULATION_TARGET,
    POPULATION_TOLERANCE,
    WIDTH_MAX,
    WIDTH_MIN,
    format_part2_table,
    part2_metrics,
)


def _synthetic_exact_samples() -> np.ndarray:
    offsets = np.array(
        [[0.3, 0.3], [0.3, -0.3], [-0.3, 0.3], [-0.3, -0.3]],
        dtype=np.float32,
    )
    centers = np.asarray(CENTERS, dtype=np.float32)
    points = centers[:, None, :] + offsets[None, :, :]
    return np.repeat(points.reshape(-1, 2), 125, axis=0)


def test_part2_targets_and_tolerances():
    assert POPULATION_TARGET == 0.25
    assert POPULATION_TOLERANCE == 0.015
    assert WIDTH_MIN == 0.29
    assert WIDTH_MAX == 0.32


def test_part2_metrics_pass_for_an_exact_target_distribution():
    metrics = part2_metrics(jnp.asarray(_synthetic_exact_samples()))

    assert metrics["passed"] is True
    assert metrics["all_population_pass"] is True
    assert metrics["all_width_pass"] is True
    assert metrics["max_population_error"] == 0.0
    for cluster in range(4):
        result = metrics["clusters"][str(cluster)]
        assert result["population_fraction"] == 0.25
        np.testing.assert_allclose(result["std_x"], 0.30, atol=1e-6)
        np.testing.assert_allclose(result["std_y"], 0.30, atol=1e-6)
        np.testing.assert_allclose(result["width"], 0.30, atol=1e-6)
        assert result["width_within_range"] is True


def test_part2_metrics_detect_an_overwide_cluster():
    samples = _synthetic_exact_samples()
    mask = np.arange(500) < 500
    cluster_zero = np.where(mask)[0]
    samples[cluster_zero] = (
        np.asarray(CENTERS[0])
        + (samples[cluster_zero] - np.asarray(CENTERS[0])) * 1.3
    )

    metrics = part2_metrics(jnp.asarray(samples))

    assert metrics["passed"] is False
    assert metrics["clusters"]["0"]["width_within_range"] is False
    assert metrics["clusters"]["1"]["width_within_range"] is True


def test_part2_table_reports_cluster_fractions_and_widths():
    metrics = part2_metrics(jnp.asarray(_synthetic_exact_samples()))

    table = format_part2_table(metrics)

    for cluster in range(4):
        assert f"Cluster {cluster}" in table
    assert "fraction" in table
    assert "width" in table
    assert "Overall Part 2 VERIFY (1): PASS" in table
