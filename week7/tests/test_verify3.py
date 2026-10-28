"""Tests for the VERIFY (3) sample metrics."""

import numpy as np
import jax.numpy as jnp

from forward_diffusion import CENTERS
from verify3 import (
    CENTER_TOLERANCE,
    NUM_SAMPLES,
    POPULATION_TARGET,
    POPULATION_TOLERANCE,
    STD_TARGET,
    STD_TOLERANCE,
    assign_to_nearest_center,
    cluster_metrics,
    format_verification_table,
)


def _synthetic_exact_samples() -> np.ndarray:
    """2000 samples with exact 1/4 weights, exact centers, and std 0.30."""
    offsets = np.array(
        [[0.3, 0.3], [0.3, -0.3], [-0.3, 0.3], [-0.3, -0.3]],
        dtype=np.float32,
    )
    centers = np.asarray(CENTERS, dtype=np.float32)
    points = centers[:, None, :] + offsets[None, :, :]
    return np.repeat(points.reshape(-1, 2), 125, axis=0)


def test_verify3_targets_and_tolerances_match_the_learning_sheet():
    assert NUM_SAMPLES == 2000
    assert POPULATION_TARGET == 0.25
    assert POPULATION_TOLERANCE == 0.04
    assert CENTER_TOLERANCE == 0.05
    assert STD_TARGET == 0.30
    assert STD_TOLERANCE == 0.03


def test_nearest_center_assignment_uses_the_four_mixture_centers():
    samples = jnp.array(
        [[2.9, 0.1], [0.1, 2.9], [-2.9, 0.1], [0.1, -2.9]],
        dtype=jnp.float32,
    )

    labels, distances = assign_to_nearest_center(samples)

    np.testing.assert_array_equal(np.asarray(labels), np.array([0, 1, 2, 3]))
    assert distances.shape == (4,)
    assert bool(jnp.all(distances >= 0.0))


def test_cluster_metrics_pass_for_an_exact_target_distribution():
    samples = _synthetic_exact_samples()

    metrics = cluster_metrics(jnp.asarray(samples))

    assert metrics["passed"] is True
    assert metrics["all_population_pass"] is True
    assert metrics["all_center_pass"] is True
    assert metrics["all_std_pass"] is True
    for cluster in range(4):
        result = metrics["clusters"][str(cluster)]
        assert result["population"] == 500
        assert result["population_fraction"] == 0.25
        np.testing.assert_allclose(result["estimated_center"], CENTERS[cluster])
        np.testing.assert_allclose(result["std_x"], STD_TARGET, atol=1e-6)
        np.testing.assert_allclose(result["std_y"], STD_TARGET, atol=1e-6)


def test_cluster_metrics_detect_a_shifted_cluster():
    samples = _synthetic_exact_samples()
    samples[:500] += np.array([0.2, 0.0], dtype=np.float32)

    metrics = cluster_metrics(jnp.asarray(samples))

    assert metrics["passed"] is False
    assert metrics["clusters"]["0"]["center_within_tolerance"] is False
    assert metrics["clusters"]["1"]["center_within_tolerance"] is True


def test_verification_table_lists_each_cluster_and_overall_status():
    metrics = cluster_metrics(jnp.asarray(_synthetic_exact_samples()))

    table = format_verification_table(metrics)

    for cluster in range(4):
        assert f"Cluster {cluster}" in table
    assert "population" in table
    assert "center" in table
    assert "std" in table
    assert "Overall VERIFY (3): PASS" in table
