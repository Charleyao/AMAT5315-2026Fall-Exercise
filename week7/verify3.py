"""VERIFY (3) metrics for generated Gaussian-mixture samples."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from forward_diffusion import CENTERS

NUM_SAMPLES = 2000
POPULATION_TARGET = 0.25
POPULATION_TOLERANCE = 0.04
CENTER_TOLERANCE = 0.05
STD_TARGET = 0.30
STD_TOLERANCE = 0.03


def assign_to_nearest_center(
    samples: jnp.ndarray | np.ndarray,
    centers: jnp.ndarray | np.ndarray = CENTERS,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Assign every sample to its nearest target center.

    Returns the integer labels and the Euclidean distance to the assigned
    center.
    """
    samples = jnp.asarray(samples, dtype=jnp.float32)
    centers = jnp.asarray(centers, dtype=jnp.float32)
    squared_distances = jnp.sum(
        (samples[:, None, :] - centers[None, :, :]) ** 2, axis=-1
    )
    labels = jnp.argmin(squared_distances, axis=1)
    distances = jnp.sqrt(jnp.min(squared_distances, axis=1))
    return labels, distances


def cluster_metrics(
    samples: jnp.ndarray | np.ndarray,
    centers: jnp.ndarray | np.ndarray = CENTERS,
    labels: jnp.ndarray | np.ndarray | None = None,
) -> dict[str, object]:
    """Compute population, center, and spread checks for every cluster."""
    samples_np = np.asarray(samples, dtype=np.float64)
    centers_np = np.asarray(centers, dtype=np.float64)
    if labels is None:
        labels_np = np.asarray(assign_to_nearest_center(samples_np, centers_np)[0])
    else:
        labels_np = np.asarray(labels)

    clusters: dict[str, dict[str, object]] = {}
    for cluster in range(centers_np.shape[0]):
        assigned = samples_np[labels_np == cluster]
        population = int(assigned.shape[0])
        population_fraction = (
            population / samples_np.shape[0] if samples_np.shape[0] else 0.0
        )

        if population:
            estimated_center = assigned.mean(axis=0)
            std = assigned.std(axis=0)
        else:
            estimated_center = np.full(2, np.nan)
            std = np.full(2, np.nan)

        population_error = abs(population_fraction - POPULATION_TARGET)
        center_error = np.abs(estimated_center - centers_np[cluster])
        std_error = np.abs(std - STD_TARGET)

        population_ok = bool(
            population_error <= POPULATION_TOLERANCE
        )
        center_ok = bool(
            population > 0 and np.all(center_error <= CENTER_TOLERANCE)
        )
        std_ok = bool(population > 0 and np.all(std_error <= STD_TOLERANCE))

        clusters[str(cluster)] = {
            "target_center": centers_np[cluster].tolist(),
            "population": population,
            "population_fraction": float(population_fraction),
            "population_error": float(population_error),
            "population_within_tolerance": population_ok,
            "estimated_center": estimated_center.tolist(),
            "center_error_x": float(center_error[0]),
            "center_error_y": float(center_error[1]),
            "max_abs_center_error": float(np.max(center_error)),
            "center_within_tolerance": center_ok,
            "std_x": float(std[0]),
            "std_y": float(std[1]),
            "std_error_x": float(std_error[0]),
            "std_error_y": float(std_error[1]),
            "max_abs_std_error": float(np.max(std_error)),
            "std_within_tolerance": std_ok,
            "passed": population_ok and center_ok and std_ok,
        }

    all_population_pass = all(
        result["population_within_tolerance"]
        for result in clusters.values()
    )
    all_center_pass = all(
        result["center_within_tolerance"] for result in clusters.values()
    )
    all_std_pass = all(
        result["std_within_tolerance"] for result in clusters.values()
    )

    return {
        "num_samples": int(samples_np.shape[0]),
        "population_target": POPULATION_TARGET,
        "population_tolerance": POPULATION_TOLERANCE,
        "center_tolerance": CENTER_TOLERANCE,
        "std_target": STD_TARGET,
        "std_tolerance": STD_TOLERANCE,
        "clusters": clusters,
        "all_population_pass": all_population_pass,
        "all_center_pass": all_center_pass,
        "all_std_pass": all_std_pass,
        "passed": all_population_pass and all_center_pass and all_std_pass,
    }


def format_verification_table(metrics: dict[str, object]) -> str:
    """Format a PASS/FAIL table covering every VERIFY (3) criterion."""
    lines = [
        "Targets: population 0.25 +/- 0.04; center each coordinate +/- 0.05; "
        "std x/y 0.30 +/- 0.03",
        (
            "Cluster | n | population | pop | center estimate | "
            "max|dcenter| | center | std x | std y | max|dstd| | std | overall"
        ),
    ]

    for cluster in range(len(metrics["clusters"])):
        result = metrics["clusters"][str(cluster)]
        center = result["estimated_center"]
        lines.append(
            f"Cluster {cluster} | "
            f"{result['population']:4d} | "
            f"{result['population_fraction']:.4f} | "
            f"{'PASS' if result['population_within_tolerance'] else 'FAIL'} | "
            f"({center[0]:+.4f}, {center[1]:+.4f}) | "
            f"{result['max_abs_center_error']:.4f} | "
            f"{'PASS' if result['center_within_tolerance'] else 'FAIL'} | "
            f"{result['std_x']:.4f} | "
            f"{result['std_y']:.4f} | "
            f"{result['max_abs_std_error']:.4f} | "
            f"{'PASS' if result['std_within_tolerance'] else 'FAIL'} | "
            f"{'PASS' if result['passed'] else 'FAIL'}"
        )

    lines.append(
        "Overall VERIFY (3): "
        f"{'PASS' if metrics['passed'] else 'FAIL'}"
    )
    return "\n".join(lines)


__all__ = [
    "CENTER_TOLERANCE",
    "NUM_SAMPLES",
    "POPULATION_TARGET",
    "POPULATION_TOLERANCE",
    "STD_TARGET",
    "STD_TOLERANCE",
    "assign_to_nearest_center",
    "cluster_metrics",
    "format_verification_table",
]
