"""Metrics for Part 2 VERIFY (1): DDIM and DDPM sample quality."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from forward_diffusion import CENTERS
from verify3 import assign_to_nearest_center

POPULATION_TARGET = 0.25
POPULATION_TOLERANCE = 0.015
WIDTH_MIN = 0.29
WIDTH_MAX = 0.32


def cluster_width(std_x: float, std_y: float) -> float:
    """Isotropic cluster width: RMS of the two coordinate std values."""
    return float(np.sqrt((std_x**2 + std_y**2) / 2.0))


def part2_metrics(
    samples: jnp.ndarray | np.ndarray,
    centers: jnp.ndarray | np.ndarray = CENTERS,
    labels: jnp.ndarray | np.ndarray | None = None,
) -> dict[str, object]:
    """Compute cluster fractions and widths for Part 2 verification."""
    samples_np = np.asarray(samples, dtype=np.float64)
    centers_np = np.asarray(centers, dtype=np.float64)
    if labels is None:
        labels_np = np.asarray(
            assign_to_nearest_center(samples_np, centers_np)[0]
        )
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
            std = assigned.std(axis=0)
            width = cluster_width(std[0], std[1])
        else:
            std = np.full(2, np.nan)
            width = float("nan")

        population_error = abs(population_fraction - POPULATION_TARGET)
        population_ok = bool(
            population_error <= POPULATION_TOLERANCE
        )
        width_ok = bool(
            population > 0
            and WIDTH_MIN <= width <= WIDTH_MAX
        )

        clusters[str(cluster)] = {
            "target_center": centers_np[cluster].tolist(),
            "population": population,
            "population_fraction": float(population_fraction),
            "population_error": float(population_error),
            "population_within_tolerance": population_ok,
            "std_x": float(std[0]),
            "std_y": float(std[1]),
            "width": width,
            "width_within_range": width_ok,
            "passed": population_ok and width_ok,
        }

    all_population_pass = all(
        result["population_within_tolerance"]
        for result in clusters.values()
    )
    all_width_pass = all(
        result["width_within_range"] for result in clusters.values()
    )
    population_errors = [
        result["population_error"] for result in clusters.values()
    ]
    widths = [
        result["width"]
        for result in clusters.values()
        if np.isfinite(result["width"])
    ]

    return {
        "num_samples": int(samples_np.shape[0]),
        "population_target": POPULATION_TARGET,
        "population_tolerance": POPULATION_TOLERANCE,
        "width_min": WIDTH_MIN,
        "width_max": WIDTH_MAX,
        "clusters": clusters,
        "max_population_error": float(max(population_errors)),
        "min_width": float(min(widths)) if widths else float("nan"),
        "max_width": float(max(widths)) if widths else float("nan"),
        "all_population_pass": all_population_pass,
        "all_width_pass": all_width_pass,
        "passed": all_population_pass and all_width_pass,
    }


def format_part2_table(metrics: dict[str, object], name: str = "") -> str:
    """Format a per-cluster fraction/width PASS-FAIL table."""
    heading = f"{name}\n" if name else ""
    lines = [
        heading
        + "Cluster | n | fraction | fraction tolerance | std x | std y | width | "
        "width range | overall",
    ]
    for cluster in range(len(metrics["clusters"])):
        result = metrics["clusters"][str(cluster)]
        lines.append(
            f"Cluster {cluster} | "
            f"{result['population']:4d} | "
            f"{result['population_fraction']:.4f} | "
            f"{'PASS' if result['population_within_tolerance'] else 'FAIL'} | "
            f"{result['std_x']:.4f} | "
            f"{result['std_y']:.4f} | "
            f"{result['width']:.4f} | "
            f"{'PASS' if result['width_within_range'] else 'FAIL'} | "
            f"{'PASS' if result['passed'] else 'FAIL'}"
        )
    lines.append(
        f"max |fraction - 0.25| = {metrics['max_population_error']:.4f} "
        f"(tolerance {metrics['population_tolerance']})"
    )
    lines.append(
        f"width range = [{metrics['min_width']:.4f}, "
        f"{metrics['max_width']:.4f}] "
        f"(required [{metrics['width_min']}, {metrics['width_max']}])"
    )
    lines.append(
        "Overall Part 2 VERIFY (1): "
        f"{'PASS' if metrics['passed'] else 'FAIL'}"
    )
    return "\n".join(lines)


__all__ = [
    "POPULATION_TARGET",
    "POPULATION_TOLERANCE",
    "WIDTH_MAX",
    "WIDTH_MIN",
    "cluster_width",
    "format_part2_table",
    "part2_metrics",
]
