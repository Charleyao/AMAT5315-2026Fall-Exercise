"""Part 2 VERIFY (1) trajectory and sample-comparison figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from forward_diffusion import CENTERS
from verify3 import assign_to_nearest_center

TRAJECTORY_AXIS_LIMIT = 4.5
_CLUSTER_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")


def _scatter_by_cluster(axis, points: np.ndarray, alpha: float = 0.5) -> None:
    labels, _ = assign_to_nearest_center(points)
    labels = np.asarray(labels)
    colors = np.asarray(_CLUSTER_COLORS)[labels % 4]
    axis.scatter(
        points[:, 0],
        points[:, 1],
        s=5,
        c=colors,
        alpha=alpha,
        linewidths=0,
    )


def _draw_centers(axis) -> None:
    centers = np.asarray(CENTERS)
    axis.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="x",
        s=70,
        c="black",
        linewidths=1.8,
        zorder=5,
    )


def render_trajectory_comparison(
    ddim_trajectory: np.ndarray,
    ddpm_trajectory_1: np.ndarray,
    ddpm_trajectory_2: np.ndarray,
    sample_index: int,
    output_path: str,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot one DDIM path and the two stochastic DDPM paths."""
    ddim_trajectory = np.asarray(ddim_trajectory)
    ddpm_trajectory_1 = np.asarray(ddpm_trajectory_1)
    ddpm_trajectory_2 = np.asarray(ddpm_trajectory_2)
    for trajectory in (
        ddim_trajectory,
        ddpm_trajectory_1,
        ddpm_trajectory_2,
    ):
        if trajectory.ndim != 3 or trajectory.shape[2] != 2:
            raise ValueError("trajectories must have shape (steps, samples, 2)")
        if not 0 <= sample_index < trajectory.shape[1]:
            raise ValueError("sample_index is out of range")

    ddim_path = ddim_trajectory[:, sample_index, :]
    ddpm_path_1 = ddpm_trajectory_1[:, sample_index, :]
    ddpm_path_2 = ddpm_trajectory_2[:, sample_index, :]

    figure, axis = plt.subplots(figsize=(7.0, 6.6))
    axis.plot(
        ddim_path[:, 0],
        ddim_path[:, 1],
        color="#1f77b4",
        linewidth=2.0,
        label="DDIM (deterministic)",
    )
    axis.plot(
        ddpm_path_1[:, 0],
        ddpm_path_1[:, 1],
        color="#ff7f0e",
        linewidth=1.4,
        alpha=0.85,
        label="DDPM noise draw 1",
    )
    axis.plot(
        ddpm_path_2[:, 0],
        ddpm_path_2[:, 1],
        color="#d62728",
        linewidth=1.4,
        alpha=0.85,
        label="DDPM noise draw 2",
    )
    axis.scatter(
        ddim_path[0, 0],
        ddim_path[0, 1],
        marker="*",
        s=180,
        c="black",
        zorder=6,
        label="shared x_T",
    )
    for path, color in (
        (ddim_path, "#1f77b4"),
        (ddpm_path_1, "#ff7f0e"),
        (ddpm_path_2, "#d62728"),
    ):
        axis.scatter(
            path[-1, 0],
            path[-1, 1],
            s=55,
            color=color,
            edgecolors="white",
            linewidths=0.7,
            zorder=7,
        )
    _draw_centers(axis)
    axis.set_xlim(-TRAJECTORY_AXIS_LIMIT, TRAJECTORY_AXIS_LIMIT)
    axis.set_ylim(-TRAJECTORY_AXIS_LIMIT, TRAJECTORY_AXIS_LIMIT)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right", fontsize=8)
    axis.set_title(
        f"Reverse trajectories for sample {sample_index} "
        "(t=1000 to t=0)"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    return figure, axis


def render_sample_comparison(
    ddim_samples: np.ndarray,
    ddpm_samples: np.ndarray,
    target_samples: np.ndarray,
    ddim_metrics: dict[str, object],
    ddpm_metrics: dict[str, object],
    output_path: str,
) -> None:
    """Compare 2000 DDIM, 2000 DDPM, and 2000 exact target samples."""
    panels = [
        (
            np.asarray(ddim_samples),
            "DDIM: 2000 samples\n"
            f"max|frac-0.25|={ddim_metrics['max_population_error']:.4f}, "
            f"widths [{ddim_metrics['min_width']:.3f}, "
            f"{ddim_metrics['max_width']:.3f}]",
        ),
        (
            np.asarray(ddpm_samples),
            "DDPM: 2000 samples\n"
            f"max|frac-0.25|={ddpm_metrics['max_population_error']:.4f}, "
            f"widths [{ddpm_metrics['min_width']:.3f}, "
            f"{ddpm_metrics['max_width']:.3f}]",
        ),
        (
            np.asarray(target_samples),
            "Target: 2000 exact Gaussian-mixture samples",
        ),
    ]

    figure, axes = plt.subplots(1, 3, figsize=(16.0, 5.4), sharex=True, sharey=True)
    for axis, (points, title) in zip(axes, panels):
        _scatter_by_cluster(axis, points)
        _draw_centers(axis)
        axis.set_xlim(-TRAJECTORY_AXIS_LIMIT, TRAJECTORY_AXIS_LIMIT)
        axis.set_ylim(-TRAJECTORY_AXIS_LIMIT, TRAJECTORY_AXIS_LIMIT)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.grid(alpha=0.25)
        axis.set_title(title, fontsize=10)

    figure.suptitle(
        "Part 2 VERIFY (1): DDIM and DDPM samples vs exact target samples",
        fontsize=13,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def render_exact_predictor_comparison(
    learned_samples: np.ndarray,
    exact_samples: np.ndarray,
    target_samples: np.ndarray,
    exact_metrics: dict[str, object],
    output_path: str,
) -> None:
    """Compare learned-DDIM, exact-predictor DDIM, and target samples."""
    panels = [
        (
            np.asarray(learned_samples),
            "Learned predictor DDIM: 2000 samples",
        ),
        (
            np.asarray(exact_samples),
            "Exact conditional-mean DDIM: 2000 samples\n"
            f"max|frac-0.25|={exact_metrics['max_population_error']:.4f}, "
            f"widths [{exact_metrics['min_width']:.3f}, "
            f"{exact_metrics['max_width']:.3f}]",
        ),
        (
            np.asarray(target_samples),
            "Target: 2000 exact Gaussian-mixture samples",
        ),
    ]

    figure, axes = plt.subplots(1, 3, figsize=(16.0, 5.4), sharex=True, sharey=True)
    for axis, (points, title) in zip(axes, panels):
        _scatter_by_cluster(axis, points)
        _draw_centers(axis)
        axis.set_xlim(-TRAJECTORY_AXIS_LIMIT, TRAJECTORY_AXIS_LIMIT)
        axis.set_ylim(-TRAJECTORY_AXIS_LIMIT, TRAJECTORY_AXIS_LIMIT)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.grid(alpha=0.25)
        axis.set_title(title, fontsize=10)

    figure.suptitle(
        "Part 2: learned vs exact conditional-mean noise predictor",
        fontsize=13,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


__all__ = [
    "TRAJECTORY_AXIS_LIMIT",
    "render_exact_predictor_comparison",
    "render_sample_comparison",
    "render_trajectory_comparison",
]
