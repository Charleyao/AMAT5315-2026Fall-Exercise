"""Plotting helpers for the Week 7 Part 1 forward-diffusion evidence."""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from forward_diffusion import AXIS_LIMIT

COMPONENT_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")


def _component_colors(components: np.ndarray) -> np.ndarray:
    colors = np.asarray(COMPONENT_COLORS)
    return colors[np.asarray(components) % len(COMPONENT_COLORS)]


def create_forward_animation(
    frames: np.ndarray,
    components: np.ndarray,
    frame_steps: list[int],
    interval_ms: int = 80,
) -> tuple[FuncAnimation, plt.Figure, plt.Axes]:
    """Create an animation of precomputed forward-diffusion frames.

    ``frames`` has shape ``(num_frames, num_samples, 2)``. Axis limits and
    aspect ratio are fixed before the animation starts.
    """
    frames_np = np.asarray(frames)
    frame_steps = list(frame_steps)
    if frames_np.ndim != 3 or frames_np.shape[2] != 2:
        raise ValueError("frames must have shape (num_frames, num_samples, 2)")
    if len(frame_steps) != frames_np.shape[0]:
        raise ValueError("frame_steps must have one entry per frame")

    figure, axis = plt.subplots(figsize=(6.4, 6.4))
    scatter = axis.scatter(
        frames_np[0, :, 0],
        frames_np[0, :, 1],
        s=6,
        c=_component_colors(components),
        alpha=0.55,
        linewidths=0,
    )

    # Fixed limits: do not autoscale as the point cloud spreads out.
    axis.set_xlim(-AXIS_LIMIT, AXIS_LIMIT)
    axis.set_ylim(-AXIS_LIMIT, AXIS_LIMIT)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.grid(alpha=0.25)
    title = axis.set_title("Forward diffusion  t = 0")

    def update(frame_index: int) -> tuple:
        points = frames_np[frame_index]
        scatter.set_offsets(points)
        title.set_text(f"Forward diffusion  t = {frame_steps[frame_index]}")
        return (scatter, title)

    animation = FuncAnimation(
        figure,
        update,
        frames=frames_np.shape[0],
        interval=interval_ms,
        blit=False,
    )
    return animation, figure, axis


def render_keyframes(
    frames: np.ndarray,
    components: np.ndarray,
    frame_steps: list[int],
    output_path: str,
) -> None:
    """Save a grid of selected forward-diffusion frames."""
    frames_np = np.asarray(frames)
    frame_steps = list(frame_steps)
    num_panels = len(frame_steps)
    num_columns = 3
    num_rows = math.ceil(num_panels / num_columns)

    figure, axes = plt.subplots(
        num_rows, num_columns, figsize=(4.2 * num_columns, 4.0 * num_rows)
    )
    axes = np.atleast_1d(axes).ravel()
    colors = _component_colors(components)

    for panel_index, axis in enumerate(axes):
        if panel_index >= num_panels:
            axis.set_visible(False)
            continue
        points = frames_np[panel_index]
        axis.scatter(
            points[:, 0],
            points[:, 1],
            s=5,
            c=colors,
            alpha=0.5,
            linewidths=0,
        )
        axis.set_xlim(-AXIS_LIMIT, AXIS_LIMIT)
        axis.set_ylim(-AXIS_LIMIT, AXIS_LIMIT)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"t = {frame_steps[panel_index]}")
        axis.grid(alpha=0.25)
        axis.set_xlabel("x")
        axis.set_ylabel("y")

    figure.suptitle("Forward diffusion of a four-component Gaussian mixture")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
