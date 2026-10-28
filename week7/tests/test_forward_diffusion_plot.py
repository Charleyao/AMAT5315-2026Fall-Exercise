"""Tests for the fixed-limit forward-diffusion animation."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from forward_diffusion import AXIS_LIMIT
from forward_diffusion_plot import create_forward_animation


def test_animation_uses_fixed_axis_limits_and_aspect_ratio():
    frames = jnp.zeros((3, 8, 2), dtype=jnp.float32)
    components = jnp.array([0, 1, 2, 3, 0, 1, 2, 3])
    frame_steps = [0, 500, 1000]

    animation, figure, axis = create_forward_animation(
        frames, components, frame_steps, interval_ms=50
    )

    try:
        np.testing.assert_allclose(axis.get_xlim(), (-AXIS_LIMIT, AXIS_LIMIT))
        np.testing.assert_allclose(axis.get_ylim(), (-AXIS_LIMIT, AXIS_LIMIT))
        assert axis.get_aspect() == 1.0
        assert animation._func(0) is not None
        animation._init_draw()
    finally:
        plt.close(figure)
