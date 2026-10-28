"""Two-dimensional Gaussian mixture and the forward diffusion process.

Part 1 only: no neural network and no reverse sampling.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Four equally weighted components on a circle of radius 3.
CENTERS = jnp.array(
    [
        [3.0, 0.0],
        [0.0, 3.0],
        [-3.0, 0.0],
        [0.0, -3.0],
    ],
    dtype=jnp.float32,
)
NUM_COMPONENTS = 4
COMPONENT_COVARIANCE = 0.09
COMPONENT_STD = COMPONENT_COVARIANCE**0.5

# Linear schedule from beta_1 to beta_T.
NUM_DIFFUSION_STEPS = 1000
BETA_START = 0.0001
BETA_END = 0.02

# Used for the fixed axes of the animation and keyframe figure.
AXIS_LIMIT = 4.5


def linear_beta_schedule(
    num_steps: int = NUM_DIFFUSION_STEPS,
    beta_start: float = BETA_START,
    beta_end: float = BETA_END,
) -> jax.Array:
    """Return the linear beta schedule beta_1, ..., beta_T."""
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2")
    if not 0.0 < beta_start <= beta_end < 1.0:
        raise ValueError("beta values must satisfy 0 < beta_start <= beta_end < 1")
    return jnp.linspace(beta_start, beta_end, num_steps, dtype=jnp.float32)


def alpha_bars_from_betas(betas: jax.Array) -> jax.Array:
    """Return alpha_bar_0, ..., alpha_bar_T with alpha_bar_0 = 1."""
    betas = jnp.asarray(betas, dtype=jnp.float32)
    cumulative = jnp.cumprod(1.0 - betas)
    return jnp.concatenate([jnp.ones((1,), dtype=jnp.float32), cumulative])


def sample_gaussian_mixture(
    key: jax.Array,
    num_samples: int,
    centers: jax.Array = CENTERS,
    component_std: float = COMPONENT_STD,
) -> tuple[jax.Array, jax.Array]:
    """Sample from the equal-weight Gaussian mixture.

    Returns the samples with shape ``(num_samples, 2)`` and the integer
    component label of every sample.
    """
    key_component, key_noise = jax.random.split(key)
    logits = jnp.zeros((centers.shape[0],), dtype=jnp.float32)

    components = jax.random.categorical(
        key_component, logits, shape=(num_samples,)
    )
    noise = jax.random.normal(key_noise, (num_samples, 2), dtype=jnp.float32)
    samples = centers[components] + component_std * noise
    return samples, components


def q_sample(
    x0: jax.Array, alpha_bar: jax.Array | float, noise: jax.Array
) -> jax.Array:
    """Sample x_t from q(x_t | x_0) using reparameterization."""
    alpha_bar = jnp.asarray(alpha_bar, dtype=jnp.float32)
    return jnp.sqrt(alpha_bar) * x0 + jnp.sqrt(1.0 - alpha_bar) * noise


def forward_trajectory(
    x0: jax.Array,
    noise: jax.Array,
    alpha_bars: jax.Array,
    frame_steps: jax.Array,
) -> jax.Array:
    """Return q(x_t | x_0) samples for each requested integer time t.

    The same noise draw is used at every frame, so an individual point follows
    a continuous path from its initial location towards the noise draw.
    """
    alpha_bars = jnp.asarray(alpha_bars, dtype=jnp.float32)
    frame_steps = jnp.asarray(frame_steps)
    selected_alpha_bars = alpha_bars[frame_steps]

    signal_scale = jnp.sqrt(selected_alpha_bars)[:, None, None]
    noise_scale = jnp.sqrt(1.0 - selected_alpha_bars)[:, None, None]
    return signal_scale * x0[None, :, :] + noise_scale * noise[None, :, :]


def coordinate_stats(samples: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return the per-coordinate mean and variance of 2-D samples."""
    return jnp.mean(samples, axis=0), jnp.var(samples, axis=0)


def verify_terminal_distribution(
    samples: jax.Array,
    mean_tolerance: float = 0.1,
    variance_tolerance: float = 0.1,
) -> dict[str, object]:
    """Check that the terminal samples are close to a standard normal.

    Each coordinate mean must be within ``mean_tolerance`` of 0 and each
    coordinate variance within ``variance_tolerance`` of 1.
    """
    means, variances = coordinate_stats(samples)
    mean_errors = jnp.abs(means)
    variance_errors = jnp.abs(variances - 1.0)

    means_within_tolerance = bool(jnp.all(mean_errors <= mean_tolerance))
    variances_within_tolerance = bool(
        jnp.all(variance_errors <= variance_tolerance)
    )

    return {
        "num_samples": int(samples.shape[0]),
        "mean_x": float(means[0]),
        "mean_y": float(means[1]),
        "variance_x": float(variances[0]),
        "variance_y": float(variances[1]),
        "max_abs_mean_error": float(jnp.max(mean_errors)),
        "max_abs_variance_error": float(jnp.max(variance_errors)),
        "mean_tolerance": float(mean_tolerance),
        "variance_tolerance": float(variance_tolerance),
        "means_within_tolerance": means_within_tolerance,
        "variances_within_tolerance": variances_within_tolerance,
        "passed": means_within_tolerance and variances_within_tolerance,
    }
