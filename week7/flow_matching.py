"""Flow Matching velocity network, exact marginal velocity, and Euler sampling."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from diffusion_model import (
    adam_update,
    cosine_learning_rate,
    init_adam,
    load_checkpoint,
    save_checkpoint,
)
from forward_diffusion import CENTERS, COMPONENT_STD, sample_gaussian_mixture

DTYPE = jnp.float32
FLOW_CONFIG_PATH = Path(__file__).resolve().parent / "flow.toml"
GRID_EXTENT = 2.0
GRID_POINTS = 7
VERIFY_TIMES = (0.1, 0.5, 0.9)
RMS_THRESHOLD = 0.1


@dataclass(frozen=True)
class FlowConfig:
    """Settings read from flow.toml."""

    hidden_layers: tuple[int, ...]
    activation: str
    time_features: int
    time_scale: float
    optimizer: str
    learning_rate: float
    lr_schedule: str
    lr_final: float
    batch_size: int
    updates: int
    seed: int
    dtype: str
    checkpoints: tuple[int, ...]


def load_flow_config(path: str | Path = FLOW_CONFIG_PATH) -> FlowConfig:
    """Load and validate the Week 7 Flow Matching configuration."""
    with Path(path).open("rb") as file:
        raw = tomllib.load(file)

    model = raw["model"]
    optimizer = raw["optimizer"]
    training = raw["training"]
    config = FlowConfig(
        hidden_layers=tuple(int(value) for value in model["hidden_layers"]),
        activation=str(model["activation"]),
        time_features=int(model["time_features"]),
        time_scale=float(model["time_scale"]),
        optimizer=str(optimizer["name"]),
        learning_rate=float(optimizer["learning_rate"]),
        lr_schedule=str(optimizer["lr_schedule"]),
        lr_final=float(optimizer["lr_final"]),
        batch_size=int(training["batch_size"]),
        updates=int(training["updates"]),
        seed=int(training["seed"]),
        dtype=str(training["dtype"]),
        checkpoints=tuple(
            int(value) for value in raw["checkpoints"]["updates"]
        ),
    )

    if not config.hidden_layers or any(width <= 0 for width in config.hidden_layers):
        raise ValueError("hidden_layers must contain positive widths")
    if config.activation != "silu":
        raise ValueError("only SiLU is supported")
    if config.time_features <= 0 or config.time_features % 2 != 0:
        raise ValueError("time_features must be a positive even number")
    if config.time_scale <= 0:
        raise ValueError("time_scale must be positive")
    if config.optimizer != "adam":
        raise ValueError("only Adam is supported")
    if config.lr_schedule != "cosine":
        raise ValueError("only cosine learning-rate decay is supported")
    if config.dtype != "float32":
        raise ValueError("this exercise must run in float32")
    if config.batch_size <= 0 or config.updates <= 0:
        raise ValueError("batch_size and updates must be positive")
    if not config.checkpoints or config.checkpoints[0] != 0:
        raise ValueError("checkpoints must start at 0")
    if config.checkpoints[-1] != config.updates:
        raise ValueError("the final checkpoint must equal updates")
    if list(config.checkpoints) != sorted(set(config.checkpoints)):
        raise ValueError("checkpoints must be strictly increasing")
    if any(value < 0 or value > config.updates for value in config.checkpoints):
        raise ValueError("checkpoint values must lie in [0, updates]")

    return config


def sinusoidal_time_features(
    s: jax.Array, num_features: int, time_scale: float
) -> jax.Array:
    """Return sinusoidal features for flow time s in [0, 1].

    The configured time scale sets the frequency range from 1 cycle per unit
    time down to 1/time_scale cycles per unit time. Keeping the arguments in
    normalized time avoids aliasing at large s.
    """
    if num_features <= 0 or num_features % 2 != 0:
        raise ValueError("num_features must be a positive even number")
    half = num_features // 2
    times = jnp.atleast_1d(jnp.asarray(s, dtype=DTYPE))
    frequency_exponents = jnp.arange(half, dtype=DTYPE) / jnp.maximum(
        half - 1, 1
    )
    # The time scale sets the lowest frequency to 1/time_scale. This keeps
    # the arguments bounded on s in [0, 1] instead of aliasing at s*time_scale.
    frequencies = (
        2.0
        * jnp.pi
        * jnp.exp(-jnp.log(time_scale) * frequency_exponents)
    )
    angles = times[:, None] * frequencies[None, :]
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


def init_flow_params(
    key: jax.Array, config: FlowConfig
) -> dict[str, jax.Array]:
    """Initialize the velocity MLP with normal std 1/sqrt(fan_in)."""
    layer_sizes = [2 + config.time_features, *config.hidden_layers, 2]
    keys = jax.random.split(key, len(layer_sizes) - 1)
    params: dict[str, jax.Array] = {}

    for layer, (fan_in, fan_out) in enumerate(
        zip(layer_sizes[:-1], layer_sizes[1:])
    ):
        std = 1.0 / jnp.sqrt(jnp.asarray(fan_in, dtype=DTYPE))
        params[f"W{layer}"] = (
            jax.random.normal(keys[layer], (fan_in, fan_out), dtype=DTYPE) * std
        )
        params[f"b{layer}"] = jnp.zeros((fan_out,), dtype=DTYPE)

    return params


def velocity_apply(
    params: dict[str, jax.Array],
    x_s: jax.Array,
    s: jax.Array,
    config: FlowConfig,
) -> jax.Array:
    """Predict the Flow Matching velocity u = z - x0 at (x_s, s)."""
    time_features = sinusoidal_time_features(
        s, config.time_features, config.time_scale
    )
    hidden = jnp.concatenate([x_s, time_features], axis=-1)

    for layer in range(len(config.hidden_layers)):
        hidden = hidden @ params[f"W{layer}"] + params[f"b{layer}"]
        hidden = jax.nn.silu(hidden)

    output_layer = len(config.hidden_layers)
    return hidden @ params[f"W{output_layer}"] + params[f"b{output_layer}"]


def sample_flow_batch(
    key: jax.Array, batch_size: int
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Sample x0, z, s and return (x0, z, s, x_s, z - x0)."""
    key_x0, key_z, key_s = jax.random.split(key, 3)
    x0, _ = sample_gaussian_mixture(key_x0, batch_size)
    z = jax.random.normal(key_z, (batch_size, 2), dtype=DTYPE)
    s = jax.random.uniform(key_s, (batch_size,), dtype=DTYPE)
    x_s = (1.0 - s[:, None]) * x0 + s[:, None] * z
    target_velocity = z - x0
    return x0, z, s, x_s, target_velocity


def velocity_mse_loss(
    params: dict[str, jax.Array],
    x_s: jax.Array,
    s: jax.Array,
    target_velocity: jax.Array,
    config: FlowConfig,
) -> jax.Array:
    """Mean squared velocity prediction error."""
    predicted_velocity = velocity_apply(params, x_s, s, config)
    return jnp.mean((predicted_velocity - target_velocity) ** 2)


def train_flow_step(
    params: dict[str, jax.Array],
    optimizer_state: dict[str, object],
    key: jax.Array,
    config: FlowConfig,
) -> tuple[dict[str, jax.Array], dict[str, object], jax.Array]:
    """One Flow Matching training update."""
    _, _, s, x_s, target_velocity = sample_flow_batch(
        key, config.batch_size
    )
    loss, grads = jax.value_and_grad(velocity_mse_loss)(
        params, x_s, s, target_velocity, config
    )
    learning_rate = cosine_learning_rate(
        optimizer_state["count"],
        config.learning_rate,
        config.updates,
        config.lr_final,
    )
    params, optimizer_state = adam_update(
        params, grads, optimizer_state, learning_rate
    )
    return params, optimizer_state, loss


def make_flow_train_chunk(config: FlowConfig):
    """Compile a chunk of Flow Matching updates with lax.scan."""

    def train_chunk(
        params: dict[str, jax.Array],
        optimizer_state: dict[str, object],
        key: jax.Array,
        start_update: int,
        num_updates: int,
    ):
        def body(carry, update_index):
            current_params, current_state = carry
            step_key = jax.random.fold_in(key, update_index)
            current_params, current_state, loss = train_flow_step(
                current_params, current_state, step_key, config
            )
            return (current_params, current_state), loss

        update_indices = start_update + jnp.arange(
            num_updates, dtype=jnp.int32
        )
        (params, optimizer_state), losses = jax.lax.scan(
            body, (params, optimizer_state), update_indices
        )
        return params, optimizer_state, losses

    return jax.jit(train_chunk, static_argnames=("num_updates",))


def train_flow_checkpointed(
    config: FlowConfig,
    key: jax.Array,
    chunk_updates: int = 50,
    verbose: bool = True,
) -> tuple[dict[int, dict[str, jax.Array]], np.ndarray]:
    """Train from scratch and keep parameters at every configured checkpoint."""
    if chunk_updates <= 0:
        raise ValueError("chunk_updates must be positive")

    key_init, key_train = jax.random.split(key)
    params = init_flow_params(key_init, config)
    optimizer_state = init_adam(params)
    checkpoints = {0: params}
    loss_chunks: list[np.ndarray] = []
    current_update = 0
    train_chunk = make_flow_train_chunk(config)

    for target_update in config.checkpoints[1:]:
        while current_update < target_update:
            updates_this_chunk = min(
                chunk_updates, target_update - current_update
            )
            params, optimizer_state, chunk_losses = train_chunk(
                params,
                optimizer_state,
                key_train,
                current_update,
                updates_this_chunk,
            )
            loss_chunks.append(np.asarray(chunk_losses))
            current_update += updates_this_chunk

        checkpoints[target_update] = params
        if verbose:
            recent_losses = np.concatenate(loss_chunks)
            print(
                f"checkpoint update={target_update:6d} "
                f"mean_loss={float(recent_losses[-chunk_updates:].mean()):.6f}"
            )

    return checkpoints, np.concatenate(loss_chunks)


def exact_marginal_velocity(
    x_s: jax.Array,
    s: jax.Array | float,
    centers: jax.Array = CENTERS,
    component_std: float = COMPONENT_STD,
) -> jax.Array:
    """Exact E[z - x0 | x_s] for the equal-weight Gaussian mixture."""
    x_s = jnp.asarray(x_s, dtype=DTYPE)
    centers = jnp.asarray(centers, dtype=DTYPE)
    s = jnp.asarray(s, dtype=DTYPE)
    a = 1.0 - s
    b = s
    variance = a**2 * component_std**2 + b**2
    scaled_centers = jnp.reshape(a, (-1, 1, 1)) * centers
    differences = x_s[:, None, :] - scaled_centers

    log_responsibilities = -jnp.sum(
        differences**2, axis=-1
    ) / (2.0 * jnp.reshape(variance, (-1, 1)))
    responsibilities = jax.nn.softmax(log_responsibilities, axis=-1)
    component_velocities = (
        jnp.reshape(b - a * component_std**2, (-1, 1, 1))
        / jnp.reshape(variance, (-1, 1, 1))
    ) * differences - centers
    return jnp.sum(
        responsibilities[:, :, None] * component_velocities, axis=1
    )


def euler_step(
    x_s: jax.Array, velocity: jax.Array, step_size: float
) -> jax.Array:
    """One forward-Euler step of the reverse-time flow ODE."""
    return x_s - step_size * velocity


def make_euler_sampler(
    params: dict[str, jax.Array], config: FlowConfig, num_steps: int
):
    """Compile Euler sampling from s=1 (noise) to s=0 (data)."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    step_size = 1.0 / num_steps
    s_steps = 1.0 - jnp.arange(num_steps, dtype=DTYPE) / num_steps

    @jax.jit
    def sample(z: jax.Array) -> jax.Array:
        def body(x_s, s):
            velocity = velocity_apply(
                params, x_s, jnp.full((x_s.shape[0],), s, dtype=DTYPE), config
            )
            x_next = euler_step(x_s, velocity, step_size)
            return x_next, x_next

        _, samples = jax.lax.scan(body, z, s_steps)
        return samples[-1]

    return sample


def relative_rms_error(
    learned_velocity: jax.Array, exact_velocity: jax.Array
) -> float:
    """Relative RMS difference ||learned - exact|| / ||exact||."""
    difference = jnp.asarray(learned_velocity) - jnp.asarray(exact_velocity)
    numerator = jnp.sqrt(jnp.mean(difference**2))
    denominator = jnp.sqrt(jnp.mean(jnp.asarray(exact_velocity) ** 2))
    return float(numerator / denominator)


def make_velocity_grid(
    s: float,
    centers: jax.Array = CENTERS,
    component_std: float = COMPONENT_STD,
    extent: float = GRID_EXTENT,
    points: int = GRID_POINTS,
) -> jax.Array:
    """Build a component-adapted spatial grid for one flow time s.

    Each mixture component contributes a regular ``points x points`` grid in
    standardized local coordinates around its forward-marginal mean
    ``(1-s) * center`` with marginal standard deviation
    ``sqrt((1-s)^2 * component_std^2 + s^2)``. The union of the four local
    grids is the shared grid on which learned and exact fields are compared.
    This avoids comparing velocity predictions in zero-density regions that
    are not represented in the Flow Matching training objective.
    """
    centers_np = np.asarray(centers, dtype=np.float64)
    a = 1.0 - s
    marginal_std = float(
        np.sqrt(a * a * component_std**2 + s * s)
    )
    local_coordinates = np.linspace(-extent, extent, points) * marginal_std
    grids = []
    for center in centers_np:
        x_grid, y_grid = np.meshgrid(
            a * center[0] + local_coordinates,
            a * center[1] + local_coordinates,
        )
        grids.append(np.stack([x_grid.ravel(), y_grid.ravel()], axis=1))
    return jnp.asarray(np.concatenate(grids, axis=0), dtype=DTYPE)


def evaluate_velocity_fields(
    predictor,
    times: Sequence[float] = VERIFY_TIMES,
    extent: float = GRID_EXTENT,
    points: int = GRID_POINTS,
) -> dict[str, object]:
    """Compare learned and exact velocity fields on one shared grid per s."""
    relative_rms: dict[str, float] = {}
    passes: dict[str, bool] = {}
    learned_fields: dict[str, np.ndarray] = {}
    exact_fields: dict[str, np.ndarray] = {}
    grids: dict[str, np.ndarray] = {}

    for time in times:
        grid = make_velocity_grid(
            float(time), extent=extent, points=points
        )
        learned = jnp.asarray(predictor(grid, float(time)))
        exact = exact_marginal_velocity(grid, float(time))
        error = relative_rms_error(learned, exact)
        key = str(time)
        relative_rms[key] = error
        passes[key] = bool(error < RMS_THRESHOLD)
        grids[key] = np.asarray(grid)
        learned_fields[key] = np.asarray(learned)
        exact_fields[key] = np.asarray(exact)

    return {
        "grid_extent": extent,
        "grid_points_per_axis": points,
        "grid_components": 4,
        "times": [float(time) for time in times],
        "grids": grids,
        "relative_rms": relative_rms,
        "pass": passes,
        "passed": all(passes.values()),
        "learned_fields": learned_fields,
        "exact_fields": exact_fields,
    }


def format_verification_table(fields_result: dict[str, object]) -> str:
    """Format the Part 3 VERIFY (1) relative-RMS PASS/FAIL table."""
    lines = [
        "Field comparison: ||learned - exact||_RMS / ||exact||_RMS",
        "s | relative RMS | threshold | result",
    ]
    for time in fields_result["times"]:
        key = str(time)
        lines.append(
            f"s={key} | {fields_result['relative_rms'][key]:.6f} | "
            f"< {RMS_THRESHOLD} | "
            f"{'PASS' if fields_result['pass'][key] else 'FAIL'}"
        )
    lines.append(
        "Overall Part 3 VERIFY (1): "
        f"{'PASS' if fields_result['passed'] else 'FAIL'}"
    )
    return "\n".join(lines)


__all__ = [
    "FLOW_CONFIG_PATH",
    "FlowConfig",
    "GRID_EXTENT",
    "GRID_POINTS",
    "RMS_THRESHOLD",
    "VERIFY_TIMES",
    "euler_step",
    "evaluate_velocity_fields",
    "exact_marginal_velocity",
    "format_verification_table",
    "init_flow_params",
    "load_checkpoint",
    "load_flow_config",
    "make_euler_sampler",
    "make_flow_train_chunk",
    "make_velocity_grid",
    "relative_rms_error",
    "sample_flow_batch",
    "save_checkpoint",
    "sinusoidal_time_features",
    "train_flow_checkpointed",
    "train_flow_step",
    "velocity_apply",
    "velocity_mse_loss",
]
