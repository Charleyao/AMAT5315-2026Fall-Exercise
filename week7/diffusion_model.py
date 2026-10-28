"""Time-conditioned noise predictor and DDPM reverse sampling (Part 1)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from forward_diffusion import (
    NUM_DIFFUSION_STEPS,
    alpha_bars_from_betas,
    sample_gaussian_mixture,
)

CONFIG_PATH = Path(__file__).resolve().parent / "diffusion.toml"
DTYPE = jnp.float32


@dataclass(frozen=True)
class DiffusionConfig:
    """Settings read from diffusion.toml."""

    hidden_layers: tuple[int, ...]
    activation: str
    time_features: int
    time_offset: float
    weight_init: str
    weight_init_std: str
    optimizer: str
    learning_rate: float
    lr_schedule: str
    lr_final: float
    batch_size: int
    updates: int
    seed: int
    dtype: str
    checkpoints: tuple[int, ...]


def load_config(path: str | Path = CONFIG_PATH) -> DiffusionConfig:
    """Load and validate the Week 7 diffusion configuration."""
    with Path(path).open("rb") as file:
        raw = tomllib.load(file)

    model = raw["model"]
    optimizer = raw["optimizer"]
    training = raw["training"]
    checkpoints = tuple(int(value) for value in raw["checkpoints"]["updates"])
    config = DiffusionConfig(
        hidden_layers=tuple(int(value) for value in model["hidden_layers"]),
        activation=str(model["activation"]),
        time_features=int(model["time_features"]),
        time_offset=float(model["time_offset"]),
        weight_init=str(model["weight_init"]),
        weight_init_std=str(model["weight_init_std"]),
        optimizer=str(optimizer["name"]),
        learning_rate=float(optimizer["learning_rate"]),
        lr_schedule=str(optimizer["lr_schedule"]),
        lr_final=float(optimizer["lr_final"]),
        batch_size=int(training["batch_size"]),
        updates=int(training["updates"]),
        seed=int(training["seed"]),
        dtype=str(training["dtype"]),
        checkpoints=checkpoints,
    )

    if not config.hidden_layers or any(width <= 0 for width in config.hidden_layers):
        raise ValueError("hidden_layers must contain positive widths")
    if config.activation != "silu":
        raise ValueError("only the specified SiLU activation is supported")
    if config.time_features <= 0 or config.time_features % 2 != 0:
        raise ValueError("time_features must be a positive even number")
    if config.weight_init != "normal":
        raise ValueError("only normal weight initialization is supported")
    if config.weight_init_std != "1/sqrt(input_width)":
        raise ValueError("weight_init_std must be 1/sqrt(input_width)")
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
    t: jax.Array, num_features: int, offset: float
) -> jax.Array:
    """Return 32 fixed-frequency sinusoidal features for diffusion time.

    Time is normalized by T, with the configured offset applied first. Thus
    ``t=1`` and ``offset=-1`` map to normalized time zero.
    """
    if num_features <= 0 or num_features % 2 != 0:
        raise ValueError("num_features must be a positive even number")
    half = num_features // 2
    times = jnp.atleast_1d(jnp.asarray(t, dtype=DTYPE))
    normalized_times = (times + offset) / NUM_DIFFUSION_STEPS
    frequencies = jnp.exp(
        -jnp.log(10_000.0)
        * jnp.arange(half, dtype=DTYPE)
        / jnp.maximum(half - 1, 1)
    )
    angles = normalized_times[:, None] * frequencies[None, :]
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


def init_model_params(
    key: jax.Array, config: DiffusionConfig
) -> dict[str, jax.Array]:
    """Initialize the MLP with normal weights of std 1/sqrt(fan_in)."""
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


def model_apply(
    params: dict[str, jax.Array],
    x_t: jax.Array,
    t: jax.Array,
    config: DiffusionConfig,
) -> jax.Array:
    """Predict the noise added to ``x_t`` at diffusion time ``t``."""
    time_features = sinusoidal_time_features(
        t, config.time_features, config.time_offset
    )
    hidden = jnp.concatenate([x_t, time_features], axis=-1)

    for layer in range(len(config.hidden_layers)):
        hidden = hidden @ params[f"W{layer}"] + params[f"b{layer}"]
        hidden = jax.nn.silu(hidden)

    output_layer = len(config.hidden_layers)
    return hidden @ params[f"W{output_layer}"] + params[f"b{output_layer}"]


def mse_loss(
    params: dict[str, jax.Array],
    x_t: jax.Array,
    t: jax.Array,
    noise: jax.Array,
    config: DiffusionConfig,
) -> jax.Array:
    """MSE between the predicted and the true forward-diffusion noise."""
    predicted_noise = model_apply(params, x_t, t, config)
    return jnp.mean((predicted_noise - noise) ** 2)


def init_adam(params: dict[str, jax.Array]) -> dict[str, object]:
    """Create first/second moments and an update counter for Adam."""
    return {
        "m": jax.tree.map(jnp.zeros_like, params),
        "v": jax.tree.map(jnp.zeros_like, params),
        "count": jnp.asarray(0, dtype=jnp.int32),
    }


def adam_update(
    params: dict[str, jax.Array],
    grads: dict[str, jax.Array],
    state: dict[str, object],
    learning_rate: jax.Array,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> tuple[dict[str, jax.Array], dict[str, object]]:
    """Apply one Adam update with bias correction."""
    count = state["count"] + jnp.asarray(1, dtype=jnp.int32)
    first_moment = jax.tree.map(
        lambda m, g: beta1 * m + (1.0 - beta1) * g, state["m"], grads
    )
    second_moment = jax.tree.map(
        lambda v, g: beta2 * v + (1.0 - beta2) * g * g, state["v"], grads
    )
    first_correction = 1.0 - jnp.power(beta1, count)
    second_correction = 1.0 - jnp.power(beta2, count)

    updated_params = jax.tree.map(
        lambda p, m, v: p
        - learning_rate
        * (m / first_correction)
        / (jnp.sqrt(v / second_correction) + epsilon),
        params,
        first_moment,
        second_moment,
    )
    return updated_params, {
        "m": first_moment,
        "v": second_moment,
        "count": count,
    }


def cosine_learning_rate(
    update: int | jax.Array,
    base_learning_rate: float,
    total_updates: int,
    final_learning_rate: float = 0.0,
) -> jax.Array:
    """Cosine decay from the base rate to the final rate."""
    if total_updates <= 0:
        raise ValueError("total_updates must be positive")
    progress = jnp.clip(
        jnp.asarray(update, dtype=DTYPE) / total_updates, 0.0, 1.0
    )
    cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
    return final_learning_rate + (
        base_learning_rate - final_learning_rate
    ) * cosine


def sample_training_batch(
    key: jax.Array,
    batch_size: int,
    alpha_bars: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Draw x0, a uniform diffusion level, noise, and return x_t."""
    key_x0, key_time, key_noise = jax.random.split(key, 3)
    x0, _ = sample_gaussian_mixture(key_x0, batch_size)
    t = jax.random.randint(
        key_time,
        (batch_size,),
        1,
        NUM_DIFFUSION_STEPS + 1,
        dtype=jnp.int32,
    )
    noise = jax.random.normal(key_noise, x0.shape, dtype=DTYPE)
    alpha_bar_t = alpha_bars[t]
    x_t = (
        jnp.sqrt(alpha_bar_t)[:, None] * x0
        + jnp.sqrt(1.0 - alpha_bar_t)[:, None] * noise
    )
    return x_t, t, noise


def train_step(
    params: dict[str, jax.Array],
    optimizer_state: dict[str, object],
    key: jax.Array,
    config: DiffusionConfig,
    alpha_bars: jax.Array,
) -> tuple[dict[str, jax.Array], dict[str, object], jax.Array]:
    """Perform one MSE training update on a fresh training batch."""
    x_t, t, noise = sample_training_batch(
        key, config.batch_size, alpha_bars
    )
    loss, grads = jax.value_and_grad(mse_loss)(
        params, x_t, t, noise, config
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


def make_train_chunk(
    config: DiffusionConfig, alpha_bars: jax.Array
):
    """Compile a chunk of consecutive training updates with lax.scan."""

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
            current_params, current_state, loss = train_step(
                current_params,
                current_state,
                step_key,
                config,
                alpha_bars,
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


def train_checkpointed(
    config: DiffusionConfig,
    alpha_bars: jax.Array,
    key: jax.Array,
    chunk_updates: int = 50,
    verbose: bool = True,
) -> tuple[dict[int, dict[str, jax.Array]], np.ndarray]:
    """Train to each checkpoint and keep the parameters at every checkpoint."""
    if chunk_updates <= 0:
        raise ValueError("chunk_updates must be positive")

    key_init, key_train = jax.random.split(key)
    params = init_model_params(key_init, config)
    optimizer_state = init_adam(params)
    checkpoints = {0: params}
    loss_chunks: list[np.ndarray] = []
    current_update = 0
    train_chunk = make_train_chunk(config, alpha_bars)

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


def ddpm_reverse_step(
    x_t: jax.Array,
    t: int | jax.Array,
    predicted_noise: jax.Array,
    betas: jax.Array,
    alpha_bars: jax.Array,
    noise: jax.Array,
) -> jax.Array:
    """One DDPM reverse step using variance beta_t and no clipping.

    The additive noise is omitted at the final step ``t = 1``.
    """
    beta_t = betas[t - 1]
    alpha_t = 1.0 - beta_t
    alpha_bar_t = alpha_bars[t]
    mean = (
        x_t - beta_t / jnp.sqrt(1.0 - alpha_bar_t) * predicted_noise
    ) / jnp.sqrt(alpha_t)
    noise_scale = jnp.sqrt(beta_t) * jnp.where(t > 1, 1.0, 0.0)
    return mean + noise_scale * noise


def make_reverse_sampler(
    config: DiffusionConfig,
    betas: jax.Array,
    alpha_bars: jax.Array,
):
    """Compile the DDPM reverse process from t=T down to t=0."""

    def reverse(
        params: dict[str, jax.Array],
        x_T: jax.Array,
        reverse_noise: jax.Array,
    ) -> jax.Array:
        reverse_times = jnp.arange(
            NUM_DIFFUSION_STEPS, 0, -1, dtype=jnp.int32
        )

        def body(x_t, t):
            t_batch = jnp.full((x_t.shape[0],), t, dtype=jnp.int32)
            predicted_noise = model_apply(params, x_t, t_batch, config)
            x_previous = ddpm_reverse_step(
                x_t,
                t,
                predicted_noise,
                betas,
                alpha_bars,
                reverse_noise[t - 1],
            )
            return x_previous, x_previous

        _, sampled_steps = jax.lax.scan(body, x_T, reverse_times)
        return jnp.concatenate([x_T[None, :, :], sampled_steps], axis=0)

    return jax.jit(reverse)


def save_checkpoint(
    params: dict[str, jax.Array], path: str | Path
) -> None:
    """Save one parameter checkpoint as an NPZ file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{name: np.asarray(value) for name, value in params.items()})


def load_checkpoint(path: str | Path) -> dict[str, jax.Array]:
    """Load a parameter checkpoint saved by save_checkpoint."""
    with np.load(path) as data:
        return {name: jnp.asarray(data[name]) for name in data.files}


__all__ = [
    "CONFIG_PATH",
    "DiffusionConfig",
    "adam_update",
    "cosine_learning_rate",
    "ddpm_reverse_step",
    "init_adam",
    "init_model_params",
    "load_checkpoint",
    "load_config",
    "make_reverse_sampler",
    "make_train_chunk",
    "model_apply",
    "mse_loss",
    "sample_training_batch",
    "save_checkpoint",
    "sinusoidal_time_features",
    "train_checkpointed",
    "train_step",
    "alpha_bars_from_betas",
]
