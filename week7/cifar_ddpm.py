"""JAX implementation of diffusers v0.40.0 UNet2DModel for CIFAR-10 DDPM.

Weights are loaded from the pinned ``google/ddpm-cifar10-32`` safetensors
checkpoint. The network is inference-only and follows the HF diffusers module
layout and operation order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file

from forward_diffusion import alpha_bars_from_betas, linear_beta_schedule

DTYPE = jnp.float32
NUM_DIFFUSION_STEPS = 1000
MODEL_DIR = Path(__file__).resolve().parent / "inputs" / "cifar-model"
CONFIG_PATH = MODEL_DIR / "config.json"
WEIGHTS_PATH = MODEL_DIR / "diffusion_pytorch_model.safetensors"


@dataclass(frozen=True)
class UNet2DConfig:
    sample_size: int
    in_channels: int
    out_channels: int
    block_out_channels: tuple[int, ...]
    down_block_types: tuple[str, ...]
    up_block_types: tuple[str, ...]
    layers_per_block: int
    norm_num_groups: int
    norm_eps: float
    attention_head_dim: int | None
    center_input_sample: bool
    flip_sin_to_cos: bool
    freq_shift: float
    time_embedding_type: str
    act_fn: str
    downsample_padding: int


def load_unet_config(path: str | Path = CONFIG_PATH) -> UNet2DConfig:
    """Load the pinned UNet2DModel configuration."""
    with Path(path).open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if raw.get("_class_name") != "UNet2DModel":
        raise ValueError("checkpoint is not a UNet2DModel")
    config = UNet2DConfig(
        sample_size=int(raw["sample_size"]),
        in_channels=int(raw["in_channels"]),
        out_channels=int(raw["out_channels"]),
        block_out_channels=tuple(int(v) for v in raw["block_out_channels"]),
        down_block_types=tuple(raw["down_block_types"]),
        up_block_types=tuple(raw["up_block_types"]),
        layers_per_block=int(raw["layers_per_block"]),
        norm_num_groups=int(raw["norm_num_groups"]),
        norm_eps=float(raw["norm_eps"]),
        attention_head_dim=(
            None
            if raw.get("attention_head_dim") is None
            else int(raw["attention_head_dim"])
        ),
        center_input_sample=bool(raw["center_input_sample"]),
        flip_sin_to_cos=bool(raw["flip_sin_to_cos"]),
        freq_shift=float(raw["freq_shift"]),
        time_embedding_type=str(raw["time_embedding_type"]),
        act_fn=str(raw["act_fn"]),
        downsample_padding=int(raw["downsample_padding"]),
    )
    if config.sample_size != 32:
        raise ValueError("this exercise requires the 32x32 CIFAR-10 model")
    if config.act_fn != "silu" or config.time_embedding_type != "positional":
        raise ValueError("unsupported activation or time embedding")
    return config


def load_unet_params(path: str | Path = WEIGHTS_PATH) -> dict[str, jax.Array]:
    """Load all safetensors tensors as float32 JAX arrays."""
    tensors = load_file(str(path))
    return {
        name: jnp.asarray(tensor, dtype=DTYPE)
        for name, tensor in tensors.items()
    }


def _add_conv(
    shapes: dict[str, tuple[int, ...]],
    prefix: str,
    in_channels: int,
    out_channels: int,
    kernel: int,
    bias: bool = True,
) -> None:
    shapes[f"{prefix}.weight"] = (out_channels, in_channels, kernel, kernel)
    if bias:
        shapes[f"{prefix}.bias"] = (out_channels,)


def _add_linear(
    shapes: dict[str, tuple[int, ...]],
    prefix: str,
    in_features: int,
    out_features: int,
) -> None:
    shapes[f"{prefix}.weight"] = (out_features, in_features)
    shapes[f"{prefix}.bias"] = (out_features,)


def _add_group_norm(
    shapes: dict[str, tuple[int, ...]], prefix: str, channels: int
) -> None:
    shapes[f"{prefix}.weight"] = (channels,)
    shapes[f"{prefix}.bias"] = (channels,)


def _add_resnet(
    shapes: dict[str, tuple[int, ...]],
    prefix: str,
    in_channels: int,
    out_channels: int,
    time_embed_dim: int,
) -> None:
    _add_group_norm(shapes, f"{prefix}.norm1", in_channels)
    _add_conv(shapes, f"{prefix}.conv1", in_channels, out_channels, 3)
    _add_linear(shapes, f"{prefix}.time_emb_proj", time_embed_dim, out_channels)
    _add_group_norm(shapes, f"{prefix}.norm2", out_channels)
    _add_conv(shapes, f"{prefix}.conv2", out_channels, out_channels, 3)
    if in_channels != out_channels:
        _add_conv(
            shapes,
            f"{prefix}.conv_shortcut",
            in_channels,
            out_channels,
            1,
        )


def _add_attention(
    shapes: dict[str, tuple[int, ...]], prefix: str, channels: int
) -> None:
    _add_group_norm(shapes, f"{prefix}.group_norm", channels)
    for name in ("query", "key", "value", "proj_attn"):
        _add_linear(shapes, f"{prefix}.{name}", channels, channels)


def unet_param_shapes(config: UNet2DConfig) -> dict[str, tuple[int, ...]]:
    """Return the expected safetensors keys and OIHW shapes."""
    shapes: dict[str, tuple[int, ...]] = {}
    time_embed_dim = config.block_out_channels[0] * 4
    _add_conv(shapes, "conv_in", config.in_channels, config.block_out_channels[0], 3)
    _add_linear(shapes, "time_embedding.linear_1", config.block_out_channels[0], time_embed_dim)
    _add_linear(shapes, "time_embedding.linear_2", time_embed_dim, time_embed_dim)
    _add_group_norm(shapes, "conv_norm_out", config.block_out_channels[0])
    _add_conv(shapes, "conv_out", config.block_out_channels[0], config.out_channels, 3)

    in_channels = config.block_out_channels[0]
    for block, block_type in enumerate(config.down_block_types):
        out_channels = config.block_out_channels[block]
        for layer in range(config.layers_per_block):
            layer_in = in_channels if layer == 0 else out_channels
            _add_resnet(
                shapes,
                f"down_blocks.{block}.resnets.{layer}",
                layer_in,
                out_channels,
                time_embed_dim,
            )
            if "Attn" in block_type:
                _add_attention(
                    shapes,
                    f"down_blocks.{block}.attentions.{layer}",
                    out_channels,
                )
        if block != len(config.down_block_types) - 1:
            _add_conv(
                shapes,
                f"down_blocks.{block}.downsamplers.0.conv",
                out_channels,
                out_channels,
                3,
            )
        in_channels = out_channels

    mid_channels = config.block_out_channels[-1]
    _add_resnet(shapes, "mid_block.resnets.0", mid_channels, mid_channels, time_embed_dim)
    _add_attention(shapes, "mid_block.attentions.0", mid_channels)
    _add_resnet(shapes, "mid_block.resnets.1", mid_channels, mid_channels, time_embed_dim)

    reversed_channels = list(reversed(config.block_out_channels))
    prev_output = reversed_channels[0]
    for block, block_type in enumerate(config.up_block_types):
        out_channels = reversed_channels[block]
        skip_channels = reversed_channels[min(block + 1, len(reversed_channels) - 1)]
        num_layers = config.layers_per_block + 1
        for layer in range(num_layers):
            res_skip = skip_channels if layer == num_layers - 1 else out_channels
            resnet_in = prev_output if layer == 0 else out_channels
            _add_resnet(
                shapes,
                f"up_blocks.{block}.resnets.{layer}",
                resnet_in + res_skip,
                out_channels,
                time_embed_dim,
            )
            if "Attn" in block_type:
                _add_attention(
                    shapes,
                    f"up_blocks.{block}.attentions.{layer}",
                    out_channels,
                )
        if block != len(config.up_block_types) - 1:
            _add_conv(
                shapes,
                f"up_blocks.{block}.upsamplers.0.conv",
                out_channels,
                out_channels,
                3,
            )
        prev_output = out_channels

    return shapes


def assert_parameter_shapes(
    params: dict[str, jax.Array], config: UNet2DConfig
) -> None:
    """Raise if the checkpoint does not match the expected JAX architecture."""
    expected = unet_param_shapes(config)
    actual = {name: tuple(value.shape) for name, value in params.items()}
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    wrong = {
        name: (actual[name], expected[name])
        for name in sorted(set(expected) & set(actual))
        if actual[name] != expected[name]
    }
    if missing or unexpected or wrong:
        raise ValueError(
            f"checkpoint mismatch: missing={missing[:10]}, "
            f"unexpected={unexpected[:10]}, wrong={list(wrong.items())[:10]}"
        )


def _conv2d(
    x: jax.Array,
    weight: jax.Array,
    bias: jax.Array,
    stride: int = 1,
    padding: int = 1,
) -> jax.Array:
    weight_hwio = jnp.transpose(weight, (2, 3, 1, 0))
    y = jax.lax.conv_general_dilated(
        x,
        weight_hwio,
        window_strides=(stride, stride),
        padding=((padding, padding), (padding, padding)),
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return y + bias


def _linear(x: jax.Array, weight: jax.Array, bias: jax.Array) -> jax.Array:
    return x @ jnp.transpose(weight) + bias


def _group_norm(
    x: jax.Array,
    weight: jax.Array,
    bias: jax.Array,
    groups: int,
    eps: float,
) -> jax.Array:
    batch, height, width, channels = x.shape
    grouped = x.reshape(batch, height, width, groups, channels // groups)
    mean = jnp.mean(grouped, axis=(1, 2, 4), keepdims=True)
    variance = jnp.mean((grouped - mean) ** 2, axis=(1, 2, 4), keepdims=True)
    normalized = (grouped - mean) / jnp.sqrt(variance + eps)
    return normalized.reshape(batch, height, width, channels) * weight + bias


def _norm(
    params: dict[str, jax.Array], prefix: str, x: jax.Array, config: UNet2DConfig
) -> jax.Array:
    return _group_norm(
        x,
        params[f"{prefix}.weight"],
        params[f"{prefix}.bias"],
        config.norm_num_groups,
        config.norm_eps,
    )


def _timestep_embedding(
    timesteps: jax.Array,
    embedding_dim: int,
    flip_sin_to_cos: bool,
    freq_shift: float,
) -> jax.Array:
    half_dim = embedding_dim // 2
    exponent = -jnp.log(10000.0) * jnp.arange(half_dim, dtype=DTYPE)
    exponent = exponent / (half_dim - freq_shift)
    frequencies = jnp.exp(exponent)
    arguments = timesteps[:, None].astype(DTYPE) * frequencies[None, :]
    embedding = jnp.concatenate(
        [jnp.sin(arguments), jnp.cos(arguments)], axis=-1
    )
    if flip_sin_to_cos:
        embedding = jnp.concatenate(
            [embedding[:, half_dim:], embedding[:, :half_dim]], axis=-1
        )
    return embedding


def _resnet(
    params: dict[str, jax.Array],
    x: jax.Array,
    temb: jax.Array,
    prefix: str,
    config: UNet2DConfig,
) -> jax.Array:
    hidden = _norm(params, f"{prefix}.norm1", x, config)
    hidden = jax.nn.silu(hidden)
    hidden = _conv2d(
        hidden,
        params[f"{prefix}.conv1.weight"],
        params[f"{prefix}.conv1.bias"],
    )
    time_emb = jax.nn.silu(temb)
    time_emb = _linear(
        time_emb,
        params[f"{prefix}.time_emb_proj.weight"],
        params[f"{prefix}.time_emb_proj.bias"],
    )
    hidden = hidden + time_emb[:, None, None, :]
    hidden = _norm(params, f"{prefix}.norm2", hidden, config)
    hidden = jax.nn.silu(hidden)
    hidden = _conv2d(
        hidden,
        params[f"{prefix}.conv2.weight"],
        params[f"{prefix}.conv2.bias"],
    )
    if f"{prefix}.conv_shortcut.weight" in params:
        shortcut = _conv2d(
            x,
            params[f"{prefix}.conv_shortcut.weight"],
            params[f"{prefix}.conv_shortcut.bias"],
            padding=0,
        )
    else:
        shortcut = x
    return shortcut + hidden


def _attention(
    params: dict[str, jax.Array],
    x: jax.Array,
    prefix: str,
    config: UNet2DConfig,
) -> jax.Array:
    residual = x
    hidden = _norm(params, f"{prefix}.group_norm", x, config)
    batch, height, width, channels = hidden.shape
    tokens = hidden.reshape(batch, height * width, channels)
    query = _linear(
        tokens, params[f"{prefix}.query.weight"], params[f"{prefix}.query.bias"]
    )
    key = _linear(
        tokens, params[f"{prefix}.key.weight"], params[f"{prefix}.key.bias"]
    )
    value = _linear(
        tokens, params[f"{prefix}.value.weight"], params[f"{prefix}.value.bias"]
    )
    scale = channels**-0.5
    scores = jnp.einsum("nld,nmd->nlm", query, key) * scale
    probabilities = jax.nn.softmax(scores, axis=-1)
    attended = jnp.einsum("nlm,nmd->nld", probabilities, value)
    attended = _linear(
        attended,
        params[f"{prefix}.proj_attn.weight"],
        params[f"{prefix}.proj_attn.bias"],
    )
    return residual + attended.reshape(batch, height, width, channels)


def _downsample(
    params: dict[str, jax.Array], x: jax.Array, prefix: str
) -> jax.Array:
    x = jnp.pad(x, ((0, 0), (0, 1), (0, 1), (0, 0)))
    return _conv2d(
        x,
        params[f"{prefix}.weight"],
        params[f"{prefix}.bias"],
        stride=2,
        padding=0,
    )


def _upsample(
    params: dict[str, jax.Array], x: jax.Array, prefix: str
) -> jax.Array:
    x = jnp.repeat(jnp.repeat(x, 2, axis=1), 2, axis=2)
    return _conv2d(
        x,
        params[f"{prefix}.weight"],
        params[f"{prefix}.bias"],
        padding=1,
    )


def unet_apply(
    params: dict[str, jax.Array],
    sample: jax.Array,
    timestep: jax.Array | int,
    config: UNet2DConfig,
) -> jax.Array:
    """Run the UNet on NHWC samples and zero-indexed model timesteps."""
    batch = sample.shape[0]
    if jnp.ndim(timestep) == 0:
        timesteps = jnp.full((batch,), timestep, dtype=jnp.int32)
    else:
        timesteps = jnp.asarray(timestep, dtype=jnp.int32)
    time_embed = _timestep_embedding(
        timesteps,
        config.block_out_channels[0],
        config.flip_sin_to_cos,
        config.freq_shift,
    )
    time_embed = _linear(
        time_embed,
        params["time_embedding.linear_1.weight"],
        params["time_embedding.linear_1.bias"],
    )
    time_embed = jax.nn.silu(time_embed)
    time_embed = _linear(
        time_embed,
        params["time_embedding.linear_2.weight"],
        params["time_embedding.linear_2.bias"],
    )

    hidden = _conv2d(
        sample, params["conv_in.weight"], params["conv_in.bias"]
    )
    down_res_samples = [hidden]

    for block, block_type in enumerate(config.down_block_types):
        for layer in range(config.layers_per_block):
            hidden = _resnet(
                params,
                hidden,
                time_embed,
                f"down_blocks.{block}.resnets.{layer}",
                config,
            )
            if "Attn" in block_type:
                hidden = _attention(
                    params,
                    hidden,
                    f"down_blocks.{block}.attentions.{layer}",
                    config,
                )
            down_res_samples.append(hidden)
        if block != len(config.down_block_types) - 1:
            hidden = _downsample(
                params,
                hidden,
                f"down_blocks.{block}.downsamplers.0.conv",
            )
            down_res_samples.append(hidden)

    hidden = _resnet(params, hidden, time_embed, "mid_block.resnets.0", config)
    hidden = _attention(params, hidden, "mid_block.attentions.0", config)
    hidden = _resnet(params, hidden, time_embed, "mid_block.resnets.1", config)

    for block, block_type in enumerate(config.up_block_types):
        num_layers = config.layers_per_block + 1
        skips = down_res_samples[-num_layers:]
        down_res_samples = down_res_samples[:-num_layers]
        for layer in range(num_layers):
            skip = skips[-1 - layer]
            hidden = jnp.concatenate([hidden, skip], axis=-1)
            hidden = _resnet(
                params,
                hidden,
                time_embed,
                f"up_blocks.{block}.resnets.{layer}",
                config,
            )
            if "Attn" in block_type:
                hidden = _attention(
                    params,
                    hidden,
                    f"up_blocks.{block}.attentions.{layer}",
                    config,
                )
        if block != len(config.up_block_types) - 1:
            hidden = _upsample(
                params,
                hidden,
                f"up_blocks.{block}.upsamplers.0.conv",
            )

    hidden = _norm(params, "conv_norm_out", hidden, config)
    hidden = jax.nn.silu(hidden)
    return _conv2d(
        hidden, params["conv_out.weight"], params["conv_out.bias"]
    )


def display_images(samples: jax.Array | np.ndarray) -> np.ndarray:
    """Map model samples from [-1,1] to [0,1] and clip for display only."""
    samples = np.asarray(samples, dtype=np.float32)
    return np.clip((samples + 1.0) / 2.0, 0.0, 1.0)


def make_ddpm_sampler(
    params: dict[str, jax.Array],
    config: UNet2DConfig,
    num_samples: int = 4,
    num_steps: int = NUM_DIFFUSION_STEPS,
):
    """Compile 1000-step DDPM sampling with model timestep t-1."""
    betas = linear_beta_schedule(num_steps)
    alpha_bars = alpha_bars_from_betas(betas)
    timesteps = jnp.arange(num_steps, 0, -1, dtype=jnp.int32)

    @jax.jit
    def sample(key: jax.Array) -> jax.Array:
        key_initial, key_noise = jax.random.split(key)
        x = jax.random.normal(
            key_initial,
            (num_samples, config.sample_size, config.sample_size, config.in_channels),
            dtype=DTYPE,
        )

        def body(x_t, t):
            predicted_noise = unet_apply(params, x_t, t - 1, config)
            beta_t = betas[t - 1]
            alpha_bar_t = alpha_bars[t]
            alpha_t = 1.0 - beta_t
            mean = (
                x_t
                - beta_t / jnp.sqrt(1.0 - alpha_bar_t) * predicted_noise
            ) / jnp.sqrt(alpha_t)
            noise = jax.random.normal(
                jax.random.fold_in(key_noise, t), x_t.shape, dtype=DTYPE
            )
            x_previous = jnp.where(
                t > 1, mean + jnp.sqrt(beta_t) * noise, mean
            )
            return x_previous, x_previous

        _, sampled_steps = jax.lax.scan(body, x, timesteps)
        return jnp.concatenate([x[None], sampled_steps], axis=0)

    return sample, betas, alpha_bars


__all__ = [
    "CONFIG_PATH",
    "MODEL_DIR",
    "NUM_DIFFUSION_STEPS",
    "UNet2DConfig",
    "WEIGHTS_PATH",
    "assert_parameter_shapes",
    "display_images",
    "load_unet_config",
    "load_unet_params",
    "make_ddpm_sampler",
    "unet_apply",
    "unet_param_shapes",
]
