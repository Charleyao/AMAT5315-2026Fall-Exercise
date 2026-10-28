"""Pre-norm decoder-only Transformer in JAX for Week 8 Part 5."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Callable

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import optax

VOCAB_SIZE = 512
CONTEXT_LENGTH = 64
N_LAYERS = 2
D_MODEL = 96
N_HEADS = 4
HEAD_DIM = 24
MLP_HIDDEN = 384


@dataclass
class TransformerConfig:
    vocab_size: int = VOCAB_SIZE
    context_length: int = CONTEXT_LENGTH
    n_layers: int = N_LAYERS
    d_model: int = D_MODEL
    n_heads: int = N_HEADS
    head_dim: int = HEAD_DIM
    mlp_hidden: int = MLP_HIDDEN
    steps: int = 6000
    batch_size: int = 32
    peak_lr: float = 0.003
    warmup_steps: int = 300
    end_lr_fraction: float = 0.1
    weight_decay: float = 0.01
    seed: int = 1730
    eval_every: int = 500
    eval_batch_windows: int = 64


def layer_norm(
    x: jax.Array,
    scale: jax.Array,
    bias: jax.Array,
    eps: float = 1e-5,
) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(variance + eps) * scale + bias


def _linear_key(key: jax.Array, fan_in: int, fan_out: int) -> jax.Array:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, (fan_in, fan_out), minval=-limit, maxval=limit)


def initialize_params(
    key: jax.Array,
    config: TransformerConfig,
) -> dict:
    """Initialize token/position embeddings and two pre-norm layers."""
    keys = jax.random.split(key, 5 + 8 * config.n_layers)
    token_key, position_key, final_ln_key, output_key = keys[:4]
    d = config.d_model

    params = {
        "token_embedding": 0.02
        * jax.random.normal(token_key, (config.vocab_size, d)),
        "position_embedding": 0.02
        * jax.random.normal(
            position_key, (config.context_length, d)
        ),
        "ln_final_scale": jnp.ones((d,)),
        "ln_final_bias": jnp.zeros((d,)),
        "output_weight": _linear_key(output_key, d, config.vocab_size),
        "output_bias": jnp.zeros((config.vocab_size,)),
        "layers": [],
    }
    key_index = 4
    for _ in range(config.n_layers):
        (
            qkv_key,
            out_key,
            mlp_in_key,
            mlp_out_key,
        ) = keys[key_index : key_index + 4]
        key_index += 4
        layer = {
            "ln1_scale": jnp.ones((d,)),
            "ln1_bias": jnp.zeros((d,)),
            "wqkv": _linear_key(qkv_key, d, 3 * d),
            "bqkv": jnp.zeros((3 * d,)),
            "wo": _linear_key(out_key, d, d),
            "bo": jnp.zeros((d,)),
            "ln2_scale": jnp.ones((d,)),
            "ln2_bias": jnp.zeros((d,)),
            "w1": _linear_key(mlp_in_key, d, config.mlp_hidden),
            "b1": jnp.zeros((config.mlp_hidden,)),
            "w2": _linear_key(
                mlp_out_key, config.mlp_hidden, d
            ),
            "b2": jnp.zeros((d,)),
        }
        params["layers"].append(layer)
    return params


def parameter_count(params: dict) -> int:
    total = 0
    for value in jax.tree_util.tree_leaves(params):
        total += int(np.prod(value.shape))
    return total


def forward(
    params: dict,
    tokens: jax.Array,
    active_length: jax.Array | int | None = None,
) -> jax.Array:
    """Return logits for a batch of token sequences.

    ``tokens`` may be right-padded to the context length; ``active_length``
    masks the padded key positions.  Causal masking prevents every query from
    attending to itself at a later position.
    """
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape (batch, time)")
    batch, time = tokens.shape
    config_d = params["token_embedding"].shape[1]
    if time > params["position_embedding"].shape[0]:
        raise ValueError("sequence longer than the position table")

    if active_length is None:
        active_length = time
    active_length = jnp.asarray(active_length, dtype=jnp.int32)
    if active_length.shape != ():
        raise ValueError("active_length must be a scalar")

    x = params["token_embedding"][tokens] + params["position_embedding"][
        :time
    ]
    heads = N_HEADS
    head_dim = config_d // heads
    causal = jnp.tril(jnp.ones((time, time), dtype=bool))
    key_valid = jnp.arange(time)[None, :] < active_length
    attention_mask = causal[None, None, :, :] & key_valid[:, None, :]

    for layer in params["layers"]:
        normed = layer_norm(x, layer["ln1_scale"], layer["ln1_bias"])
        qkv = normed @ layer["wqkv"] + layer["bqkv"]
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(batch, time, heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, time, heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, time, heads, head_dim).transpose(0, 2, 1, 3)

        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / np.sqrt(head_dim)
        scores = jnp.where(attention_mask, scores, -1e30)
        attention = jax.nn.softmax(scores, axis=-1)
        attended = jnp.einsum("bhqk,bhkd->bhqd", attention, v)
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, time, -1)
        x = x + attended @ layer["wo"] + layer["bo"]

        normed = layer_norm(x, layer["ln2_scale"], layer["ln2_bias"])
        hidden = jax.nn.gelu(
            normed @ layer["w1"] + layer["b1"], approximate=True
        )
        x = x + hidden @ layer["w2"] + layer["b2"]

    x = layer_norm(x, params["ln_final_scale"], params["ln_final_bias"])
    return x @ params["output_weight"] + params["output_bias"]


def batch_loss(
    params: dict,
    inputs: jax.Array,
    targets: jax.Array,
) -> jax.Array:
    logits = forward(params, inputs)
    return optax.softmax_cross_entropy_with_integer_labels(
        logits, targets
    ).mean()


def _window_batch_losses(
    params: dict,
    inputs: jax.Array,
    targets: jax.Array,
) -> jax.Array:
    logits = forward(params, inputs)
    return optax.softmax_cross_entropy_with_integer_labels(logits, targets)


def evaluate_by_position(
    params: dict,
    tokens: np.ndarray,
    context_length: int = CONTEXT_LENGTH,
    batch_windows: int = 64,
) -> dict:
    """Evaluate consecutive 65-token windows and average loss by position.

    Each window contributes 64 targets: position 1 uses one context token,
    ..., position 64 uses all 64 context tokens.
    """
    tokens = np.asarray(tokens, dtype=np.int32)
    n_windows = (len(tokens) - (context_length + 1)) // context_length + 1
    if n_windows < 1:
        raise ValueError("token stream is too short for one window")
    starts = np.arange(n_windows, dtype=np.int64) * context_length
    offsets = np.arange(context_length + 1, dtype=np.int64)
    position_sum = np.zeros(context_length, dtype=np.float64)
    total_windows = 0

    @jax.jit
    def _losses(params, inputs, targets):
        return _window_batch_losses(params, inputs, targets)

    for begin in range(0, n_windows, batch_windows):
        batch_starts = starts[begin : begin + batch_windows]
        sequences = tokens[batch_starts[:, None] + offsets[None, :]]
        inputs = jnp.asarray(sequences[:, :-1], dtype=jnp.int32)
        targets = jnp.asarray(sequences[:, 1:], dtype=jnp.int32)
        losses = np.asarray(_losses(params, inputs, targets))
        position_sum += losses.sum(axis=0)
        total_windows += len(batch_starts)

    position_mean = position_sum / total_windows
    total_nats = float(position_sum.sum())
    scored_tokens = int(total_windows * context_length)
    return {
        "position_loss": position_mean,
        "overall_loss": float(position_mean.mean()),
        "total_nats": total_nats,
        "scored_tokens": scored_tokens,
        "windows": int(total_windows),
        "final8_mean": float(position_mean[-8:].mean()),
        "after32_mean": float(position_mean[31:].mean()),
    }


def train_transformer(
    train_tokens: np.ndarray,
    valid_tokens: np.ndarray,
    valid_characters: int,
    config: TransformerConfig | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Train the Transformer and track validation loss by context position."""
    if config is None:
        config = TransformerConfig()
    train_tokens = np.asarray(train_tokens, dtype=np.int32)
    valid_tokens = np.asarray(valid_tokens, dtype=np.int32)
    if len(train_tokens) < config.context_length + 1:
        raise ValueError("training stream is too short")
    if len(valid_tokens) < config.context_length + 1:
        raise ValueError("validation stream is too short")

    key = jax.random.PRNGKey(config.seed)
    params = initialize_params(key, config)
    end_value = config.peak_lr * config.end_lr_fraction
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.peak_lr,
        warmup_steps=config.warmup_steps,
        decay_steps=config.steps,
        end_value=end_value,
    )
    optimizer = optax.adamw(
        learning_rate=schedule,
        weight_decay=config.weight_decay,
        b1=0.9,
        b2=0.999,
        eps=1e-8,
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, inputs, targets):
        loss, grads = jax.value_and_grad(batch_loss)(
            params, inputs, targets
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    rng = np.random.default_rng(config.seed + 1)
    offsets = np.arange(config.context_length + 1, dtype=np.int64)
    n_starts = len(train_tokens) - (config.context_length + 1) + 1
    history = {
        "step": [],
        "train_loss": [],
        "learning_rate": [],
        "validation_step": [],
        "validation_loss": [],
        "validation_bpc": [],
        "validation_final8": [],
    }

    initial_eval = evaluate_by_position(
        params,
        valid_tokens,
        context_length=config.context_length,
        batch_windows=config.eval_batch_windows,
    )
    initial_bpc = initial_eval["total_nats"] / (
        np.log(2.0) * valid_characters
    )
    history["validation_step"].append(0)
    history["validation_loss"].append(initial_eval["overall_loss"])
    history["validation_bpc"].append(initial_bpc)
    history["validation_final8"].append(initial_eval["final8_mean"])
    log(
        f"[trf ] step 0: validation {initial_eval['overall_loss']:.6f} "
        f"nats/token, {initial_bpc:.6f} bits/char"
    )

    started = time.time()
    for step in range(1, config.steps + 1):
        starts = rng.integers(0, n_starts, size=config.batch_size)
        sequences = train_tokens[starts[:, None] + offsets[None, :]]
        inputs = jnp.asarray(sequences[:, :-1], dtype=jnp.int32)
        targets = jnp.asarray(sequences[:, 1:], dtype=jnp.int32)
        params, opt_state, loss = train_step(
            params, opt_state, inputs, targets
        )

        history["step"].append(step)
        history["train_loss"].append(float(loss))
        history["learning_rate"].append(float(schedule(step)))

        if step % config.eval_every == 0 or step == config.steps:
            evaluation = evaluate_by_position(
                params,
                valid_tokens,
                context_length=config.context_length,
                batch_windows=config.eval_batch_windows,
            )
            bpc = evaluation["total_nats"] / (
                np.log(2.0) * valid_characters
            )
            history["validation_step"].append(step)
            history["validation_loss"].append(evaluation["overall_loss"])
            history["validation_bpc"].append(bpc)
            history["validation_final8"].append(
                evaluation["final8_mean"]
            )
            elapsed = time.time() - started
            log(
                f"[trf ] step {step:4d}/{config.steps}: "
                f"train {float(loss):.6f}, val {evaluation['overall_loss']:.6f}, "
                f"bpc {bpc:.6f}, final8 {evaluation['final8_mean']:.6f}, "
                f"lr {float(schedule(step)):.6f}, {elapsed:6.1f}s"
            )

    final_eval = evaluate_by_position(
        params,
        valid_tokens,
        context_length=config.context_length,
        batch_windows=config.eval_batch_windows,
    )
    final_bpc = final_eval["total_nats"] / (
        np.log(2.0) * valid_characters
    )
    if history["validation_step"][-1] != config.steps:
        history["validation_step"].append(config.steps)
        history["validation_loss"].append(final_eval["overall_loss"])
        history["validation_bpc"].append(final_bpc)
        history["validation_final8"].append(final_eval["final8_mean"])
    else:
        history["validation_loss"][-1] = final_eval["overall_loss"]
        history["validation_bpc"][-1] = final_bpc
        history["validation_final8"][-1] = final_eval["final8_mean"]

    return {
        "params": params,
        "config": asdict(config),
        "parameter_count": parameter_count(params),
        "final_evaluation": final_eval,
        "final_validation_loss": final_eval["overall_loss"],
        "final_bits_per_char": final_bpc,
        "training_time_seconds": float(time.time() - started),
        "history": history,
    }


def load_checkpoint(path) -> tuple[dict, TransformerConfig]:
    """Reconstruct the parameter tree and config from a saved NPZ checkpoint."""
    data = np.load(path, allow_pickle=False)
    config = TransformerConfig(**json.loads(str(data["config_json"])))
    base_names = (
        "token_embedding",
        "position_embedding",
        "ln_final_scale",
        "ln_final_bias",
        "output_weight",
        "output_bias",
    )
    params = {name: jnp.asarray(data[name]) for name in base_names}
    layer_names = (
        "ln1_scale",
        "ln1_bias",
        "wqkv",
        "bqkv",
        "wo",
        "bo",
        "ln2_scale",
        "ln2_bias",
        "w1",
        "b1",
        "w2",
        "b2",
    )
    params["layers"] = [
        {
            name: jnp.asarray(data[f"layer{layer_index}_{name}"])
            for name in layer_names
        }
        for layer_index in range(config.n_layers)
    ]
    return params, config


def forward_with_attention(
    params: dict,
    tokens: jax.Array,
    requests: set[tuple[int, int]],
    active_length: jax.Array | int | None = None,
) -> tuple[jax.Array, dict[tuple[int, int], jax.Array]]:
    """Return logits and selected causal attention maps.

    ``requests`` contains zero-based ``(layer_index, head_index)`` pairs.  Each
    returned attention map has shape ``(batch, time, time)`` and maps query
    tokens (rows) to key tokens (columns).
    """
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape (batch, time)")
    batch, time = tokens.shape
    config_d = params["token_embedding"].shape[1]
    if time > params["position_embedding"].shape[0]:
        raise ValueError("sequence longer than the position table")

    if active_length is None:
        active_length = time
    active_length = jnp.asarray(active_length, dtype=jnp.int32)
    if active_length.shape != ():
        raise ValueError("active_length must be a scalar")

    x = params["token_embedding"][tokens] + params["position_embedding"][
        :time
    ]
    heads = N_HEADS
    head_dim = config_d // heads
    causal = jnp.tril(jnp.ones((time, time), dtype=bool))
    key_valid = jnp.arange(time)[None, :] < active_length
    attention_mask = causal[None, None, :, :] & key_valid[:, None, :]
    captured: dict[tuple[int, int], jax.Array] = {}

    for layer_index, layer in enumerate(params["layers"]):
        normed = layer_norm(x, layer["ln1_scale"], layer["ln1_bias"])
        qkv = normed @ layer["wqkv"] + layer["bqkv"]
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(batch, time, heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, time, heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, time, heads, head_dim).transpose(0, 2, 1, 3)

        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / np.sqrt(head_dim)
        scores = jnp.where(attention_mask, scores, -1e30)
        attention = jax.nn.softmax(scores, axis=-1)
        for requested_layer, requested_head in requests:
            if requested_layer == layer_index:
                captured[(requested_layer, requested_head)] = attention[
                    :, requested_head, :, :
                ]

        attended = jnp.einsum("bhqk,bhkd->bhqd", attention, v)
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, time, -1)
        x = x + attended @ layer["wo"] + layer["bo"]

        normed = layer_norm(x, layer["ln2_scale"], layer["ln2_bias"])
        hidden = jax.nn.gelu(
            normed @ layer["w1"] + layer["b1"], approximate=True
        )
        x = x + hidden @ layer["w2"] + layer["b2"]

    x = layer_norm(x, params["ln_final_scale"], params["ln_final_bias"])
    logits = x @ params["output_weight"] + params["output_bias"]
    missing = set(requests) - set(captured)
    if missing:
        raise ValueError(f"unavailable attention requests: {sorted(missing)}")
    return logits, captured
