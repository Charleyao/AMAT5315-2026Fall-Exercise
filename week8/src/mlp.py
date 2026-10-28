"""Previous-token MLP language model trained with JAX on CPU."""

from __future__ import annotations

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
EMBED_DIM = 64
HIDDEN_DIM = 512


@dataclass
class MLPConfig:
    steps: int = 4000
    batch_size: int = 16384
    peak_lr: float = 0.006
    warmup_steps: int = 200
    embed_dim: int = EMBED_DIM
    hidden_dim: int = HIDDEN_DIM
    vocab_size: int = VOCAB_SIZE
    weight_decay: float = 0.01
    seed: int = 1730
    eval_every: int = 250
    eval_batch_size: int = 32768


def initialize_params(key: jax.Array, config: MLPConfig) -> dict[str, jax.Array]:
    """Xavier-uniform weights and zero biases."""
    embed_key, hidden_key, output_key = jax.random.split(key, 3)
    hidden_limit = np.sqrt(6.0 / (config.embed_dim + config.hidden_dim))
    output_limit = np.sqrt(6.0 / (config.hidden_dim + config.vocab_size))
    return {
        "E": 0.02 * jax.random.normal(
            embed_key, (config.vocab_size, config.embed_dim)
        ),
        "W1": jax.random.uniform(
            hidden_key,
            (config.embed_dim, config.hidden_dim),
            minval=-hidden_limit,
            maxval=hidden_limit,
        ),
        "b1": jnp.zeros((config.hidden_dim,)),
        "W2": jax.random.uniform(
            output_key,
            (config.hidden_dim, config.vocab_size),
            minval=-output_limit,
            maxval=output_limit,
        ),
        "b2": jnp.zeros((config.vocab_size,)),
    }


def forward(params: dict[str, jax.Array], inputs: jax.Array) -> jax.Array:
    hidden = jnp.tanh(params["E"][inputs] @ params["W1"] + params["b1"])
    return hidden @ params["W2"] + params["b2"]


def batch_loss(
    params: dict[str, jax.Array],
    inputs: jax.Array,
    targets: jax.Array,
) -> jax.Array:
    logits = forward(params, inputs)
    return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()


def evaluate(
    params: dict[str, jax.Array],
    tokens: np.ndarray,
    batch_size: int = 32768,
) -> float:
    """Mean held-out cross-entropy in nats/token over all consecutive pairs."""
    tokens = np.asarray(tokens, dtype=np.int32)
    inputs_all = tokens[:-1]
    targets_all = tokens[1:]
    total_loss = 0.0
    total_tokens = 0

    @jax.jit
    def _sum_loss(
        params: dict[str, jax.Array], inputs: jax.Array, targets: jax.Array
    ) -> jax.Array:
        logits = forward(params, inputs)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits, targets
        ).sum()

    for start in range(0, len(inputs_all), batch_size):
        stop = min(start + batch_size, len(inputs_all))
        inputs = jnp.asarray(inputs_all[start:stop])
        targets = jnp.asarray(targets_all[start:stop])
        total_loss += float(_sum_loss(params, inputs, targets))
        total_tokens += stop - start
    return total_loss / total_tokens


def train_mlp(
    train_tokens: np.ndarray,
    valid_tokens: np.ndarray,
    config: MLPConfig | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Train the MLP and return parameters, final loss, and history."""
    if config is None:
        config = MLPConfig()
    train_tokens = np.asarray(train_tokens, dtype=np.int32)
    valid_tokens = np.asarray(valid_tokens, dtype=np.int32)
    if len(train_tokens) < 2 or len(valid_tokens) < 2:
        raise ValueError("token streams must contain at least two tokens")

    key = jax.random.PRNGKey(config.seed)
    params = initialize_params(key, config)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.peak_lr,
        warmup_steps=config.warmup_steps,
        decay_steps=config.steps,
        end_value=0.0,
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
        loss, grads = jax.value_and_grad(batch_loss)(params, inputs, targets)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    rng = np.random.default_rng(config.seed + 1)
    n_pairs = len(train_tokens) - 1
    history = {
        "step": [],
        "train_loss": [],
        "learning_rate": [],
        "validation_step": [],
        "validation_loss": [],
    }
    initial_validation = evaluate(params, valid_tokens, config.eval_batch_size)
    history["validation_step"].append(0)
    history["validation_loss"].append(initial_validation)
    log(f"[mlp ] step 0: validation {initial_validation:.6f} nats/token")

    started = time.time()
    for step in range(1, config.steps + 1):
        indices = rng.integers(0, n_pairs, size=config.batch_size)
        inputs = jnp.asarray(train_tokens[indices], dtype=jnp.int32)
        targets = jnp.asarray(train_tokens[indices + 1], dtype=jnp.int32)
        params, opt_state, loss = train_step(params, opt_state, inputs, targets)

        history["step"].append(step)
        history["train_loss"].append(float(loss))
        history["learning_rate"].append(float(schedule(step)))

        should_evaluate = step % config.eval_every == 0 or step == config.steps
        if should_evaluate:
            validation_loss = evaluate(
                params, valid_tokens, config.eval_batch_size
            )
            history["validation_step"].append(step)
            history["validation_loss"].append(validation_loss)
            elapsed = time.time() - started
            log(
                f"[mlp ] step {step:4d}/{config.steps}: "
                f"train {float(loss):.6f}, validation {validation_loss:.6f}, "
                f"lr {float(schedule(step)):.6f}, {elapsed:6.1f}s"
            )

    final_validation = evaluate(params, valid_tokens, config.eval_batch_size)
    if history["validation_step"][-1] != config.steps:
        history["validation_step"].append(config.steps)
        history["validation_loss"].append(final_validation)
    else:
        history["validation_loss"][-1] = final_validation

    return {
        "params": params,
        "config": asdict(config),
        "final_validation_loss": float(final_validation),
        "training_time_seconds": float(time.time() - started),
        "history": history,
    }
