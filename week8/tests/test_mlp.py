"""Tests for the JAX previous-token MLP."""

from __future__ import annotations

import numpy as np
import pytest

from src.mlp import (
    MLPConfig,
    batch_loss,
    evaluate,
    forward,
    initialize_params,
    train_mlp,
)


def test_default_config_matches_assignment():
    config = MLPConfig()
    assert config.steps == 4000
    assert config.batch_size == 16384
    assert config.peak_lr == 0.006
    assert config.embed_dim == 64
    assert config.hidden_dim == 512
    assert config.vocab_size == 512


def test_parameter_shapes():
    import jax

    config = MLPConfig(embed_dim=8, hidden_dim=16)
    params = initialize_params(jax.random.PRNGKey(0), config)
    assert params["E"].shape == (512, 8)
    assert params["W1"].shape == (8, 16)
    assert params["b1"].shape == (16,)
    assert params["W2"].shape == (16, 512)
    assert params["b2"].shape == (512,)


def test_forward_shape_and_zero_logits_loss():
    import jax
    import jax.numpy as jnp

    config = MLPConfig(embed_dim=8, hidden_dim=16)
    params = initialize_params(jax.random.PRNGKey(1), config)
    params = {name: jnp.zeros_like(value) for name, value in params.items()}
    inputs = jnp.asarray(np.array([1, 2, 3, 4], dtype=np.int32))
    targets = jnp.asarray(np.array([2, 3, 4, 5], dtype=np.int32))
    logits = forward(params, inputs)
    assert logits.shape == (4, 512)
    assert float(batch_loss(params, inputs, targets)) == pytest.approx(
        np.log(512.0), abs=1e-6
    )


def test_evaluate_on_token_stream():
    import jax

    config = MLPConfig(embed_dim=8, hidden_dim=16, eval_batch_size=16)
    params = initialize_params(jax.random.PRNGKey(2), config)
    import jax.numpy as jnp

    params = {name: jnp.zeros_like(value) for name, value in params.items()}
    tokens = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    assert evaluate(params, tokens, batch_size=16) == pytest.approx(
        np.log(512.0), abs=1e-6
    )


def test_train_mlp_smoke_test(tmp_path):
    config = MLPConfig(
        steps=6,
        batch_size=64,
        warmup_steps=1,
        embed_dim=8,
        hidden_dim=16,
        eval_every=3,
        eval_batch_size=256,
    )
    rng = np.random.default_rng(0)
    train_tokens = rng.integers(0, 512, size=2000, dtype=np.int32)
    valid_tokens = rng.integers(0, 512, size=500, dtype=np.int32)
    result = train_mlp(
        train_tokens,
        valid_tokens,
        config,
        log=lambda message: None,
    )
    assert np.isfinite(result["final_validation_loss"])
    assert len(result["history"]["step"]) == config.steps
    assert result["history"]["validation_step"][-1] == config.steps
    assert all(np.isfinite(value) for value in result["history"]["train_loss"])
    for value in result["params"].values():
        assert np.all(np.isfinite(np.asarray(value)))
