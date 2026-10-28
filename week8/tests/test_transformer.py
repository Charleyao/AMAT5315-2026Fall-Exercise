"""Tests for the Part 5 Transformer implementation."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.transformer import (
    TransformerConfig,
    evaluate_by_position,
    forward,
    initialize_params,
    layer_norm,
    load_checkpoint,
    parameter_count,
)

WEEK_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT = WEEK_DIR / "artifacts" / "part5_transformer_checkpoint.npz"


def test_config_matches_assignment():
    config = TransformerConfig()
    assert config.vocab_size == 512
    assert config.n_layers == 2
    assert config.d_model == 96
    assert config.n_heads == 4
    assert config.head_dim == 24
    assert config.context_length == 64
    assert config.mlp_hidden == 384
    assert config.steps == 6000
    assert config.batch_size == 32
    assert config.peak_lr == 0.003
    assert config.warmup_steps == 300


def test_parameter_count_and_shapes():
    params = initialize_params(
        jax.random.PRNGKey(0), TransformerConfig()
    )
    assert parameter_count(params) == 328_832
    assert params["token_embedding"].shape == (512, 96)
    assert params["position_embedding"].shape == (64, 96)
    assert len(params["layers"]) == 2
    assert params["layers"][0]["wqkv"].shape == (96, 288)
    assert params["layers"][0]["w1"].shape == (96, 384)
    assert params["output_weight"].shape == (96, 512)


def test_layer_norm_shape_and_normalization():
    x = jnp.asarray(np.random.default_rng(0).normal(size=(4, 7)))
    result = layer_norm(
        x, jnp.ones((7,)), jnp.zeros((7,))
    )
    assert result.shape == (4, 7)
    np.testing.assert_allclose(
        np.asarray(jnp.mean(result, axis=-1)), 0.0, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(jnp.std(result, axis=-1)), 1.0, atol=1e-3
    )


def test_causal_masking_blocks_future_tokens():
    params = initialize_params(
        jax.random.PRNGKey(1), TransformerConfig()
    )
    inputs = np.random.default_rng(0).integers(
        0, 512, size=(2, 64), dtype=np.int32
    )
    changed = inputs.copy()
    changed[:, 10] = (changed[:, 10] + 1) % 512

    original_logits = np.asarray(forward(params, jnp.asarray(inputs)))
    changed_logits = np.asarray(forward(params, jnp.asarray(changed)))
    np.testing.assert_allclose(
        original_logits[:, :10], changed_logits[:, :10], atol=1e-6
    )
    assert np.max(np.abs(original_logits[:, 10:] - changed_logits[:, 10:])) > 1e-4


def test_padded_forward_uses_active_length():
    params = initialize_params(
        jax.random.PRNGKey(2), TransformerConfig()
    )
    short = np.array([[5, 6, 7]] + [[0] * 3], dtype=np.int32)
    padded = np.zeros((2, 64), dtype=np.int32)
    padded[0, :3] = short[0]
    padded[1, :3] = short[0]
    padded[1, 3:] = 123
    logits = forward(
        params, jnp.asarray(padded), jnp.asarray(3)
    )
    np.testing.assert_allclose(
        logits[0, 2], logits[1, 2], atol=1e-6
    )


def test_evaluate_by_position_shape():
    params = initialize_params(
        jax.random.PRNGKey(3), TransformerConfig()
    )
    tokens = np.random.default_rng(0).integers(
        0, 512, size=200, dtype=np.int32
    )
    evaluation = evaluate_by_position(params, tokens, batch_windows=2)
    assert evaluation["position_loss"].shape == (64,)
    assert evaluation["scored_tokens"] == evaluation["windows"] * 64
    assert np.isfinite(evaluation["overall_loss"])


@pytest.mark.skipif(
    not CHECKPOINT.exists(), reason="run `python -m src.part5` first"
)
def test_saved_checkpoint_roundtrip():
    params, config = load_checkpoint(CHECKPOINT)
    assert config.n_layers == 2
    assert parameter_count(params) == 328_832
    assert params["layers"][1]["w2"].shape == (384, 96)
