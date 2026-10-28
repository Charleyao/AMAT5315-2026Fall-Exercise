"""Tests for the Part 5 Figure 18 attention-head extension."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.transformer import (
    TransformerConfig,
    forward_with_attention,
    initialize_params,
)

WEEK_DIR = Path(__file__).resolve().parents[1]
CHECKS = WEEK_DIR / "artifacts" / "part5_figure18_checks.json"
FIGURE = WEEK_DIR / "evidence" / "figure18_attention_heads.png"


def test_forward_with_attention_is_causal_and_row_normalized():
    params = initialize_params(
        jax.random.PRNGKey(7), TransformerConfig()
    )
    tokens = np.random.default_rng(0).integers(
        0, 512, size=(1, 24), dtype=np.int32
    )
    _, attention = forward_with_attention(
        params, jnp.asarray(tokens), {(0, 0), (1, 1)}
    )
    assert set(attention) == {(0, 0), (1, 1)}
    for matrix in attention.values():
        matrix = np.asarray(matrix[0])
        assert matrix.shape == (24, 24)
        assert np.max(np.triu(matrix, k=1)) <= 1e-12
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.skipif(
    not (CHECKS.exists() and FIGURE.exists()),
    reason="run `python -m src.part5_figure18` first",
)
def test_saved_figure18_checks_and_outputs():
    checks = json.loads(CHECKS.read_text(encoding="utf-8"))
    assert checks["all_checks_passed"]
    assert checks["checkpoint_reused_without_training"]
    assert checks["n_tokens"] == 24
    assert len(checks["labels"]) == 24
    assert set(checks["heads"]) == {"layer1_head1", "layer2_head2"}
    for record in checks["heads"].values():
        assert record["future_attention_zero"]
        assert record["rows_sum_to_one"]
        assert record["future_attention_max"] <= 1e-12
        assert record["row_sum_error"] <= 1e-6
    assert FIGURE.exists() and FIGURE.stat().st_size > 0
    assert "figure18_attention_heads.png" in (
        WEEK_DIR / "README.md"
    ).read_text(encoding="utf-8")
