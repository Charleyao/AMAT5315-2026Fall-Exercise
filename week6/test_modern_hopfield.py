"""Tests for Week 6 Part 4 VERIFY (1): modern Hopfield / softmax attention.

Covers pattern construction/order, the update formula, equivalence with
``jax.nn.dot_product_attention`` using ``scale=beta``, weight normalisation,
beta-driven concentration and the saved artifacts/figure.

Run with::

    cd week6
    uv run python -m pytest -q
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import modern_hopfield as mh
import modern_hopfield_experiment as experiment
import rbm

WEEK_DIR = Path(__file__).resolve().parent
MNIST_DIR = WEEK_DIR / "inputs" / "mnist"


def test_patterns_are_first_training_images_as_pm_one_spins():
    images, labels = rbm.load_binary_mnist("train", MNIST_DIR)
    expected = rbm.to_spins(images[:17]).astype(np.float32)
    patterns, pattern_labels = mh.mnist_spin_patterns(17)
    np.testing.assert_array_equal(np.asarray(patterns), expected)
    np.testing.assert_array_equal(pattern_labels, labels[:17])
    assert patterns.shape == (17, 784)
    assert set(np.unique(np.asarray(patterns)).tolist()) == {-1.0, 1.0}


def test_update_matches_explicit_formula_and_weights_sum_to_one():
    key = jax.random.PRNGKey(0)
    patterns = jnp.where(
        jax.random.bernoulli(key, 0.5, (12, 20)), 1.0, -1.0
    )
    queries = jnp.where(
        jax.random.bernoulli(jax.random.PRNGKey(1), 0.5, (5, 20)), 1.0, -1.0
    )
    beta = 0.07
    updated, weights = mh.modern_hopfield_update(patterns, queries, beta)
    expected_weights = jax.nn.softmax(beta * (queries @ patterns.T), axis=-1)
    np.testing.assert_allclose(np.asarray(weights), np.asarray(expected_weights), atol=1e-7)
    np.testing.assert_allclose(
        np.asarray(updated), np.asarray(expected_weights @ patterns), atol=1e-7
    )
    np.testing.assert_allclose(
        np.asarray(weights.sum(axis=-1)), np.ones(5), atol=1e-6
    )
    assert updated.shape == (5, 20)
    assert weights.shape == (5, 12)


def test_single_query_matches_batched_call():
    patterns = jnp.where(
        jax.random.bernoulli(jax.random.PRNGKey(2), 0.5, (8, 10)), 1.0, -1.0
    )
    query = patterns[3]
    single_update, single_weights = mh.modern_hopfield_update(patterns, query, 0.2)
    batched_update, batched_weights = mh.modern_hopfield_update(
        patterns, query[None, :], 0.2
    )
    np.testing.assert_array_equal(np.asarray(single_update), np.asarray(batched_update[0]))
    np.testing.assert_array_equal(
        np.asarray(single_weights), np.asarray(batched_weights[0])
    )


def test_attention_equivalence_with_scale_beta():
    patterns, _ = mh.mnist_spin_patterns(100)
    queries, _ = mh.held_out_spins(20)
    for beta in (0.001, 0.03, 0.3, 1.0):
        manual, _weights = mh.modern_hopfield_update(patterns, queries, beta)
        attention = mh.attention_update(patterns, queries, beta)
        assert float(jnp.max(jnp.abs(manual - attention))) < 1e-6


def test_attention_uses_explicit_beta_not_default_scale():
    patterns, _ = mh.mnist_spin_patterns(100)
    queries, _ = mh.held_out_spins(4)
    beta = 0.3
    manual, _ = mh.modern_hopfield_update(patterns, queries, beta)
    scaled = jax.nn.dot_product_attention(
        queries[:, None, :],
        patterns[:, None, :],
        patterns[:, None, :],
        scale=beta,
    )[:, 0, :]
    default = jax.nn.dot_product_attention(
        queries[:, None, :],
        patterns[:, None, :],
        patterns[:, None, :],
        scale=None,  # default 1/sqrt(784) ~= 0.0357
    )[:, 0, :]
    np.testing.assert_allclose(np.asarray(scaled), np.asarray(manual), atol=1e-6)
    assert float(jnp.max(jnp.abs(default - manual))) > 1e-4


def test_larger_beta_concentrates_weight_on_most_similar_pattern():
    patterns, _ = mh.mnist_spin_patterns(100)
    query, _ = mh.held_out_spins(1)
    similarities = np.asarray(patterns @ query[0])
    most_similar = int(np.argmax(similarities))
    top_weights = []
    entropies = []
    for beta in experiment.PLOT_BETAS:
        weights = np.asarray(mh.softmax_weights(patterns, query, beta)[0])
        top_weights.append(float(weights.max()))
        entropies.append(mh.attention_entropy(weights))
        assert int(np.argmax(weights)) == most_similar
    assert top_weights[0] < top_weights[1] < top_weights[2]
    assert entropies[0] > entropies[1] > entropies[2]


@pytest.mark.skipif(
    not experiment.RESULTS_PATH.is_file(), reason="run modern_hopfield_experiment.py"
)
def test_saved_experiment_verifies_requirements():
    data = json.loads(experiment.RESULTS_PATH.read_text())
    settings = data["settings"]
    assert settings["P"] == [3, 10, 30, 100, 300, 1000]
    assert settings["betas"] == [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    assert settings["n_queries"] == 20
    assert settings["plot_P"] == 100
    assert settings["plot_betas"] == [0.003, 0.03, 0.3]

    assert data["max_abs_difference"] < 1e-5
    assert data["max_weight_sum_error"] < 1e-5
    assert data["argmax_matches_most_similar"] is True

    for p in settings["P"]:
        for beta in settings["betas"]:
            assert data["difference_by_point"][str(p)][str(beta)] < 1e-5
            assert data["weight_sum_error_by_point"][str(p)][str(beta)] < 1e-5
        entry = data["concentration"][str(p)]
        tops = [entry[str(beta)]["top_weight"] for beta in settings["betas"]]
        entropies = [entry[str(beta)]["entropy"] for beta in settings["betas"]]
        assert all(tops[i + 1] >= tops[i] for i in range(len(tops) - 1))
        assert all(
            entropies[i + 1] <= entropies[i] for i in range(len(entropies) - 1)
        )
        assert all(
            entry[str(beta)]["top_index"] == entry[str(beta)]["most_similar_index"]
            for beta in settings["betas"]
        )

    plot = data["plot"]
    for beta in settings["plot_betas"]:
        weights = np.asarray(plot["weights"][str(beta)])
        assert weights.shape == (100,)
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert plot["top_indices"][str(beta)] == int(np.argmax(weights))
        assert plot["top_weights"][str(beta)] == pytest.approx(float(weights.max()))
    tops = [plot["top_weights"][str(beta)] for beta in settings["plot_betas"]]
    entropies = [plot["entropies"][str(beta)] for beta in settings["plot_betas"]]
    assert tops[0] < tops[1] < tops[2]
    assert entropies[0] > entropies[1] > entropies[2]


@pytest.mark.skipif(
    not experiment.FIGURE_PATH.is_file(), reason="run modern_hopfield_experiment.py"
)
def test_figure_has_three_beta_panels():
    assert experiment.FIGURE_PATH.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    data = json.loads(experiment.RESULTS_PATH.read_text())
    fig = experiment.build_figure(data)
    try:
        assert len(fig.axes) == 3
        titles = [axis.get_title() for axis in fig.axes]
        assert titles == [r"$\beta$ = 0.003", r"$\beta$ = 0.03", r"$\beta$ = 0.3"]
        for axis in fig.axes:
            arrays = [patch for patch in axis.patches]
            assert len(arrays) == 100
    finally:
        experiment.plt.close(fig)
