"""Tests for Week 6 Part 3 VERIFY (1): Hopfield network and alpha sweep.

Covers couplings (symmetric, zero diagonal), fields and unstable-bit
definition, the Gaussian prediction, asynchronous zero-temperature recall,
the finite-size convergence of the experiment, and the saved artifacts.

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

import hopfield
import hopfield_experiment as experiment

WEEK_DIR = Path(__file__).resolve().parent


def test_random_patterns_are_independent_pm_one():
    key = jax.random.PRNGKey(0)
    patterns = np.asarray(hopfield.random_patterns(key, 5, 100))
    assert patterns.shape == (5, 100)
    assert set(np.unique(patterns).tolist()) == {-1.0, 1.0}
    other = np.asarray(hopfield.random_patterns(jax.random.PRNGKey(1), 5, 100))
    assert not np.array_equal(patterns, other)
    # Independent random patterns have small mutual overlap.
    overlaps = patterns @ patterns.T / 100
    assert np.allclose(np.diag(overlaps), 1.0)
    assert np.max(np.abs(overlaps - np.eye(5))) < 0.4


def test_couplings_are_symmetric_with_zero_diagonal():
    patterns = np.array([[1, -1, 1, 1], [1, 1, -1, 1]], dtype=np.float32)
    J = np.asarray(hopfield.build_couplings(jnp.asarray(patterns)))
    assert J.shape == (4, 4)
    np.testing.assert_array_equal(J, J.T)
    np.testing.assert_array_equal(np.diag(J), np.zeros(4))
    expected = patterns.T @ patterns / 4
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_allclose(J, expected, atol=1e-7)


def test_local_fields_match_matrix_product():
    patterns = np.array([[1, -1, 1, 1], [1, 1, -1, 1]], dtype=np.float32)
    J = hopfield.build_couplings(jnp.asarray(patterns))
    fields = np.asarray(hopfield.local_fields(J, patterns))
    np.testing.assert_allclose(fields, patterns @ np.asarray(J).T, atol=1e-7)


def test_unstable_definition_is_strictly_negative():
    patterns = jnp.array([[1.0, -1.0, 1.0]])
    fields = jnp.array([[ -0.5, 0.5, 0.0]])  # products: -0.5, -0.5, 0.0
    mask = np.asarray(hopfield.unstable_mask(patterns, fields))
    np.testing.assert_array_equal(mask, [[True, True, False]])
    J = jnp.zeros((3, 3))
    assert hopfield.unstable_fraction(patterns, J) == 0.0


def test_gaussian_prediction_matches_normal_cdf():
    from math import erf, sqrt

    def phi(z):
        return 0.5 * (1.0 + erf(z / sqrt(2.0)))

    for alpha in (0.05, 0.1, 0.25, 1.0):
        assert hopfield.gaussian_prediction(alpha) == pytest.approx(
            phi(-1.0 / sqrt(alpha)), rel=1e-12
        )
    predictions = [hopfield.gaussian_prediction(a) for a in (0.05, 0.1, 0.25)]
    assert predictions[2] > predictions[1] > predictions[0]


def test_async_recall_leaves_zero_field_spins_unchanged():
    J = jnp.zeros((5, 5))
    spins = jnp.array([1.0, -1.0, 1.0, 1.0, -1.0])
    recalled, sweeps, converged = hopfield.async_recall(J, spins, jax.random.PRNGKey(0))
    np.testing.assert_array_equal(np.asarray(recalled), np.asarray(spins))
    assert converged
    assert sweeps == 1


def test_async_recall_keeps_a_single_stored_pattern_fixed():
    key = jax.random.PRNGKey(2)
    pattern = hopfield.random_patterns(key, 1, 50)[0]
    J = hopfield.build_couplings(pattern[None, :])
    recalled, sweeps, converged = hopfield.async_recall(J, pattern, jax.random.PRNGKey(3))
    np.testing.assert_array_equal(np.asarray(recalled), np.asarray(pattern))
    assert converged
    assert sweeps == 1


def test_async_recall_recovers_a_pattern_from_noise():
    pattern = hopfield.random_patterns(jax.random.PRNGKey(4), 1, 100)[0]
    J = hopfield.build_couplings(pattern[None, :])
    corrupted = np.asarray(pattern).copy()
    corrupted[:15] *= -1
    recalled, sweeps, converged = hopfield.async_recall(
        J, jnp.asarray(corrupted), jax.random.PRNGKey(5)
    )
    np.testing.assert_array_equal(np.asarray(recalled), np.asarray(pattern))
    assert converged
    assert sweeps >= 1
    assert hopfield.pattern_overlap(recalled, pattern) == pytest.approx(1.0)


def test_async_recall_is_reproducible_and_returns_spins():
    key = jax.random.PRNGKey(6)
    patterns = hopfield.random_patterns(key, 20, 50)
    J = hopfield.build_couplings(patterns)
    spins = jnp.array(np.asarray(patterns[0]).copy())
    first, _, _ = hopfield.async_recall(J, spins, jax.random.PRNGKey(7))
    second, _, _ = hopfield.async_recall(J, spins, jax.random.PRNGKey(7))
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
    assert set(np.unique(np.asarray(first)).tolist()) <= {-1.0, 1.0}


@pytest.mark.skipif(
    not experiment.RESULTS_PATH.is_file(), reason="run hopfield_experiment.py first"
)
def test_saved_experiment_matches_theory_and_converges_with_N():
    data = json.loads(experiment.RESULTS_PATH.read_text())
    settings = data["settings"]
    assert settings["n_spins"] == [100, 200, 400, 800]
    assert settings["pattern_sets_per_point"] == 20
    assert settings["seed"] == 5315
    assert len(settings["alpha"]) == 21
    assert settings["alpha"][0] == pytest.approx(0.05)
    assert settings["alpha"][-1] == pytest.approx(0.25)

    theory = np.asarray(data["theory"])
    for index, alpha in enumerate(settings["alpha"]):
        assert theory[index] == pytest.approx(
            hopfield.gaussian_prediction(alpha), rel=1e-12
        )

    deviations = {}
    for n_spins in settings["n_spins"]:
        result = data["results"][str(n_spins)]
        means = np.asarray(result["mean"])
        ses = np.asarray(result["se"])
        per_set = np.asarray(result["per_set"])
        assert means.shape == (21,)
        assert ses.shape == (21,)
        assert per_set.shape == (21, 20)
        assert np.isfinite(means).all() and np.isfinite(ses).all()
        assert means[-1] > means[0]  # more load -> more unstable bits
        deviations[n_spins] = float(np.mean(np.abs(means - theory)))

    # Finite-size convergence: deviations shrink monotonically with N.
    assert deviations[800] < deviations[400] < deviations[200] < deviations[100]

    # Close to the Gaussian prediction at alpha = 0.25 (reference regime).
    final_index = len(settings["alpha"]) - 1
    assert data["results"]["800"]["mean"][final_index] == pytest.approx(
        theory[final_index], abs=0.005
    )


@pytest.mark.skipif(
    not experiment.FIGURE_PATH.is_file(), reason="run hopfield_experiment.py first"
)
def test_figure_has_curves_theory_and_log_panel():
    assert experiment.FIGURE_PATH.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    data = json.loads(experiment.RESULTS_PATH.read_text())
    fig = experiment.build_figure(data)
    try:
        assert len(fig.axes) == 2
        linear, logarithmic = fig.axes
        assert logarithmic.get_yscale() == "log"
        for axis in (linear, logarithmic):
            labels = [text.get_text() for text in axis.get_legend().get_texts()]
            for n_spins in (100, 200, 400, 800):
                assert f"N = {n_spins}" in labels
            assert r"$\Phi(-1/\sqrt{\alpha})$" in labels
    finally:
        experiment.plt.close(fig)
