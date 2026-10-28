"""Tests for Week 6 Part 4 VERIFY (3): retrieval success criterion.

Covers nearest-target selection with ties, the per-query success threshold
(not mean sign agreement), the saved success sweep, and the success/gallery
figures.

Run with::

    cd week6
    uv run python -m pytest -q
"""

import json
from pathlib import Path

import numpy as np
import pytest

import modern_hopfield as mh
import modern_hopfield_success as success

WEEK_DIR = Path(__file__).resolve().parent


def test_nearest_targets_resolve_ties_to_lowest_index():
    patterns = np.array([[1, 1, 1], [1, 1, -1]], dtype=np.float32)
    queries = np.array([[1, 1, 0], [1, -1, 1]], dtype=np.float32)
    targets, ties, similarities = success.nearest_targets(queries, patterns)
    np.testing.assert_array_equal(similarities, [[2, 2], [1, -1]])
    np.testing.assert_array_equal(targets, [0, 0])
    np.testing.assert_array_equal(ties, [1, 0])


def test_success_is_per_query_fraction_not_mean_overlap():
    # Two queries with target all-ones and sign overlaps 1.0 and 0.9.
    patterns = np.ones((1, 20), dtype=np.float32)
    targets = np.zeros(2, dtype=int)
    outputs = np.ones((2, 20), dtype=np.float32)
    outputs[1, 0] = -1.0
    overlaps, passed = success.sign_retrieval_success(outputs, targets, patterns)
    np.testing.assert_allclose(overlaps, [1.0, 0.9])
    assert overlaps.mean() == pytest.approx(0.95)
    # Mean overlap is 0.95, but only one of two queries meets the threshold.
    assert passed.tolist() == [True, False]
    assert passed.mean() == pytest.approx(0.5)


def test_saved_success_results_verify_requirements():
    if not success.RESULTS_PATH.is_file():
        pytest.skip("run modern_hopfield_success.py")
    data = json.loads(success.RESULTS_PATH.read_text())
    settings = data["settings"]
    assert settings["P"] == [3, 10, 30, 100, 300, 1000]
    assert settings["betas"] == success.BETAS
    assert settings["n_queries"] == 100
    assert settings["success_threshold"] == 0.95
    assert settings["required_success_rate"] == 0.90
    assert settings["gallery_P"] == 100
    assert settings["gallery_beta"] == 0.3
    assert "per query" in settings["success_definition"]

    for p in settings["P"]:
        result = data["results"][str(p)]
        assert result["classical"]["rate"] == 0.0
        assert result["modern"][str(success.GALLERY_BETA)]["rate"] >= 0.90
        for beta in success.BETAS:
            item = result["modern"][str(beta)]
            overlaps = np.asarray(item["sign_overlap"])
            passed = np.asarray(item["success"], dtype=bool)
            assert overlaps.shape == (100,)
            assert passed.shape == (100,)
            np.testing.assert_array_equal(passed, overlaps >= 0.95)
            assert item["rate"] == pytest.approx(passed.mean())
            assert item["n_success"] == int(passed.sum())
        classical_overlaps = np.asarray(result["classical"]["sign_overlap"])
        np.testing.assert_array_equal(
            np.asarray(result["classical"]["success"], dtype=bool),
            classical_overlaps >= 0.95,
        )

        # Targets really are nearest stored patterns, with lowest-index ties.
        patterns, _labels = mh.mnist_spin_patterns(p)
        queries, _idx, _orig = mh.corrupted_stored_queries(
            patterns, mh.random_lower_spins(100), 100
        )
        similarity = np.asarray(queries @ patterns.T)
        for query_index, target in enumerate(result["targets"]):
            maximum = similarity[query_index].max()
            tied = np.flatnonzero(similarity[query_index] == maximum)
            assert target == tied[0]


def test_success_figure_has_classical_and_all_betas():
    if not success.FIGURE_PATH.is_file():
        pytest.skip("run modern_hopfield_success.py")
    data = json.loads(success.RESULTS_PATH.read_text())
    figure_data = {
        "results": data["results"],
        "settings": data["settings"],
    }
    fig = success.build_success_figure(figure_data)
    try:
        assert len(fig.axes) == 1
        axis = fig.axes[0]
        labels = [text.get_text() for text in axis.get_legend().get_texts()]
        assert any("classical" in label for label in labels)
        assert any("required rate 0.90" in label for label in labels)
        for beta in success.BETAS:
            assert any(f"{beta:g}" in label and "modern" in label for label in labels)
        # A horizontal reference line at the required 0.90 rate.
        assert any(
            len(line.get_ydata()) == 2
            and np.allclose(line.get_ydata(), 0.90)
            for line in axis.lines
        )
        # 0.95 must not be drawn as the success-rate threshold.
        assert not any(
            len(line.get_ydata()) == 2
            and np.allclose(line.get_ydata(), 0.95)
            for line in axis.lines
        )
        assert axis.get_xscale() == "log"
        assert [tick.get_text() for tick in axis.get_xticklabels()] == [
            "3", "10", "30", "100", "300", "1000"
        ]
    finally:
        success.plt.close(fig)


def test_gallery_layout_has_target_query_classical_and_modern():
    rng = np.random.default_rng(0)
    patterns = rng.choice([-1.0, 1.0], size=(100, 784))
    labels = np.arange(100) % 10
    queries = patterns.copy()
    queries[:, mh.LOWER_START:] = rng.choice([-1.0, 1.0], size=(100, 14 * 28))
    gallery = {
        "patterns": patterns,
        "queries": queries,
        "classical": patterns,
        "modern": patterns,
        "targets": np.arange(100) % 10,
        "source_indices": np.arange(100),
        "labels": labels,
    }
    fig = success.build_gallery({"gallery": gallery})
    try:
        axes = [axis for axis in fig.axes if axis.images]
        assert len(axes) == 10 * 4
        titles = [text.get_text() for text in fig.texts]
        assert any("nearest target" in title for title in titles)
        assert any("corrupted query" in title for title in titles)
        assert any("classical" in title for title in titles)
        assert any("modern" in title for title in titles)
    finally:
        success.plt.close(fig)
