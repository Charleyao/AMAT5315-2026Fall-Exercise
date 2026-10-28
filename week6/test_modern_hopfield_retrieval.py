"""Tests for Week 6 Part 4 VERIFY (2): retrieval with corrected queries.

Checks the corrupted-stored-digit query protocol, modern-vs-classical
retrieval metrics, the blur/retrieval regimes across beta, the gallery and
metrics figures, and the saved artifacts.

Run with::

    cd week6
    uv run python -m pytest -q
"""

import json
from pathlib import Path

import numpy as np
import pytest

import modern_hopfield as mh
import modern_hopfield_retrieval as retrieval

WEEK_DIR = Path(__file__).resolve().parent


def test_run_retrieval_small_case():
    random_lower = mh.random_lower_spins(20)
    result = retrieval.run_retrieval(10, random_lower, n_queries=20)
    assert result["queries"].shape == (20, 784)
    assert result["originals"].shape == (20, 784)
    assert sorted(result["modern"]) == sorted(retrieval.BETAS)
    for beta in retrieval.BETAS:
        item = result["modern"][beta]
        assert len(item["sign_agreement"]["per_query"]) == 20
        assert np.isfinite(item["sign_agreement"]["mean"])
        assert np.isfinite(item["distance_to_mean_pattern"])
    # Tiny load: large beta retrieves stored patterns, small beta is blurred.
    assert result["modern"][retrieval.BETAS[-1]]["exact_match_fraction"] >= 0.8
    assert result["modern"][retrieval.BETAS[0]]["distance_to_mean_pattern"] < 0.02
    assert result["modern"][retrieval.BETAS[-1]]["distance_to_mean_pattern"] > 0.2
    for key in (
        "sign_agreement", "ink_fraction", "pairwise_overlap",
        "unique_states", "exact_match_fraction",
    ):
        assert key in result["classical"]


def test_gallery_layout_has_all_columns_and_digits():
    rng = np.random.default_rng(0)
    patterns = rng.choice([-1.0, 1.0], size=(100, 784))
    labels = np.array([d for d in range(10) for _ in range(10)])
    queries = patterns.copy()
    queries[:, mh.LOWER_START:] = rng.choice([-1.0, 1.0], size=(100, 14 * 28))
    data = {
        "gallery_result": {
            "patterns": patterns,
            "pattern_labels": labels,
            "queries": queries,
            "originals": patterns,
            "classical_outputs": patterns,
            "modern_outputs": {beta: patterns for beta in retrieval.BETAS},
        },
        "gallery_indices": list(range(10)),
    }
    fig = retrieval.build_gallery(data)
    try:
        axes = [axis for axis in fig.axes if axis.images]
        assert len(axes) == 10 * (2 + len(retrieval.BETAS) + 1)
        titles = [text.get_text() for text in fig.texts]
        assert "stored" in titles and "query" in titles and "classical" in titles
        for beta in retrieval.BETAS:
            assert any(f"{beta:g}" in title for title in titles)
    finally:
        retrieval.plt.close(fig)


@pytest.mark.skipif(
    not retrieval.RESULTS_PATH.is_file(), reason="run modern_hopfield_retrieval.py"
)
def test_saved_retrieval_results_verify_requirements():
    data = json.loads(retrieval.RESULTS_PATH.read_text())
    settings = data["settings"]
    assert settings["P"] == [3, 10, 30, 100, 300, 1000]
    assert settings["betas"] == retrieval.BETAS
    assert settings["n_queries"] == 100
    assert settings["query_seed"] == mh.QUERY_SEED
    assert "cycling through P" in settings["query_source"]
    assert settings["gallery_P"] == retrieval.GALLERY_P

    for p in settings["P"]:
        result = data["results"][str(p)]
        modern = result["modern"]
        assert modern[str(retrieval.BETAS[0])]["distance_to_mean_pattern"] < 0.02
        assert modern[str(retrieval.BETAS[-1])]["distance_to_mean_pattern"] > 0.2
        assert modern[str(retrieval.BETAS[-1])]["sign_agreement"]["mean"] > 0.9
        assert modern[str(retrieval.BETAS[-1])]["exact_match_fraction"] >= 0.8
        for beta in retrieval.BETAS:
            assert len(modern[str(beta)]["sign_agreement"]["per_query"]) == 100
            assert np.isfinite(modern[str(beta)]["sign_agreement"]["mean"])
            assert np.isfinite(modern[str(beta)]["overlap"]["mean"])

    gallery = data["results"][str(retrieval.GALLERY_P)]
    classical = gallery["classical"]
    assert classical["ink_fraction"] < 0.5 * gallery["stored_ink_fraction"]
    assert classical["pairwise_overlap"] > 0.99
    assert classical["unique_states"] == 1
    assert classical["exact_match_fraction"] < 0.1
    assert (
        gallery["modern"][str(retrieval.BETAS[-1])]["sign_agreement"]["mean"]
        > classical["sign_agreement"]["mean"] + 0.1
    )

    with np.load(retrieval.ARRAYS_PATH) as arrays:
        assert np.asarray(arrays["gallery_stored"]).shape == (10, 784)
        assert np.asarray(arrays["gallery_query"]).shape == (10, 784)
        assert np.asarray(arrays["gallery_classical"]).shape == (10, 784)
        stored = np.asarray(arrays["gallery_stored"])
        query = np.asarray(arrays["gallery_query"])
        # Correct protocol: upper half preserved, lower half replaced.
        np.testing.assert_array_equal(
            query[:, : mh.LOWER_START], stored[:, : mh.LOWER_START]
        )
        assert not np.array_equal(
            query[:, mh.LOWER_START:], stored[:, mh.LOWER_START:]
        )
        assert set(np.unique(query[:, mh.LOWER_START:]).tolist()) <= {-1.0, 1.0}
        for beta in retrieval.BETAS:
            assert np.asarray(arrays[f"gallery_beta_{beta:g}"]).shape == (10, 784)

    for path in (retrieval.GALLERY_PATH, retrieval.METRICS_PATH):
        assert path.is_file()
        assert path.stat().st_size > 10_000
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.skipif(
    not retrieval.METRICS_PATH.is_file(), reason="run modern_hopfield_retrieval.py"
)
def test_metrics_figure_has_quality_and_blur_panels():
    data = json.loads(retrieval.RESULTS_PATH.read_text())
    figure_data = {"results": data["results"]}
    fig = retrieval.build_metrics_figure(figure_data)
    try:
        assert len(fig.axes) == 2
        for axis in fig.axes:
            assert axis.get_xscale() == "log"
            labels = [text.get_text() for text in axis.get_legend().get_texts()]
            for p in retrieval.P_VALUES:
                assert f"P = {p}" in labels
        assert "sign agreement" in fig.axes[0].get_ylabel()
        assert "average pattern" in fig.axes[1].get_ylabel()
        assert any(
            text.get_text() == "classical" for text in fig.axes[0].get_legend().get_texts()
        )
    finally:
        retrieval.plt.close(fig)
