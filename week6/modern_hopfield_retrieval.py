"""VERIFY (2): modern Hopfield retrieval vs classical recall on corrupted digits.

Query protocol (learning-sheet version):

* query **100 stored MNIST training digits**, cycling through the ``P`` stored
  patterns;
* keep the upper 14 rows of the stored digit and replace the lower 14 rows
  with random +/-1 spins;
* never use held-out test digits as retrieval targets.

Modern Hopfield retrieval is one update ``X^T softmax(beta X x)`` for each
beta. Classical retrieval uses async zero-temperature recall on the same
patterns and queries. A gallery shows stored digits, corrupted queries,
modern retrievals for every beta and the classical result.

Usage::

    cd week6
    uv run python modern_hopfield_retrieval.py
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")  # headless rendering; must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

import hopfield
import modern_hopfield as mh

WEEK_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = WEEK_DIR / "artifacts"
EVIDENCE_DIR = WEEK_DIR / "evidence"
RESULTS_PATH = ARTIFACTS_DIR / "modern_hopfield_retrieval.json"
ARRAYS_PATH = ARTIFACTS_DIR / "modern_hopfield_retrieval.npz"
GALLERY_PATH = EVIDENCE_DIR / "modern_hopfield_retrieval_gallery.png"
METRICS_PATH = EVIDENCE_DIR / "modern_hopfield_retrieval_metrics.png"

P_VALUES = [3, 10, 30, 100, 300, 1000]
BETAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
N_QUERIES = 100
GALLERY_P = 1000
COLORS = {
    3: "#8c564b", 10: "#17becf", 30: "#bcbd22",
    100: "#1f77b4", 300: "#2ca02c", 1000: "#d62728",
}


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "se": float(values.std(ddof=1) / np.sqrt(len(values))),
        "per_query": values.tolist(),
    }


def run_retrieval(
    p: int,
    random_lower: jax.Array,
    n_queries: int = N_QUERIES,
    betas: list[float] = BETAS,
) -> dict:
    """Modern and classical retrieval metrics for one pattern-set size."""
    patterns, pattern_labels = mh.mnist_spin_patterns(p)
    queries, source_indices, originals = mh.corrupted_stored_queries(
        patterns, random_lower, n_queries
    )
    patterns_np = np.asarray(patterns)
    originals_np = np.asarray(originals)
    mean_pattern = patterns_np.mean(axis=0)
    stored_ink = float((patterns_np > 0).mean())

    modern = {}
    modern_outputs = {}
    for beta in betas:
        updated, _weights = mh.modern_hopfield_update(patterns, queries, beta)
        updated = np.asarray(updated)
        modern_outputs[beta] = updated
        signs = np.sign(updated)
        modern[beta] = {
            "overlap": _summary((updated * originals_np).mean(axis=1)),
            "lower_overlap": _summary(
                (updated[:, mh.LOWER_START:] * originals_np[:, mh.LOWER_START:]).mean(axis=1)
            ),
            "sign_agreement": _summary((signs == originals_np).mean(axis=1)),
            "exact_match_fraction": float((signs == originals_np).all(axis=1).mean()),
            "ink_fraction": float((updated > 0).mean()),
            "distance_to_mean_pattern": float(np.abs(updated - mean_pattern).mean()),
        }

    # Classical Hopfield recall on the same patterns and queries.
    J = hopfield.build_couplings(patterns)
    recalled, sweeps, converged = hopfield.async_recall_batch(
        J, queries, jax.random.fold_in(jax.random.PRNGKey(mh.QUERY_SEED), 1000 + p)
    )
    recalled = np.asarray(recalled)
    recalled_signs = np.sign(recalled)
    pairwise = recalled @ recalled.T / rbm_n_visible()
    upper = np.triu_indices(n_queries, 1)
    classical = {
        "overlap": _summary((recalled * originals_np).mean(axis=1)),
        "lower_overlap": _summary(
            (recalled[:, mh.LOWER_START:] * originals_np[:, mh.LOWER_START:]).mean(axis=1)
        ),
        "sign_agreement": _summary((recalled_signs == originals_np).mean(axis=1)),
        "exact_match_fraction": float((recalled_signs == originals_np).all(axis=1).mean()),
        "ink_fraction": float((recalled > 0).mean()),
        "pairwise_overlap": float(pairwise[upper].mean()),
        "unique_states": int(len(np.unique(recalled > 0, axis=0))),
        "mean_sweeps": float(sweeps),
        "converged_fraction": float(converged),
    }

    return {
        "patterns": patterns,
        "pattern_labels": pattern_labels,
        "queries": queries,
        "source_indices": source_indices,
        "originals": originals,
        "modern": modern,
        "modern_outputs": modern_outputs,
        "classical": classical,
        "classical_outputs": recalled,
        "stored_ink_fraction": stored_ink,
    }


def rbm_n_visible() -> int:
    """Number of visible units (784) without importing the RBM training code."""
    return 28 * 28


def build_gallery(data: dict) -> plt.Figure:
    """Stored / corrupted / modern betas / classical gallery for P=GALLERY_P."""
    result = data["gallery_result"]
    patterns = np.asarray(result["patterns"])
    labels = np.asarray(result["pattern_labels"])
    queries = np.asarray(result["queries"])
    originals = np.asarray(result["originals"])
    classical = np.asarray(result["classical_outputs"])
    modern = result["modern_outputs"]

    row_indices = []
    for digit in range(10):
        candidates = np.where(labels[:N_QUERIES] == digit)[0]
        if candidates.size == 0:
            raise ValueError(f"digit {digit} missing from the first 100 stored patterns")
        row_indices.append(int(candidates[0]))

    columns = [
        ("stored", lambda i: originals[i]),
        ("query", lambda i: queries[i]),
        *[(rf"$\beta$={beta:g}", lambda i, b=beta: modern[b][i]) for beta in BETAS],
        ("classical", lambda i: classical[i]),
    ]
    n_rows, n_cols = len(row_indices), len(columns)
    fig = plt.figure(figsize=(n_cols * 1.15, n_rows * 1.18 + 1.6))
    grid = GridSpec(
        n_rows, n_cols, wspace=0.04, hspace=0.04,
        left=0.045, right=0.995, top=0.92, bottom=0.04,
    )

    for column, (title, getter) in enumerate(columns):
        fig.text(
            (column + 0.5) / n_cols, 0.945, title,
            ha="center", va="bottom", fontsize=8.5,
        )

    for row, index in enumerate(row_indices):
        for column, (_title, getter) in enumerate(columns):
            axis = fig.add_subplot(grid[row, column])
            axis.imshow(
                getter(index).reshape(28, 28),
                cmap="gray", vmin=-1, vmax=1, interpolation="nearest",
            )
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_gid(f"row={row};column={column}")
            if _title == "query":
                axis.axhline(mh.UPPER_ROWS - 0.5, color="#d62728", linewidth=0.5)
        fig.text(0.008, 1 - (row + 0.5) / n_rows * 0.88 - 0.04, f"digit {labels[index]}",
                 rotation=90, va="center", fontsize=8)

    fig.suptitle(
        f"Modern vs classical Hopfield retrieval (P = {GALLERY_P}, corrupted "
        "stored-digit queries; upper rows kept, lower rows random)",
        fontsize=11,
    )
    return fig


def build_metrics_figure(data: dict) -> plt.Figure:
    """Sign agreement and blur distance vs beta, with classical baselines."""
    results = data["results"]
    fig, (ax_agreement, ax_blur) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    def modern_item(result, beta):
        modern = result["modern"]
        return modern[beta] if beta in modern else modern[str(beta)]

    for p in P_VALUES:
        result = results[str(p)]
        agreement = [modern_item(result, beta)["sign_agreement"]["mean"] for beta in BETAS]
        agreement_se = [modern_item(result, beta)["sign_agreement"]["se"] for beta in BETAS]
        distance = [modern_item(result, beta)["distance_to_mean_pattern"] for beta in BETAS]
        ax_agreement.errorbar(
            BETAS, agreement, yerr=agreement_se, color=COLORS[p],
            marker="o", markersize=3, linewidth=1.2, capsize=2, label=f"P = {p}",
        )
        ax_blur.plot(
            BETAS, distance, color=COLORS[p], marker="o", markersize=3,
            linewidth=1.2, label=f"P = {p}",
        )

    classical_agreement = np.mean(
        [results[str(p)]["classical"]["sign_agreement"]["mean"] for p in P_VALUES]
    )
    ax_agreement.axhline(
        classical_agreement, color="black", linestyle="--", linewidth=1.2,
        label="classical",
    )
    for axis in (ax_agreement, ax_blur):
        axis.set_xscale("log")
        axis.set_xticks(BETAS)
        axis.set_xticklabels([f"{beta:g}" for beta in BETAS])
        axis.set_xlabel(r"$\beta$")
        axis.grid(alpha=0.3)
        axis.legend(frameon=False, fontsize=8)
    ax_agreement.set_ylabel("sign agreement with stored digit")
    ax_agreement.set_title("(a) Retrieval quality")
    ax_blur.set_ylabel("mean |output - average pattern|")
    ax_blur.set_title("(b) Blur toward the average digit")
    fig.suptitle("Modern Hopfield retrieval across beta (classical shown for reference)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def save_results(data: dict) -> tuple[Path, Path]:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "settings": {
            "P": P_VALUES,
            "betas": BETAS,
            "n_queries": N_QUERIES,
            "query_seed": mh.QUERY_SEED,
            "query_source": (
                "100 stored MNIST training digits cycling through P; upper 14 "
                "rows kept, lower 14 rows random +/-1"
            ),
            "gallery_P": GALLERY_P,
            "modern_retrieval": "one update X^T softmax(beta X x)",
            "classical_retrieval": "async zero-temperature recall to no-flip sweep",
        },
        "results": data["results"],
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    gallery_result = data["gallery_result"]
    gallery_indices = data["gallery_indices"]
    arrays = {
        "gallery_stored": np.asarray(gallery_result["originals"])[gallery_indices],
        "gallery_query": np.asarray(gallery_result["queries"])[gallery_indices],
        "gallery_classical": np.asarray(gallery_result["classical_outputs"])[gallery_indices],
    }
    for beta in BETAS:
        arrays[f"gallery_beta_{beta:g}"] = np.asarray(
            gallery_result["modern_outputs"][beta]
        )[gallery_indices]
    np.savez_compressed(ARRAYS_PATH, **arrays)
    return RESULTS_PATH, ARRAYS_PATH


def main() -> None:
    random_lower = mh.random_lower_spins(N_QUERIES)
    results = {}
    gallery_result = None
    for p in P_VALUES:
        result = run_retrieval(p, random_lower)
        results[str(p)] = {
            "modern": result["modern"],
            "classical": result["classical"],
            "stored_ink_fraction": result["stored_ink_fraction"],
        }
        if p == GALLERY_P:
            gallery_result = result

    labels = np.asarray(gallery_result["pattern_labels"])
    gallery_indices = [
        int(np.where(labels[:N_QUERIES] == digit)[0][0]) for digit in range(10)
    ]
    data = {
        "results": results,
        "gallery_result": gallery_result,
        "gallery_indices": gallery_indices,
    }

    results_path, arrays_path = save_results(data)

    gallery = build_gallery(data)
    gallery.savefig(GALLERY_PATH, dpi=200, bbox_inches="tight")
    plt.close(gallery)

    metrics_figure = build_metrics_figure(data)
    metrics_figure.savefig(METRICS_PATH, dpi=200, bbox_inches="tight")
    plt.close(metrics_figure)

    print(f"Modern vs classical Hopfield retrieval (P={GALLERY_P} gallery)")
    print("  beta     sign-agreement   exact-match   distance-to-mean")
    gallery_modern = results[str(GALLERY_P)]["modern"]
    for beta in BETAS:
        item = gallery_modern[beta]
        print(
            f"  {beta:<7} {item['sign_agreement']['mean']:.3f}"
            f"+/-{item['sign_agreement']['se']:.3f}      "
            f"{item['exact_match_fraction']:.3f}        "
            f"{item['distance_to_mean_pattern']:.3f}"
        )
    classical = results[str(GALLERY_P)]["classical"]
    print(
        f"  classical: sign agreement {classical['sign_agreement']['mean']:.3f}, "
        f"exact {classical['exact_match_fraction']:.3f}, "
        f"ink {classical['ink_fraction']:.3f} (stored "
        f"{results[str(GALLERY_P)]['stored_ink_fraction']:.3f}), "
        f"pairwise {classical['pairwise_overlap']:.3f}, "
        f"unique states {classical['unique_states']}"
    )

    # --- explicit verification -------------------------------------------
    for p in P_VALUES:
        result = results[str(p)]
        modern = result["modern"]
        assert modern[BETAS[0]]["distance_to_mean_pattern"] < 0.02, p
        assert modern[BETAS[-1]]["distance_to_mean_pattern"] > 0.2, p
        assert modern[BETAS[-1]]["sign_agreement"]["mean"] > 0.9, p
        assert modern[BETAS[-1]]["exact_match_fraction"] >= 0.8, p

    # At high load (the gallery setting) classical recall collapses.
    gallery_metrics = results[str(GALLERY_P)]
    classical = gallery_metrics["classical"]
    modern_final = gallery_metrics["modern"][BETAS[-1]]
    assert classical["ink_fraction"] < 0.5 * gallery_metrics["stored_ink_fraction"]
    assert classical["pairwise_overlap"] > 0.99
    assert classical["unique_states"] == 1
    assert classical["exact_match_fraction"] < 0.1
    assert modern_final["sign_agreement"]["mean"] > classical["sign_agreement"]["mean"] + 0.1
    assert modern_final["exact_match_fraction"] > 0.8

    assert gallery_result is not None
    print("\n  [ok] small beta gives the blurred average pattern")
    print("  [ok] large beta retrieves the stored digit")
    print("  [ok] classical recall collapses to a common, incomplete pattern")
    print("  [ok] query protocol uses stored digits with random lower halves")
    print(f"\n  data:    {results_path.relative_to(WEEK_DIR)}")
    print(f"  arrays:  {arrays_path.relative_to(WEEK_DIR)}")
    print(f"  gallery: {GALLERY_PATH.relative_to(WEEK_DIR)}")
    print(f"  metrics: {METRICS_PATH.relative_to(WEEK_DIR)}")


if __name__ == "__main__":
    main()
