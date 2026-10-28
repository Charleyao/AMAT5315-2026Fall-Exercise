"""VERIFY (3): retrieval success with the exact learning-sheet criterion.

Queries follow the corrected protocol (100 stored digits cycling through the
``P`` stored patterns, upper half kept, lower half random +/-1). For every
query the **target is the nearest stored digit** (largest dot product; ties
resolved by the lowest pattern index).

For each method the retrieved sign pattern ``sign(output)`` is compared with
that target: a query is a success when the normalized overlap is
``>= 0.95``. Success rate is the fraction of the 100 queries that pass — not
the average sign agreement.

Modern Hopfield is evaluated for every beta; classical Hopfield uses async
zero-temperature recall on the same patterns and the identical queries.

Usage::

    cd week6
    uv run python modern_hopfield_success.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering; must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

import modern_hopfield as mh
import modern_hopfield_retrieval as retrieval

WEEK_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = WEEK_DIR / "artifacts"
EVIDENCE_DIR = WEEK_DIR / "evidence"
RESULTS_PATH = ARTIFACTS_DIR / "modern_hopfield_success.json"
FIGURE_PATH = EVIDENCE_DIR / "modern_hopfield_success_vs_P.png"
GALLERY_PATH = EVIDENCE_DIR / "modern_hopfield_success_gallery.png"

P_VALUES = [3, 10, 30, 100, 300, 1000]
BETAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
N_QUERIES = 100
SUCCESS_THRESHOLD = 0.95          # per-query sign-overlap criterion
REQUIRED_SUCCESS_RATE = 0.90      # required success rate at beta = 0.3
GALLERY_P = 100
GALLERY_BETA = 0.3
COLORS = {
    3: "#8c564b", 10: "#17becf", 30: "#bcbd22",
    100: "#1f77b4", 300: "#2ca02c", 1000: "#d62728",
}


def nearest_targets(queries: np.ndarray, patterns: np.ndarray):
    """Nearest stored pattern per query; ties resolved by the lowest index.

    Returns ``(targets, tie_counts, similarities)``.
    """
    similarities = np.asarray(queries) @ np.asarray(patterns).T
    max_similarity = similarities.max(axis=1, keepdims=True)
    tie_counts = (similarities == max_similarity).sum(axis=1) - 1
    targets = np.argmax(similarities, axis=1)
    return targets, tie_counts, similarities


def sign_retrieval_success(
    outputs: np.ndarray,
    targets: np.ndarray,
    patterns: np.ndarray,
    threshold: float = SUCCESS_THRESHOLD,
):
    """Per-query sign overlap with the target and the success mask."""
    signs = np.where(np.asarray(outputs) >= 0.0, 1.0, -1.0)
    overlaps = (signs * np.asarray(patterns)[targets]).mean(axis=1)
    return overlaps, overlaps >= threshold


def run_experiment() -> dict:
    random_lower = mh.random_lower_spins(N_QUERIES)
    results = {}
    gallery = None

    for p in P_VALUES:
        retrieval_result = retrieval.run_retrieval(p, random_lower)
        patterns = np.asarray(retrieval_result["patterns"])
        queries = np.asarray(retrieval_result["queries"])
        targets, tie_counts, similarities = nearest_targets(queries, patterns)
        for query_index in range(N_QUERIES):
            tied = np.flatnonzero(
                similarities[query_index] == similarities[query_index].max()
            )
            assert targets[query_index] == tied[0], "nearest-target tie not lowest index"

        modern = {}
        for beta in BETAS:
            overlaps, success = sign_retrieval_success(
                retrieval_result["modern_outputs"][beta], targets, patterns
            )
            modern[str(beta)] = {
                "rate": float(success.mean()),
                "n_success": int(success.sum()),
                "sign_overlap": overlaps.tolist(),
                "success": success.tolist(),
            }

        classical_overlaps, classical_success = sign_retrieval_success(
            retrieval_result["classical_outputs"], targets, patterns
        )
        classical = {
            "rate": float(classical_success.mean()),
            "n_success": int(classical_success.sum()),
            "sign_overlap": classical_overlaps.tolist(),
            "success": classical_success.tolist(),
        }

        results[str(p)] = {
            "targets": targets.tolist(),
            "tie_counts": tie_counts.tolist(),
            "n_tied_queries": int((tie_counts > 0).sum()),
            "modern": modern,
            "classical": classical,
            "stored_labels": np.asarray(retrieval_result["pattern_labels"])[
                targets
            ].tolist(),
            "source_labels": np.asarray(retrieval_result["pattern_labels"])[
                retrieval_result["source_indices"]
            ].tolist(),
        }
        if p == GALLERY_P:
            gallery = {
                "patterns": patterns,
                "queries": queries,
                "classical": np.asarray(retrieval_result["classical_outputs"]),
                "modern": np.asarray(
                    retrieval_result["modern_outputs"][GALLERY_BETA]
                ),
                "targets": targets,
                "source_indices": np.asarray(retrieval_result["source_indices"]),
                "labels": np.asarray(retrieval_result["pattern_labels"]),
            }

    return {
        "settings": {
            "P": P_VALUES,
            "betas": BETAS,
            "n_queries": N_QUERIES,
            "query_seed": mh.QUERY_SEED,
            "query_protocol": (
                "100 stored MNIST digits cycling through P; upper 14 rows kept, "
                "lower 14 rows random +/-1"
            ),
            "target": "nearest stored pattern by dot product; ties -> lowest index",
            "success_threshold": SUCCESS_THRESHOLD,
            "required_success_rate": REQUIRED_SUCCESS_RATE,
            "success_definition": (
                "fraction of queries whose sign(output) overlap with the nearest "
                "stored target is >= 0.95 (per query, not mean sign agreement)"
            ),
            "gallery_P": GALLERY_P,
            "gallery_beta": GALLERY_BETA,
        },
        "results": results,
        "gallery": gallery,
    }


def build_success_figure(data: dict) -> plt.Figure:
    """Retrieval success vs P for classical and modern (all beta)."""
    fig, axis = plt.subplots(figsize=(8.4, 4.9))
    for beta in BETAS:
        rates = [data["results"][str(p)]["modern"][str(beta)]["rate"] for p in P_VALUES]
        axis.plot(
            P_VALUES, rates, marker="o", markersize=4,
            linewidth=2.2 if beta == GALLERY_BETA else 1.2,
            label=rf"modern $\beta$={beta:g}",
        )
    classical_rates = [data["results"][str(p)]["classical"]["rate"] for p in P_VALUES]
    axis.plot(
        P_VALUES, classical_rates, color="black", linestyle="--",
        marker="x", markersize=6, linewidth=1.6, label="classical",
    )
    axis.axhline(
        REQUIRED_SUCCESS_RATE, color="#777777", linestyle=":", linewidth=1.0,
        label=rf"required rate {REQUIRED_SUCCESS_RATE:.2f} at $\beta$={GALLERY_BETA:g}",
    )
    axis.text(
        0.02, 0.06,
        "0.95 = per-query overlap criterion\n"
        rf"0.90 = required success rate at $\beta$={GALLERY_BETA:g}",
        transform=axis.transAxes, fontsize=8, color="#444444", va="bottom",
    )
    axis.set_xscale("log")
    axis.set_xticks(P_VALUES)
    axis.set_xticklabels([str(p) for p in P_VALUES])
    axis.set_xlabel("P (stored patterns)")
    axis.set_ylabel("retrieval success rate (per-query overlap >= 0.95)")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.3)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    axis.set_title("Retrieval success: nearest stored target, 100 corrupted queries per P")
    fig.tight_layout()
    return fig


def build_gallery(data: dict) -> plt.Figure:
    """P=100 gallery: target, corrupted query, classical, modern beta=0.3."""
    gallery = data["gallery"]
    rows = 10
    columns = [
        ("nearest target", lambda i: gallery["patterns"][gallery["targets"][i]]),
        ("corrupted query", lambda i: gallery["queries"][i]),
        ("classical", lambda i: gallery["classical"][i]),
        (rf"modern $\beta$={GALLERY_BETA:g}", lambda i: gallery["modern"][i]),
    ]
    fig = plt.figure(figsize=(5.0, rows * 1.05 + 1.3))
    grid = GridSpec(
        rows, len(columns), wspace=0.05, hspace=0.05,
        left=0.14, right=0.98, top=0.93, bottom=0.03,
    )
    for column, (title, _getter) in enumerate(columns):
        fig.text(
            (column + 0.5) / len(columns) * 0.84 + 0.14, 0.955, title,
            ha="center", va="bottom", fontsize=8.5,
        )

    for row in range(rows):
        for column, (_title, getter) in enumerate(columns):
            axis = fig.add_subplot(grid[row, column])
            axis.imshow(
                getter(row).reshape(28, 28),
                cmap="gray", vmin=-1, vmax=1, interpolation="nearest",
            )
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_gid(f"row={row};column={column}")
            if column == 1:
                axis.axhline(mh.UPPER_ROWS - 0.5, color="#d62728", linewidth=0.5)
        source = int(gallery["labels"][gallery["source_indices"][row]])
        target = int(gallery["labels"][gallery["targets"][row]])
        fig.text(
            0.012, 1 - (row + 0.5) * (0.90 / rows) - 0.03,
            f"q{row}: src {source} -> tgt {target}",
            rotation=90, va="center", fontsize=7.5,
        )

    fig.suptitle(
        f"Retrieval success gallery (P={GALLERY_P}, beta={GALLERY_BETA:g}); "
        "target = nearest stored digit",
        fontsize=10,
    )
    return fig


def save_results(data: dict) -> Path:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "settings": data["settings"],
        "results": data["results"],
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return RESULTS_PATH


def main() -> None:
    data = run_experiment()
    RESULTS_PATH_OUT = save_results(data)

    figure = build_success_figure(data)
    figure.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)
    gallery = build_gallery(data)
    gallery.savefig(GALLERY_PATH, dpi=200, bbox_inches="tight")
    plt.close(gallery)

    print("Retrieval success: sign overlap >= 0.95 with nearest stored target")
    print(f"{'P':>5} {'classical':>10} " + " ".join(f"b={beta:<5}" for beta in BETAS))
    for p in P_VALUES:
        result = data["results"][str(p)]
        modern = " ".join(
            f"{result['modern'][str(beta)]['rate']:6.3f}" for beta in BETAS
        )
        print(f"{p:>5} {result['classical']['rate']:>10.3f} {modern}")

    print("\nNearest-target ties (resolved to lowest index):")
    for p in P_VALUES:
        result = data["results"][str(p)]
        print(f"  P={p:>4}: {result['n_tied_queries']} tied queries")

    # --- explicit verification -------------------------------------------
    for p in P_VALUES:
        result = data["results"][str(p)]
        assert result["classical"]["rate"] == 0.0, f"P={p}: classical success is not 0"
        rate = result["modern"][str(GALLERY_BETA)]["rate"]
        assert rate >= 0.90, f"P={p}: modern beta=0.3 success {rate} < 0.90"
        for beta in BETAS:
            item = result["modern"][str(beta)]
            success = np.asarray(item["success"])
            overlaps = np.asarray(item["sign_overlap"])
            assert item["rate"] == np.mean(overlaps >= SUCCESS_THRESHOLD)
            assert item["n_success"] == int(success.sum())
        # Ties are resolved to the lowest pattern index.
        for query_index, target in enumerate(result["targets"]):
            assert result["tie_counts"][query_index] >= 0

    print("\n  [ok] modern beta=0.3 success >= 0.90 for every P")
    print("  [ok] classical success = 0 for every P")
    print("  [ok] success is the per-query fraction with overlap >= 0.95")
    print("  [ok] nearest-target ties resolved to the lowest stored index")
    print(f"\n  data:    {RESULTS_PATH_OUT.relative_to(WEEK_DIR)}")
    print(f"  figure:  {FIGURE_PATH.relative_to(WEEK_DIR)}")
    print(f"  gallery: {GALLERY_PATH.relative_to(WEEK_DIR)}")


if __name__ == "__main__":
    main()
