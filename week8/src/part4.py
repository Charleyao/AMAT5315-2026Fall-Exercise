"""AMAT5315 Week 8, Part 4: attention as a memory built from the text.

Run from the repository root with::

    cd week8 && uv run python -m src.part4
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

from .part4_memory import (
    KEY_DIM,
    METHOD_TEXT_LABELS,
    METHODS,
    M_VALUES,
    PICTURE_SIZE,
    REFERENCE_R,
    SOFTMAX_BETA,
    THEORY_R,
    run_picture_experiment,
)
from .part4_plotting import plot_attention_recall

WEEK_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = WEEK_DIR / "artifacts"
EVIDENCE_DIR = WEEK_DIR / "evidence"

M1024 = 1024
M4096 = 4096
THEORY_TOLERANCE = 0.15
REFERENCE_TOLERANCE = 0.15
SOFTMAX_R_MAX = 0.10
SOFTMAX_CLEAR_COSINE = 0.95


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def run() -> dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    experiment = run_picture_experiment()
    records = experiment["records"]
    m_values = experiment["m_values"]

    # ------------------------------------------------------- VERIFY (1)
    theory_errors = {
        "outer_product": (
            records[M1024]["outer_product"]["R"]
            - THEORY_R[M1024]["outer_product"]
        )
        / THEORY_R[M1024]["outer_product"],
        "relu": (
            records[M1024]["relu"]["R"] - THEORY_R[M1024]["relu"]
        )
        / THEORY_R[M1024]["relu"],
    }
    m4096_reference_errors = {
        method: (
            records[M4096][method]["R"] - REFERENCE_R[M4096][method]
        )
        / REFERENCE_R[M4096][method]
        for method in METHODS
    }
    softmax_r = {
        m: records[m]["softmax"]["R"] for m in m_values
    }
    softmax_cosines = {
        m: records[m]["softmax"]["cosine_similarity"] for m in m_values
    }

    trends_ok = all(
        all(
            records[m_values[index + 1]][method]["R"]
            > records[m_values[index]][method]["R"]
            for index in range(len(m_values) - 1)
        )
        for method in ("outer_product", "relu", "relu_squared")
    )

    checks = {
        "outer_r_m1024_within_15_percent_of_4": (
            abs(theory_errors["outer_product"]) <= THEORY_TOLERANCE
        ),
        "relu_r_m1024_within_15_percent_of_2.83": (
            abs(theory_errors["relu"]) <= THEORY_TOLERANCE
        ),
        "relu_squared_r_gt_1_at_m4096": (
            records[M4096]["relu_squared"]["R"] > 1.0
        ),
        "softmax_r_lt_0.1_at_every_m": all(
            value < SOFTMAX_R_MAX for value in softmax_r.values()
        ),
        "softmax_reconstructs_image": all(
            value >= SOFTMAX_CLEAR_COSINE
            for value in softmax_cosines.values()
        ),
        "non_softmax_r_increases_with_m": trends_ok,
        "m4096_matches_references_within_15_percent": all(
            abs(value) <= REFERENCE_TOLERANCE
            for value in m4096_reference_errors.values()
        ),
    }
    verify1_passed = bool(all(checks.values()))

    verify1 = {
        "m1024_theory": THEORY_R[M1024],
        "m1024_relative_errors": theory_errors,
        "m4096_references": REFERENCE_R[M4096],
        "m4096_relative_errors": m4096_reference_errors,
        "softmax_r": softmax_r,
        "softmax_cosine_similarity": softmax_cosines,
        "softmax_r_max_allowed": SOFTMAX_R_MAX,
        "softmax_clear_cosine_min": SOFTMAX_CLEAR_COSINE,
        "checks": checks,
        "passed": verify1_passed,
    }

    # ----------------------------------------------------------- figure
    figure_path = plot_attention_recall(
        records,
        EVIDENCE_DIR / "figure13_attention_recall.png",
        m_values=tuple(m_values),
    )

    # --------------------------------------------------------- artifacts
    r_table = np.array(
        [
            [records[m][method]["R"] for method in METHODS]
            for m in m_values
        ]
    )
    cosine_table = np.array(
        [
            [records[m][method]["cosine_similarity"] for method in METHODS]
            for m in m_values
        ]
    )
    target_weights = np.array(
        [
            [records[m][method]["target_weight"] for method in METHODS]
            for m in m_values
        ]
    )
    normalized_images = np.array(
        [
            [records[m][method]["normalized_image"] for method in METHODS]
            for m in m_values
        ]
    )
    recall_npz_path = ARTIFACT_DIR / "part4_recall_data.npz"
    np.savez_compressed(
        recall_npz_path,
        character=experiment["character"],
        m_values=np.asarray(m_values),
        methods=np.asarray(METHODS, dtype=str),
        R=r_table,
        cosine=cosine_table,
        target_weight=target_weights,
        normalized_images=normalized_images,
    )

    records_payload = {
        str(m): {
            method: {
                "R": records[m][method]["R"],
                "cosine_similarity": records[m][method][
                    "cosine_similarity"
                ],
                "target_weight": records[m][method]["target_weight"],
                "signal_norm": records[m][method]["signal_norm"],
                "interference_norm": records[m][method][
                    "interference_norm"
                ],
                "recalled_norm": records[m][method]["recalled_norm"],
            }
            for method in METHODS
        }
        for m in m_values
    }
    results = {
        "part": "Week 8 Part 4 - attention: a memory built from the text",
        "character": {
            "shape": [PICTURE_SIZE, PICTURE_SIZE],
            "encoding": "ink = +1, background = -1",
            "ink_pixels": experiment["character_ink_pixels"],
            "substitution": (
                "Learning-sheet glyph unavailable; used an analytically "
                "rasterized block letter 'A'."
            ),
        },
        "settings": {
            "dimension": KEY_DIM,
            "M_values": m_values,
            "methods": list(METHODS),
            "softmax_beta": SOFTMAX_BETA,
            "seed": experiment["seed"],
            "query": "stored character key (memory index 0)",
            "R_definition": (
                "||recalled - target_weight*target_value|| / "
                "||target_weight*target_value||"
            ),
            "visualization_normalization": "recalled / target_weight",
        },
        "records": records_payload,
        "R_table": {
            str(m): {
                method: records[m][method]["R"] for method in METHODS
            }
            for m in m_values
        },
        "verification": {"1": verify1},
        "acceptance": {
            "verify1": verify1_passed,
            "all_passed": verify1_passed,
        },
        "figure": str(figure_path),
        "recall_data": recall_npz_path.name,
    }
    results_path = ARTIFACT_DIR / "part4_results.json"
    results_path.write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )

    settings = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in ("numpy", "matplotlib")
        },
        "picture": {
            "size": PICTURE_SIZE,
            "encoding": {"ink": 1, "background": -1},
            "glyph": "block letter A",
            "substitution_note": (
                "Figure 13 glyph could not be extracted because the learning "
                "sheet PDF is not in the repository; the substitution is "
                "documented here and in the README."
            ),
        },
        "memory": {
            "M_values": m_values,
            "key_dimension": KEY_DIM,
            "seed": experiment["seed"],
            "draw_seed_rule": "seed + M",
            "value_pictures": "target character plus random +/-1 pictures",
            "query": "own key of target memory (index 0)",
        },
        "mechanisms": {
            "outer_product": "weight = key similarity",
            "relu": "weight = max(0, similarity)",
            "relu_squared": "weight = max(0, similarity)^2",
            "softmax": f"weight = softmax({SOFTMAX_BETA:g} * similarity)",
            "softmax_numerical_stability": "subtract maximum score",
        },
        "acceptance": {
            "m1024_theory_tolerance": THEORY_TOLERANCE,
            "m4096_reference_tolerance": REFERENCE_TOLERANCE,
            "softmax_r_max": SOFTMAX_R_MAX,
            "softmax_clear_cosine_min": SOFTMAX_CLEAR_COSINE,
        },
    }
    settings_path = ARTIFACT_DIR / "part4_settings.json"
    settings_path.write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8"
    )

    # ------------------------------------------------------------ report
    lines: list[str] = []
    lines.append(
        "AMAT5315 Week 8 Part 4 - attention: a memory built from the text"
    )
    lines.append("=" * 80)
    lines.append("")
    lines.append("Picture memory")
    lines.append("-" * 80)
    lines.append(
        f"  character: {PICTURE_SIZE}x{PICTURE_SIZE} block letter A, "
        f"{experiment['character_ink_pixels']} ink pixels (+1), background -1"
    )
    lines.append(
        "  note: learning-sheet glyph unavailable; documented substitution"
    )
    lines.append(
        f"  keys: d={KEY_DIM}, M in {m_values}, seed={experiment['seed']} "
        f"(draw seed = seed + M)"
    )
    lines.append(
        "  query: stored character key; normalize recalled vector by target weight"
    )
    lines.append(
        "  R = ||interference|| / ||signal||, "
        "signal = target_weight * target_value"
    )
    lines.append("")
    lines.append("All 16 interference-to-signal ratios")
    lines.append("-" * 80)
    lines.append(
        f"  {'M':>6} " + " ".join(f"{METHOD_TEXT_LABELS[m]:>16}" for m in METHODS)
    )
    for m in m_values:
        values = " ".join(
            f"{records[m][method]['R']:>16.4f}" for method in METHODS
        )
        lines.append(f"  {m:>6} {values}")
    lines.append("")
    lines.append("Softmax reconstruction quality")
    lines.append("-" * 80)
    for m in m_values:
        lines.append(
            f"  M={m:>5}: R={softmax_r[m]:.4f}, "
            f"cosine(normalized recall, target)={softmax_cosines[m]:.6f}"
        )
    lines.append("")
    lines.append("VERIFY (1) - comparison with reference behavior")
    lines.append("-" * 80)
    lines.append(
        f"  M=1024 outer-product: R={records[M1024]['outer_product']['R']:.4f}, "
        f"theory 4.0, relative error "
        f"{theory_errors['outer_product']:+.2%} (limit 15%)"
    )
    lines.append(
        f"  M=1024 ReLU:          R={records[M1024]['relu']['R']:.4f}, "
        f"theory 2.83, relative error "
        f"{theory_errors['relu']:+.2%} (limit 15%)"
    )
    lines.append(
        "  M=4096 measured vs reference:"
    )
    for method in METHODS:
        lines.append(
            f"    {METHOD_TEXT_LABELS[method]:<18} "
            f"R={records[M4096][method]['R']:.4f}, "
            f"reference {REFERENCE_R[M4096][method]:.3f}, "
            f"relative error {m4096_reference_errors[method]:+.2%}"
        )
    lines.append(
        f"  ReLU^2 R > 1 at M=4096: "
        f"{records[M4096]['relu_squared']['R'] > 1.0}"
    )
    lines.append(
        f"  Softmax R < 0.1 at every M: "
        f"{all(value < SOFTMAX_R_MAX for value in softmax_r.values())} "
        f"(max {max(softmax_r.values()):.4f})"
    )
    lines.append(
        f"  Softmax clearly reconstructs image (cosine >= "
        f"{SOFTMAX_CLEAR_COSINE}): "
        f"{all(value >= SOFTMAX_CLEAR_COSINE for value in softmax_cosines.values())}"
    )
    lines.append(
        f"  Non-softmax R increases with M: {trends_ok}"
    )
    for name, value in checks.items():
        lines.append(f"  {name}: {value}")
    lines.append(
        f"  VERIFY (1): {'PASS' if verify1_passed else 'FAIL'}"
    )
    lines.append("")
    lines.append("Outputs")
    lines.append("-" * 80)
    lines.append(f"  Figure 13: {figure_path}")
    lines.append(f"  recall data: {recall_npz_path}")
    lines.append(f"  results: {results_path}")
    lines.append(f"  settings: {settings_path}")
    lines.append("")
    lines.append("Summary")
    lines.append("-" * 80)
    lines.append(
        f"  VERIFY (1): {'PASS' if verify1_passed else 'FAIL'}"
    )

    report = "\n".join(lines) + "\n"
    report_path = EVIDENCE_DIR / "part4_verify.txt"
    report_path.write_text(report, encoding="utf-8")
    print()
    print(report, end="")
    return results


if __name__ == "__main__":
    run()
