"""AMAT5315 Week 8, Part 1: from tokens to vectors (embeddings).

Reproducible pipeline:

1. Load ``potion-base-8M`` embeddings directly from safetensors (NumPy).
2. Tokenize exactly 27 ordered word pairs across three relations.
3. Normalize word vectors, form offsets ``unit(B) - unit(A)``, and score each
   offset against the mean offset of the other eight pairs.
4. Repeat with a row-shuffled embedding table (seed 1730) as a control.
5. Verify acceptance criteria, write JSON artifacts, a text report, and the
   three-panel Figure 5 reproduction.

Run from the repository root with::

    cd week8 && uv run python -m src.part1
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

from .embeddings import (
    analyze_relation,
    load_embedding_table,
    load_tokenizer,
    shuffled_table,
)
from .plotting import plot_relation_fans
from .relations import RELATION_TITLES, RELATIONS

WEEK_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = WEEK_DIR / "data"
ARTIFACT_DIR = WEEK_DIR / "artifacts"
EVIDENCE_DIR = WEEK_DIR / "evidence"

MODEL_REPO = "minishlab/potion-base-8M"
MODEL_REVISION = "bf8b056"
SHUFFLE_SEED = 1730

EXPECTED_MEANS = {
    "country_capital": 0.47,
    "male_female": 0.55,
    "adjective_comparative": 0.43,
}
PRETRAINED_MIN_MEAN = 0.30
SHUFFLED_MEAN_RANGE = (-0.15, 0.15)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def _file_metadata(path: Path) -> dict:
    return {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _format_relation_report(relation: str, pretrained: dict, shuffled: dict) -> str:
    lines = [
        f"Relation: {relation} -- {RELATION_TITLES[relation]}",
        f"  {'pair':<20} {'token ids':<16} {'cosine':>9}",
    ]
    for pair, cosine in zip(pretrained["pairs"], pretrained["cosines"]):
        ids = f"{pair['token_id_a']}/{pair['token_id_b']}"
        lines.append(f"  {pair['a'] + '-' + pair['b']:<20} {ids:<16} {cosine:>+9.3f}")
    lines.append(
        f"  pretrained mean = {pretrained['mean_cosine']:.4f}  "
        f"(std {pretrained['std_cosine']:.4f})"
    )
    lines.append(f"  shuffled control cosines (same pairs, seed {SHUFFLE_SEED}):")
    lines.append(
        "    " + ", ".join(f"{value:+.3f}" for value in shuffled["cosines"])
    )
    lines.append(f"  shuffled control mean = {shuffled['mean_cosine']:.4f}")
    return "\n".join(lines)


def run() -> dict:
    """Run the full Part 1 analysis and write all outputs."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    model_path = DATA_DIR / "model.safetensors"
    tokenizer_path = DATA_DIR / "tokenizer.json"
    for required in (model_path, tokenizer_path):
        if not required.exists():
            raise FileNotFoundError(
                f"missing {required}; run `uv run python -m src.download_model` first"
            )

    table = load_embedding_table(model_path)
    tokenizer = load_tokenizer(tokenizer_path)
    control_table = shuffled_table(table, seed=SHUFFLE_SEED)

    pretrained = {
        relation: analyze_relation(table, tokenizer, pairs)
        for relation, pairs in RELATIONS.items()
    }
    shuffled = {
        relation: analyze_relation(control_table, tokenizer, pairs)
        for relation, pairs in RELATIONS.items()
    }

    # Every word must map to a single non-special token id inside the table.
    all_token_ids = [
        token_id
        for result in pretrained.values()
        for pair in result["pairs"]
        for token_id in (pair["token_id_a"], pair["token_id_b"])
    ]
    all_token_ids_valid = bool(all_token_ids) and all(
        0 <= token_id < table.shape[0] for token_id in all_token_ids
    )
    pair_count = sum(len(result["pairs"]) for result in pretrained.values())

    pretrained_means = {
        relation: pretrained[relation]["mean_cosine"] for relation in RELATIONS
    }
    shuffled_means = {
        relation: shuffled[relation]["mean_cosine"] for relation in RELATIONS
    }
    pretrained_ok = all(value >= PRETRAINED_MIN_MEAN for value in pretrained_means.values())
    shuffled_ok = all(
        SHUFFLED_MEAN_RANGE[0] <= value <= SHUFFLED_MEAN_RANGE[1]
        for value in shuffled_means.values()
    )
    deltas_from_expected = {
        relation: pretrained_means[relation] - expected
        for relation, expected in EXPECTED_MEANS.items()
    }

    verification = {
        "pair_count": pair_count,
        "expected_pair_count": 27,
        "all_27_pairs_accounted_for": pair_count == 27,
        "all_token_ids_valid": all_token_ids_valid,
        "pretrained_min_mean": PRETRAINED_MIN_MEAN,
        "pretrained_all_means_ge_min": pretrained_ok,
        "shuffled_mean_range": list(SHUFFLED_MEAN_RANGE),
        "shuffled_all_means_in_range": shuffled_ok,
        "expected_means": EXPECTED_MEANS,
        "deltas_from_expected": {
            key: round(float(value), 6) for key, value in deltas_from_expected.items()
        },
        "matches_expected_means_within_0_05": all(
            abs(value) <= 0.05 for value in deltas_from_expected.values()
        ),
        "passed": bool(
            pair_count == 27
            and all_token_ids_valid
            and pretrained_ok
            and shuffled_ok
        ),
    }

    results = {
        "part": "Week 8 Part 1 - From tokens to vectors (embeddings)",
        "model": {
            "repo_id": MODEL_REPO,
            "revision": MODEL_REVISION,
            "embedding_shape": list(table.shape),
            "embedding_dtype": str(table.dtype),
            "files": {
                "model.safetensors": _file_metadata(model_path),
                "tokenizer.json": _file_metadata(tokenizer_path),
            },
        },
        "method": {
            "normalization": "unit-normalize each word embedding before subtraction",
            "offset": "unit(embedding(B)) - unit(embedding(A))",
            "score": (
                "cosine(offset_i, mean(offset_j for j != i)); "
                "leave-one-out over the other eight pairs"
            ),
            "shuffle_control": (
                "random permutation of embedding-table rows with "
                f"numpy.random.default_rng({SHUFFLE_SEED})"
            ),
        },
        "relations": {
            relation: {
                "title": RELATION_TITLES[relation],
                "pairs": pretrained[relation]["pairs"],
                "cosines": pretrained[relation]["cosines"],
                "mean_cosine": pretrained[relation]["mean_cosine"],
                "std_cosine": pretrained[relation]["std_cosine"],
                "shuffled_control": {
                    "pairs": [
                        {
                            "a": pair["a"],
                            "b": pair["b"],
                            "cosine": cosine,
                        }
                        for pair, cosine in zip(
                            shuffled[relation]["pairs"],
                            shuffled[relation]["cosines"],
                        )
                    ],
                    "cosines": shuffled[relation]["cosines"],
                    "mean_cosine": shuffled[relation]["mean_cosine"],
                    "std_cosine": shuffled[relation]["std_cosine"],
                },
            }
            for relation in RELATIONS
        },
        "verification": verification,
    }

    settings = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in ("numpy", "matplotlib", "safetensors", "tokenizers")
        },
        "model": {
            "repo_id": MODEL_REPO,
            "revision": MODEL_REVISION,
            "source_urls": [
                f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}",
                f"https://hf-mirror.com/{MODEL_REPO}/resolve/{MODEL_REVISION}",
            ],
            "files": ["model.safetensors", "tokenizer.json"],
        },
        "shuffle_seed": SHUFFLE_SEED,
        "relations": {
            relation: [f"{a}-{b}" for a, b in pairs]
            for relation, pairs in RELATIONS.items()
        },
        "normalization": "L2 unit length per word vector (float64 accumulation)",
        "acceptance_criteria": {
            "pretrained_mean_cosine_min": PRETRAINED_MIN_MEAN,
            "shuffled_mean_cosine_range": list(SHUFFLED_MEAN_RANGE),
        },
    }

    results_path = ARTIFACT_DIR / "part1_results.json"
    settings_path = ARTIFACT_DIR / "part1_settings.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")

    figure_path = plot_relation_fans(
        pretrained,
        shuffled,
        EVIDENCE_DIR / "figure5_word_relations.png",
    )

    report_lines = [
        "AMAT5315 Week 8 Part 1 - embedding relation verification",
        "=" * 68,
        f"model: {MODEL_REPO} @ {MODEL_REVISION}",
        f"embedding table: {table.shape[0]} x {table.shape[1]} ({table.dtype})",
        f"embedding files: {', '.join(sorted(p.name for p in (model_path, tokenizer_path)))}",
        f"shuffle control seed: {SHUFFLE_SEED}",
        f"token pairs checked: {pair_count}",
        "",
    ]
    for relation in RELATIONS:
        report_lines.append(_format_relation_report(
            relation, pretrained[relation], shuffled[relation]
        ))
        report_lines.append("")

    report_lines.extend(
        [
            "Acceptance checks",
            "-" * 68,
            f"  all 27 pairs accounted for: {verification['all_27_pairs_accounted_for']}",
            f"  all token ids valid:         {verification['all_token_ids_valid']}",
            f"  pretrained means >= {PRETRAINED_MIN_MEAN:.2f}:    "
            f"{verification['pretrained_all_means_ge_min']}",
            f"  shuffled means in [-0.15, 0.15]: "
            f"{verification['shuffled_all_means_in_range']}",
            f"  expected means reproduced (|delta| <= 0.05): "
            f"{verification['matches_expected_means_within_0_05']}",
            "",
            f"PART 1 VERIFY (1): {'PASS' if verification['passed'] else 'FAIL'}",
            "",
            f"figure: {figure_path}",
            f"results: {results_path}",
            f"settings: {settings_path}",
        ]
    )
    report = "\n".join(report_lines) + "\n"
    (EVIDENCE_DIR / "part1_verify.txt").write_text(report, encoding="utf-8")

    print(report, end="")
    return results


if __name__ == "__main__":
    run()
