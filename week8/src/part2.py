"""AMAT5315 Week 8, Part 2: count models, n-grams, and four verification tasks.

Run from the repository root with::

    cd week8 && uv run python -m src.part2
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

from .chi_square import pooled_chi_square
from .ngram import DEFAULT_SMOOTHING_GRID, NGramModel, verify_bigram_normalization
from .part2_data import (
    DATA_DIR,
    EXPECTED_TRAIN_STORIES,
    EXPECTED_VALID_STORIES,
    EXPECTED_VOCAB_SIZE,
    TINYSTORIES_FILE,
    TINYSTORIES_REPO,
    TINYSTORIES_REVISION,
    TOKENIZER_FILE,
    TOKENIZER_REPO,
    TOKENIZER_REVISION,
    TRAIN_FRACTION,
    encode_stories,
    load_stories,
    load_tokenizer,
    split_stories,
    tokenizer_statistics,
)
from .part2_plotting import (
    plot_chi_square_diagnostics,
    plot_curse_of_dimensionality,
)

WEEK_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = WEEK_DIR / "artifacts"
EVIDENCE_DIR = WEEK_DIR / "evidence"

SMOOTHING_GRID = list(DEFAULT_SMOOTHING_GRID)
ORDERS = list(range(1, 7))
SAMPLING_SEED = 1730
SAMPLED_TOKENS = 250_000
TOP_CONTEXTS = 20
MIN_EXPECTED = 5.0

REFERENCE_VERIFY2 = {
    "uniform": {"nats": 6.238, "ppl": 512.0, "bpc": 4.386},
    "unigram": {"nats": 4.839, "ppl": 126.4, "bpc": 3.403},
    "bigram": {"nats": 3.038, "ppl": 20.9, "bpc": 2.136},
    "trigram": {"nats": 2.094, "ppl": 8.12, "bpc": 1.473},
}

REFERENCE_VERIFY4 = {
    3: {"bpc": 1.473, "unseen": 0.006},
    4: {"bpc": 1.310, "unseen": 0.042},
    5: {"bpc": 1.46, "unseen": 0.122},
    6: {"bpc": 1.79, "unseen": 0.235},
}


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_metadata(path: Path) -> dict:
    return {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def _select_smoothing(nats_by_k: list[float], grid: list[float]) -> tuple[float, int]:
    """Choose the k with the lowest selection loss; ties go to the smaller k."""
    best_index = min(
        range(len(grid)), key=lambda index: (nats_by_k[index], grid[index])
    )
    return grid[best_index], best_index


def _short_piece(piece: str) -> str:
    return piece or "∅"


def run() -> dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = DATA_DIR / TINYSTORIES_FILE
    tokenizer_path = DATA_DIR / TOKENIZER_FILE
    for required in (dataset_path, tokenizer_path):
        if not required.exists():
            raise FileNotFoundError(
                f"missing {required}; run `uv run python -m src.download_part2_data`"
            )

    # ---------------------------------------------------------------- data
    stories = load_stories(dataset_path)
    train_stories, valid_stories = split_stories(stories, TRAIN_FRACTION)
    tokenizer = load_tokenizer(tokenizer_path)
    train = encode_stories(tokenizer, train_stories)
    valid = encode_stories(tokenizer, valid_stories)
    tokenizer_stats = tokenizer_statistics(tokenizer, train, valid)

    split_correct = (
        train.stories == EXPECTED_TRAIN_STORIES
        and valid.stories == EXPECTED_VALID_STORIES
        and len(stories) == EXPECTED_TRAIN_STORIES + EXPECTED_VALID_STORIES
    )
    tokenizer_ok = (
        tokenizer_stats["vocab_size"] == EXPECTED_VOCAB_SIZE
        and tokenizer_stats["normalized_roundtrip"] == tokenizer_stats["stories"]
        and tokenizer_stats["train_unk"] == 0
        and tokenizer_stats["valid_unk"] == 0
        and abs(tokenizer_stats["overall_chars_per_token"] - 2.05) <= 0.01
    )

    vocab_size = tokenizer_stats["vocab_size"]
    valid_chars_per_token = tokenizer_stats["valid_chars_per_token"]

    uniform_metrics = {
        "nats_per_token": float(np.log(vocab_size)),
        "perplexity": float(vocab_size),
        "bits_per_char": float(
            np.log(vocab_size) / (np.log(2.0) * valid_chars_per_token)
        ),
        "scored_tokens": int(len(valid.tokens)),
        "unseen_fraction": 0.0,
        "smoothing": None,
    }

    # ------------------------------------------- smoothing on training tail
    selection_cut = int(len(train.tokens) * 0.9)
    selection_targets = len(train.tokens) - selection_cut
    model_records: dict[int, dict] = {}
    bigram_model: NGramModel | None = None

    for order in ORDERS:
        selection_model = NGramModel(order, vocab_size=vocab_size).fit(
            train.tokens[:selection_cut]
        )
        counts, denominators, seen = selection_model.score_counts(
            train.tokens, target_start=selection_cut
        )
        selection_nats = [
            NGramModel.metrics_from_counts(
                counts,
                denominators,
                seen,
                smoothing,
                vocab_size=vocab_size,
            )["nats_per_token"]
            for smoothing in SMOOTHING_GRID
        ]
        chosen_k, chosen_index = _select_smoothing(selection_nats, SMOOTHING_GRID)

        full_model = NGramModel(order, vocab_size=vocab_size).fit(train.tokens)
        metrics = full_model.evaluate(
            valid.tokens,
            chosen_k,
            char_count=valid.chars,
        )
        model_records[order] = {
            "order": order,
            "context_length": order - 1,
            "chosen_k": chosen_k,
            "selection_losses": [
                {"k": float(k), "nats_per_token": float(value)}
                for k, value in zip(SMOOTHING_GRID, selection_nats)
            ],
            "chosen_selection_nats": float(selection_nats[chosen_index]),
            "joint_states": int(len(full_model.joint_keys)),
            "context_states": int(len(full_model.context_keys)),
            "metrics": metrics,
        }
        if order == 2:
            bigram_model = full_model
        print(
            f"[fit ] n={order}: k={chosen_k:g}, "
            f"held-out {metrics['nats_per_token']:.4f} nats/token, "
            f"bpc={metrics['bits_per_char']:.4f}, "
            f"unseen={metrics['unseen_fraction']:.4%}"
        )

    assert bigram_model is not None

    # ------------------------------------------------------- VERIFY (1)
    chosen_bigram_k = model_records[2]["chosen_k"]
    conditional = bigram_model.conditional_matrix(chosen_bigram_k)
    verify1 = verify_bigram_normalization(conditional)
    verify1_by_k = {}
    for smoothing in SMOOTHING_GRID:
        check = verify_bigram_normalization(
            bigram_model.conditional_matrix(smoothing)
        )
        verify1_by_k[f"{smoothing:g}"] = {
            "max_error": check["max_error"],
            "sums": check["sums"],
        }
    verify1["by_k"] = verify1_by_k
    verify1["chosen_k"] = chosen_bigram_k
    verify1["passed"] = bool(verify1["max_error"] <= 1e-12)

    # ------------------------------------------------------- VERIFY (2)
    actual_by_name = {
        "uniform": uniform_metrics,
        "unigram": model_records[1]["metrics"],
        "bigram": model_records[2]["metrics"],
        "trigram": model_records[3]["metrics"],
    }
    verify2_rows = []
    for name in ("uniform", "unigram", "bigram", "trigram"):
        actual = actual_by_name[name]
        reference = REFERENCE_VERIFY2[name]
        nats_diff = actual["nats_per_token"] - reference["nats"]
        ppl_relative_diff = (
            abs(actual["perplexity"] - reference["ppl"]) / reference["ppl"]
        )
        bpc_diff = actual["bits_per_char"] - reference["bpc"]
        verify2_rows.append(
            {
                "model": name,
                "actual_nats": actual["nats_per_token"],
                "reference_nats": reference["nats"],
                "nats_diff": nats_diff,
                "actual_ppl": actual["perplexity"],
                "reference_ppl": reference["ppl"],
                "ppl_relative_diff": ppl_relative_diff,
                "actual_bpc": actual["bits_per_char"],
                "reference_bpc": reference["bpc"],
                "bpc_diff": bpc_diff,
                "loss_ok": abs(nats_diff) <= 0.03,
                "ppl_ok": ppl_relative_diff <= 0.03,
            }
        )
    verify2_passed = bool(
        all(row["loss_ok"] and row["ppl_ok"] for row in verify2_rows)
        and split_correct
        and tokenizer_ok
    )
    verify2 = {
        "rows": verify2_rows,
        "split_correct": split_correct,
        "tokenizer_ok": tokenizer_ok,
        "loss_tolerance": 0.03,
        "perplexity_relative_tolerance": 0.03,
        "passed": verify2_passed,
    }

    # ------------------------------------------------------- VERIFY (3)
    token_frequencies = np.bincount(train.tokens, minlength=vocab_size)
    top_context_ids = np.argsort(-token_frequencies, kind="stable")[:TOP_CONTEXTS]
    samples = bigram_model.sample_bigram(
        SAMPLED_TOKENS,
        chosen_bigram_k,
        seed=SAMPLING_SEED,
    )
    previous = samples[:-1]
    following = samples[1:]
    row_lookup = np.full(vocab_size, -1, dtype=np.int64)
    row_lookup[top_context_ids] = np.arange(TOP_CONTEXTS)
    sample_rows = row_lookup[previous]
    selected = sample_rows >= 0
    observed = np.zeros((TOP_CONTEXTS, vocab_size), dtype=np.int64)
    np.add.at(
        observed,
        (sample_rows[selected], following[selected]),
        1,
    )

    context_observations = observed.sum(axis=1)
    expected = (
        context_observations[:, None] * conditional[top_context_ids]
    )
    chi_result = pooled_chi_square(observed, expected, min_expected=MIN_EXPECTED)
    per_row_statistic = np.array(
        [detail["statistic"] for detail in chi_result["row_details"]]
    )
    context_pieces = [
        _short_piece(tokenizer.id_to_piece(int(token_id)))
        for token_id in top_context_ids
    ]
    verify3 = {
        "sampled_tokens": SAMPLED_TOKENS,
        "sampling_seed": SAMPLING_SEED,
        "top_contexts": [
            {
                "token_id": int(token_id),
                "piece": piece,
                "train_frequency": int(token_frequencies[token_id]),
                "sampled_occurrences": int(count),
            }
            for token_id, piece, count in zip(
                top_context_ids, context_pieces, context_observations
            )
        ],
        "transitions_tested": int(observed.sum()),
        "chi_squared": chi_result["statistic"],
        "dof": chi_result["dof"],
        "p_value": chi_result["p_value"],
        "min_expected": chi_result["min_expected"],
        "all_individual_expected_ge_min": chi_result[
            "all_individual_expected_ge_min"
        ],
        "grouping": (
            "within each context, next-token categories with expected "
            f"count < {MIN_EXPECTED:g} pooled into one other category"
        ),
        "passed": bool(chi_result["p_value"] > 0.001),
    }

    chi_table_path = ARTIFACT_DIR / "part2_chi2_table.npz"
    np.savez_compressed(
        chi_table_path,
        top_context_ids=top_context_ids,
        context_pieces=np.array(context_pieces, dtype=str),
        per_row_statistic=per_row_statistic,
        observed=observed,
        expected=expected,
        context_observations=context_observations,
    )

    chi_figure_path = plot_chi_square_diagnostics(
        observed,
        expected,
        per_row_statistic,
        context_pieces,
        chi_result,
        EVIDENCE_DIR / "figure_verify3_bigram_chi2.png",
    )

    # ------------------------------------------------------- VERIFY (4)
    verify4_rows = []
    bpc_values = {
        order: model_records[order]["metrics"]["bits_per_char"]
        for order in ORDERS
    }
    unseen_values = {
        order: model_records[order]["metrics"]["unseen_fraction"]
        for order in ORDERS
    }
    minimum_order = min(ORDERS, key=lambda order: bpc_values[order])
    for order, reference in REFERENCE_VERIFY4.items():
        bpc_diff = bpc_values[order] - reference["bpc"]
        unseen_diff = unseen_values[order] - reference["unseen"]
        verify4_rows.append(
            {
                "order": order,
                "bits_per_char": bpc_values[order],
                "reference_bpc": reference["bpc"],
                "bpc_diff": bpc_diff,
                "unseen_fraction": unseen_values[order],
                "reference_unseen": reference["unseen"],
                "unseen_diff": unseen_diff,
                "loss_ok": abs(bpc_diff) <= 0.03,
                "unseen_ok": abs(unseen_diff) <= 0.02,
            }
        )
    verify4_passed = bool(
        minimum_order == 4
        and all(row["loss_ok"] and row["unseen_ok"] for row in verify4_rows)
    )
    verify4 = {
        "rows": verify4_rows,
        "minimum_order": minimum_order,
        "loss_tolerance": 0.03,
        "unseen_tolerance": 0.02,
        "passed": verify4_passed,
    }

    figure7_path = plot_curse_of_dimensionality(
        {
            order: {
                "bits_per_char": bpc_values[order],
                "unseen_fraction": unseen_values[order],
            }
            for order in ORDERS
        },
        EVIDENCE_DIR / "figure7_curse_of_dimensionality.png",
    )

    # ------------------------------------------------------------ artifacts
    results = {
        "part": "Week 8 Part 2 - count models: Markov chains and n-grams",
        "dataset": {
            "repo_id": TINYSTORIES_REPO,
            "revision": TINYSTORIES_REVISION,
            "file": _file_metadata(dataset_path),
            "train_stories": train.stories,
            "valid_stories": valid.stories,
            "total_stories": len(stories),
            "train_tokens": len(train.tokens),
            "valid_tokens": len(valid.tokens),
            "train_chars": train.chars,
            "valid_chars": valid.chars,
            "train_fraction": TRAIN_FRACTION,
            "split": "first 90% stories train, last 10% stories validation",
            "shuffled": False,
        },
        "tokenizer": {
            "repo_id": TOKENIZER_REPO,
            "revision": TOKENIZER_REVISION,
            "file": _file_metadata(tokenizer_path),
            **tokenizer_stats,
            "eos_inserted_between_stories": True,
            "roundtrip_note": (
                "exact decode equals input for stories without repeated "
                "whitespace; after whitespace normalization all stories match"
            ),
        },
        "training": {
            "smoothing_grid": SMOOTHING_GRID,
            "selection": "last 10% of training tokens",
            "selection_cut_token_index": selection_cut,
            "selection_targets": selection_targets,
            "orders": ORDERS,
            "models": {
                str(order): model_records[order] for order in ORDERS
            },
        },
        "verification": {
            "1": {
                **verify1,
                "description": (
                    "bigram probabilities summed over all sequences of "
                    "lengths 1, 2, and 3; the first token is uniform"
                ),
            },
            "2": verify2,
            "3": verify3,
            "4": verify4,
        },
        "acceptance": {
            "verify1": verify1["passed"],
            "verify2": verify2["passed"],
            "verify3": verify3["passed"],
            "verify4": verify4["passed"],
            "all_passed": bool(
                verify1["passed"]
                and verify2["passed"]
                and verify3["passed"]
                and verify4["passed"]
            ),
        },
        "figures": {
            "figure7": str(figure7_path),
            "verify3_chi2": str(chi_figure_path),
        },
    }

    settings = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in ("numpy", "matplotlib", "sentencepiece", "scipy")
        },
        "dataset": {
            "repo_id": TINYSTORIES_REPO,
            "revision": TINYSTORIES_REVISION,
            "file": TINYSTORIES_FILE,
            "delimiter": "<|endoftext|>",
            "train_fraction": TRAIN_FRACTION,
            "shuffle": False,
        },
        "tokenizer": {
            "repo_id": TOKENIZER_REPO,
            "revision": TOKENIZER_REVISION,
            "path": f"stories260K/{TOKENIZER_FILE}",
            "eos_inserted_after_each_story": True,
        },
        "models": {
            "orders": ORDERS,
            "vocab_size": vocab_size,
            "add_k_formula": "(count(context,b)+k)/(count(context)+k*512)",
            "smoothing_grid": SMOOTHING_GRID,
            "smoothing_selection": "last 10% of training token stream",
        },
        "sampling": {
            "tokens": SAMPLED_TOKENS,
            "seed": SAMPLING_SEED,
            "initial_token": "uniform over vocabulary",
            "top_contexts": TOP_CONTEXTS,
            "chi_square_min_expected": MIN_EXPECTED,
        },
        "acceptance": {
            "verify2_loss_tolerance_nats": 0.03,
            "verify2_perplexity_relative_tolerance": 0.03,
            "verify3_p_value_min": 0.001,
            "verify4_bpc_tolerance": 0.03,
            "verify4_unseen_tolerance": 0.02,
        },
    }

    results_path = ARTIFACT_DIR / "part2_results.json"
    settings_path = ARTIFACT_DIR / "part2_settings.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")

    # --------------------------------------------------------------- report
    lines: list[str] = []
    lines.append("AMAT5315 Week 8 Part 2 - count models: Markov chains and n-grams")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Dataset")
    lines.append("-" * 78)
    lines.append(f"  source: {TINYSTORIES_REPO} @ {TINYSTORIES_REVISION}")
    lines.append(f"  file: {TINYSTORIES_FILE}")
    lines.append(f"  stories: {len(stories)} total, {train.stories} train, {valid.stories} validation")
    lines.append(f"  split: first 90% / last 10%, no shuffling")
    lines.append(f"  tokens: {len(train.tokens):,} train, {len(valid.tokens):,} validation")
    lines.append(f"  characters: {train.chars:,} train, {valid.chars:,} validation")
    lines.append(f"  exact split: {split_correct}")
    lines.append("")
    lines.append("Tokenizer")
    lines.append("-" * 78)
    lines.append(f"  source: {TOKENIZER_REPO} @ {TOKENIZER_REVISION}, stories260K/{TOKENIZER_FILE}")
    lines.append(f"  vocabulary size: {tokenizer_stats['vocab_size']}")
    lines.append(f"  EOS id: {tokenizer_stats['eos_id']}, UNK id: {tokenizer_stats['unk_id']}")
    lines.append(f"  EOS inserted after every story: train {train.eos_count:,}, validation {valid.eos_count:,}")
    lines.append(
        f"  average characters/token: overall {tokenizer_stats['overall_chars_per_token']:.6f}, "
        f"train {tokenizer_stats['train_chars_per_token']:.6f}, "
        f"validation {tokenizer_stats['valid_chars_per_token']:.6f}"
    )
    lines.append(
        f"  round-trip: exact {tokenizer_stats['raw_roundtrip']:,}/{tokenizer_stats['stories']:,}, "
        f"after whitespace normalization "
        f"{tokenizer_stats['normalized_roundtrip']:,}/{tokenizer_stats['stories']:,}"
    )
    lines.append(
        f"  UNK tokens: train {tokenizer_stats['train_unk']}, validation {tokenizer_stats['valid_unk']}"
    )
    lines.append(f"  tokenizer checks: {tokenizer_ok}")
    lines.append("")
    lines.append("Smoothing selection (last 10% of training tokens)")
    lines.append("-" * 78)
    for order in ORDERS:
        record = model_records[order]
        losses = ", ".join(
            f"{item['k']:g}:{item['nats_per_token']:.4f}"
            for item in record["selection_losses"]
        )
        lines.append(
            f"  n={order}: selected k={record['chosen_k']:g} "
            f"(selection nats/token {record['chosen_selection_nats']:.4f}); "
            f"grid [{losses}]"
        )
    lines.append("")
    lines.append("Held-out metrics")
    lines.append("-" * 78)
    lines.append(
        f"  {'model':<10} {'k':>8} {'nats/token':>12} {'perplexity':>12} {'bits/char':>11} {'unseen':>9}"
    )
    lines.append(
        f"  {'uniform':<10} {'-':>8} {uniform_metrics['nats_per_token']:>12.4f} "
        f"{uniform_metrics['perplexity']:>12.4f} {uniform_metrics['bits_per_char']:>11.4f} "
        f"{uniform_metrics['unseen_fraction']:>8.2%}"
    )
    for order in ORDERS:
        metrics = model_records[order]["metrics"]
        lines.append(
            f"  {'n=' + str(order):<10} {model_records[order]['chosen_k']:>8g} "
            f"{metrics['nats_per_token']:>12.4f} {metrics['perplexity']:>12.4f} "
            f"{metrics['bits_per_char']:>11.4f} {metrics['unseen_fraction']:>8.2%}"
        )
    lines.append("")
    lines.append("VERIFY (1) - bigram normalization over all sequences of lengths 1, 2, 3")
    lines.append("-" * 78)
    for name, value in verify1["sums"].items():
        lines.append(f"  sum({name}) = {value:.17f}, error = {verify1['errors'][name]:.3e}")
    lines.append(f"  maximum conditional row-sum error: {verify1['max_conditional_row_sum_error']:.3e}")
    lines.append(f"  maximum sequence-sum error: {verify1['max_error']:.3e} (limit 1e-12)")
    lines.append(f"  allocation: {verify1['allocated_cells']} cells for length-3 contraction")
    lines.append(f"  initial token distribution: uniform over {vocab_size} tokens")
    lines.append(f"  VERIFY (1): {'PASS' if verify1['passed'] else 'FAIL'}")
    lines.append("")
    lines.append("VERIFY (2) - held-out comparison with reference values")
    lines.append("-" * 78)
    lines.append(
        f"  {'model':<10} {'nats':>8} {'ref':>8} {'diff':>8} {'ppl':>9} {'ref':>8} {'rel.diff':>9} "
        f"{'bpc':>8} {'ref':>7} {'bpc diff':>9} status"
    )
    for row in verify2_rows:
        status = "ok" if row["loss_ok"] and row["ppl_ok"] else "FAIL"
        lines.append(
            f"  {row['model']:<10} {row['actual_nats']:>8.4f} {row['reference_nats']:>8.3f} "
            f"{row['nats_diff']:>+8.4f} {row['actual_ppl']:>9.3f} {row['reference_ppl']:>8.3f} "
            f"{row['ppl_relative_diff']:>9.2%} {row['actual_bpc']:>8.4f} "
            f"{row['reference_bpc']:>7.3f} {row['bpc_diff']:>+9.4f} {status}"
        )
    lines.append(
        f"  split correct: {split_correct}; tokenizer checks: {tokenizer_ok}; "
        f"perplexity within 3%: {all(row['ppl_ok'] for row in verify2_rows)}"
    )
    lines.append(f"  VERIFY (2): {'PASS' if verify2['passed'] else 'FAIL'}")
    lines.append("")
    lines.append("VERIFY (3) - ancestral sampling from the bigram model")
    lines.append("-" * 78)
    lines.append(
        f"  sampled tokens: {SAMPLED_TOKENS:,} (seed {SAMPLING_SEED}); "
        f"transitions after the 20 most frequent contexts: {verify3['transitions_tested']:,}"
    )
    lines.append(
        f"  top-context occurrences: min {context_observations.min():,}, "
        f"max {context_observations.max():,}"
    )
    lines.append(
        f"  pooled chi-squared: {verify3['chi_squared']:.4f}, dof: {verify3['dof']}, "
        f"p-value: {verify3['p_value']:.6f}"
    )
    lines.append(
        f"  expected-count grouping: {verify3['grouping']}; "
        f"all kept expected counts >= {MIN_EXPECTED:g}: "
        f"{verify3['all_individual_expected_ge_min']}"
    )
    lines.append(f"  acceptance p > 0.001: {verify3['passed']}")
    lines.append(f"  VERIFY (3): {'PASS' if verify3['passed'] else 'FAIL'}")
    lines.append("")
    lines.append("VERIFY (4) - curse of dimensionality")
    lines.append("-" * 78)
    lines.append(
        f"  {'n':>2} {'bits/char':>10} {'ref':>7} {'diff':>9} {'unseen':>9} {'ref':>8} {'diff (pp)':>10} status"
    )
    for row in verify4_rows:
        status = "ok" if row["loss_ok"] and row["unseen_ok"] else "FAIL"
        lines.append(
            f"  {row['order']:>2} {row['bits_per_char']:>10.4f} {row['reference_bpc']:>7.3f} "
            f"{row['bpc_diff']:>+9.4f} {row['unseen_fraction']:>8.3%} "
            f"{row['reference_unseen']:>8.2%} {100.0 * row['unseen_diff']:>+9.2f} {status}"
        )
    lines.append(f"  minimum held-out bits/char at n={minimum_order} (required: 4)")
    lines.append(f"  VERIFY (4): {'PASS' if verify4['passed'] else 'FAIL'}")
    lines.append("")
    lines.append("Outputs")
    lines.append("-" * 78)
    lines.append(f"  figure 7: {figure7_path}")
    lines.append(f"  verify 3 figure: {chi_figure_path}")
    lines.append(f"  results: {results_path}")
    lines.append(f"  settings: {settings_path}")
    lines.append(f"  chi-square table: {chi_table_path}")
    lines.append("")
    lines.append("Summary")
    lines.append("-" * 78)
    for number in (1, 2, 3, 4):
        passed = results["acceptance"][f"verify{number}"]
        lines.append(f"  VERIFY ({number}): {'PASS' if passed else 'FAIL'}")

    report = "\n".join(lines) + "\n"
    (EVIDENCE_DIR / "part2_verify.txt").write_text(report, encoding="utf-8")
    print()
    print(report, end="")
    return results


if __name__ == "__main__":
    run()
