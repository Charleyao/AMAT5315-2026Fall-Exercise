"""AMAT5315 Week 8, Part 3: associative memory stores and a previous-token MLP.

Run from the repository root with::

    cd week8 && uv run python -m src.part3
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

from .memory import (
    DIMENSIONS,
    EXPECTED_CAPACITIES_D64,
    EXPECTED_SLOPES,
    MECHANISMS,
    MECHANISM_LABELS,
    M_VALUES,
    capacity_from_curve,
    fit_capacity_slopes,
    local_growth_slopes,
    run_memory_experiment,
)
from .mlp import MLPConfig, train_mlp
from .part2_data import (
    DATA_DIR,
    TINYSTORIES_FILE,
    TINYSTORIES_REPO,
    TINYSTORIES_REVISION,
    TOKENIZER_FILE,
    TOKENIZER_REPO,
    TOKENIZER_REVISION,
    encode_stories,
    load_stories,
    load_tokenizer,
    split_stories,
)
from .part3_plotting import (
    plot_capacity_vs_dimension,
    plot_expected_capacity_table,
    plot_memory_capacity_d64,
    plot_mlp_training,
)

WEEK_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = WEEK_DIR / "artifacts"
EVIDENCE_DIR = WEEK_DIR / "evidence"

BIGRAM_REFERENCE = 3.038
MLP_REFERENCE = 3.041
VERIFY1_OUTER_RANGE = (50.0, 85.0)
VERIFY1_LS_RANGE = (100.0, 170.0)
VERIFY1_LS_R_MAX = 1e-10
CAPACITY_APPROACH_TOLERANCE = 0.10
GROWTH_SLOPE_RANGE = (0.35, 0.65)


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


def _mechanism_values(result: dict, mechanism: str) -> dict:
    return {
        str(d): float(result["capacities"][d][mechanism])
        for d in result["dimensions"]
    }


def run() -> dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------- memory stores
    memory = run_memory_experiment(
        dimensions=DIMENSIONS,
        m_values=M_VALUES,
        draws=3,
        n_queries=256,
        seed=1730,
    )
    dimensions = memory["dimensions"]
    m_values = np.asarray(memory["m_values"], dtype=np.int64)
    ratios_median = memory["ratios_median"]
    capacities = memory["capacities"]
    slopes = fit_capacity_slopes(capacities)
    d64_index = dimensions.index(64)
    capacities_d64 = capacities[64]
    ratios_d64 = ratios_median[d64_index]

    growth_slopes = {}
    for mechanism_index, mechanism in enumerate(MECHANISMS):
        if mechanism == "least_squares":
            growth_slopes[mechanism] = local_growth_slopes(
                ratios_d64[:, mechanism_index], lower=1.0, upper=10.0
            )
        else:
            growth_slopes[mechanism] = local_growth_slopes(
                ratios_d64[:, mechanism_index], lower=0.15, upper=1.0
            )

    # ------------------------------------------------------- VERIFY (1)
    ls_small_m_mask = m_values <= 64
    ls_r_max_small_m = float(
        ratios_d64[ls_small_m_mask, MECHANISMS.index("least_squares")].max()
    )
    p2_relative_error = abs(
        capacities_d64["relu_p2"] - EXPECTED_CAPACITIES_D64["relu_p2"]
    ) / EXPECTED_CAPACITIES_D64["relu_p2"]
    p3_relative_error = abs(
        capacities_d64["relu_p3"] - EXPECTED_CAPACITIES_D64["relu_p3"]
    ) / EXPECTED_CAPACITIES_D64["relu_p3"]

    verify1_checks = {
        "outer_capacity_in_range": (
            VERIFY1_OUTER_RANGE[0]
            <= capacities_d64["outer_product"]
            <= VERIFY1_OUTER_RANGE[1]
        ),
        "least_squares_capacity_in_range": (
            VERIFY1_LS_RANGE[0]
            <= capacities_d64["least_squares"]
            <= VERIFY1_LS_RANGE[1]
        ),
        "least_squares_r_below_1e-10_for_m_le_64": (
            ls_r_max_small_m < VERIFY1_LS_R_MAX
        ),
        "relu_p2_capacity_within_10_percent": (
            p2_relative_error <= CAPACITY_APPROACH_TOLERANCE
        ),
        "relu_p3_capacity_within_10_percent": (
            p3_relative_error <= CAPACITY_APPROACH_TOLERANCE
        ),
        "curves_show_sqrt_growth": all(
            GROWTH_SLOPE_RANGE[0]
            <= value
            <= GROWTH_SLOPE_RANGE[1]
            for value in growth_slopes.values()
        ),
    }
    verify1_passed = bool(all(verify1_checks.values()))
    verify1 = {
        "dimension": 64,
        "capacities": capacities_d64,
        "expected_capacities": EXPECTED_CAPACITIES_D64,
        "least_squares_max_r_for_m_le_64": ls_r_max_small_m,
        "relu_p2_relative_error": p2_relative_error,
        "relu_p3_relative_error": p3_relative_error,
        "growth_slopes": growth_slopes,
        "growth_slope_expected": 0.5,
        "checks": verify1_checks,
        "passed": verify1_passed,
    }

    # ------------------------------------------------------- VERIFY (2)
    verify2_checks = {
        "linear_stores_and_relu_slope_within_0.1_of_1": all(
            abs(slopes[mechanism] - 1.0) <= 0.10
            for mechanism in ("outer_product", "least_squares", "relu_p1")
        ),
        "relu_p2_slope_within_0.15_of_2": (
            abs(slopes["relu_p2"] - 2.0) <= 0.15
        ),
        "relu_p3_slope_above_2.5": slopes["relu_p3"] > 2.5,
    }
    verify2_passed = bool(all(verify2_checks.values()))
    verify2 = {
        "dimensions": dimensions,
        "capacities": {
            str(d): capacities[d] for d in dimensions
        },
        "slopes": slopes,
        "expected_slopes": EXPECTED_SLOPES,
        "checks": verify2_checks,
        "passed": verify2_passed,
    }

    # ------------------------------------------------------- MLP training
    stories = load_stories(DATA_DIR / TINYSTORIES_FILE)
    train_stories, valid_stories = split_stories(stories)
    tokenizer = load_tokenizer(DATA_DIR / TOKENIZER_FILE)
    train = encode_stories(tokenizer, train_stories)
    valid = encode_stories(tokenizer, valid_stories)

    mlp_config = MLPConfig()
    training = train_mlp(
        train.tokens,
        valid.tokens,
        mlp_config,
    )
    mlp_params = training["params"]
    final_mlp_loss = float(training["final_validation_loss"])
    bigram_difference = abs(final_mlp_loss - BIGRAM_REFERENCE)
    mlp_reference_difference = abs(final_mlp_loss - MLP_REFERENCE)

    verify3 = {
        "final_validation_loss": final_mlp_loss,
        "bigram_reference": BIGRAM_REFERENCE,
        "absolute_difference_from_bigram": bigram_difference,
        "mlp_reference": MLP_REFERENCE,
        "absolute_difference_from_mlp_reference": mlp_reference_difference,
        "acceptance_tolerance": 0.01,
        "passed": bool(bigram_difference <= 0.01),
    }

    # ----------------------------------------------------------- figures
    figure9_path = plot_memory_capacity_d64(
        m_values,
        ratios_d64,
        capacities_d64,
        memory["theoretical"][64],
        EVIDENCE_DIR / "figure9_memory_capacity_d64.png",
        d=64,
    )
    figure10_path = plot_capacity_vs_dimension(
        capacities,
        slopes,
        EVIDENCE_DIR / "figure10_capacity_vs_dimension.png",
    )
    cap_table_path = plot_expected_capacity_table(
        capacities_d64,
        EVIDENCE_DIR / "figure_verify1_capacity_bars.png",
    )
    mlp_figure_path = plot_mlp_training(
        training["history"],
        final_mlp_loss,
        BIGRAM_REFERENCE,
        EVIDENCE_DIR / "figure_mlp_training.png",
    )

    # --------------------------------------------------------- artifacts
    memory_npz_path = ARTIFACT_DIR / "part3_memory_curves.npz"
    np.savez_compressed(
        memory_npz_path,
        m_values=m_values,
        dimensions=np.asarray(dimensions),
        mechanisms=np.asarray(MECHANISMS, dtype=str),
        ratios_median=ratios_median,
        ratios_draws=memory["ratios_draws"],
        theoretical=np.stack(
            [
                np.stack(
                    [memory["theoretical"][d][m] for m in MECHANISMS],
                    axis=-1,
                )
                for d in dimensions
            ],
            axis=0,
        ),
        capacities=np.array(
            [
                [capacities[d][m] for m in MECHANISMS]
                for d in dimensions
            ]
        ),
    )

    median_csv_path = ARTIFACT_DIR / "part3_memory_median_R.csv"
    with median_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dimension", "M", "mechanism", "median_R"])
        for d_index, d in enumerate(dimensions):
            for m_index, m in enumerate(m_values):
                for mechanism_index, mechanism in enumerate(MECHANISMS):
                    writer.writerow(
                        [
                            d,
                            int(m),
                            mechanism,
                            f"{ratios_median[d_index, m_index, mechanism_index]:.12g}",
                        ]
                    )

    checkpoint_path = ARTIFACT_DIR / "part3_mlp_checkpoint.npz"
    checkpoint_payload = {
        name: np.asarray(value)
        for name, value in mlp_params.items()
    }
    checkpoint_payload["config_json"] = np.asarray(
        json.dumps(training["config"])
    )
    np.savez_compressed(checkpoint_path, **checkpoint_payload)

    mlp_results_path = ARTIFACT_DIR / "part3_mlp_results.json"
    mlp_results = {
        "config": training["config"],
        "final_validation_loss": final_mlp_loss,
        "bigram_reference": BIGRAM_REFERENCE,
        "absolute_difference_from_bigram": bigram_difference,
        "training_time_seconds": training["training_time_seconds"],
        "validation_tokens": int(len(valid.tokens) - 1),
        "dataset": {
            "train_stories": train.stories,
            "validation_stories": valid.stories,
            "train_tokens": int(len(train.tokens)),
            "validation_tokens_total": int(len(valid.tokens)),
        },
        "history": training["history"],
        "checkpoint": checkpoint_path.name,
    }
    mlp_results_path.write_text(
        json.dumps(mlp_results, indent=2) + "\n", encoding="utf-8"
    )

    memory_payload = {
        str(d): {
            mechanism: [
                float(value)
                for value in ratios_median[dimensions.index(d), :, index]
            ]
            for index, mechanism in enumerate(MECHANISMS)
        }
        for d in dimensions
    }
    results = {
        "part": "Week 8 Part 3 - the MLP: a memory in the weights",
        "memory": {
            "dimensions": dimensions,
            "m_values": [int(m) for m in m_values],
            "mechanisms": list(MECHANISMS),
            "mechanism_labels": MECHANISM_LABELS,
            "draws": memory["draws"],
            "query_samples": memory["n_queries"],
            "seed": memory["seed"],
            "ratio_definition": (
                "R = ||interference|| / ||signal|| = sqrt(energy ratio)"
            ),
            "median_ratios": memory_payload,
            "capacities": {str(d): capacities[d] for d in dimensions},
            "slopes": slopes,
            "growth_slopes_d64": growth_slopes,
            "expected_capacities_d64": EXPECTED_CAPACITIES_D64,
            "expected_slopes": EXPECTED_SLOPES,
        },
        "verification": {
            "1": verify1,
            "2": verify2,
            "3": verify3,
        },
        "mlp": {
            "config": training["config"],
            "final_validation_loss": final_mlp_loss,
            "bigram_reference": BIGRAM_REFERENCE,
            "absolute_difference_from_bigram": bigram_difference,
            "mlp_reference": MLP_REFERENCE,
            "absolute_difference_from_mlp_reference": mlp_reference_difference,
            "training_time_seconds": training["training_time_seconds"],
            "checkpoint": checkpoint_path.name,
        },
        "acceptance": {
            "verify1": verify1_passed,
            "verify2": verify2_passed,
            "verify3": verify3["passed"],
            "all_passed": bool(
                verify1_passed and verify2_passed and verify3["passed"]
            ),
        },
        "figures": {
            "figure9": str(figure9_path),
            "figure10": str(figure10_path),
            "capacity_bars": str(cap_table_path),
            "mlp_training": str(mlp_figure_path),
        },
    }
    results_path = ARTIFACT_DIR / "part3_results.json"
    results_path.write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )

    settings = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in ("numpy", "matplotlib", "jax", "jaxlib", "optax", "sentencepiece")
        },
        "data": {
            "dataset_repo": TINYSTORIES_REPO,
            "dataset_revision": TINYSTORIES_REVISION,
            "dataset_file": TINYSTORIES_FILE,
            "dataset_file_sha256": _file_metadata(
                DATA_DIR / TINYSTORIES_FILE
            )["sha256"],
            "tokenizer_repo": TOKENIZER_REPO,
            "tokenizer_revision": TOKENIZER_REVISION,
            "tokenizer_file": TOKENIZER_FILE,
            "tokenizer_file_sha256": _file_metadata(
                DATA_DIR / TOKENIZER_FILE
            )["sha256"],
            "train_stories": train.stories,
            "validation_stories": valid.stories,
            "train_tokens": int(len(train.tokens)),
            "validation_tokens": int(len(valid.tokens)),
        },
        "memory": {
            "dimensions": dimensions,
            "m_values": [int(m) for m in m_values],
            "draws": memory["draws"],
            "query_samples": memory["n_queries"],
            "seed": memory["seed"],
            "capacity_definition": "M where R crosses 1, log-log interpolation",
            "batch_cell_limit": 8_000_000,
        },
        "mlp": {
            "framework": "JAX CPU",
            "architecture": {
                "vocab_size": mlp_config.vocab_size,
                "embedding_dim": mlp_config.embed_dim,
                "hidden_units": mlp_config.hidden_dim,
                "hidden_activation": "tanh",
                "output_logits": mlp_config.vocab_size,
                "loss": "softmax cross-entropy",
                "input": "previous token only",
            },
            "training": {
                "steps": mlp_config.steps,
                "batch_size": mlp_config.batch_size,
                "optimizer": "AdamW",
                "peak_learning_rate": mlp_config.peak_lr,
                "warmup_steps": mlp_config.warmup_steps,
                "schedule": "linear warmup then cosine decay",
                "weight_decay": mlp_config.weight_decay,
                "seed": mlp_config.seed,
            },
        },
    }
    settings_path = ARTIFACT_DIR / "part3_settings.json"
    settings_path.write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8"
    )

    # ------------------------------------------------------------ report
    lines: list[str] = []
    lines.append("AMAT5315 Week 8 Part 3 - the MLP: a memory in the weights")
    lines.append("=" * 82)
    lines.append("")
    lines.append("Memory experiment")
    lines.append("-" * 82)
    lines.append(
        "  R = ||interference|| / ||signal||; median over 3 independent draws; "
        "capacity at R=1"
    )
    lines.append("")
    lines.append("  d=64 capacities and reference values")
    lines.append(
        f"  {'store':<18} {'capacity':>12} {'expected':>12} {'rel.error':>11}"
    )
    for mechanism in MECHANISMS:
        measured = capacities_d64[mechanism]
        expected = EXPECTED_CAPACITIES_D64[mechanism]
        relative = abs(measured - expected) / expected
        lines.append(
            f"  {MECHANISM_LABELS[mechanism]:<18} {measured:>12.3f} "
            f"{expected:>12.3f} {relative:>10.2%}"
        )
    lines.append("")
    lines.append("  all dimensions")
    lines.append(
        f"  {'store':<18} " + " ".join(f"{'d=' + str(d):>12}" for d in dimensions)
    )
    for mechanism in MECHANISMS:
        values = " ".join(
            f"{capacities[d][mechanism]:>12.3f}" for d in dimensions
        )
        lines.append(f"  {MECHANISM_LABELS[mechanism]:<18} {values}")
    lines.append("")
    lines.append("VERIFY (1) - d=64 memory curves")
    lines.append("-" * 82)
    lines.append(
        f"  outer-product capacity: {capacities_d64['outer_product']:.3f} "
        f"(required {VERIFY1_OUTER_RANGE[0]:.0f}-{VERIFY1_OUTER_RANGE[1]:.0f})"
    )
    lines.append(
        f"  least-squares capacity: {capacities_d64['least_squares']:.3f} "
        f"(required {VERIFY1_LS_RANGE[0]:.0f}-{VERIFY1_LS_RANGE[1]:.0f})"
    )
    lines.append(
        f"  least-squares max R for M<=64: {ls_r_max_small_m:.3e} "
        f"(required < {VERIFY1_LS_R_MAX:.0e})"
    )
    lines.append(
        f"  ReLU^2 capacity: {capacities_d64['relu_p2']:.1f} "
        f"(reference {EXPECTED_CAPACITIES_D64['relu_p2']:.0f}, "
        f"rel.error {p2_relative_error:.2%})"
    )
    lines.append(
        f"  ReLU^3 capacity: {capacities_d64['relu_p3']:.1f} "
        f"(reference {EXPECTED_CAPACITIES_D64['relu_p3']:.0f}, "
        f"rel.error {p3_relative_error:.2%})"
    )
    lines.append(
        "  local log-log growth slopes (expected ~0.5): "
        + ", ".join(
            f"{MECHANISM_LABELS[m]}={growth_slopes[m]:.3f}"
            for m in MECHANISMS
        )
    )
    for name, value in verify1_checks.items():
        lines.append(f"  {name}: {value}")
    lines.append(f"  VERIFY (1): {'PASS' if verify1_passed else 'FAIL'}")
    lines.append("")
    lines.append("VERIFY (2) - capacity scaling with dimension")
    lines.append("-" * 82)
    lines.append(
        f"  {'store':<18} {'slope':>8} {'expected':>10} check"
    )
    for mechanism in MECHANISMS:
        check = ""
        if mechanism in ("outer_product", "least_squares", "relu_p1"):
            check = "within 0.1 of 1"
        elif mechanism == "relu_p2":
            check = "within 0.15 of 2"
        else:
            check = "> 2.5"
        lines.append(
            f"  {MECHANISM_LABELS[mechanism]:<18} {slopes[mechanism]:>8.3f} "
            f"{EXPECTED_SLOPES[mechanism]:>10.2f} {check}"
        )
    for name, value in verify2_checks.items():
        lines.append(f"  {name}: {value}")
    lines.append(f"  VERIFY (2): {'PASS' if verify2_passed else 'FAIL'}")
    lines.append("")
    lines.append("VERIFY (3) - previous-token MLP")
    lines.append("-" * 82)
    lines.append(
        f"  data: {train.stories} train stories, {valid.stories} validation stories"
    )
    lines.append(
        f"  tokens: {len(train.tokens):,} train, {len(valid.tokens):,} validation"
    )
    lines.append(
        f"  architecture: embedding {mlp_config.embed_dim}, hidden "
        f"{mlp_config.hidden_dim} tanh, output {mlp_config.vocab_size}"
    )
    lines.append(
        f"  optimizer: AdamW, {mlp_config.steps} steps, batch "
        f"{mlp_config.batch_size}, peak lr {mlp_config.peak_lr}, "
        f"warmup {mlp_config.warmup_steps}, cosine decay"
    )
    lines.append(
        f"  final validation cross-entropy: {final_mlp_loss:.6f} nats/token"
    )
    lines.append(f"  Part 2 bigram reference: {BIGRAM_REFERENCE:.3f} nats/token")
    lines.append(
        f"  absolute difference: {bigram_difference:.6f} "
        f"(required <= 0.01): {verify3['passed']}"
    )
    lines.append(
        f"  reference MLP value: {MLP_REFERENCE:.3f} "
        f"(difference {mlp_reference_difference:.6f})"
    )
    lines.append(f"  VERIFY (3): {'PASS' if verify3['passed'] else 'FAIL'}")
    lines.append("")
    lines.append("Outputs")
    lines.append("-" * 82)
    lines.append(f"  Figure 9: {figure9_path}")
    lines.append(f"  Figure 10: {figure10_path}")
    lines.append(f"  capacity bars: {cap_table_path}")
    lines.append(f"  MLP training figure: {mlp_figure_path}")
    lines.append(f"  memory curves: {memory_npz_path}")
    lines.append(f"  median R table: {median_csv_path}")
    lines.append(f"  MLP checkpoint: {checkpoint_path}")
    lines.append(f"  MLP results: {mlp_results_path}")
    lines.append(f"  results: {results_path}")
    lines.append(f"  settings: {settings_path}")
    lines.append("")
    lines.append("Summary")
    lines.append("-" * 82)
    for number in (1, 2, 3):
        passed = results["acceptance"][f"verify{number}"]
        lines.append(f"  VERIFY ({number}): {'PASS' if passed else 'FAIL'}")

    report = "\n".join(lines) + "\n"
    report_path = EVIDENCE_DIR / "part3_verify.txt"
    report_path.write_text(report, encoding="utf-8")
    print()
    print(report, end="")
    return results


if __name__ == "__main__":
    run()
