"""AMAT5315 Week 8, Part 5: the Transformer - training and generating.

Run from the repository root with::

    cd week8 && uv run python -m src.part5
"""

from __future__ import annotations

import json
import os
import platform
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .ngram import NGramModel
from .part2_data import (
    DATA_DIR,
    TINYSTORIES_FILE,
    TOKENIZER_FILE,
    encode_stories,
    load_stories,
    load_tokenizer,
    split_stories,
)
from .part5_generation import (
    T1_SETTING,
    generate_scored_sequences,
    generate_transformer_tokens,
    get_context_distribution,
    z_score_from_sequences,
)
from .part5_plotting import (
    plot_context_length,
    plot_model_bpc_comparison,
    plot_text_samples,
    plot_transformer_training,
    plot_verify4,
)
from .sampling import (
    SAMPLING_SETTINGS,
    apply_truncation,
    decode_token_stream,
    grouped_chi_square,
    ngram_next_distribution,
    sample_indices,
    temperature_probs,
)
from .transformer import (
    CONTEXT_LENGTH,
    TransformerConfig,
    forward,
    load_checkpoint,
    train_transformer,
)

WEEK_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = WEEK_DIR / "artifacts"
EVIDENCE_DIR = WEEK_DIR / "evidence"

GENERATION_SEED = 1848
VERIFY4_SEQUENCES = 128
VERIFY4_TOKENS = 128
VERIFY3_SAMPLES = 100_000
BIGRAM_REFERENCE = 3.038
TRIGRAM_REFERENCE = 2.094
BPC_THRESHOLD = 1.310
TRANSFORMER_BPC_REFERENCE = 1.207
FINAL8_REFERENCE = 1.65
AFTER32_THRESHOLD = 2.09
FALLBACK_PART2_BPC = {
    "Uniform": 4.386,
    "Unigram": 3.403,
    "Bigram": 2.136,
    "Trigram": 1.473,
}


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def _sanitize_text(text: str) -> str:
    cleaned = []
    for character in text:
        if character == "\ufffd":
            cleaned.append("?")
        elif character.isprintable() or character in "\n\t":
            cleaned.append(character)
        else:
            cleaned.append(" ")
    return "".join(cleaned).strip()


def _part2_bpc_values() -> dict[str, float]:
    path = ARTIFACT_DIR / "part2_results.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = data["verification"]["2"]["rows"]
            loaded = {
                row["model"].capitalize(): float(row["actual_bpc"])
                for row in rows
            }
            if set(loaded) == set(FALLBACK_PART2_BPC):
                return loaded
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
    return dict(FALLBACK_PART2_BPC)


def _part2_smoothing() -> dict[int, float]:
    path = ARTIFACT_DIR / "part2_results.json"
    defaults = {1: 0.0001, 2: 0.01, 3: 0.01}
    if not path.exists():
        return defaults
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        models = data["training"]["models"]
        return {
            int(order): float(models[str(order)]["chosen_k"])
            for order in defaults
        }
    except (KeyError, ValueError, json.JSONDecodeError):
        return defaults


def _sample_uniform(n_tokens: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 512, size=n_tokens, dtype=np.int32).tolist()


def _sample_ngram(
    model: NGramModel,
    prefix: list[int],
    n_tokens: int,
    seed: int,
    smoothing: float,
) -> list[int]:
    rng = np.random.default_rng(seed)
    context = list(map(int, prefix))
    generated = []
    for _ in range(n_tokens):
        probabilities = ngram_next_distribution(model, context, smoothing)
        token = int(sample_indices(probabilities, 1, rng)[0])
        generated.append(token)
        context.append(token)
    return generated


def _find_broad_context(
    params: dict,
    valid_tokens: np.ndarray,
    batch_size: int = 128,
) -> dict:
    """Find the 64-token validation context with maximum predictive entropy."""
    n_windows = len(valid_tokens) // CONTEXT_LENGTH
    windows = valid_tokens[: n_windows * CONTEXT_LENGTH].reshape(
        n_windows, CONTEXT_LENGTH
    )
    forward_jit = jax.jit(forward)
    best_entropy = -1.0
    best_index = 0
    for start in range(0, n_windows, batch_size):
        block = windows[start : start + batch_size]
        logits = np.asarray(
            forward_jit(params, jnp.asarray(block, dtype=jnp.int32)),
            dtype=np.float64,
        )
        last = logits[:, -1, :]
        shifted = last - last.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        entropies = -np.sum(
            probabilities * np.log(probabilities + 1e-300), axis=1
        )
        index = int(np.argmax(entropies))
        if entropies[index] > best_entropy:
            best_entropy = float(entropies[index])
            best_index = start + index
    return {
        "index": int(best_index),
        "tokens": windows[best_index].astype(int).tolist(),
        "entropy": best_entropy,
    }


def run() -> dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------- data
    stories = load_stories(DATA_DIR / TINYSTORIES_FILE)
    train_stories, valid_stories = split_stories(stories)
    tokenizer = load_tokenizer(DATA_DIR / TOKENIZER_FILE)
    train = encode_stories(tokenizer, train_stories)
    valid = encode_stories(tokenizer, valid_stories)
    valid_characters = valid.chars

    # ------------------------------------------------------- training
    checkpoint_path = ARTIFACT_DIR / "part5_transformer_checkpoint.npz"
    saved_results_path = ARTIFACT_DIR / "part5_transformer_results.json"
    reuse_checkpoint = (
        os.environ.get("PART5_REUSE_CHECKPOINT") == "1"
        and checkpoint_path.exists()
        and saved_results_path.exists()
    )

    if reuse_checkpoint:
        saved = json.loads(saved_results_path.read_text(encoding="utf-8"))
        params, config = load_checkpoint(checkpoint_path)
        position_loss = np.asarray(saved["position_loss"], dtype=np.float64)
        final_eval = {
            "position_loss": position_loss,
            "overall_loss": float(saved["final_validation_loss"]),
            "final8_mean": float(saved["final8_mean_loss"]),
            "after32_mean": float(saved["after32_mean_loss"]),
            "windows": int(saved["validation_windows"]),
            "scored_tokens": int(saved["scored_tokens"]),
        }
        training = {
            "params": params,
            "config": saved["config"],
            "parameter_count": int(saved["parameter_count"]),
            "final_evaluation": final_eval,
            "final_validation_loss": float(saved["final_validation_loss"]),
            "final_bits_per_char": float(saved["final_bits_per_char"]),
            "training_time_seconds": float(saved["training_time_seconds"]),
            "history": saved["history"],
        }
        print("[trf ] reused saved checkpoint; skipped retraining")
    else:
        config = TransformerConfig()
        training = train_transformer(
            train.tokens,
            valid.tokens,
            valid_characters,
            config,
        )
    params = training["params"]
    final_eval = training["final_evaluation"]
    final_loss = training["final_validation_loss"]
    final_bpc = training["final_bits_per_char"]
    history = training["history"]

    # Save the model immediately so later verification failures cannot lose it.
    checkpoint_path = ARTIFACT_DIR / "part5_transformer_checkpoint.npz"
    early_checkpoint = {}
    for name, value in params.items():
        if name == "layers":
            for layer_index, layer in enumerate(value):
                for parameter_name, parameter in layer.items():
                    early_checkpoint[
                        f"layer{layer_index}_{parameter_name}"
                    ] = np.asarray(jax.device_get(parameter))
        else:
            early_checkpoint[name] = np.asarray(jax.device_get(value))
    early_checkpoint["config_json"] = np.asarray(
        json.dumps(training["config"])
    )
    np.savez_compressed(checkpoint_path, **early_checkpoint)

    # ------------------------------------------------------- VERIFY (1)
    part2_bpc = _part2_bpc_values()
    model_names = ["Uniform", "Unigram", "Bigram", "Trigram", "Transformer"]
    bpc_values = [
        part2_bpc["Uniform"],
        part2_bpc["Unigram"],
        part2_bpc["Bigram"],
        part2_bpc["Trigram"],
        final_bpc,
    ]
    verify1_passed = bool(final_bpc < BPC_THRESHOLD)
    bpc_figure_path = plot_model_bpc_comparison(
        model_names,
        bpc_values,
        BPC_THRESHOLD,
        EVIDENCE_DIR / "figure_verify1_bits_per_char.png",
    )

    # Fit Part 2 n-gram models for temperature-1 story samples.
    smoothing = _part2_smoothing()
    ngram_models = {
        order: NGramModel(order, vocab_size=512).fit(train.tokens)
        for order in (1, 2, 3)
    }
    eos = tokenizer.eos_id()
    # A trigram starting from [EOS, EOS] has an unseen context and falls back
    # to a uniform first token. Seed it with EOS plus the most common token
    # that follows EOS in training, so the opening starts from a real context.
    eos_positions = np.flatnonzero(train.tokens[:-1] == eos)
    start_token = int(
        np.bincount(
            train.tokens[eos_positions + 1], minlength=512
        ).argmax()
    )
    uniform_tokens = _sample_uniform(60, GENERATION_SEED)
    unigram_tokens = _sample_ngram(
        ngram_models[1], [], 60, GENERATION_SEED, smoothing[1]
    )
    bigram_tokens = _sample_ngram(
        ngram_models[2], [eos], 60, GENERATION_SEED, smoothing[2]
    )
    trigram_tokens = _sample_ngram(
        ngram_models[3],
        [eos, start_token],
        60,
        GENERATION_SEED,
        smoothing[3],
    )
    transformer_tokens = generate_transformer_tokens(
        params, [eos], 60, T1_SETTING, GENERATION_SEED
    )
    story_samples = [
        ("Uniform", uniform_tokens),
        ("Unigram", unigram_tokens),
        ("Bigram", bigram_tokens),
        ("Trigram", trigram_tokens),
        ("Transformer", transformer_tokens),
    ]
    sample_records = [
        {
            "model": name,
            "tokens": list(map(int, tokens)),
            "text": _sanitize_text(decode_token_stream(tokenizer, tokens)),
        }
        for name, tokens in story_samples
    ]
    samples_json_path = ARTIFACT_DIR / "part5_samples_temperature1.json"
    samples_json_path.write_text(
        json.dumps(
            {
                "seed": GENERATION_SEED,
                "temperature": 1.0,
                "n_tokens": 60,
                "samples_in_quality_order": sample_records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    samples_text_path = ARTIFACT_DIR / "part5_samples_temperature1.txt"
    samples_text_path.write_text(
        "\n\n".join(
            f"[{sample['model']}]\n{sample['text']}"
            for sample in sample_records
        )
        + "\n",
        encoding="utf-8",
    )
    samples_figure_path = plot_text_samples(
        [
            {"label": sample["model"], "text": sample["text"]}
            for sample in sample_records
        ],
        EVIDENCE_DIR / "figure_verify1_model_samples.png",
        "Temperature-1 story samples (same seed 1848), increasing quality",
    )

    # ------------------------------------------------------- VERIFY (2)
    position_loss = final_eval["position_loss"]
    after32_mean = float(position_loss[31:].mean())
    final8_mean = float(position_loss[-8:].mean())
    verify2_checks = {
        "mean_loss_after_32_tokens_below_2.09": after32_mean < AFTER32_THRESHOLD,
        "final8_close_to_reference": abs(final8_mean - FINAL8_REFERENCE) <= 0.15,
    }
    verify2_passed = bool(all(verify2_checks.values()))
    context_figure_path = plot_context_length(
        position_loss,
        BIGRAM_REFERENCE,
        TRIGRAM_REFERENCE,
        EVIDENCE_DIR / "figure17_context_length.png",
    )
    training_figure_path = plot_transformer_training(
        history,
        EVIDENCE_DIR / "figure_transformer_training.png",
    )

    # ------------------------------------------------------- VERIFY (3)
    broad_context = _find_broad_context(params, valid.tokens)
    chi_results = []
    for index, (name, setting) in enumerate(SAMPLING_SETTINGS.items()):
        probabilities = get_context_distribution(
            params, broad_context["tokens"], setting
        )
        rng = np.random.default_rng(GENERATION_SEED + index)
        sampled = sample_indices(probabilities, VERIFY3_SAMPLES, rng)
        observed = np.bincount(sampled, minlength=512)
        expected = VERIFY3_SAMPLES * probabilities
        chi = grouped_chi_square(observed, expected, min_expected=5.0)
        chi_results.append(
            {
                "name": name,
                "label": setting["label"],
                "temperature": setting["temperature"],
                "p_value": chi["p_value"],
                "statistic": chi["statistic"],
                "dof": chi["dof"],
                "categories_kept": chi["categories_kept"],
                "rare_categories_pooled": chi["rare_categories_pooled"],
                "sampled_tokens": VERIFY3_SAMPLES,
                "distribution_max": float(probabilities.max()),
                "distribution_support": int(
                    np.count_nonzero(probabilities > 0)
                ),
                "passed": bool(chi["p_value"] > 0.001),
            }
        )
    tiny_setting = {
        "kind": "temperature",
        "temperature": 0.0001,
        "label": "Temperature T=0.0001",
    }
    tiny_probs = get_context_distribution(
        params, broad_context["tokens"], tiny_setting
    )
    tiny_argmax = int(np.argmax(tiny_probs))
    tiny_samples = sample_indices(
        tiny_probs, 1000, np.random.default_rng(GENERATION_SEED + 10)
    )
    tiny_all_argmax = bool(np.all(tiny_samples == tiny_argmax))
    tiny_result = {
        "temperature": 0.0001,
        "argmax_token": tiny_argmax,
        "max_probability": float(tiny_probs.max()),
        "samples_tested": 1000,
        "all_samples_argmax": tiny_all_argmax,
        "passed": bool(tiny_all_argmax and tiny_probs.max() > 1.0 - 1e-9),
    }
    verify3_passed = bool(
        all(result["passed"] for result in chi_results)
        and tiny_result["passed"]
    )

    sampling_examples = []
    for name, setting in SAMPLING_SETTINGS.items():
        tokens = generate_transformer_tokens(
            params, [eos], 60, setting, GENERATION_SEED
        )
        sampling_examples.append(
            {
                "name": name,
                "label": setting["label"],
                "tokens": list(map(int, tokens)),
                "text": _sanitize_text(
                    decode_token_stream(tokenizer, tokens)
                ),
            }
        )
    sampling_examples_path = ARTIFACT_DIR / "part5_sampling_method_examples.json"
    sampling_examples_path.write_text(
        json.dumps(sampling_examples, indent=2) + "\n", encoding="utf-8"
    )
    sampling_examples_figure = plot_text_samples(
        [
            {"label": example["label"], "text": example["text"]}
            for example in sampling_examples
        ],
        EVIDENCE_DIR / "figure_sampling_method_examples.png",
        "Transformer examples: four sampling settings (seed 1848)",
    )

    # ------------------------------------------------------- VERIFY (4)
    prefixes = valid.tokens[
        : VERIFY4_SEQUENCES * CONTEXT_LENGTH
    ].reshape(VERIFY4_SEQUENCES, CONTEXT_LENGTH)
    result_t1 = generate_scored_sequences(
        params, prefixes, VERIFY4_TOKENS, 1.0, GENERATION_SEED
    )
    result_t07 = generate_scored_sequences(
        params, prefixes, VERIFY4_TOKENS, 0.7, GENERATION_SEED + 1
    )
    stats_t1 = z_score_from_sequences(result_t1)
    stats_t07 = z_score_from_sequences(result_t07)
    verify4_checks = {
        "t1_z_in_interval": -4.0 < stats_t1["z_score"] < 4.0,
        "t07_abs_z_gt_40": abs(stats_t07["z_score"]) > 40.0,
    }
    verify4_passed = bool(all(verify4_checks.values()))
    verify4_npz_path = ARTIFACT_DIR / "part5_verify4_sequences.npz"
    np.savez_compressed(
        verify4_npz_path,
        t1_surprise=result_t1["surprise"],
        t1_entropy=result_t1["entropy"],
        t07_surprise=result_t07["surprise"],
        t07_entropy=result_t07["entropy"],
    )
    verify4_figure_path = plot_verify4(
        result_t1["surprise"].mean(axis=1),
        result_t1["entropy"].mean(axis=1),
        result_t07["surprise"].mean(axis=1),
        result_t07["entropy"].mean(axis=1),
        stats_t1["z_score"],
        stats_t07["z_score"],
        EVIDENCE_DIR / "figure_verify4_surprise_entropy.png",
    )

    # --------------------------------------------------------- artifacts
    checkpoint_path = ARTIFACT_DIR / "part5_transformer_checkpoint.npz"
    checkpoint_payload = {}
    for name, value in params.items():
        if name == "layers":
            for layer_index, layer in enumerate(value):
                for parameter_name, parameter in layer.items():
                    checkpoint_payload[
                        f"layer{layer_index}_{parameter_name}"
                    ] = np.asarray(jax.device_get(parameter))
        else:
            checkpoint_payload[name] = np.asarray(jax.device_get(value))
    checkpoint_payload["config_json"] = np.asarray(
        json.dumps(training["config"])
    )
    np.savez_compressed(checkpoint_path, **checkpoint_payload)

    transformer_results = {
        "config": training["config"],
        "parameter_count": training["parameter_count"],
        "final_validation_loss": final_loss,
        "final_bits_per_char": final_bpc,
        "final8_mean_loss": final8_mean,
        "after32_mean_loss": after32_mean,
        "position_loss": [float(value) for value in position_loss],
        "training_time_seconds": training["training_time_seconds"],
        "history": history,
        "checkpoint": checkpoint_path.name,
        "validation_windows": final_eval["windows"],
        "scored_tokens": final_eval["scored_tokens"],
    }
    transformer_results_path = ARTIFACT_DIR / "part5_transformer_results.json"
    transformer_results_path.write_text(
        json.dumps(transformer_results, indent=2) + "\n", encoding="utf-8"
    )

    results = {
        "part": "Week 8 Part 5 - the Transformer: training and generating",
        "model": {
            "parameter_count": training["parameter_count"],
            "final_validation_loss": final_loss,
            "final_bits_per_char": final_bpc,
            "final8_mean_loss": final8_mean,
            "after32_mean_loss": after32_mean,
            "checkpoint": checkpoint_path.name,
        },
        "verification": {
            "1": {
                "models": model_names,
                "bits_per_char": bpc_values,
                "transformer_bpc": final_bpc,
                "bpc_threshold": BPC_THRESHOLD,
                "transformer_reference_bpc": TRANSFORMER_BPC_REFERENCE,
                "samples": sample_records,
                "trigram_seed_context": [int(eos), int(start_token)],
                "trigram_seed_piece": tokenizer.id_to_piece(start_token),
                "passed": verify1_passed,
            },
            "2": {
                "position_loss": [float(value) for value in position_loss],
                "after32_mean": after32_mean,
                "final8_mean": final8_mean,
                "final8_reference": FINAL8_REFERENCE,
                "bigram_reference": BIGRAM_REFERENCE,
                "trigram_reference": TRIGRAM_REFERENCE,
                "checks": verify2_checks,
                "passed": verify2_passed,
            },
            "3": {
                "broad_context": {
                    "index": broad_context["index"],
                    "tokens": broad_context["tokens"],
                    "text": _sanitize_text(
                        decode_token_stream(
                            tokenizer, broad_context["tokens"]
                        )
                    ),
                    "entropy": broad_context["entropy"],
                },
                "chi_square": chi_results,
                "tiny_temperature": tiny_result,
                "passed": verify3_passed,
            },
            "4": {
                "sequences": VERIFY4_SEQUENCES,
                "tokens_per_sequence": VERIFY4_TOKENS,
                "temperature_1": stats_t1,
                "temperature_0_7": stats_t07,
                "checks": verify4_checks,
                "passed": verify4_passed,
            },
        },
        "acceptance": {
            "verify1": verify1_passed,
            "verify2": verify2_passed,
            "verify3": verify3_passed,
            "verify4": verify4_passed,
            "all_passed": bool(
                verify1_passed
                and verify2_passed
                and verify3_passed
                and verify4_passed
            ),
        },
        "figures": {
            "bpc_comparison": str(bpc_figure_path),
            "model_samples": str(samples_figure_path),
            "context_length": str(context_figure_path),
            "training": str(training_figure_path),
            "sampling_examples": str(sampling_examples_figure),
            "verify4": str(verify4_figure_path),
        },
        "artifacts": {
            "samples_temperature1": str(samples_json_path),
            "samples_temperature1_text": str(samples_text_path),
            "sampling_method_examples": str(sampling_examples_path),
            "verify4_sequences": str(verify4_npz_path),
            "checkpoint": str(checkpoint_path),
            "transformer_results": str(transformer_results_path),
        },
    }
    results_path = ARTIFACT_DIR / "part5_results.json"
    results_path.write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )

    settings = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in ("numpy", "matplotlib", "jax", "jaxlib", "optax")
        },
        "data": {
            "dataset": TINYSTORIES_FILE,
            "tokenizer": TOKENIZER_FILE,
            "train_tokens": int(len(train.tokens)),
            "validation_tokens": int(len(valid.tokens)),
            "train_stories": train.stories,
            "validation_stories": valid.stories,
        },
        "architecture": {
            "type": "pre-norm decoder-only Transformer",
            "vocab_size": config.vocab_size,
            "layers": config.n_layers,
            "d_model": config.d_model,
            "heads": config.n_heads,
            "head_dim": config.head_dim,
            "context_length": config.context_length,
            "mlp_hidden": config.mlp_hidden,
            "mlp_activation": "GELU",
            "linear_bias": True,
            "causal_masking": True,
            "position_embeddings": "learned",
            "parameter_count": training["parameter_count"],
        },
        "training": {
            "steps": config.steps,
            "batch_sequences": config.batch_size,
            "tokens_per_example": config.context_length + 1,
            "optimizer": "AdamW",
            "peak_lr": config.peak_lr,
            "warmup_steps": config.warmup_steps,
            "cosine_end_lr": config.peak_lr * config.end_lr_fraction,
            "weight_decay": config.weight_decay,
            "seed": config.seed,
        },
        "generation": {
            "seed": GENERATION_SEED,
            "temperature_1_samples": 60,
            "sampling_settings": SAMPLING_SETTINGS,
        },
        "verification": {
            "bpc_threshold": BPC_THRESHOLD,
            "after32_threshold": AFTER32_THRESHOLD,
            "chi_square_p_min": 0.001,
            "verify4_sequences": VERIFY4_SEQUENCES,
            "verify4_tokens": VERIFY4_TOKENS,
        },
    }
    settings_path = ARTIFACT_DIR / "part5_settings.json"
    settings_path.write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8"
    )

    # ------------------------------------------------------------ report
    lines = [
        "AMAT5315 Week 8 Part 5 - the Transformer: training and generating",
        "=" * 84,
        "",
        "Model",
        "-" * 84,
        f"  parameter count: {training['parameter_count']:,}",
        f"  architecture: {config.n_layers} layers, d_model={config.d_model}, "
        f"{config.n_heads} heads, head_dim={config.head_dim}, "
        f"context={config.context_length}, MLP={config.mlp_hidden} GELU",
        f"  training: {config.steps} steps, batch={config.batch_size}, "
        f"peak lr={config.peak_lr}, warmup={config.warmup_steps}",
        f"  final validation loss: {final_loss:.6f} nats/token",
        f"  final validation bits/char: {final_bpc:.6f}",
        f"  trigram sample seed context: [EOS, {start_token}] "
        f"({tokenizer.id_to_piece(start_token)!r})",
        f"  validation windows: {final_eval['windows']:,}",
        "",
        "VERIFY (1) - bits per character comparison",
        "-" * 84,
    ]
    for name, value in zip(model_names, bpc_values):
        lines.append(f"  {name:<12} {value:.6f}")
    lines.append(
        f"  Transformer reference: {TRANSFORMER_BPC_REFERENCE:.3f}; "
        f"threshold < {BPC_THRESHOLD:.3f}"
    )
    lines.append(
        f"  VERIFY (1): {'PASS' if verify1_passed else 'FAIL'}"
    )
    lines.append("")
    lines.append("VERIFY (2) - context-length dependence")
    lines.append("-" * 84)
    lines.append(
        f"  after 32 tokens mean loss: {after32_mean:.6f} "
        f"(threshold < {AFTER32_THRESHOLD:.3f})"
    )
    lines.append(
        f"  final 8 positions mean: {final8_mean:.6f} "
        f"(reference approximately {FINAL8_REFERENCE:.2f})"
    )
    lines.append(
        f"  Bigram reference: {BIGRAM_REFERENCE:.3f}; "
        f"Trigram reference: {TRIGRAM_REFERENCE:.3f}"
    )
    lines.append(
        "  position losses (1..64): "
        + ", ".join(f"{value:.3f}" for value in position_loss)
    )
    for name, value in verify2_checks.items():
        lines.append(f"  {name}: {value}")
    lines.append(
        f"  VERIFY (2): {'PASS' if verify2_passed else 'FAIL'}"
    )
    lines.append("")
    lines.append("VERIFY (3) - exact sampling distributions")
    lines.append("-" * 84)
    lines.append(
        f"  broad context (validation index {broad_context['index']}): "
        f"H={broad_context['entropy']:.4f}"
    )
    lines.append(
        "  context text: "
        + _sanitize_text(
            decode_token_stream(tokenizer, broad_context["tokens"])
        )
    )
    lines.append(
        f"  {'method':<28} {'chi2':>10} {'dof':>6} {'p-value':>12} "
        f"{'categories':>11} status"
    )
    for result in chi_results:
        lines.append(
            f"  {result['label']:<28} {result['statistic']:>10.3f} "
            f"{result['dof']:>6d} {result['p_value']:>12.6g} "
            f"{result['categories_kept']:>11d} "
            f"{'PASS' if result['passed'] else 'FAIL'}"
        )
    lines.append(
        f"  T=0.0001: argmax token {tiny_result['argmax_token']}, "
        f"max probability {tiny_result['max_probability']:.12g}, "
        f"1000/1000 argmax: {tiny_result['all_samples_argmax']}"
    )
    lines.append(
        f"  VERIFY (3): {'PASS' if verify3_passed else 'FAIL'}"
    )
    lines.append("")
    lines.append("VERIFY (4) - surprise versus entropy")
    lines.append("-" * 84)
    for label, stats in (
        ("T=1.0", stats_t1),
        ("T=0.7, scored at T=1", stats_t07),
    ):
        lines.append(
            f"  {label}: mean surprise={stats['mean_surprise']:.6f}, "
            f"mean entropy={stats['mean_entropy']:.6f}, "
            f"difference={stats['mean_difference']:.6f}, "
            f"SE={stats['standard_error']:.6f}, z={stats['z_score']:.3f}"
        )
    lines.append(
        f"  checks: T=1 -4<z<4: {verify4_checks['t1_z_in_interval']}; "
        f"T=0.7 |z|>40: {verify4_checks['t07_abs_z_gt_40']}"
    )
    lines.append(
        f"  VERIFY (4): {'PASS' if verify4_passed else 'FAIL'}"
    )
    lines.append("")
    lines.append("Temperature-1 story samples (seed 1848, increasing quality)")
    lines.append("-" * 84)
    for sample in sample_records:
        lines.append(f"  [{sample['model']}] {sample['text']}")
    lines.append("")
    lines.append("Sampling-method examples (seed 1848)")
    lines.append("-" * 84)
    for example in sampling_examples:
        lines.append(f"  [{example['label']}] {example['text']}")
    lines.append("")
    lines.append("Outputs")
    lines.append("-" * 84)
    lines.append(f"  bits/char figure: {bpc_figure_path}")
    lines.append(f"  model samples figure: {samples_figure_path}")
    lines.append(f"  context-length figure: {context_figure_path}")
    lines.append(f"  training figure: {training_figure_path}")
    lines.append(f"  sampling examples figure: {sampling_examples_figure}")
    lines.append(f"  verify4 figure: {verify4_figure_path}")
    lines.append(f"  checkpoint: {checkpoint_path}")
    lines.append(f"  results: {results_path}")
    lines.append(f"  settings: {settings_path}")
    lines.append("")
    lines.append("Summary")
    lines.append("-" * 84)
    for number in (1, 2, 3, 4):
        passed = results["acceptance"][f"verify{number}"]
        lines.append(f"  VERIFY ({number}): {'PASS' if passed else 'FAIL'}")

    report = "\n".join(lines) + "\n"
    report_path = EVIDENCE_DIR / "part5_verify.txt"
    report_path.write_text(report, encoding="utf-8")
    print()
    print(report, end="")
    return results


if __name__ == "__main__":
    run()
