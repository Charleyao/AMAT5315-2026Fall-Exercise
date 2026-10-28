"""Optional Part 5 extension: Figure 18 attention-head heatmaps.

Loads the existing Part 5 Transformer checkpoint; it never retrains the model.

Run from the repository root with::

    cd week8 && uv run python -m src.part5_figure18
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .part2_data import (
    DATA_DIR,
    TINYSTORIES_FILE,
    TOKENIZER_FILE,
    encode_stories,
    load_stories,
    load_tokenizer,
    split_stories,
)
from .transformer import forward_with_attention, load_checkpoint

WEEK_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = WEEK_DIR / "artifacts"
EVIDENCE_DIR = WEEK_DIR / "evidence"
CHECKPOINT = ARTIFACT_DIR / "part5_transformer_checkpoint.npz"
FIGURE = EVIDENCE_DIR / "figure18_attention_heads.png"
CHECKS = ARTIFACT_DIR / "part5_figure18_checks.json"
N_OPENING_TOKENS = 24
REQUESTS = ((0, 0), (1, 1))  # zero-based: layer 1/head 1, layer 2/head 2


def load_opening_tokens(
    n_tokens: int = N_OPENING_TOKENS,
) -> tuple[list[int], list[str], str, int]:
    """Return the opening BPE tokens of the first held-out story."""
    stories = load_stories(DATA_DIR / TINYSTORIES_FILE)
    _, valid_stories = split_stories(stories)
    tokenizer = load_tokenizer(DATA_DIR / TOKENIZER_FILE)

    for story_index, story in enumerate(valid_stories):
        encoded = encode_stories(tokenizer, [story], check_roundtrip=False)
        tokens = [int(token) for token in encoded.tokens[:n_tokens]]
        if len(tokens) < n_tokens or tokenizer.eos_id() in tokens:
            continue
        pieces = [tokenizer.id_to_piece(token) for token in tokens]
        labels = [piece.replace("▁", "·") for piece in pieces]
        text = tokenizer.decode(tokens).strip()
        return tokens, labels, text, story_index
    raise RuntimeError("no suitable held-out story opening found")


def extract_attention() -> dict:
    """Load the checkpoint and extract the two requested causal heads."""
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"missing {CHECKPOINT}; complete Part 5 first"
        )
    params, config = load_checkpoint(CHECKPOINT)
    tokens, labels, text, story_index = load_opening_tokens()
    logits, attention = forward_with_attention(
        params,
        np.asarray([tokens], dtype=np.int32),
        set(REQUESTS),
    )

    maps = {}
    for layer_index, head_index in REQUESTS:
        matrix = np.asarray(attention[(layer_index, head_index)][0])
        future_max = float(np.max(np.triu(matrix, k=1)))
        row_sum_error = float(
            np.max(np.abs(matrix.sum(axis=1) - 1.0))
        )
        maps[f"layer{layer_index + 1}_head{head_index + 1}"] = {
            "matrix": matrix,
            "future_attention_max": future_max,
            "row_sum_error": row_sum_error,
            "future_attention_zero": future_max <= 1e-12,
            "rows_sum_to_one": row_sum_error <= 1e-6,
        }

    return {
        "params": params,
        "config": config,
        "tokens": tokens,
        "labels": labels,
        "text": text,
        "story_index": story_index,
        "logits_shape": list(np.asarray(logits).shape),
        "maps": maps,
    }


def plot_figure18(result: dict) -> Path:
    """Draw the two requested attention heads side by side."""
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(18.0, 7.8))
    labels = result["labels"]
    n_tokens = len(labels)

    for ax, (key, record) in zip(axes, result["maps"].items()):
        matrix = record["matrix"]
        image = ax.imshow(
            matrix,
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            origin="upper",
            aspect="equal",
            interpolation="nearest",
        )
        ax.set_xticks(np.arange(n_tokens))
        ax.set_yticks(np.arange(n_tokens))
        ax.set_xticklabels(labels, rotation=90, fontsize=7.2)
        ax.set_yticklabels(labels, fontsize=7.2)
        ax.set_xlabel("key token (past context)", fontsize=9)
        ax.set_ylabel("query token", fontsize=9)
        title = key.replace("layer", "Layer ").replace("_head", ", Head ")
        ax.set_title(
            f"{title}\nfuture max = {record['future_attention_max']:.1e}; "
            f"row-sum error = {record['row_sum_error']:.1e}",
            fontsize=10.5,
        )
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        colorbar.set_label("attention weight", fontsize=8.5)
        ax.set_xticks(np.arange(-0.5, n_tokens, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_tokens, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.5, alpha=0.45)

    fig.suptitle(
        "Figure 18: causal attention on the opening of a held-out TinyStories story\n"
        f"tokens: {' '.join(result['labels'])}",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.012,
        "Rows are queries and columns are keys. The upper triangle is zero "
        "because future tokens are masked; every row sums to 1. "
        "'·' marks a word boundary.",
        ha="center",
        fontsize=8.5,
        color="0.3",
    )
    fig.tight_layout(rect=(0.01, 0.035, 0.99, 0.9))
    fig.savefig(FIGURE, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return FIGURE


def run() -> dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    result = extract_attention()
    figure_path = plot_figure18(result)

    checks = {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_reused_without_training": True,
        "story_index": result["story_index"],
        "n_tokens": len(result["tokens"]),
        "tokens": result["tokens"],
        "labels": result["labels"],
        "decoded_text": result["text"],
        "logits_shape": result["logits_shape"],
        "figure": str(figure_path),
        "heads": {
            key: {
                "future_attention_max": record["future_attention_max"],
                "row_sum_error": record["row_sum_error"],
                "future_attention_zero": record["future_attention_zero"],
                "rows_sum_to_one": record["rows_sum_to_one"],
            }
            for key, record in result["maps"].items()
        },
    }
    checks["all_checks_passed"] = bool(
        all(
            record["future_attention_zero"] and record["rows_sum_to_one"]
            for record in checks["heads"].values()
        )
    )
    CHECKS.write_text(
        json.dumps(checks, indent=2) + "\n", encoding="utf-8"
    )

    print(f"checkpoint: {CHECKPOINT} (loaded, not retrained)")
    print(f"held-out story index: {result['story_index']}")
    print(f"opening tokens ({len(result['tokens'])}):")
    print("  " + " ".join(result["labels"]))
    for key, record in checks["heads"].items():
        print(
            f"{key}: future max={record['future_attention_max']:.3e}, "
            f"row-sum error={record['row_sum_error']:.3e}, "
            f"future-zero={record['future_attention_zero']}, "
            f"rows-normalized={record['rows_sum_to_one']}"
        )
    print(f"figure: {figure_path}")
    print(f"checks: {CHECKS}")
    print(f"ALL CHECKS: {'PASS' if checks['all_checks_passed'] else 'FAIL'}")
    return checks


if __name__ == "__main__":
    run()
