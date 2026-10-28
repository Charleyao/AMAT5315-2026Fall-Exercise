"""TinyStories loading, splitting, and tokenization for Week 8 Part 2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sentencepiece as spm

WEEK_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = WEEK_DIR / "data"

TINYSTORIES_REPO = "roneneldan/TinyStories"
TINYSTORIES_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
TINYSTORIES_FILE = "TinyStoriesV2-GPT4-valid.txt"

TOKENIZER_REPO = "karpathy/tinyllamas"
TOKENIZER_REVISION = "0bd21da"
TOKENIZER_FILE = "tok512.model"

EXPECTED_TRAIN_STORIES = 24867
EXPECTED_VALID_STORIES = 2763
EXPECTED_VOCAB_SIZE = 512
TRAIN_FRACTION = 0.9


@dataclass
class EncodedSplit:
    """Token stream and summary statistics for one dataset split."""

    tokens: np.ndarray
    stories: int
    chars: int
    eos_count: int
    unk_count: int
    unique_tokens: int
    raw_roundtrip: int
    normalized_roundtrip: int

    @property
    def avg_chars_per_token(self) -> float:
        return self.chars / len(self.tokens)


def normalize_whitespace(text: str) -> str:
    """Collapse all runs of whitespace, matching SentencePiece decoding."""
    return " ".join(text.split())


def load_stories(path: str | Path = DATA_DIR / TINYSTORIES_FILE) -> list[str]:
    """Split on <|endoftext|>, strip, and drop empty stories."""
    text = Path(path).read_text(encoding="utf-8")
    return [part.strip() for part in text.split("<|endoftext|>") if part.strip()]


def split_stories(
    stories: list[str], train_fraction: float = TRAIN_FRACTION
) -> tuple[list[str], list[str]]:
    """Take the first 90% as training and the remaining 10% as validation."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    cut = int(len(stories) * train_fraction)
    return stories[:cut], stories[cut:]


def load_tokenizer(path: str | Path = DATA_DIR / TOKENIZER_FILE) -> spm.SentencePieceProcessor:
    """Load the 512-token SentencePiece model."""
    return spm.SentencePieceProcessor(model_file=str(path))


def encode_stories(
    tokenizer: spm.SentencePieceProcessor,
    stories: list[str],
    *,
    check_roundtrip: bool = True,
) -> EncodedSplit:
    """Encode stories and append the EOS token after every story.

    SentencePiece collapses repeated whitespace when decoding.  Round-trip is
    therefore checked both exactly and after whitespace normalization;
    normalization is the meaningful check for this tokenizer.
    """
    eos_id = tokenizer.eos_id()
    pieces: list[np.ndarray] = []
    raw_roundtrip = 0
    normalized_roundtrip = 0
    total_length = 0
    unknown = 0
    chars = 0

    for story in stories:
        ids = np.asarray(tokenizer.encode(story, out_type=int), dtype=np.int64)
        pieces.append(ids)
        total_length += len(ids) + 1
        chars += len(story)
        unknown += int(np.count_nonzero(ids == tokenizer.unk_id()))
        if check_roundtrip:
            decoded = tokenizer.decode(ids.tolist())
            raw_roundtrip += int(decoded == story)
            normalized_roundtrip += int(
                normalize_whitespace(decoded) == normalize_whitespace(story)
            )

    tokens = np.empty(total_length, dtype=np.int64)
    position = 0
    for ids in pieces:
        tokens[position : position + len(ids)] = ids
        position += len(ids)
        tokens[position] = eos_id
        position += 1

    return EncodedSplit(
        tokens=tokens,
        stories=len(stories),
        chars=chars,
        eos_count=len(stories),
        unk_count=unknown,
        unique_tokens=int(np.unique(tokens).size),
        raw_roundtrip=raw_roundtrip,
        normalized_roundtrip=normalized_roundtrip,
    )


def tokenizer_statistics(
    tokenizer: spm.SentencePieceProcessor,
    train: EncodedSplit,
    valid: EncodedSplit,
) -> dict:
    """Collect the tokenizer facts required by VERIFY (2)."""
    total_tokens = len(train.tokens) + len(valid.tokens)
    total_chars = train.chars + valid.chars
    return {
        "vocab_size": tokenizer.get_piece_size(),
        "eos_id": tokenizer.eos_id(),
        "unk_id": tokenizer.unk_id(),
        "train_tokens": len(train.tokens),
        "valid_tokens": len(valid.tokens),
        "total_tokens": total_tokens,
        "train_chars": train.chars,
        "valid_chars": valid.chars,
        "total_chars": total_chars,
        "train_chars_per_token": train.avg_chars_per_token,
        "valid_chars_per_token": valid.avg_chars_per_token,
        "overall_chars_per_token": total_chars / total_tokens,
        "train_unk": train.unk_count,
        "valid_unk": valid.unk_count,
        "train_unique_tokens": train.unique_tokens,
        "valid_unique_tokens": valid.unique_tokens,
        "train_eos": train.eos_count,
        "valid_eos": valid.eos_count,
        "raw_roundtrip": train.raw_roundtrip + valid.raw_roundtrip,
        "normalized_roundtrip": (
            train.normalized_roundtrip + valid.normalized_roundtrip
        ),
        "stories": train.stories + valid.stories,
    }
