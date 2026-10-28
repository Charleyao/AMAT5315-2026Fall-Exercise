"""Download and checksum-verify the Week 8 Part 2 data files.

Files are written to ``week8/data/`` and are excluded from Git.  Hugging Face
is tried first; the ``hf-mirror.com`` mirror is used if that endpoint fails.

Run from the repository root with::

    cd week8 && uv run python -m src.download_part2_data
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

from .part2_data import (
    DATA_DIR,
    TINYSTORIES_FILE,
    TINYSTORIES_REPO,
    TINYSTORIES_REVISION,
    TOKENIZER_FILE,
    TOKENIZER_REPO,
    TOKENIZER_REVISION,
)

FILES = {
    TINYSTORIES_FILE: {
        "size": 22502601,
        "sha256": "6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d",
        "urls": [
            f"https://huggingface.co/datasets/{TINYSTORIES_REPO}/resolve/"
            f"{TINYSTORIES_REVISION}/{TINYSTORIES_FILE}",
            f"https://hf-mirror.com/datasets/{TINYSTORIES_REPO}/resolve/"
            f"{TINYSTORIES_REVISION}/{TINYSTORIES_FILE}",
        ],
    },
    TOKENIZER_FILE: {
        "size": 7645,
        "sha256": "dfff07d929db979913f166ec94a6f5ecad4c70cfed8eb5c9cbe7e464455e46f5",
        "urls": [
            f"https://huggingface.co/{TOKENIZER_REPO}/resolve/"
            f"{TOKENIZER_REVISION}/stories260K/{TOKENIZER_FILE}",
            f"https://hf-mirror.com/{TOKENIZER_REPO}/resolve/"
            f"{TOKENIZER_REVISION}/stories260K/{TOKENIZER_FILE}",
        ],
    },
}


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid(path: Path, expected: dict) -> bool:
    return (
        path.exists()
        and path.stat().st_size == expected["size"]
        and sha256_file(path) == expected["sha256"]
    )


def download_file(name: str, expected: dict) -> Path:
    target = DATA_DIR / name
    if is_valid(target, expected):
        print(f"[skip] {name}: already present and checksum-verified")
        return target

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in expected["urls"]:
        temporary = target.with_suffix(target.suffix + ".part")
        print(f"[get ] {url}")
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "amat5315-week8/0.1"}
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                with temporary.open("wb") as handle:
                    while True:
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        handle.write(chunk)
            if not is_valid(temporary, expected):
                raise RuntimeError(
                    f"checksum/size mismatch for {name}: "
                    f"size={temporary.stat().st_size}, "
                    f"sha256={sha256_file(temporary)}"
                )
            os.replace(temporary, target)
            print(f"[ ok ] {name}: {target.stat().st_size} bytes, checksum verified")
            return target
        except Exception as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            print(f"[fail] {url}: {error}")

    raise RuntimeError(f"could not download {name}") from last_error


def main() -> None:
    for name, expected in FILES.items():
        download_file(name, expected)


if __name__ == "__main__":
    main()
