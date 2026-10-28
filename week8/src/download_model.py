"""Download the pinned potion-base-8M files used in Week 8, Part 1.

The files are stored in ``week8/data/`` and intentionally kept out of Git.
Hugging Face is tried first; ``hf-mirror.com`` is used as a fallback when the
primary endpoint is unreachable.

Run from the repository root with::

    cd week8 && uv run python -m src.download_model
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

WEEK_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = WEEK_DIR / "data"

MODEL_REPO = "minishlab/potion-base-8M"
MODEL_REVISION = "bf8b056"

# Expected size and SHA-256.  model.safetensors matches the Git LFS OID reported
# for revision bf8b056 on the Hugging Face API.
FILES = {
    "model.safetensors": {
        "size": 30236760,
        "sha256": "f65d0f325faadc1e121c319e2faa41170d3fa07d8c89abd48ca5358d9a223de2",
    },
    "tokenizer.json": {
        "size": 683666,
        "sha256": "e67e803f624fb4d67dea1c730d06e1067e1b14d830e2c2202569e3ef0f70bb50",
    },
}

BASES = (
    "https://huggingface.co",
    "https://hf-mirror.com",
)


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
    for base in BASES:
        url = f"{base}/{MODEL_REPO}/resolve/{MODEL_REVISION}/{name}"
        temporary = target.with_suffix(target.suffix + ".part")
        print(f"[get ] {url}")
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "amat5315-week8/0.1"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                with temporary.open("wb") as handle:
                    while True:
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        handle.write(chunk)
            if not is_valid(temporary, expected):
                raise RuntimeError(
                    f"checksum/size mismatch for {name}: "
                    f"size={temporary.stat().st_size}, sha256={sha256_file(temporary)}"
                )
            os.replace(temporary, target)
            print(f"[ ok ] {name}: {target.stat().st_size} bytes, checksum verified")
            return target
        except Exception as error:  # try the mirror if the primary fails
            last_error = error
            temporary.unlink(missing_ok=True)
            print(f"[fail] {base}: {error}")

    raise RuntimeError(f"could not download {name}") from last_error


def main() -> None:
    for name, expected in FILES.items():
        download_file(name, expected)


if __name__ == "__main__":
    main()
