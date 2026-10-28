"""Download and checksum-verify the Chinchilla runs CSV for Part 6."""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

WEEK_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = WEEK_DIR / "data"

URL = (
    "https://raw.githubusercontent.com/epoch-research/analyzing-chinchilla/"
    "92258837425e1b5f2851d624287f0120583a3d0e/data/svg_extracted_data.csv"
)
TARGET = DATA_DIR / "chinchilla-runs.csv"
SIZE = 23224
SHA256 = "193478df4e90f040016431996c45682659306840ecdadf83cfb9cd5ca0aa1b30"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid(path: Path) -> bool:
    return (
        path.exists()
        and path.stat().st_size == SIZE
        and sha256_file(path) == SHA256
    )


def main() -> None:
    if is_valid(TARGET):
        print(f"[skip] {TARGET.name}: already present and checksum-verified")
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temporary = TARGET.with_suffix(TARGET.suffix + ".part")
    print(f"[get ] {URL}")
    request = urllib.request.Request(
        URL, headers={"User-Agent": "amat5315-week8/0.1"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with temporary.open("wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
    if not is_valid(temporary):
        raise RuntimeError(
            f"checksum mismatch: size={temporary.stat().st_size}, "
            f"sha256={sha256_file(temporary)}"
        )
    os.replace(temporary, TARGET)
    print(f"[ ok ] {TARGET.name}: checksum verified")


if __name__ == "__main__":
    main()
