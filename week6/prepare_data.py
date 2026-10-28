"""Download and validate the Week 6 MNIST and CIFAR-10 datasets.

This module only downloads the datasets and checks their integrity by
parsing the *original* binary formats:

* MNIST: gzip-compressed IDX files (magic numbers 0x00000803 / 0x00000801).
* CIFAR-10: ``cifar-10-binary.tar.gz`` with 10000 records of 1 label byte
  plus 3 x 1024 channel bytes (channel-first: R plane, G plane, B plane).

The original train/test splits and the order of the records (image-label
pairing) are preserved exactly as stored in the files.

Usage::

    cd week6
    .venv/bin/python prepare_data.py
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import socket
import struct
import tarfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

WEEK_DIR = Path(__file__).resolve().parent
INPUTS_DIR = WEEK_DIR / "inputs"
MNIST_DIR = INPUTS_DIR / "mnist"
CIFAR_DIR = INPUTS_DIR / "cifar10"

# --- Provenance -----------------------------------------------------------
# MNIST, mirrored by the Common Visual Data Foundation (CVDF) on Google
# Cloud Storage. Original source: Yann LeCun et al., "The MNIST database of
# handwritten digits", http://yann.lecun.com/exdb/mnist/ .
MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"

# MD5 sums published with the MNIST files. They also appear in the
# ``x-goog-hash`` response header of the Google Cloud Storage objects.
MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}

# CIFAR-10 binary version, mirrored by the MindSpore (Huawei Cloud OBS)
# notebook dataset bucket. Original source: Alex Krizhevsky, "Learning
# Multiple Layers of Features from Tiny Images" (2009), University of
# Toronto, https://www.cs.toronto.edu/~kriz/cifar.html .
CIFAR_URL = (
    "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/"
    "notebook/datasets/cifar-10-binary.tar.gz"
)
CIFAR_ARCHIVE = "cifar-10-binary.tar.gz"
# Published MD5 of the binary tarball (the Python version has a different one).
CIFAR_MD5 = "c32a1d4ab5d03f1284b67883e8d87530"
CIFAR_EXTRACTED = "cifar-10-batches-bin"
CIFAR_RECORD_BYTES = 1 + 3 * 32 * 32  # label byte + 3072 channel bytes

CIFAR_CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# --- Small helpers --------------------------------------------------------
def md5sum(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hexadecimal MD5 checksum of ``path``."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gzip_payload(raw: bytes) -> bytes:
    """Decompress an in-memory gzip payload (used for the IDX headers)."""
    return gzip.decompress(raw)


def _dns_query_via_udp(name: str, servers=("8.8.8.8", "1.1.1.1")) -> list[str]:
    """Resolve ``name`` with a raw UDP DNS query, bypassing the WSL resolver.

    This environment's WSL resolver silently drops ``storage.googleapis.com``
    while other hosts resolve normally, so we ask a public resolver directly.
    """
    for server in servers:
        try:
            tid = int.from_bytes(hashlib.md5(name.encode()).digest()[:2], "big")
            header = struct.pack("!HHHHHH", tid, 0x0100, 1, 0, 0, 0)
            question = (
                b"".join(bytes([len(part)]) + part.encode() for part in name.split("."))
                + b"\x00"
                + struct.pack("!HH", 1, 1)
            )
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.settimeout(8)
                sock.sendto(header + question, (server, 53))
                data, _ = sock.recvfrom(4096)

            answers = struct.unpack("!H", data[6:8])[0]
            offset = 12
            while data[offset] != 0:  # skip the question name
                if data[offset] & 0xC0 == 0xC0:
                    offset += 1
                    break
                offset += data[offset] + 1
            offset += 1 + 4  # terminating zero + qtype/qclass

            addresses = []
            for _ in range(answers):
                if data[offset] & 0xC0 == 0xC0:
                    offset += 2
                else:
                    while data[offset] != 0:
                        offset += data[offset] + 1
                    offset += 1
                rtype, _rclass, _ttl, rdlen = struct.unpack(
                    "!HHIH", data[offset:offset + 10]
                )
                offset += 10
                if rtype == 1 and rdlen == 4:
                    addresses.append(socket.inet_ntoa(data[offset:offset + 4]))
                offset += rdlen
            if addresses:
                return addresses
        except OSError:
            continue
    return []


@contextlib.contextmanager
def _dns_fallback(hostnames, force=False):
    """Patch ``socket.getaddrinfo`` for hosts the local resolver cannot reach.

    HTTP keeps using the original hostname for SNI and certificate checks;
    only the address lookup is redirected to an IP found via public DNS.
    With ``force=True`` the override is installed even if the local resolver
    answers once, because the WSL resolver is flaky under repeated lookups.
    """
    original = socket.getaddrinfo
    overrides = {}
    for host in hostnames:
        if not force:
            try:
                original(host, 443, socket.AF_INET)
                continue
            except socket.gaierror:
                pass
        addresses = _dns_query_via_udp(host)
        if addresses:
            overrides[host] = addresses

    if not overrides:
        yield
        return

    def patched(host, port, family=0, type=0, proto=0, flags=0):
        if host in overrides:
            socktype = type or socket.SOCK_STREAM
            return [
                (socket.AF_INET, socktype, proto or socket.IPPROTO_TCP, "", (ip, port))
                for ip in overrides[host]
            ]
        return original(host, port, family, type, proto, flags)

    socket.getaddrinfo = patched
    try:
        yield
    finally:
        socket.getaddrinfo = original


def download(url: str, dest: Path, expected_md5: str | None = None) -> Path:
    """Download ``url`` to ``dest`` and verify its MD5 checksum.

    Files that already exist with the expected checksum are reused. Failed
    attempts are retried with the public-DNS workaround forced on.
    """
    if dest.is_file() and expected_md5 and md5sum(dest) == expected_md5:
        print(f"  [reuse] {dest.relative_to(WEEK_DIR)} (checksum OK)")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    host = urllib.parse.urlsplit(url).hostname
    print(f"  [get]   {url}")

    request = urllib.request.Request(url, headers={"User-Agent": "amat5315-week6/1.0"})
    for attempt in range(1, 4):
        try:
            with _dns_fallback([host], force=attempt > 1):
                with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as out:
                    total = int(response.headers.get("Content-Length", 0))
                    done = 0
                    next_mark = 10
                    while True:
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
                        done += len(chunk)
                        if total:
                            percent = 100 * done // total
                            if percent >= next_mark or done == total:
                                amount = f"{done / 2**20:6.1f} MiB" if total >= 2**20 else f"{done / 2**10:6.0f} KiB"
                                print(f"           {percent:3d}%  {amount}")
                                next_mark = (percent // 10 + 1) * 10
            break
        except (urllib.error.URLError, socket.gaierror, ConnectionError, TimeoutError) as exc:
            tmp.unlink(missing_ok=True)
            if attempt == 3:
                raise
            print(f"  [retry] attempt {attempt} failed ({exc}); trying public DNS")
            time.sleep(2)

    if expected_md5:
        actual = md5sum(tmp)
        if actual != expected_md5:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"checksum mismatch for {dest.name}: expected {expected_md5}, got {actual}"
            )
    tmp.replace(dest)
    return dest


# --- Original binary format parsers --------------------------------------
def parse_idx_images(path: Path) -> np.ndarray:
    """Parse an MNIST IDX image file into a ``(N, rows, cols)`` uint8 array."""
    with gzip.open(path, "rb") as handle:
        data = handle.read()
    magic, count, rows, cols = struct.unpack(">IIII", data[:16])
    if magic != 0x00000803:
        raise ValueError(f"{path.name}: unexpected IDX image magic {magic:#010x}")
    expected = count * rows * cols
    if len(data) - 16 != expected:
        raise ValueError(
            f"{path.name}: expected {expected} pixel bytes, found {len(data) - 16}"
        )
    return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(count, rows, cols)


def parse_idx_labels(path: Path) -> np.ndarray:
    """Parse an MNIST IDX label file into an ``(N,)`` uint8 array."""
    with gzip.open(path, "rb") as handle:
        data = handle.read()
    magic, count = struct.unpack(">II", data[:8])
    if magic != 0x00000801:
        raise ValueError(f"{path.name}: unexpected IDX label magic {magic:#010x}")
    if len(data) - 8 != count:
        raise ValueError(
            f"{path.name}: expected {count} label bytes, found {len(data) - 8}"
        )
    return np.frombuffer(data, dtype=np.uint8, offset=8)


def load_mnist(directory: Path = MNIST_DIR) -> dict[str, np.ndarray]:
    """Load the four MNIST splits from their IDX files."""
    return {
        "train_images": parse_idx_images(directory / "train-images-idx3-ubyte.gz"),
        "train_labels": parse_idx_labels(directory / "train-labels-idx1-ubyte.gz"),
        "test_images": parse_idx_images(directory / "t10k-images-idx3-ubyte.gz"),
        "test_labels": parse_idx_labels(directory / "t10k-labels-idx1-ubyte.gz"),
    }


def _parse_cifar_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse one CIFAR-10 binary batch into ``(N, 32, 32, 3)`` images + labels."""
    raw = path.read_bytes()
    if len(raw) % CIFAR_RECORD_BYTES:
        raise ValueError(
            f"{path.name}: size {len(raw)} is not a multiple of {CIFAR_RECORD_BYTES}"
        )
    records = np.frombuffer(raw, dtype=np.uint8).reshape(-1, CIFAR_RECORD_BYTES)
    labels = records[:, 0].copy()
    # 3072 channel bytes are stored channel-first (R plane, G plane, B plane).
    images = (
        records[:, 1:]
        .reshape(-1, 3, 32, 32)
        .transpose(0, 2, 3, 1)
        .copy()
    )
    return images, labels


def load_cifar10(directory: Path = CIFAR_DIR) -> dict[str, object]:
    """Load CIFAR-10 train (batches 1-5) and test splits from binary batches."""
    batch_dir = directory / CIFAR_EXTRACTED
    train_images, train_labels = [], []
    for index in range(1, 6):
        images, labels = _parse_cifar_batch(batch_dir / f"data_batch_{index}.bin")
        train_images.append(images)
        train_labels.append(labels)
    test_images, test_labels = _parse_cifar_batch(batch_dir / "test_batch.bin")

    # The binary distribution ships the class names as plain text, one per line.
    class_names = (batch_dir / "batches.meta.txt").read_text(encoding="utf-8").split()
    if class_names != CIFAR_CLASS_NAMES:
        raise ValueError(f"unexpected CIFAR-10 class names: {class_names}")

    return {
        "train_images": np.concatenate(train_images),
        "train_labels": np.concatenate(train_labels),
        "test_images": test_images,
        "test_labels": test_labels,
        "class_names": class_names,
    }


# --- Validation ----------------------------------------------------------
def validate_mnist(data: dict[str, np.ndarray]) -> None:
    """Check the MNIST properties required by the exercise."""
    assert data["train_images"].shape == (60000, 28, 28), data["train_images"].shape
    assert data["test_images"].shape == (10000, 28, 28), data["test_images"].shape
    assert data["train_labels"].shape == (60000,)
    assert data["test_labels"].shape == (10000,)
    assert data["train_images"].dtype == np.uint8
    assert data["train_labels"].dtype == np.uint8
    assert data["train_images"].min() == 0 and data["train_images"].max() == 255
    assert data["test_images"].min() == 0 and data["test_images"].max() == 255
    assert set(np.unique(data["train_labels"]).tolist()) == set(range(10))
    assert set(np.unique(data["test_labels"]).tolist()) == set(range(10))


def validate_cifar10(data: dict[str, object]) -> None:
    """Check the CIFAR-10 properties required by the exercise."""
    train_images = data["train_images"]
    test_images = data["test_images"]
    train_labels = data["train_labels"]
    test_labels = data["test_labels"]
    assert train_images.shape == (50000, 32, 32, 3), train_images.shape
    assert test_images.shape == (10000, 32, 32, 3), test_images.shape
    assert train_labels.shape == (50000,) and test_labels.shape == (10000,)
    assert train_images.dtype == np.uint8 and train_labels.dtype == np.uint8
    assert train_images.min() == 0 and train_images.max() == 255
    assert test_images.min() == 0 and test_images.max() == 255
    assert list(np.bincount(train_labels, minlength=10)) == [5000] * 10
    assert list(np.bincount(test_labels, minlength=10)) == [1000] * 10
    assert len(data["class_names"]) == 10


# --- Reporting -----------------------------------------------------------
def _print_split(name: str, images: np.ndarray, labels: np.ndarray) -> None:
    print(
        f"  {name:<6} images={images.shape[0]:>5}  shape={images.shape}  "
        f"pixels=[{images.min()}, {images.max()}]  labels={labels.shape[0]}"
    )
    counts = np.bincount(labels, minlength=10)
    print(f"         per-class counts: {counts.tolist()}")


def download_all(mnist_dir: Path = MNIST_DIR, cifar_dir: Path = CIFAR_DIR) -> None:
    """Download both datasets and verify their checksums."""
    print("MNIST")
    for filename, checksum in MNIST_FILES.items():
        download(f"{MNIST_URL}/{filename}", mnist_dir / filename, checksum)

    print("CIFAR-10")
    archive = download(f"{CIFAR_URL}", cifar_dir / CIFAR_ARCHIVE, CIFAR_MD5)
    extracted = cifar_dir / CIFAR_EXTRACTED
    if not (extracted / "test_batch.bin").is_file():
        print(f"  [untar] {archive.name} -> {extracted.name}")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(cifar_dir, filter="data")
    else:
        print(f"  [reuse] {extracted.relative_to(WEEK_DIR)} (already extracted)")


def main() -> None:
    download_all()

    print("\nParsing and validating the original binary formats")
    mnist = load_mnist()
    validate_mnist(mnist)
    print("MNIST (28x28 grayscale, 10 digit classes)")
    _print_split("train", mnist["train_images"], mnist["train_labels"])
    _print_split("test", mnist["test_images"], mnist["test_labels"])

    cifar = load_cifar10()
    validate_cifar10(cifar)
    print("CIFAR-10 (32x32x3 RGB, 10 object classes)")
    _print_split("train", cifar["train_images"], cifar["train_labels"])
    _print_split("test", cifar["test_images"], cifar["test_labels"])
    print(f"  classes: {cifar['class_names']}")

    print("\nAll downloads and integrity checks passed.")


if __name__ == "__main__":
    main()
