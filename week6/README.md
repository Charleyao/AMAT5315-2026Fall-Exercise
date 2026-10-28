# Week 6 — Part 1: Dataset download and validation

This part only **downloads and validates** the two image datasets used by the
Week 6 exercise. No models, figures, or Part 2 work are included here.

| Dataset  | Train images | Test images | Image shape   | Pixel range | Classes |
|----------|-------------:|------------:|---------------|-------------|--------:|
| MNIST    | 60,000       | 10,000      | `28 x 28` (1) | 0–255       | 10      |
| CIFAR-10 | 50,000       | 10,000      | `32 x 32 x 3` | 0–255       | 10      |

All downloads and integrity checks passed on 2026-10-28 (UTC).

## Provenance and download URLs

### MNIST

- **Original source:** Yann LeCun, Corinna Cortes, Christopher J.C. Burges,
  *The MNIST database of handwritten digits*,
  <http://yann.lecun.com/exdb/mnist/>.
- **Downloaded from (mirror):** the Common Visual Data Foundation (CVDF)
  mirror on Google Cloud Storage,
  <https://storage.googleapis.com/cvdf-datasets/mnist>.
- **Files:** `train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`,
  `t10k-images-idx3-ubyte.gz`, `t10k-labels-idx1-ubyte.gz`.
- **Local location:** `week6/inputs/mnist/`.
- The MD5 sums below match the `md5=` value in the objects' `x-goog-hash`
  response headers, so the mirror is byte-identical to the published files.

### CIFAR-10 (binary version)

- **Original source:** Alex Krizhevsky, *Learning Multiple Layers of Features
  from Tiny Images* (2009), University of Toronto,
  <https://www.cs.toronto.edu/~kriz/cifar.html>.
- **Downloaded from (mirror):** the MindSpore notebook dataset bucket on
  Huawei Cloud OBS,
  <https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-binary.tar.gz>.
- **Local location:** `week6/inputs/cifar10/` (tarball plus the extracted
  `cifar-10-batches-bin/` directory).
- The binary version is used (not `cifar-10-python.tar.gz`): its class names
  ship as plain text in `batches.meta.txt`, and each batch stores records as
  `1 label byte + 3 x 1024 channel bytes` (R plane, G plane, B plane).

## Checksums (verified)

| File                                   | Size (bytes) | MD5                                |
|----------------------------------------|-------------:|------------------------------------|
| `train-images-idx3-ubyte.gz`           |    9,912,422 | `f68b3c2dcbeaaa9fbdd348bbdeb94873` |
| `train-labels-idx1-ubyte.gz`           |       28,881 | `d53e105ee54ea40749a09fcbcd1e9432` |
| `t10k-images-idx3-ubyte.gz`            |    1,648,877 | `9fb629c4189551a2d022fa330f9573f3` |
| `t10k-labels-idx1-ubyte.gz`            |        4,542 | `ec29112dd5afa0611ce80d1b7f02629c` |
| `cifar-10-binary.tar.gz`               |  170,052,171 | `c32a1d4ab5d03f1284b67883e8d87530` |

Note: `c58f30108f718f92721af3b95e74349a` is the MD5 of the *Python* CIFAR-10
tarball, not the binary one. The binary tarball used here is `c32a1d4a…`.

## Validation results (parsed from the original binary formats)

MNIST IDX format: 4-byte magic (`0x00000803` for images, `0x00000801` for
labels), then dimensions/count, then raw `uint8` pixels or labels.

```
MNIST (28x28 grayscale, 10 digit classes)
  train  images=60000  shape=(60000, 28, 28)  pixels=[0, 255]  labels=60000
         per-class counts: [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]
  test   images=10000  shape=(10000, 28, 28)  pixels=[0, 255]  labels=10000
         per-class counts: [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]
```

CIFAR-10 binary format: each record is 3,073 bytes (`1` label byte + `3,072`
pixel bytes), reshaped to `(32, 32, 3)` RGB.

```
CIFAR-10 (32x32x3 RGB, 10 object classes)
  train  images=50000  shape=(50000, 32, 32, 3)  pixels=[0, 255]  labels=50000
         per-class counts: [5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000]
  test   images=10000  shape=(10000, 32, 32, 3)  pixels=[0, 255]  labels=10000
         per-class counts: [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
  classes: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
```

The full run is saved in `evidence/dataset-validation.txt`.

## What is preserved

- The **original train/test splits** (60k/10k for MNIST, 50k/10k for CIFAR-10);
  batches are concatenated in file order (`data_batch_1` … `data_batch_5`),
  with no shuffling.
- **Image-label pairing** follows the record order of the source files. The
  tests check parsed labels against the raw label bytes/records, and that
  re-reading the data is deterministic.
- **CIFAR-10 class balance**: exactly 5,000 training and 1,000 test images per
  class.
- **MNIST class counts** match the official training/test histograms.

## VERIFY (1): MNIST distribution figure

`plot_mnist_distribution.py` reuses the verified files in `inputs/mnist/`
(no re-download) and renders a two-panel figure following the reference on
page 4 of the Week 6 learning sheet:

- **Left:** the first two *training* images of every digit 0–9, in stored
  order, at native 28×28 resolution with `interpolation="nearest"` (no
  smoothing), each row labeled with its digit.
- **Right:** grouped bars comparing the per-class fractions of the 60,000
  training and 10,000 test images.

![MNIST sample gallery and empirical class distribution](evidence/mnist_distribution.png)

Reproduce with:

```sh
cd week6
.venv/bin/python plot_mnist_distribution.py   # writes evidence/mnist_distribution.png
.venv/bin/python -m pytest -q                 # includes the VERIFY (1) tests
```

The later `plot_mnist_distribution.py` run also writes the
`evidence/mnist-distribution-verify.txt` log. Numerical verification:

| digit | train n | train fraction | test n | test fraction |
|------:|--------:|---------------:|-------:|--------------:|
| 0 | 5923 | 0.0987 | 980 | 0.0980 |
| 1 | 6742 | **0.1124** | 1135 | **0.1135** |
| 2 | 5958 | 0.0993 | 1032 | 0.1032 |
| 3 | 6131 | 0.1022 | 1010 | 0.1010 |
| 4 | 5842 | 0.0974 | 982 | 0.0982 |
| 5 | 5421 | **0.0903** | 892 | **0.0892** |
| 6 | 5918 | 0.0986 | 958 | 0.0958 |
| 7 | 6265 | 0.1044 | 1028 | 0.1028 |
| 8 | 5851 | 0.0975 | 974 | 0.0974 |
| 9 | 5949 | 0.0992 | 1009 | 0.1009 |
| **sum** | **60000** | **1.0000** | **10000** | **1.0000** |

Verified properties:

- All ten digit classes 0–9 are present and labeled; each gallery image is
  the exact source `uint8` array of a training row whose label equals the row
  label (tested against the raw data).
- Digit **1** has the highest frequency and digit **5** the lowest, in both
  splits.
- Fractions sum to 1 for each split (60,000 and 10,000 images).
- The train/test assignments and the MNIST files are unchanged (counts match
  the official histograms and the MD5 checksums still pass).

## VERIFY (2): CIFAR-10 distribution figure

`plot_cifar10_distribution.py` reuses both the verified CIFAR-10 files in
`inputs/cifar10/` (no re-download) and the shared layout utilities in
`distribution_plotting.py` (the same code behind the MNIST figure).

- **Left:** the first two *training* images of each of the 10 object classes,
  in stored order, at native **32×32×3 RGB** resolution with
  `interpolation="nearest"` (no smoothing), each row labeled with its class.
- **Right:** grouped bars comparing the per-class fractions of the 50,000
  training and 10,000 test images; every bar is exactly 0.1.

![CIFAR-10 sample gallery and class balance](evidence/cifar10_distribution.png)

Reproduce with:

```sh
cd week6
.venv/bin/python plot_cifar10_distribution.py   # writes evidence/cifar10_distribution.png
.venv/bin/python -m pytest -q                   # includes the VERIFY (2) tests
```

A run also writes the `evidence/cifar10-distribution-verify.txt` log.

| class | train n | train fraction | test n | test fraction |
|-------|--------:|---------------:|-------:|--------------:|
| airplane | 5000 | 0.1 | 1000 | 0.1 |
| automobile | 5000 | 0.1 | 1000 | 0.1 |
| bird | 5000 | 0.1 | 1000 | 0.1 |
| cat | 5000 | 0.1 | 1000 | 0.1 |
| deer | 5000 | 0.1 | 1000 | 0.1 |
| dog | 5000 | 0.1 | 1000 | 0.1 |
| frog | 5000 | 0.1 | 1000 | 0.1 |
| horse | 5000 | 0.1 | 1000 | 0.1 |
| ship | 5000 | 0.1 | 1000 | 0.1 |
| truck | 5000 | 0.1 | 1000 | 0.1 |
| **sum** | **50000** | **1.0** | **10000** | **1.0** |

Verified properties:

- Training set has **50,000** images and the test set **10,000**.
- Every class has exactly **5,000** training and **1,000** test images;
  each class frequency is exactly **0.1** in both splits.
- All three RGB channels are decoded in the documented order (R plane, then
  G, then B), independently re-decoded from the raw `.bin` records in the
  tests; pixel values span 0–255.
- The original image-label pairing and stored order are preserved, and the
  archive MD5 checksum still matches.

## MNIST vs CIFAR-10

- **Image representation and resolution.** MNIST images are single-channel
  grayscale at 28×28 (784 values each), so a gallery mostly shows stroke
  shape and intensity. CIFAR-10 images are three-channel RGB at 32×32×3
  (3,072 values each), so the gallery also conveys colour and richer texture.
- **Class frequency distributions.** MNIST is imbalanced: digit 1 is the most
  frequent (≈0.112 train / 0.114 test) and digit 5 the least (≈0.090 / 0.089),
  with visible train/test differences. CIFAR-10 is exactly balanced: every
  class is 0.1 in both splits, so all bars have the same height.
- **What the panels show.** The gallery reveals what the pixels look like and
  whether labels match the visual content (digit shapes vs. objects and
  colours), while the frequency plot reveals dataset balance and split
  consistency. The gallery cannot show class proportions, and the bar chart
  cannot show image content — together they characterize the dataset.

## How to reproduce

```sh
cd week6
python3 -m venv .venv
.venv/bin/python -m pip install numpy matplotlib pytest
.venv/bin/python prepare_data.py                # download + checksum + parse
.venv/bin/python plot_mnist_distribution.py     # VERIFY (1) figure
.venv/bin/python plot_cifar10_distribution.py   # VERIFY (2) figure
.venv/bin/python -m pytest -q                   # 38 tests
```

`prepare_data.py` is idempotent: files that already exist with the correct MD5
are reused. Downloads are written to a temporary `.part` file and renamed only
after the checksum matches.

## Notes

- Datasets live in `inputs/` and are **gitignored** (only `.gitkeep` is
  tracked); `.venv/` is also ignored. Re-run `prepare_data.py` after a fresh
  clone.
- Environment quirk: the WSL resolver in this machine drops DNS lookups for
  `storage.googleapis.com`. `prepare_data.py` detects that and resolves the
  host through a public DNS server while keeping the original hostname for TLS
  (SNI and certificate validation are unchanged).
- Scope: Part 1 dataset preparation plus VERIFY (1) and VERIFY (2) only —
  no model training and no Part 2.
