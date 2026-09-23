#!/usr/bin/env bash
# Reproduce the Part 3 standard seeded random-flow run.
#
#     bash run_random.sh
#
# Uses the installed `field` / `fluid` binaries when they are on PATH
# (`cargo install --path .`), otherwise falls back to `cargo run`.
set -euo pipefail

WEEK4="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WEEK4"
mkdir -p artifacts evidence

if command -v field >/dev/null 2>&1; then
    run_field() { field "$@"; }
else
    run_field() { cargo run --quiet --bin field -- "$@"; }
fi
if command -v fluid >/dev/null 2>&1; then
    run_fluid() { fluid "$@"; }
else
    run_fluid() { cargo run --quiet --bin fluid -- "$@"; }
fi

run_field random --n 128 --seed 2026 --k-min 2 --k-max 6 \
    | run_fluid --method rk4 --nu 0.004 --dt 0.01 --t-end 10 --every 0.1 \
                --out artifacts/random \
    > artifacts/random.tsv

echo "wrote artifacts/random.tsv"
head -2 artifacts/random.tsv
tail -1 artifacts/random.tsv
