#!/usr/bin/env bash
# Reproduce the Part 2 Taylor-Green t=0 -> t=1 run.
#
# Run from week4/scripts/ (paths are resolved from the script location):
#
#     bash run_taylor_green.sh
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

# Numerical run: t = 0 -> t = 1, RK4, fixed dt = 0.01, snapshots every 0.1.
run_field taylor-green --n 64 \
    | run_fluid --method rk4 --nu 0.1 --dt 0.01 --t-end 1 --every 0.1 \
                --out artifacts/taylor-green \
    > artifacts/taylor-green.tsv

# Exact t = 1 field for the relative-velocity comparison.
run_field taylor-green --n 64 --nu 0.1 --t 1 \
    > artifacts/taylor-green/exact-t1.json

echo "wrote artifacts/taylor-green.tsv and artifacts/taylor-green/exact-t1.json"
head -2 artifacts/taylor-green.tsv
tail -1 artifacts/taylor-green.tsv
