#!/usr/bin/env bash
# Reproduce the Part 4 random-flow self-convergence runs.
#
#     bash run_convergence.sh
#
# One shared random initial field, four fixed RK4 steps. `--every 2` keeps only
# the t = 0 and t = 2 frames.
set -euo pipefail

WEEK4="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WEEK4"
mkdir -p artifacts/convergence evidence

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
    > artifacts/convergence/random-initial.json

for dt in 0.02 0.0125 0.01 0.0025; do
    cat artifacts/convergence/random-initial.json \
        | run_fluid --method rk4 --nu 0.004 --dt "$dt" --t-end 2 --every 2 \
                    --out "artifacts/convergence/rk4-dt$dt" \
        > "artifacts/convergence/rk4-dt$dt.tsv"
    echo "rk4 dt=$dt: $(tail -1 "artifacts/convergence/rk4-dt$dt.tsv")"
done

echo "wrote artifacts/convergence/{random-initial.json,rk4-dt*/}"
