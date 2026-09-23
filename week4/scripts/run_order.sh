#!/usr/bin/env bash
# Reproduce the Part 4 Taylor-Green RK4 order experiment.
#
#     bash run_order.sh
#
# Three fixed steps (0.4, 0.25, 0.2) all divide t_end = 2 exactly; `--every 2`
# keeps only the t = 0 and t = 2 frames.
set -euo pipefail

WEEK4="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WEEK4"
mkdir -p artifacts/order evidence

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

for dt in 0.4 0.25 0.2; do
    run_field taylor-green --n 8 \
        | run_fluid --method rk4 --nu 0.5 --dt "$dt" --t-end 2 --every 2 \
                    --out "artifacts/order/rk4-dt$dt" \
        > "artifacts/order/rk4-dt$dt.tsv"
    echo "rk4 dt=$dt: $(tail -1 "artifacts/order/rk4-dt$dt.tsv")"
done

run_field taylor-green --n 8 --nu 0.5 --t 2 \
    > artifacts/order/exact-t2.json

echo "wrote artifacts/order/{rk4-dt0.4,rk4-dt0.25,rk4-dt0.2}/ and exact-t2.json"
