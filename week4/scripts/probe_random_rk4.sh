#!/usr/bin/env bash
# Probe the random-flow RK4 stability boundary from the shared initial field.
# Exploration helper for scripts/run_scan.sh; writes throwaway runs under
# artifacts/scan/probe/.
set -uo pipefail

WEEK4="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WEEK4"
mkdir -p artifacts/scan/probe

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

if [ ! -s artifacts/scan/random-initial.json ]; then
    run_field random --n 128 --seed 2026 --k-min 2 --k-max 6 \
        > artifacts/scan/random-initial.json
fi

printf '%-8s %-6s %-10s\n' dt exit last_finite_t
for dt in 0.036 0.034 0.032 0.030 0.028 0.026 0.024 0.022 0.020 0.018; do
    out="artifacts/scan/probe/rk4-dt$dt"
    tsv="$out.tsv"
    cat artifacts/scan/random-initial.json \
        | run_fluid --method rk4 --nu 0.004 --dt "$dt" --t-end 10 --every 0.5 \
                    --out "$out" > "$tsv"
    code=$?
    last=$(awk -F'\t' 'NR>1 && $2 ~ /^[0-9.eE+-]+$/ {t=$1} END{print t+0}' "$tsv")
    printf '%-8s %-6s %-10s\n' "$dt" "$code" "$last"
done
