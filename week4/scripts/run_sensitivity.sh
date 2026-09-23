#!/usr/bin/env bash
# Reproduce the Part 3 sensitivity experiment: two stable dt = 0.01 RK4 runs
# (original vs perturbed) each for Taylor-Green and the seeded random flow.
#
#     bash run_sensitivity.sh
set -euo pipefail

WEEK4="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WEEK4"
mkdir -p artifacts/sensitivity evidence

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
run_prep() { cargo run --quiet --bin sensitivity-prep -- "$@"; }

# --- initial fields -----------------------------------------------------------
run_field taylor-green --n 64 > artifacts/sensitivity/tg-original.json
run_field random --n 128 --seed 2026 --k-min 2 --k-max 6 \
    > artifacts/sensitivity/random-original.json

# --- divergence-free vorticity ripple -----------------------------------------
run_prep artifacts/sensitivity/tg-original.json artifacts/sensitivity/tg-perturbed.json
run_prep artifacts/sensitivity/random-original.json artifacts/sensitivity/random-perturbed.json

# --- four stable runs ---------------------------------------------------------
run_fluid_case() { # name, in-file, nu
    local name="$1" in="$2" nu="$3"
    cat "$in" \
        | run_fluid --method rk4 --nu "$nu" --dt 0.01 --t-end 20 --every 0.5 \
                    --out "artifacts/sensitivity/$name" \
        > "artifacts/sensitivity/$name.tsv"
    echo "$name: $(tail -1 "artifacts/sensitivity/$name.tsv")"
}

run_fluid_case tg-original  artifacts/sensitivity/tg-original.json  0.1
run_fluid_case tg-perturbed artifacts/sensitivity/tg-perturbed.json 0.1
run_fluid_case random-original  artifacts/sensitivity/random-original.json  0.004
run_fluid_case random-perturbed artifacts/sensitivity/random-perturbed.json 0.004

echo "wrote artifacts/sensitivity/{tg,random}-{original,perturbed}/"
