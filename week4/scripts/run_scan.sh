#!/usr/bin/env bash
# Reproduce the Part 3 stability-limit scan.
#
#     bash run_scan.sh
#
# A non-finite energy makes `fluid` exit 1 by contract, so this script does NOT
# use `set -e`; each run's exit code is recorded instead.
set -uo pipefail

WEEK4="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WEEK4"
mkdir -p artifacts/scan artifacts/unstable evidence

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

SUMMARY="artifacts/scan/summary.txt"
: > "$SUMMARY"

# One shared random initial field for every random stability run.
run_field random --n 128 --seed 2026 --k-min 2 --k-max 6 \
    > artifacts/scan/random-initial.json
echo "random-initial: $(wc -c < artifacts/scan/random-initial.json) bytes" | tee -a "$SUMMARY"

# --- Taylor-Green: obviously over the diffusive limit -------------------------
run_field taylor-green --n 64 \
    | run_fluid --method rk4 --nu 0.1 --dt 0.04 --t-end 4 --every 0.1 \
                --out artifacts/unstable/taylor-green \
    > artifacts/unstable/taylor-green.tsv
echo "tg  rk4 dt=0.040  exit=$?" | tee -a "$SUMMARY"

# --- Taylor-Green: the two sides of the boundary ------------------------------
for dt in 0.032 0.033; do
    run_field taylor-green --n 64 \
        | run_fluid --method rk4 --nu 0.1 --dt "$dt" --t-end 8 --every 0.5 \
                    --out "artifacts/scan/tg-rk4-dt$dt" \
        > "artifacts/scan/tg-rk4-dt$dt.tsv"
    echo "tg  rk4 dt=$dt  exit=$?" | tee -a "$SUMMARY"
done

# --- Random: RK4 scan from the shared initial field ---------------------------
# The prescribed first probes 0.038/0.040 both blow up; stepping down finds the
# largest stable step 0.030 and the smallest unstable one 0.032.
for dt in 0.038 0.040 0.032 0.030; do
    cat artifacts/scan/random-initial.json \
        | run_fluid --method rk4 --nu 0.004 --dt "$dt" --t-end 10 --every 0.5 \
                    --out "artifacts/scan/random-rk4-dt$dt" \
        > "artifacts/scan/random-rk4-dt$dt.tsv"
    echo "rnd rk4 dt=$dt  exit=$?" | tee -a "$SUMMARY"
done

# --- Random: forward Euler ----------------------------------------------------
cat artifacts/scan/random-initial.json \
    | run_fluid --method euler --nu 0.004 --dt 0.01 --t-end 10 --every 0.5 \
                --out artifacts/scan/random-euler-dt0.01 \
    > artifacts/scan/random-euler-dt0.01.tsv
echo "rnd euler dt=0.010  exit=$?" | tee -a "$SUMMARY"

echo "--- summary ---"
cat "$SUMMARY"
