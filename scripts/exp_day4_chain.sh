#!/usr/bin/env bash
# Day-4 chain: retrain the format direction with the regularised objective after the first
# one collapsed into a <|"|> loop (IMPLEMENTATION.md 6.13), gated by the trainer's own sanity
# generation, then evaluate it on dev24. Waits for the day-3 chain (pilot) to finish.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
while pgrep -f "^bash scripts/exp_day3_chain.sh" >/dev/null; do sleep 60; done
echo "=== $(date '+%F %T') day-3 chain finished"
# reuse the day-2 chain's stop/train/restart/evaluate flow with the new trainer defaults
PROMPT_TOKENS=4096 PAIRS_PER_CTX=2 EPOCHS=1 OUT=adapters_out/format_r16_v2 scripts/exp_day2_chain.sh
echo "=== $(date '+%F %T') day-4 chain done rc=$?"
