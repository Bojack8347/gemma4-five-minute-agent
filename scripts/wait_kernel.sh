#!/usr/bin/env bash
# Poll a Kaggle kernel until it reaches a terminal state, then download its output.
# Usage: scripts/wait_kernel.sh <owner/slug> <output-dir>
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KERNEL="$1"
OUT="$2"
export KAGGLE_API_TOKEN="$(cat "$ROOT/kaggle.json")"
KAGGLE="$ROOT/.venv/bin/kaggle"

last=""
while true; do
  status="$("$KAGGLE" kernels status "$KERNEL" 2>&1 | tail -1)"
  if [ "$status" != "$last" ]; then echo "$(date +%H:%M:%S) $status"; last="$status"; fi
  case "$status" in
    *COMPLETE*|*ERROR*|*CANCEL*) break ;;
  esac
  sleep 60
done

mkdir -p "$OUT"
"$KAGGLE" kernels output "$KERNEL" -p "$OUT" 2>&1 | tail -3
ls -la "$OUT" | head -20
