#!/usr/bin/env bash
# Block until the local vLLM server answers /health or its process exits.
# Tracks the server by pid file (scripts/serve.py writes it) rather than by
# command-line match, which would also match this script's own shell.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIDFILE="$ROOT/logs/vllm_server.pid"
LOG="$ROOT/logs/vllm_server.log"
PORT="${1:-8000}"

for _ in $(seq 1 30); do [ -s "$PIDFILE" ] && break; sleep 1; done
PID="$(cat "$PIDFILE" 2>/dev/null)"

while true; do
  if curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "SERVER_READY pid=$PID"
    grep -E "GPU KV cache size|Maximum concurrency|Model loading took|Loaded new LoRA|lora" "$LOG" | cut -c1-200 | tail -8
    exit 0
  fi
  if [ -z "$PID" ] || ! kill -0 "$PID" 2>/dev/null; then
    echo "SERVER_DIED pid=$PID"
    grep -E "Error|error:|Traceback" "$LOG" | grep -v FutureWarning | cut -c1-240 | head -12
    exit 1
  fi
  sleep 5
done
