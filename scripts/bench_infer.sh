#!/usr/bin/env sh
set -eu

MODEL=${BITNET_BENCH_MODEL:-testdata/ggml-model-i2_s.gguf}
PROMPT=${BITNET_BENCH_PROMPT:-Write a short 4-line poem about rivers.}
SYSTEM=${BITNET_BENCH_SYSTEM:-You are a concise assistant.}
MAX_TOKENS=${BITNET_BENCH_TOKENS:-64}
TEMP=${BITNET_BENCH_TEMP:-0}
PROCS=${BITNET_BENCH_PROCS:-0}
BATCH=${BITNET_BENCH_BATCH:-1}

if [ -n "${BITNET_BENCH_PROMPT_FILE:-}" ]; then
  PROMPT=$(cat "$BITNET_BENCH_PROMPT_FILE")
fi

start=$(date +%s%N)
out=$(GOCACHE=/tmp/go-build go run ./cmd/bitnet \
  --model "$MODEL" \
  --chat-template \
  --system "$SYSTEM" \
  --user "$PROMPT" \
  --max-tokens "$MAX_TOKENS" \
  --temp "$TEMP" \
  --procs "$PROCS" \
  --batch "$BATCH")
end=$(date +%s%N)

elapsed=$(awk "BEGIN{printf \"%.3f\", ($end-$start)/1e9}")
tokens=$(printf "%s\n" "$out" | awk 'match($0,/ tokens=([0-9]+)/,m){print m[1]; exit}')
if [ -z "${tokens:-}" ]; then
  tokens=$MAX_TOKENS
fi
tokps=$(awk "BEGIN{if ($elapsed>0) printf \"%.3f\", $tokens/$elapsed; else print \"0\"}")

printf "bench: tokens=%s elapsed=%ss tok/s=%s\n" "$tokens" "$elapsed" "$tokps"
printf "%s\n" "$out"
