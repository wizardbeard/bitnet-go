#!/usr/bin/env sh
set -eu

OUT_FILE=${BITNET_I2S_SWEEP_OUT:-.bench/i2s-kernels-sweep.txt}
BENCH_TIME=${BITNET_I2S_BENCH_TIME:-60ms}
THREADS=${BITNET_I2S_SWEEP_THREADS:-6}
GOCACHE=${GOCACHE:-/tmp/go-build}

mkdir -p "$(dirname "$OUT_FILE")"
: >"$OUT_FILE"

bench_pat='BenchmarkMatVec(T)?I2SI8SVariants'

run_cfg() {
  label=$1
  rows_min=$2
  cols_min=$3
  chunk_rows=$4
  chunk_cols=$5
  block_min_rows=$6

  printf "\n=== %s ===\n" "$label" | tee -a "$OUT_FILE"
  printf "rows_min=%s cols_min=%s chunk_rows=%s chunk_cols=%s block_min_rows=%s threads=%s\n" \
    "$rows_min" "$cols_min" "$chunk_rows" "$chunk_cols" "$block_min_rows" "$THREADS" | tee -a "$OUT_FILE"

  tmp=$(mktemp)
  BITNET_I2S_I8S_DISABLE_FAST=1 \
  BITNET_MATVEC_THREADS="$THREADS" \
  BITNET_I2S_I8S_PAR_ROWS_MIN="$rows_min" \
  BITNET_I2S_I8S_PAR_COLS_MIN="$cols_min" \
  BITNET_I2S_I8S_PAR_CHUNK_ROWS="$chunk_rows" \
  BITNET_I2S_I8S_PAR_CHUNK_COLS="$chunk_cols" \
  BITNET_I2S_I8S_BLOCK_MIN_ROWS="$block_min_rows" \
  GOCACHE=$GOCACHE go test ./internal/kernels -run '^$' -bench "$bench_pat" -benchmem -benchtime "$BENCH_TIME" | tee "$tmp"

  cat "$tmp" >>"$OUT_FILE"
  avg_ns=$(awk '/BenchmarkMatVec(T)?I2SI8SVariants\/r=.*\/dispatch-/ {sum+=$3; n++} END {if (n>0) printf "%.0f", sum/n; else print "NA"}' "$tmp")
  printf "avg_dispatch_ns=%s\n" "$avg_ns" | tee -a "$OUT_FILE"
  rm -f "$tmp"
}

run_cfg auto_default 512 512 0 0 256
run_cfg min_1024 1024 1024 0 0 256
run_cfg chunk_256 512 512 256 256 256
run_cfg chunk_512 512 512 512 512 256
run_cfg block_128 512 512 0 0 128
