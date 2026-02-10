#!/usr/bin/env sh
set -eu

OUT_FILE=${BITNET_I2S_BENCH_OUT:-.bench/i2s-kernels.txt}
BENCH_TIME=${BITNET_I2S_BENCH_TIME:-200ms}
GOCACHE=${GOCACHE:-/tmp/go-build}

mkdir -p "$(dirname "$OUT_FILE")"

GOCACHE=$GOCACHE go test ./internal/kernels \
  -run '^$' \
  -bench 'BenchmarkMatVecI2SI8SVariants|BenchmarkMatVecTI2SI8SVariants|BenchmarkQuantizeRowI8S' \
  -benchmem \
  -benchtime "$BENCH_TIME" | tee "$OUT_FILE"
