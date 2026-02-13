#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"

echo "[parity-profile] profile=cpu_parity_v1"

env \
  BITNET_PARITY_PROFILE=cpu_parity_v1 \
  BITNET_ENFORCE_I2S=1 \
  BITNET_PARITY_FORCE=1 \
  BITNET_PARITY_STRICT=0 \
  BITNET_PARITY_FORCE_RELAX_TOPK=1 \
  go test ./pkg/bitnet -run TestParityAgainstI2SVectors -count=1 -v

env \
  BITNET_PARITY_PROFILE=cpu_parity_v1 \
  BITNET_ENFORCE_I2S_2B=1 \
  BITNET_PARITY_FORCE=1 \
  BITNET_PARITY_STRICT=0 \
  BITNET_PARITY_FORCE_RELAX_TOPK=1 \
  go test ./pkg/bitnet -run TestParityAgainstI2S2BVectors -count=1 -v

echo "[parity-profile] PASS"
