#!/usr/bin/env sh
set -eu

FAMILY=${1:-i2s}
STEP=${BITNET_DRIFT_TRACE_STEP:-14}
TOKEN=${BITNET_DRIFT_TRACE_TOKEN:-55358}
ATOL=${BITNET_DRIFT_TRACE_ATOL:-6e-2}
RTOL=${BITNET_DRIFT_TRACE_RTOL:-6e-2}
STRICT_K=${BITNET_DRIFT_TRACE_TOPK_STRICT:-3}
PARITY_STRICT=${BITNET_DRIFT_TRACE_PARITY_STRICT:-1}
OUT=${BITNET_DRIFT_TRACE_OUT:-.bench/i2s-drift-trace-${FAMILY}.log}
ENFORCE_EXIT=${BITNET_DRIFT_TRACE_ENFORCE_EXIT:-0}
VERBOSE=${BITNET_DRIFT_TRACE_VERBOSE:-1}

mkdir -p "$(dirname "$OUT")"

case "$FAMILY" in
  i2s)
    TEST_RE='TestParityAgainstI2SVectors'
    ENFORCE_NAME='BITNET_ENFORCE_I2S'
    ;;
  i2s_2b)
    TEST_RE='TestParityAgainstI2S2BVectors'
    ENFORCE_NAME='BITNET_ENFORCE_I2S_2B'
    ;;
  *)
    echo "unknown family: $FAMILY (use i2s or i2s_2b)" >&2
    exit 1
    ;;
esac

set +e
if [ "$VERBOSE" = "1" ]; then
  env \
    "$ENFORCE_NAME=1" \
    GOCACHE=/tmp/go-build \
    BITNET_PARITY_FORCE=1 \
    BITNET_PARITY_STRICT="$PARITY_STRICT" \
    BITNET_I2S_FORCE_LOGIT_ATOL="$ATOL" \
    BITNET_I2S_FORCE_LOGIT_RTOL="$RTOL" \
    BITNET_I2S_TOPK_STRICT="$STRICT_K" \
    BITNET_PARITY_FORCE_RELAX_TOPK=1 \
    BITNET_DRIFT_TRACE_STEP="$STEP" \
    BITNET_DRIFT_TRACE_TOKEN="$TOKEN" \
    go test ./pkg/bitnet -run "$TEST_RE" -count=1 -v >"$OUT" 2>&1
else
  env \
    "$ENFORCE_NAME=1" \
    GOCACHE=/tmp/go-build \
    BITNET_PARITY_FORCE=1 \
    BITNET_PARITY_STRICT="$PARITY_STRICT" \
    BITNET_I2S_FORCE_LOGIT_ATOL="$ATOL" \
    BITNET_I2S_FORCE_LOGIT_RTOL="$RTOL" \
    BITNET_I2S_TOPK_STRICT="$STRICT_K" \
    BITNET_PARITY_FORCE_RELAX_TOPK=1 \
    BITNET_DRIFT_TRACE_STEP="$STEP" \
    BITNET_DRIFT_TRACE_TOKEN="$TOKEN" \
    go test ./pkg/bitnet -run "$TEST_RE" -count=1 >"$OUT" 2>&1
fi
STATUS=$?
set -e

echo "[drift-trace] family=$FAMILY status=$STATUS step=$STEP token=$TOKEN atol=$ATOL rtol=$RTOL"
echo "[drift-trace] log: $OUT"
grep -E 'drift_trace|topk logit mismatch|^--- FAIL|^FAIL|^ok' "$OUT" || true

if [ "$ENFORCE_EXIT" = "1" ]; then
  exit "$STATUS"
fi
exit 0
