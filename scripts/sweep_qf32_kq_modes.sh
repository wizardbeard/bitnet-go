#!/usr/bin/env sh
set -eu

FAMILY=${1:-i2s}
STEP=${BITNET_QF32_KQ_MODE_STEP:-14}
TOKEN=${BITNET_QF32_KQ_MODE_TOKEN:-55358}
Q_LAYER_MAX=${BITNET_QF32_KQ_MODE_Q_LAYER_MAX:-6}
KQ_LAYER_MAX=${BITNET_QF32_KQ_MODE_KQ_LAYER_MAX:-12}
MODES=${BITNET_QF32_KQ_MODE_LIST:-ggml naive f64}
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUT_BASE=${BITNET_QF32_KQ_MODE_OUT_BASE:-.bench/qf32-kq-modes}
SUMMARY=${BITNET_QF32_KQ_MODE_SUMMARY:-.bench/qf32-kq-modes-${FAMILY}.tsv}

mkdir -p "$OUT_BASE" "$(dirname "$SUMMARY")"

case "$FAMILY" in
  i2s)
    ENFORCE_NAME=BITNET_ENFORCE_I2S
    TEST_RE='TestParityAgainstI2SVectors'
    ;;
  i2s_2b)
    ENFORCE_NAME=BITNET_ENFORCE_I2S_2B
    TEST_RE='TestParityAgainstI2S2BVectors'
    ;;
  *)
    echo "unknown family: $FAMILY" >&2
    exit 1
    ;;
esac

run_case() {
  name=$1
  shift
  log="$OUT_BASE/${FAMILY}-$name.log"
  set +e
  env \
    "$ENFORCE_NAME=1" \
    GOCACHE=/tmp/go-build \
    BITNET_PARITY_PROFILE=cpu_parity_v1 \
    BITNET_PARITY_FORCE=1 \
    BITNET_PARITY_STRICT=0 \
    BITNET_PARITY_FORCE_RELAX_TOPK=1 \
    BITNET_STRICT_Q_F32=1 \
    BITNET_STRICT_Q_F32_LAYER_MAX="$Q_LAYER_MAX" \
    BITNET_STRICT_KQ=1 \
    BITNET_STRICT_KQ_LAYER_MAX="$KQ_LAYER_MAX" \
    BITNET_DRIFT_TRACE_STEP="$STEP" \
    BITNET_DRIFT_TRACE_TOKEN="$TOKEN" \
    "$@" \
    go test ./pkg/bitnet -run "$TEST_RE" -count=1 -v >"$log" 2>&1
  status=$?
  set -e

  line=$(awk '/topk logit mismatch step=/{print; exit}' "$log")
  mismatch_step=$(printf "%s\n" "$line" | sed -n 's/.*step=\([0-9][0-9]*\) token=.*/\1/p')
  mismatch_token=$(printf "%s\n" "$line" | sed -n 's/.*token=\([0-9][0-9]*\): got=.*/\1/p')
  got=$(printf "%s\n" "$line" | sed -n 's/.* got=\([^ ]*\) .*/\1/p')
  want=$(printf "%s\n" "$line" | sed -n 's/.* want=\([^ ]*\) .*/\1/p')
  abs_err=""
  if [ -n "$got" ] && [ -n "$want" ]; then
    abs_err=$(awk -v g="$got" -v w="$want" 'BEGIN{d=g-w; if(d<0)d=-d; printf "%.9g\n", d}')
  fi
  step_logit=$(awk '/drift_trace logits step='"$STEP"' /{for(i=1;i<=NF;i++){if($i~/^logit=/){split($i,a,"=");print a[2]; exit}}}' "$log")
  printf "%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$name" "$status" "${mismatch_step:-}" "${mismatch_token:-}" "${got:-}" "${want:-}" "${abs_err:-}" "${step_logit:-}"
}

{
  printf "case\ttest_status\tmismatch_step\tmismatch_token\tmismatch_got\tmismatch_want\tmismatch_abs_err\tstep_logit\n"
  run_case "qf32_l${Q_LAYER_MAX}_kq_default"
  for mode in $MODES; do
    run_case "qf32_l${Q_LAYER_MAX}_kq_${mode}" BITNET_STRICT_KQ_MODE="$mode"
  done
} > "$SUMMARY"

echo "qf32+kq mode sweep summary: $SUMMARY"
cat "$SUMMARY"
