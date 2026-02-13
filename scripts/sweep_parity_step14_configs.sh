#!/usr/bin/env sh
set -eu

FAMILY=${1:-i2s}
STEP=${BITNET_PARITY_SWEEP_STEP:-14}
TOKEN=${BITNET_PARITY_SWEEP_TOKEN:-55358}
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUT_BASE=${BITNET_PARITY_SWEEP_OUT_BASE:-.bench/parity-step-sweep}
SUMMARY=${BITNET_PARITY_SWEEP_SUMMARY:-.bench/parity-step-sweep-summary.tsv}

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
  log="$OUT_BASE/$name.log"
  set +e
  env \
    "$ENFORCE_NAME=1" \
    BITNET_PARITY_FORCE=1 \
    BITNET_PARITY_STRICT=0 \
    BITNET_PARITY_FORCE_RELAX_TOPK=1 \
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
  step14_logit=$(awk '/drift_trace logits step='"$STEP"' /{for(i=1;i<=NF;i++){if($i~/^logit=/){split($i,a,"=");print a[2]; exit}}}' "$log")
  printf "%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$name" "$status" "${mismatch_step:-}" "${mismatch_token:-}" "${got:-}" "${want:-}" "${abs_err:-}" "${step14_logit:-}"
}

{
  printf "case\ttest_status\tmismatch_step\tmismatch_token\tmismatch_got\tmismatch_want\tmismatch_abs_err\tstep14_logit\n"
  run_case "baseline"
  run_case "kq_l7" BITNET_STRICT_KQ=1 BITNET_STRICT_KQ_LAYER_MAX=7
  run_case "kq_l14" BITNET_STRICT_KQ=1 BITNET_STRICT_KQ_LAYER_MAX=14
  run_case "expf_l0" BITNET_STRICT_EXPF=1 BITNET_STRICT_EXPF_LAYER_MAX=0
  run_case "kq_l7_expf_l0" BITNET_STRICT_KQ=1 BITNET_STRICT_KQ_LAYER_MAX=7 BITNET_STRICT_EXPF=1 BITNET_STRICT_EXPF_LAYER_MAX=0
  run_case "kq_l14_expf_l0" BITNET_STRICT_KQ=1 BITNET_STRICT_KQ_LAYER_MAX=14 BITNET_STRICT_EXPF=1 BITNET_STRICT_EXPF_LAYER_MAX=0
} > "$SUMMARY"

echo "parity step sweep summary: $SUMMARY"
cat "$SUMMARY"
