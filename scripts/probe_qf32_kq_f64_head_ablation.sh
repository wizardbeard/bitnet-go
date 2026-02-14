#!/usr/bin/env sh
set -eu

FAMILY=${1:-i2s}
Q_LAYER_MAX=${BITNET_QF32_KQ_F64_HEAD_Q_LAYER_MAX:-6}
KQ_LAYER_MAX=${BITNET_QF32_KQ_F64_HEAD_KQ_LAYER_MAX:-12}
STEP=${BITNET_QF32_KQ_F64_HEAD_STEP:-2}
TOKEN=${BITNET_QF32_KQ_F64_HEAD_TOKEN:-40}
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUT_DIR=${BITNET_QF32_KQ_F64_HEAD_OUT_DIR:-.bench/qf32-kq-f64-head-ablation}
SUMMARY=${BITNET_QF32_KQ_F64_HEAD_SUMMARY:-$OUT_DIR/${FAMILY}.tsv}

mkdir -p "$OUT_DIR" "$(dirname "$SUMMARY")"

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
  q_head=$2
  log="$OUT_DIR/${FAMILY}-${name}.log"
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
    BITNET_STRICT_Q_F32_HEAD="$q_head" \
    BITNET_STRICT_KQ=1 \
    BITNET_STRICT_KQ_LAYER_MAX="$KQ_LAYER_MAX" \
    BITNET_STRICT_KQ_MODE=f64 \
    BITNET_DRIFT_TRACE_STEP="$STEP" \
    BITNET_DRIFT_TRACE_TOKEN="$TOKEN" \
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
  printf "%s\t%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$name" "$q_head" "$status" \
    "${mismatch_step:-}" "${mismatch_token:-}" "${got:-}" "${want:-}" "${abs_err:-}" "${step_logit:-}"
}

{
  printf "case\tq_head\ttest_status\tmismatch_step\tmismatch_token\tmismatch_got\tmismatch_want\tmismatch_abs_err\tstep_logit\n"
  run_case "all_q_heads" -1
  run_case "q_head0_only" 0
  run_case "q_head1_only" 1
  run_case "q_head2_only" 2
  run_case "q_head3_only" 3
} > "$SUMMARY"

echo "qf32+kq f64 head ablation summary: $SUMMARY"
cat "$SUMMARY"
