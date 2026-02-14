#!/usr/bin/env sh
set -eu

FAMILY=${1:-i2s}
STEP=${BITNET_QF32_KQ_F64_BOUNDARY_STEP:-2}
LAYER=${BITNET_QF32_KQ_F64_BOUNDARY_LAYER:-7}
VALUES_N=${BITNET_QF32_KQ_F64_BOUNDARY_VALUES_N:-16}
KQ_LAYER_MAX=${BITNET_QF32_KQ_F64_BOUNDARY_KQ_LAYER_MAX:-12}
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUT_DIR=${BITNET_QF32_KQ_F64_BOUNDARY_OUT_DIR:-.bench/qf32-kq-f64-boundary}
SUMMARY=${BITNET_QF32_KQ_F64_BOUNDARY_SUMMARY:-$OUT_DIR/${FAMILY}-L${LAYER}-S${STEP}.tsv}

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
  q_layer=$1
  log="$OUT_DIR/${FAMILY}-q${q_layer}.log"
  set +e
  env \
    "$ENFORCE_NAME=1" \
    GOCACHE=/tmp/go-build \
    BITNET_PARITY_PROFILE=cpu_parity_v1 \
    BITNET_PARITY_FORCE=1 \
    BITNET_PARITY_STRICT=0 \
    BITNET_PARITY_FORCE_RELAX_TOPK=1 \
    BITNET_STRICT_Q_F32=1 \
    BITNET_STRICT_Q_F32_LAYER_MAX="$q_layer" \
    BITNET_STRICT_KQ=1 \
    BITNET_STRICT_KQ_LAYER_MAX="$KQ_LAYER_MAX" \
    BITNET_STRICT_KQ_MODE=f64 \
    BITNET_DRIFT_TRACE_STEP="$STEP" \
    BITNET_DRIFT_TRACE_TOKEN=-1 \
    BITNET_DRIFT_TRACE_LAYER="$LAYER" \
    BITNET_DRIFT_TRACE_VALUES_N="$VALUES_N" \
    go test ./pkg/bitnet -run "$TEST_RE" -count=1 -v >"$log" 2>&1
  status=$?
  set -e

  mismatch=$(awk '/topk logit mismatch step=/{print; exit}' "$log")
  logit=$(awk '/drift_trace logits step='"$STEP"' /{for(i=1;i<=NF;i++){if($i~/^logit=/){split($i,a,"=");print a[2]; exit}}}' "$log")
  printf "%s\t%d\t%s\t%s\n" "$q_layer" "$status" "${mismatch:-}" "${logit:-}"
}

extract_values() {
  file=$1
  name=$2
  awk -v layer="$LAYER" -v n="$name" '
    $1=="drift_trace" {
      gotLayer=0; gotName=0
      for(i=1;i<=NF;i++){
        if($i=="layer="layer) gotLayer=1
        if($i=="name="n) gotName=1
        if(index($i,"values=")==1 && gotLayer==1 && gotName==1){
          sub(/^values=/,"",$i)
          print $i
          exit
        }
      }
    }
  ' "$file"
}

csv_diff_stats() {
  a=$1
  b=$2
  awk -v av="$a" -v bv="$b" 'BEGIN{
    na=split(av,A,","); nb=split(bv,B,","); n=na; if(nb<n)n=nb;
    if(n<=0){print "\t"; exit}
    sum=0; max=0;
    for(i=1;i<=n;i++){
      d=A[i]-B[i]; if(d<0)d=-d;
      sum+=d; if(d>max)max=d;
    }
    printf "%.9g\t%.9g\n", sum/n, max;
  }'
}

meta_file="$OUT_DIR/${FAMILY}-meta.tsv"
{
  printf "q_layer\ttest_status\tmismatch\tstep_logit\n"
  run_case 6
  run_case 7
} > "$meta_file"

log6="$OUT_DIR/${FAMILY}-q6.log"
log7="$OUT_DIR/${FAMILY}-q7.log"

{
  printf "name\tmean_abs_q6_vs_q7\tmax_abs_q6_vs_q7\n"
  for name in attn_norm Qcur Kcur Vcur attn_softmax_h0 attn_o_out x_post_attn; do
    v6=$(extract_values "$log6" "$name")
    v7=$(extract_values "$log7" "$name")
    if [ -n "$v6" ] && [ -n "$v7" ]; then
      stats=$(csv_diff_stats "$v6" "$v7")
      printf "%s\t%s\n" "$name" "$stats"
    fi
  done
} > "$SUMMARY"

echo "qf32+kq f64 boundary meta: $meta_file"
cat "$meta_file"
echo "qf32+kq f64 boundary summary: $SUMMARY"
cat "$SUMMARY"
