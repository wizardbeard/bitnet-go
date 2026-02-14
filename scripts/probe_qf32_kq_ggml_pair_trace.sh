#!/usr/bin/env sh
set -eu

FAMILY=${1:-i2s}
Q_LAYER_MAX=${BITNET_QF32_KQ_GGML_PAIR_Q_LAYER_MAX:-6}
KQ_LAYER_MAX=${BITNET_QF32_KQ_GGML_PAIR_KQ_LAYER_MAX:-12}
STEP=${BITNET_QF32_KQ_GGML_PAIR_STEP:-2}
LAYER=${BITNET_QF32_KQ_GGML_PAIR_LAYER:-7}
VALUES_N=${BITNET_QF32_KQ_GGML_PAIR_VALUES_N:-16}
SOFTMAX_HEADS=${BITNET_QF32_KQ_GGML_PAIR_SOFTMAX_HEADS:-4}
ATTN_OUT_HEADS=${BITNET_QF32_KQ_GGML_PAIR_ATTN_OUT_HEADS:-4}
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUT_DIR=${BITNET_QF32_KQ_GGML_PAIR_OUT_DIR:-.bench/qf32-kq-ggml-pair-trace}
SUMMARY=${BITNET_QF32_KQ_GGML_PAIR_SUMMARY:-$OUT_DIR/${FAMILY}-L${LAYER}-S${STEP}.tsv}
META=${BITNET_QF32_KQ_GGML_PAIR_META:-$OUT_DIR/${FAMILY}-meta.tsv}

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
  q_heads=$2
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
    BITNET_STRICT_Q_F32_HEAD=-1 \
    BITNET_STRICT_Q_F32_HEADS="$q_heads" \
    BITNET_STRICT_KQ=1 \
    BITNET_STRICT_KQ_LAYER_MAX="$KQ_LAYER_MAX" \
    BITNET_STRICT_KQ_MODE=ggml \
    BITNET_DRIFT_TRACE_STEP="$STEP" \
    BITNET_DRIFT_TRACE_TOKEN=-1 \
    BITNET_DRIFT_TRACE_LAYER="$LAYER" \
    BITNET_DRIFT_TRACE_VALUES_N="$VALUES_N" \
    BITNET_DRIFT_TRACE_SOFTMAX_HEADS="$SOFTMAX_HEADS" \
    BITNET_DRIFT_TRACE_ATTN_OUT_HEADS="$ATTN_OUT_HEADS" \
    go test ./pkg/bitnet -run "$TEST_RE" -count=1 -v >"$log" 2>&1
  status=$?
  set -e

  mismatch=$(awk '/topk logit mismatch step=/{print; exit}' "$log")
  logit=$(awk '/drift_trace logits step='"$STEP"' /{for(i=1;i<=NF;i++){if($i~/^logit=/){split($i,a,"=");print a[2]; exit}}}' "$log")
  printf "%s\t%s\t%d\t%s\t%s\n" "$name" "$q_heads" "$status" "${mismatch:-}" "${logit:-}"
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

extract_head_metric() {
  file=$1
  head=$2
  metric=$3
  awk -v layer="$LAYER" -v h="$head" -v m="$metric" '
    $1=="drift_trace" && $2=="attn_out_head" {
      gotLayer=0; gotHead=0; val=""
      for(i=1;i<=NF;i++){
        if($i=="layer="layer) gotLayer=1
        if($i=="head="h) gotHead=1
        if(index($i,m"=")==1){ sub("^"m"=","",$i); val=$i }
      }
      if(gotLayer==1 && gotHead==1 && val!=""){
        print val
        exit
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

{
  printf "case\tq_heads\ttest_status\tmismatch\tstep_logit\n"
  run_case "all_q_heads" ""
  run_case "q_head0_2" "0,2"
  run_case "q_head0_1" "0,1"
  run_case "q_head2_3" "2,3"
} > "$META"

base_log="$OUT_DIR/${FAMILY}-all_q_heads.log"
{
  printf "case\tmetric\tmean_abs_vs_all\tmax_abs_vs_all\n"
  for case in q_head0_2 q_head0_1 q_head2_3; do
    log="$OUT_DIR/${FAMILY}-${case}.log"
    for name in Qcur Kcur Vcur attn_o_out x_post_attn; do
      vb=$(extract_values "$base_log" "$name")
      vc=$(extract_values "$log" "$name")
      if [ -n "$vb" ] && [ -n "$vc" ]; then
        stats=$(csv_diff_stats "$vb" "$vc")
        printf "%s\t%s\t%s\n" "$case" "$name" "$stats"
      fi
    done
    h=0
    while [ "$h" -lt "$SOFTMAX_HEADS" ]; do
      name="attn_softmax_h${h}"
      vb=$(extract_values "$base_log" "$name")
      vc=$(extract_values "$log" "$name")
      if [ -n "$vb" ] && [ -n "$vc" ]; then
        stats=$(csv_diff_stats "$vb" "$vc")
        printf "%s\t%s\t%s\n" "$case" "$name" "$stats"
      fi
      h=$((h + 1))
    done
    h=0
    while [ "$h" -lt "$ATTN_OUT_HEADS" ]; do
      sb=$(extract_head_metric "$base_log" "$h" "subnorm_l2")
      sc=$(extract_head_metric "$log" "$h" "subnorm_l2")
      if [ -n "$sb" ] && [ -n "$sc" ]; then
        stats=$(csv_diff_stats "$sb" "$sc")
        printf "%s\tattn_out_head%d_subnorm_l2\t%s\n" "$case" "$h" "$stats"
      fi
      pb=$(extract_head_metric "$base_log" "$h" "proj_l2")
      pc=$(extract_head_metric "$log" "$h" "proj_l2")
      if [ -n "$pb" ] && [ -n "$pc" ]; then
        stats=$(csv_diff_stats "$pb" "$pc")
        printf "%s\tattn_out_head%d_proj_l2\t%s\n" "$case" "$h" "$stats"
      fi
      h=$((h + 1))
    done
  done
} > "$SUMMARY"

echo "qf32+kq ggml pair trace meta: $META"
cat "$META"
echo "qf32+kq ggml pair trace summary: $SUMMARY"
cat "$SUMMARY"
