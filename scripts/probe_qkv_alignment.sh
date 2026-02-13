#!/usr/bin/env sh
set -eu

FAMILY=${1:-i2s}
LAYER=${BITNET_QKV_PROBE_LAYER:-14}
STEP=${BITNET_QKV_PROBE_STEP:-14}
OUT_DIR=${BITNET_QKV_PROBE_OUT_DIR:-.bench}
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TESTDATA_DIR="$ROOT_DIR/testdata"

count_json_items() {
  file=$1
  if [ ! -f "$file" ]; then
    echo 0
    return
  fi
  awk '
    {
      gsub(/[[:space:]]/, "", $0)
      gsub(/^\[/, "", $0)
      gsub(/\]$/, "", $0)
      if ($0 == "") next
      n += gsub(/,/, "&") + 1
    }
    END { print n + 0 }
  ' "$file"
}

extract_go_values() {
  name=$1
  out=$2
  awk -v layer="$LAYER" -v name="$name" '
    $1=="drift_trace" && $2=="values" {
      gotLayer = ""
      gotName = ""
      gotValues = ""
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^layer=/) { split($i, a, "="); gotLayer = a[2] }
        if ($i ~ /^name=/) { split($i, a, "="); gotName = a[2] }
        if ($i ~ /^values=/) { sub(/^values=/, "", $i); gotValues = $i }
      }
      if (gotLayer == layer && gotName == name) {
        print gotValues
        exit
      }
    }
  ' "$GO_LOG" > "$out"
}

extract_ref_values() {
  name=$1
  out=$2
  awk -v layer="$LAYER" -v name="$name" '
    $1=="DEBUG_VALUES" {
      gotName = ""
      gotValues = ""
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^name=/) { split($i, a, "="); gotName = a[2] }
        if ($i ~ /^values=/) { sub(/^values=/, "", $i); gotValues = $i }
      }
      target = name "-" layer
      if (gotName == target) {
        print gotValues
        exit
      }
    }
  ' "$REF_LOG" > "$out"
}

compare_csv() {
  left=$1
  right=$2
  label=$3
  awk -v a_file="$left" -v b_file="$right" -v label="$label" '
    function abs(x) { return x < 0 ? -x : x }
    BEGIN {
      if ((getline a < a_file) <= 0) { print "qkvprobe", "label=" label, "error=empty_left"; exit 1 }
      if ((getline b < b_file) <= 0) { print "qkvprobe", "label=" label, "error=empty_right"; exit 1 }
      na = split(a, aa, ",")
      nb = split(b, bb, ",")
      n = na
      if (nb < n) n = nb
      if (n <= 0) {
        print "qkvprobe", "label=" label, "error=no_overlap"
        exit 1
      }
      sum = 0
      mx = -1
      mxi = -1
      for (i = 1; i <= n; i++) {
        d = abs((aa[i] + 0) - (bb[i] + 0))
        sum += d
        if (d > mx) {
          mx = d
          mxi = i - 1
        }
      }
      mean = sum / n
      printf "qkvprobe label=%s n=%d mean_abs=%g max_abs=%g max_idx=%d left=%g right=%g\n", label, n, mean, mx, mxi, aa[mxi+1] + 0, bb[mxi+1] + 0
    }
  '
}

case "$FAMILY" in
  i2s)
    FIXTURE_FILE="$TESTDATA_DIR/model_fixture_i2s.txt"
    PROMPT_TOKENS_JSON="$TESTDATA_DIR/expected.i2s.prompt_tokens.json"
    ;;
  i2s_2b)
    FIXTURE_FILE="$TESTDATA_DIR/model_fixture_i2s_2b.txt"
    PROMPT_TOKENS_JSON="$TESTDATA_DIR/expected.i2s_2b.prompt_tokens.json"
    ;;
  *)
    echo "unknown family: $FAMILY (use i2s or i2s_2b)" >&2
    exit 1
    ;;
esac

fixture=$(sed -n '1p' "$FIXTURE_FILE")
case "$fixture" in
  /*) MODEL_PATH="$fixture" ;;
  *) MODEL_PATH="$TESTDATA_DIR/$fixture" ;;
esac

PROMPT_TOKENS_COUNT=$(count_json_items "$PROMPT_TOKENS_JSON")
POS=${BITNET_QKV_PROBE_POS:-$((PROMPT_TOKENS_COUNT - 1 + STEP))}
VALUES_N=${BITNET_QKV_PROBE_VALUES_N:-4096}

mkdir -p "$OUT_DIR"
GO_LOG="$OUT_DIR/i2s-drift-qkv-go-${FAMILY}.log"
REF_LOG="$OUT_DIR/i2s-drift-qkv-ref-${FAMILY}.log"
REPORT="$OUT_DIR/qkvprobe-${FAMILY}-L${LAYER}-S${STEP}.txt"

BITNET_DRIFT_TRACE_OUT="$GO_LOG" \
BITNET_DRIFT_TRACE_STEP="$STEP" \
BITNET_DRIFT_TRACE_LAYER="$LAYER" \
BITNET_DRIFT_TRACE_VALUES_N="$VALUES_N" \
BITNET_DRIFT_TRACE_PARITY_STRICT=1 \
./scripts/trace_i2s_drift_step.sh "$FAMILY" >/dev/null

BITNET_REF_DRIFT_TRACE_OUT="$REF_LOG" \
BITNET_DRIFT_TRACE_STEP="$STEP" \
BITNET_REF_DRIFT_TRACE_POS="$POS" \
BITNET_REF_DRIFT_VALUES_N="$VALUES_N" \
./scripts/trace_ref_i2s_drift_step.sh "$FAMILY" >/dev/null

GO_ATTN="$OUT_DIR/go-attn_norm-L${LAYER}.csv"
GO_Q="$OUT_DIR/go-qcur-L${LAYER}.csv"
GO_K="$OUT_DIR/go-kcur-L${LAYER}.csv"
GO_V="$OUT_DIR/go-vcur-L${LAYER}.csv"
REF_ATTN="$OUT_DIR/ref-attn_norm-L${LAYER}.csv"
REF_Q="$OUT_DIR/ref-qcur-L${LAYER}.csv"
REF_K="$OUT_DIR/ref-kcur-L${LAYER}.csv"
REF_V="$OUT_DIR/ref-vcur-L${LAYER}.csv"

extract_go_values "attn_norm" "$GO_ATTN"
extract_go_values "Qcur" "$GO_Q"
extract_go_values "Kcur" "$GO_K"
extract_go_values "Vcur" "$GO_V"
extract_ref_values "attn_norm" "$REF_ATTN"
extract_ref_values "Qcur" "$REF_Q"
extract_ref_values "Kcur" "$REF_K"
extract_ref_values "Vcur" "$REF_V"

for f in "$GO_ATTN" "$GO_Q" "$GO_K" "$GO_V" "$REF_ATTN" "$REF_Q" "$REF_K" "$REF_V"; do
  if [ ! -s "$f" ]; then
    echo "missing extracted vector: $f" >&2
    exit 1
  fi
done

{
  echo "qkvprobe family=$FAMILY layer=$LAYER step=$STEP pos=$POS model=$MODEL_PATH"
  compare_csv "$GO_ATTN" "$REF_ATTN" "go_attn_norm_vs_ref_attn_norm"
  go run ./cmd/qkvprobe --model "$MODEL_PATH" --layer "$LAYER" --input-csv "$GO_ATTN" --q-ref-csv "$GO_Q" --k-ref-csv "$GO_K" --v-ref-csv "$GO_V" --label "replay_go_input_vs_go_qkv"
  go run ./cmd/qkvprobe --model "$MODEL_PATH" --layer "$LAYER" --input-csv "$REF_ATTN" --q-ref-csv "$REF_Q" --k-ref-csv "$REF_K" --v-ref-csv "$REF_V" --label "replay_ref_input_vs_ref_qkv"
  go run ./cmd/qkvprobe --model "$MODEL_PATH" --layer "$LAYER" --input-csv "$GO_ATTN" --q-ref-csv "$REF_Q" --k-ref-csv "$REF_K" --v-ref-csv "$REF_V" --label "replay_go_input_vs_ref_qkv"
  go run ./cmd/qkvprobe --model "$MODEL_PATH" --layer "$LAYER" --input-csv "$REF_ATTN" --q-ref-csv "$GO_Q" --k-ref-csv "$GO_K" --v-ref-csv "$GO_V" --label "replay_ref_input_vs_go_qkv"
} | tee "$REPORT"

echo "qkvprobe report: $REPORT"
