#!/usr/bin/env sh
set -eu

FAMILY=${1:-i2s}
LAYER=${BITNET_VCUR_PROBE_LAYER:-14}
STEP=${BITNET_VCUR_PROBE_STEP:-14}
OUT_DIR=${BITNET_VCUR_PROBE_OUT_DIR:-.bench}
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

PROMPT_FILE=${BITNET_VCUR_PROBE_PROMPT_FILE:-$TESTDATA_DIR/prompt.txt}
PROMPT=$(cat "$PROMPT_FILE")
PROMPT_TOKENS_COUNT=$(count_json_items "$PROMPT_TOKENS_JSON")
POS=${BITNET_VCUR_PROBE_POS:-$((PROMPT_TOKENS_COUNT - 1 + STEP))}
MAX_TOKENS=${BITNET_VCUR_PROBE_MAX_TOKENS:-$((STEP + 1))}

mkdir -p "$OUT_DIR"
NORM_LOG="$OUT_DIR/ref-vcur-probe-attn_norm-L${LAYER}.log"
VCUR_LOG="$OUT_DIR/ref-vcur-probe-vcur-L${LAYER}.log"
NORM_CSV="$OUT_DIR/ref-vcur-probe-attn_norm-L${LAYER}.csv"
VCUR_CSV="$OUT_DIR/ref-vcur-probe-vcur-L${LAYER}.csv"

"$ROOT_DIR/scripts/build_ref.sh" >/dev/null
"$ROOT_DIR/scripts/build_ref_tracer.sh" >/dev/null

env \
  BITNET_REF_MODEL="$MODEL_PATH" \
  BITNET_REF_PROMPT="$PROMPT" \
  BITNET_REF_SEED=1 \
  BITNET_REF_MAX_TOKENS="$MAX_TOKENS" \
  BITNET_REF_TOPK=5 \
  BITNET_REF_DEBUG=1 \
  BITNET_REF_DEBUG_VALUES=1 \
  BITNET_REF_DEBUG_VALUES_N=4096 \
  BITNET_REF_DEBUG_VALUES_NAME="attn_norm-${LAYER}" \
  BITNET_REF_DEBUG_POS="$POS" \
  "$ROOT_DIR/.ref/bin/ref-trace" >"$NORM_LOG" 2>&1

env \
  BITNET_REF_MODEL="$MODEL_PATH" \
  BITNET_REF_PROMPT="$PROMPT" \
  BITNET_REF_SEED=1 \
  BITNET_REF_MAX_TOKENS="$MAX_TOKENS" \
  BITNET_REF_TOPK=5 \
  BITNET_REF_DEBUG=1 \
  BITNET_REF_DEBUG_VALUES=1 \
  BITNET_REF_DEBUG_VALUES_N=4096 \
  BITNET_REF_DEBUG_VALUES_NAME="Vcur-${LAYER}" \
  BITNET_REF_DEBUG_POS="$POS" \
  "$ROOT_DIR/.ref/bin/ref-trace" >"$VCUR_LOG" 2>&1

sed -n 's/^DEBUG_VALUES name=attn_norm-'"$LAYER"' values=//p' "$NORM_LOG" | head -n1 >"$NORM_CSV"
sed -n 's/^DEBUG_VALUES name=Vcur-'"$LAYER"' values=//p' "$VCUR_LOG" | head -n1 >"$VCUR_CSV"

if [ ! -s "$NORM_CSV" ] || [ ! -s "$VCUR_CSV" ]; then
  echo "failed to extract attn_norm/Vcur vectors" >&2
  exit 1
fi

go run ./cmd/vcurprobe \
  --model "$MODEL_PATH" \
  --layer "$LAYER" \
  --attn-norm-csv "$NORM_CSV" \
  --vcur-ref-csv "$VCUR_CSV"

