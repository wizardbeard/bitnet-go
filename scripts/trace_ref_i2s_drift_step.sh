#!/usr/bin/env sh
set -eu

FAMILY=${1:-i2s}
STEP=${BITNET_DRIFT_TRACE_STEP:-14}
TOKEN=${BITNET_DRIFT_TRACE_TOKEN:-55358}
MAX_TOKENS=${BITNET_REF_DRIFT_MAX_TOKENS:-$((STEP + 1))}
VALUES_N=${BITNET_REF_DRIFT_VALUES_N:-16}
OUT=${BITNET_REF_DRIFT_TRACE_OUT:-.bench/ref-i2s-drift-trace-${FAMILY}.log}
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

if [ ! -f "$FIXTURE_FILE" ]; then
  echo "missing fixture file: $FIXTURE_FILE" >&2
  exit 1
fi
fixture=$(sed -n '1p' "$FIXTURE_FILE")
if [ -z "$fixture" ]; then
  echo "empty fixture file: $FIXTURE_FILE" >&2
  exit 1
fi
case "$fixture" in
  /*) MODEL_PATH="$fixture" ;;
  *) MODEL_PATH="$TESTDATA_DIR/$fixture" ;;
esac
if [ ! -f "$MODEL_PATH" ]; then
  echo "model not found: $MODEL_PATH" >&2
  exit 1
fi

PROMPT_FILE=${BITNET_REF_DRIFT_PROMPT_FILE:-$TESTDATA_DIR/prompt.txt}
if [ ! -f "$PROMPT_FILE" ]; then
  echo "prompt file not found: $PROMPT_FILE" >&2
  exit 1
fi
PROMPT=$(cat "$PROMPT_FILE")

PROMPT_TOKENS_COUNT=$(count_json_items "$PROMPT_TOKENS_JSON")
if [ "$PROMPT_TOKENS_COUNT" -le 0 ]; then
  echo "prompt token vector missing/empty: $PROMPT_TOKENS_JSON" >&2
  echo "run scripts/run_ref_${FAMILY}.sh first to freeze vectors" >&2
  exit 1
fi
POS=${BITNET_REF_DRIFT_TRACE_POS:-$((PROMPT_TOKENS_COUNT - 1 + STEP))}

mkdir -p "$(dirname "$OUT")"

"$ROOT_DIR/scripts/build_ref.sh" >/dev/null
"$ROOT_DIR/scripts/build_ref_tracer.sh" >/dev/null

if [ ! -x "$ROOT_DIR/.ref/bin/ref-trace" ]; then
  echo "missing tracer binary: $ROOT_DIR/.ref/bin/ref-trace" >&2
  exit 1
fi

env \
  BITNET_REF_MODEL="$MODEL_PATH" \
  BITNET_REF_PROMPT="$PROMPT" \
  BITNET_REF_SEED=1 \
  BITNET_REF_MAX_TOKENS="$MAX_TOKENS" \
  BITNET_REF_TOPK=5 \
  BITNET_REF_DEBUG=1 \
  BITNET_REF_DEBUG_VALUES=1 \
  BITNET_REF_DEBUG_VALUES_N="$VALUES_N" \
  BITNET_REF_DEBUG_POS="$POS" \
  "$ROOT_DIR/.ref/bin/ref-trace" >"$OUT" 2>&1

echo "[ref-drift-trace] family=$FAMILY step=$STEP pos=$POS token=$TOKEN prompt_tokens=$PROMPT_TOKENS_COUNT values_n=$VALUES_N"
echo "[ref-drift-trace] log: $OUT"
grep -E '^TOPK step='"$STEP"'|^TOKEN step='"$STEP"'|^DEBUG name=result_norm|^DEBUG_VALUES name=(result_norm|attn_norm|Qcur|Kcur|Vcur|kq_soft_max_ext|attn_sub_norm|attn_o_out|ffn_inp|ffn_sub_norm|ffn_out|l_out)-|^DEBUG name=(attn_o_out|ffn_gate|ffn_up|ffn_out|ffn_down|l_out)-' "$OUT" || true
