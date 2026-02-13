#!/usr/bin/env sh
set -eu

FAMILY=${1:-i2s}
LAYER=${BITNET_QKV_KQ_MODE_LAYER:-14}
STEP=${BITNET_QKV_KQ_MODE_STEP:-14}
KQ_LAYER_MAX=${BITNET_QKV_KQ_MODE_LAYER_MAX:-14}
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUT_BASE=${BITNET_QKV_KQ_MODE_OUT_BASE:-.bench/qkv-kq-modes}
SUMMARY=${BITNET_QKV_KQ_MODE_SUMMARY:-.bench/qkv-kq-modes-summary.tsv}

mkdir -p "$OUT_BASE" "$(dirname "$SUMMARY")"

extract_mean_abs() {
  file=$1
  label=$2
  awk -v target="$label" '
    {
      found = ""
      mean = ""
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^label=/) { split($i, a, "="); found = a[2] }
        if ($i ~ /^mean_abs=/) { split($i, a, "="); mean = a[2] }
      }
      if (found == target && mean != "") {
        print mean
        exit
      }
    }
  ' "$file"
}

run_case() {
  mode=$1
  out_dir="$OUT_BASE/$mode"
  mkdir -p "$out_dir"

  env \
    BITNET_QKV_PROBE_OUT_DIR="$out_dir" \
    BITNET_QKV_PROBE_LAYER="$LAYER" \
    BITNET_QKV_PROBE_STEP="$STEP" \
    BITNET_QKV_PROBE_PARITY_STRICT=0 \
    BITNET_STRICT_KQ=1 \
    BITNET_STRICT_KQ_LAYER_MAX="$KQ_LAYER_MAX" \
    BITNET_STRICT_KQ_MODE="$mode" \
    "$ROOT_DIR/scripts/probe_qkv_alignment.sh" "$FAMILY" >/dev/null

  report="$out_dir/qkvprobe-${FAMILY}-L${LAYER}-S${STEP}.txt"
  if [ ! -f "$report" ]; then
    echo "missing report for mode=$mode: $report" >&2
    exit 1
  fi

  go_q=$(extract_mean_abs "$report" "replay_go_input_vs_ref_qkv.Qcur")
  go_k=$(extract_mean_abs "$report" "replay_go_input_vs_ref_qkv.Kcur")
  go_v=$(extract_mean_abs "$report" "replay_go_input_vs_ref_qkv.Vcur")
  ref_q=$(extract_mean_abs "$report" "replay_ref_input_vs_go_qkv.Qcur")
  ref_k=$(extract_mean_abs "$report" "replay_ref_input_vs_go_qkv.Kcur")
  ref_v=$(extract_mean_abs "$report" "replay_ref_input_vs_go_qkv.Vcur")

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$mode" "$go_q" "$go_k" "$go_v" "$ref_q" "$ref_k" "$ref_v"
}

{
  printf "mode\tgo_in_ref_q\tgo_in_ref_k\tgo_in_ref_v\tref_in_go_q\tref_in_go_k\tref_in_go_v\n"
  run_case "ggml"
  run_case "naive"
  run_case "f64"
} > "$SUMMARY"

echo "qkv strict-kq mode summary: $SUMMARY"
cat "$SUMMARY"
