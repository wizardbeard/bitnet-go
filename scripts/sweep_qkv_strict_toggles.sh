#!/usr/bin/env sh
set -eu

FAMILY=${1:-i2s}
LAYER=${BITNET_QKV_SWEEP_LAYER:-14}
STEP=${BITNET_QKV_SWEEP_STEP:-14}
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUT_BASE=${BITNET_QKV_SWEEP_OUT_BASE:-.bench/qkv-strict-sweep}
SUMMARY=${BITNET_QKV_SWEEP_SUMMARY:-.bench/qkv-strict-sweep-summary.tsv}

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
  name=$1
  env_key=$2
  env_val=$3
  out_dir="$OUT_BASE/$name"
  mkdir -p "$out_dir"

  if [ -n "$env_key" ]; then
    env \
      BITNET_QKV_PROBE_OUT_DIR="$out_dir" \
      BITNET_QKV_PROBE_LAYER="$LAYER" \
      BITNET_QKV_PROBE_STEP="$STEP" \
      BITNET_QKV_PROBE_PARITY_STRICT=0 \
      "$env_key=$env_val" \
      "$ROOT_DIR/scripts/probe_qkv_alignment.sh" "$FAMILY" >/dev/null
  else
    env \
      BITNET_QKV_PROBE_OUT_DIR="$out_dir" \
      BITNET_QKV_PROBE_LAYER="$LAYER" \
      BITNET_QKV_PROBE_STEP="$STEP" \
      BITNET_QKV_PROBE_PARITY_STRICT=0 \
      "$ROOT_DIR/scripts/probe_qkv_alignment.sh" "$FAMILY" >/dev/null
  fi

  report="$out_dir/qkvprobe-${FAMILY}-L${LAYER}-S${STEP}.txt"
  if [ ! -f "$report" ]; then
    echo "missing report for case=$name: $report" >&2
    exit 1
  fi

  go_q=$(extract_mean_abs "$report" "replay_go_input_vs_ref_qkv.Qcur")
  go_k=$(extract_mean_abs "$report" "replay_go_input_vs_ref_qkv.Kcur")
  go_v=$(extract_mean_abs "$report" "replay_go_input_vs_ref_qkv.Vcur")
  ref_q=$(extract_mean_abs "$report" "replay_ref_input_vs_go_qkv.Qcur")
  ref_k=$(extract_mean_abs "$report" "replay_ref_input_vs_go_qkv.Kcur")
  ref_v=$(extract_mean_abs "$report" "replay_ref_input_vs_go_qkv.Vcur")

  if [ -z "$go_q" ] || [ -z "$go_k" ] || [ -z "$go_v" ] || [ -z "$ref_q" ] || [ -z "$ref_k" ] || [ -z "$ref_v" ]; then
    echo "failed to parse report for case=$name: $report" >&2
    exit 1
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$name" "$go_q" "$go_k" "$go_v" "$ref_q" "$ref_k" "$ref_v"
}

{
  printf "case\tgo_in_ref_q\tgo_in_ref_k\tgo_in_ref_v\tref_in_go_q\tref_in_go_k\tref_in_go_v\n"
  run_case "baseline" "" ""
  run_case "strict_kq" "BITNET_STRICT_KQ" "1"
  run_case "strict_attention" "BITNET_STRICT_ATTENTION" "1"
  run_case "strict_expf" "BITNET_STRICT_EXPF" "1"
  run_case "strict_attention_ref" "BITNET_STRICT_ATTENTION_REF" "1"
} > "$SUMMARY"

echo "qkv strict sweep summary: $SUMMARY"
cat "$SUMMARY"
