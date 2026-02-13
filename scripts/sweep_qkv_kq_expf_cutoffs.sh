#!/usr/bin/env sh
set -eu

FAMILY=${1:-i2s}
LAYER=${BITNET_QKV_CUTOFF_LAYER:-14}
STEP=${BITNET_QKV_CUTOFF_STEP:-14}
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUT_BASE=${BITNET_QKV_CUTOFF_OUT_BASE:-.bench/qkv-kq-expf-cutoffs}
SUMMARY=${BITNET_QKV_CUTOFF_SUMMARY:-.bench/qkv-kq-expf-cutoffs-summary.tsv}
KQ_MAX_LIST=${BITNET_QKV_CUTOFF_KQ_MAXS:-0 7 14 29}

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
  shift
  out_dir="$OUT_BASE/$name"
  mkdir -p "$out_dir"

  env \
    BITNET_QKV_PROBE_OUT_DIR="$out_dir" \
    BITNET_QKV_PROBE_LAYER="$LAYER" \
    BITNET_QKV_PROBE_STEP="$STEP" \
    BITNET_QKV_PROBE_PARITY_STRICT=0 \
    "$@" \
    "$ROOT_DIR/scripts/probe_qkv_alignment.sh" "$FAMILY" >/dev/null

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
  run_case "baseline"
  run_case "expf_l0" BITNET_STRICT_EXPF=1 BITNET_STRICT_EXPF_LAYER_MAX=0
  for m in $KQ_MAX_LIST; do
    run_case "kq_l${m}" BITNET_STRICT_KQ=1 BITNET_STRICT_KQ_LAYER_MAX="$m"
    run_case "kq_l${m}_expf_l0" BITNET_STRICT_KQ=1 BITNET_STRICT_KQ_LAYER_MAX="$m" BITNET_STRICT_EXPF=1 BITNET_STRICT_EXPF_LAYER_MAX=0
  done
} > "$SUMMARY"

echo "qkv kq+expf cutoff summary: $SUMMARY"
cat "$SUMMARY"
