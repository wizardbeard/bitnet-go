#!/usr/bin/env sh
set -eu

SUMMARY=${1:-.bench/perf-repeat-summary.tsv}
OUT_ENV=${BITNET_REPEAT_BEST_ENV:-.bench/perf-repeat-best.env}

if [ ! -f "$SUMMARY" ]; then
  echo "summary not found: $SUMMARY" >&2
  exit 1
fi

best_line=$(awk -F'\t' '
  NR == 1 { next }
  NF < 5 { next }
  $5 == "" { next }
  {
    thread = $1 + 0
    runtime_default = $2 + 0
    runtime_full = $3 + 0
    elapsed = $4 + 0
    tokps = $5 + 0
    if (!seen || tokps > best_tokps || (tokps == best_tokps && elapsed < best_elapsed)) {
      seen = 1
      best_thread = thread
      best_tokps = tokps
      best_elapsed = elapsed
      best_runtime_default = runtime_default
      best_runtime_full = runtime_full
    }
  }
  END {
    if (!seen) exit 1
    printf "%d\t%.6f\t%.6f\t%.6f\t%.6f\n", best_thread, best_runtime_default, best_runtime_full, best_elapsed, best_tokps
  }
' "$SUMMARY")

best_thread=$(printf "%s" "$best_line" | awk -F'\t' '{print $1}')
best_runtime_default=$(printf "%s" "$best_line" | awk -F'\t' '{print $2}')
best_runtime_full=$(printf "%s" "$best_line" | awk -F'\t' '{print $3}')
best_elapsed=$(printf "%s" "$best_line" | awk -F'\t' '{print $4}')
best_tokps=$(printf "%s" "$best_line" | awk -F'\t' '{print $5}')

mkdir -p "$(dirname "$OUT_ENV")"
{
  printf "BITNET_MATVEC_THREADS=%s\n" "$best_thread"
  printf "BITNET_MATVEC_THREADS_MEDIAN_TOKPS=%s\n" "$best_tokps"
  printf "BITNET_MATVEC_THREADS_MEDIAN_ELAPSED_S=%s\n" "$best_elapsed"
  printf "BITNET_MATVEC_THREADS_MEDIAN_RUNTIME_DEFAULT_NS=%s\n" "$best_runtime_default"
  printf "BITNET_MATVEC_THREADS_MEDIAN_RUNTIME_FULL_NS=%s\n" "$best_runtime_full"
} > "$OUT_ENV"

echo "[perf-repeat-select] best_thread=$best_thread tokps=$best_tokps elapsed_s=$best_elapsed runtime_default_ns=$best_runtime_default runtime_full_ns=$best_runtime_full"
echo "[perf-repeat-select] env: $OUT_ENV"
