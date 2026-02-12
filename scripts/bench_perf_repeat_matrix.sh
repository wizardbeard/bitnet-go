#!/usr/bin/env sh
set -eu

THREADS=${BITNET_REPEAT_THREADS:-"1 4 6 8"}
SUMMARY=${BITNET_REPEAT_SUMMARY:-.bench/perf-repeat-summary.tsv}
OUT_PREFIX=${BITNET_REPEAT_OUT_PREFIX:-.bench/perf-repeat-threads}

mkdir -p "$(dirname "$SUMMARY")"

printf "thread\truntime_default_prefix_ns\truntime_full_sort_ns\te2e_elapsed_s\te2e_tokps\traw_tsv\n" > "$SUMMARY"

median_col() {
  file=$1
  col=$2
  awk -F'\t' -v c="$col" '
    NR == 1 { next }
    $0 == "" { next }
    {
      v = $c
      if (v == "") next
      n++
      a[n] = v + 0
    }
    END {
      if (n == 0) {
        print ""
        exit
      }
      for (i = 1; i <= n; i++) {
        for (j = i + 1; j <= n; j++) {
          if (a[j] < a[i]) {
            t = a[i]
            a[i] = a[j]
            a[j] = t
          }
        }
      }
      if (n % 2 == 1) {
        m = a[(n + 1) / 2]
      } else {
        m = (a[n / 2] + a[n / 2 + 1]) / 2
      }
      printf "%.6f\n", m
    }
  ' "$file"
}

for t in $THREADS; do
  out_tsv="${OUT_PREFIX}${t}.tsv"
  echo "[perf-repeat-matrix] BITNET_MATVEC_THREADS=$t -> $out_tsv"
  BITNET_MATVEC_THREADS="$t" BITNET_REPEAT_OUT="$out_tsv" ./scripts/bench_perf_repeat.sh

  med_runtime_default=$(median_col "$out_tsv" 2)
  med_runtime_full=$(median_col "$out_tsv" 3)
  med_e2e_elapsed=$(median_col "$out_tsv" 4)
  med_e2e_tokps=$(median_col "$out_tsv" 5)

  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$t" \
    "$med_runtime_default" \
    "$med_runtime_full" \
    "$med_e2e_elapsed" \
    "$med_e2e_tokps" \
    "$out_tsv" >> "$SUMMARY"

done

echo ""
echo "[perf-repeat-matrix] summary: $SUMMARY"
cat "$SUMMARY"
