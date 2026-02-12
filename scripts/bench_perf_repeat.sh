#!/usr/bin/env sh
set -eu

RUNS=${BITNET_REPEAT_RUNS:-5}
MODEL=${BITNET_REPEAT_MODEL:-testdata/ggml-model-i2_s.gguf}
PROMPT=${BITNET_REPEAT_PROMPT:-Hello BitNet}
PROMPT_FILE=${BITNET_REPEAT_PROMPT_FILE:-}
TOKENS=${BITNET_REPEAT_TOKENS:-15}
TEMP=${BITNET_REPEAT_TEMP:-0}
TOP_P=${BITNET_REPEAT_TOP_P:-1}
PROCS=${BITNET_REPEAT_PROCS:-6}
GO_BIN=${BITNET_REPEAT_GO_BIN:-.bench/bitnet-go}
GOCACHE_DIR=${BITNET_REPEAT_GOCACHE:-/tmp/go-build}
RUNTIME_PKG=${BITNET_REPEAT_RUNTIME_PKG:-./internal/runtime}
RUNTIME_BENCH=${BITNET_REPEAT_RUNTIME_BENCH:-BenchmarkGenerateTopPCompare}
RUNTIME_BENCH_TIME=${BITNET_REPEAT_RUNTIME_BENCH_TIME:-2x}
SKIP_RUNTIME=${BITNET_REPEAT_SKIP_RUNTIME:-0}
SKIP_E2E=${BITNET_REPEAT_SKIP_E2E:-0}
OUT_TSV=${BITNET_REPEAT_OUT:-.bench/perf-repeat.tsv}

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

if [ -n "$PROMPT_FILE" ]; then
  PROMPT=$(cat "$PROMPT_FILE")
fi

case "$MODEL" in
  /*) MODEL_ABS=$MODEL ;;
  *) MODEL_ABS=$ROOT_DIR/$MODEL ;;
esac

mkdir -p "$(dirname "$OUT_TSV")"

tmpdir=$(mktemp -d)
cleanup() {
  rm -rf "$tmpdir"
}
trap cleanup EXIT INT TERM

runtime_default_file="$tmpdir/runtime_default_prefix.ns"
runtime_full_file="$tmpdir/runtime_full_sort.ns"
e2e_elapsed_file="$tmpdir/e2e_elapsed.s"
e2e_tokps_file="$tmpdir/e2e_tokps"

touch "$runtime_default_file" "$runtime_full_file" "$e2e_elapsed_file" "$e2e_tokps_file"

stats() {
  file=$1
  label=$2
  unit=$3
  if [ ! -s "$file" ]; then
    printf "%s: n=0\n" "$label"
    return
  fi
  sorted="$tmpdir/sorted.$$"
  sort -n "$file" > "$sorted"
  awk -v label="$label" -v unit="$unit" '
    { a[NR] = $1; sum += $1 }
    END {
      n = NR
      if (n == 0) {
        printf "%s: n=0\n", label
        exit
      }
      if (n % 2 == 1) {
        med = a[(n + 1) / 2]
      } else {
        med = (a[n / 2] + a[n / 2 + 1]) / 2
      }
      p95i = int((95 * n + 99) / 100)
      if (p95i < 1) p95i = 1
      if (p95i > n) p95i = n
      p95 = a[p95i]
      min = a[1]
      max = a[n]
      mean = sum / n
      printf "%s: n=%d median=%.6f%s p95=%.6f%s mean=%.6f%s min=%.6f%s max=%.6f%s\n",
        label, n, med, unit, p95, unit, mean, unit, min, unit, max, unit
    }
  ' "$sorted"
  rm -f "$sorted"
}

printf "run\truntime_default_prefix_ns\truntime_full_sort_ns\te2e_elapsed_s\te2e_tokps\n" > "$OUT_TSV"

printf "Building %s...\n" "$GO_BIN"
GOCACHE="$GOCACHE_DIR" go build -o "$GO_BIN" ./cmd/bitnet

i=1
while [ "$i" -le "$RUNS" ]; do
  runtime_default=""
  runtime_full=""
  e2e_elapsed=""
  e2e_tokps=""

  if [ "$SKIP_RUNTIME" != "1" ]; then
    runtime_out="$tmpdir/runtime.$i.txt"
    GOCACHE="$GOCACHE_DIR" BITNET_BENCH_MODEL="$MODEL_ABS" go test "$RUNTIME_PKG" \
      -run '^$' \
      -bench "$RUNTIME_BENCH" \
      -benchmem \
      -benchtime "$RUNTIME_BENCH_TIME" > "$runtime_out"

    runtime_default=$(awk '/BenchmarkGenerateTopPCompare\/default_prefix/ {print $3; exit}' "$runtime_out")
    runtime_full=$(awk '/BenchmarkGenerateTopPCompare\/full_sort/ {print $3; exit}' "$runtime_out")
    if [ -z "${runtime_default:-}" ] || [ -z "${runtime_full:-}" ]; then
      echo "failed to parse runtime benchmark output for run $i" >&2
      cat "$runtime_out" >&2
      exit 1
    fi
    printf "%s\n" "$runtime_default" >> "$runtime_default_file"
    printf "%s\n" "$runtime_full" >> "$runtime_full_file"
  fi

  if [ "$SKIP_E2E" != "1" ]; then
    start=$(date +%s%N)
    out=$("$GO_BIN" \
      --model "$MODEL_ABS" \
      --prompt "$PROMPT" \
      --max-tokens "$TOKENS" \
      --seed 1 \
      --temp "$TEMP" \
      --top-p "$TOP_P" \
      --procs "$PROCS")
    end=$(date +%s%N)

    e2e_elapsed=$(awk "BEGIN{printf \"%.6f\", ($end-$start)/1e9}")
    tokens=$(printf "%s\n" "$out" | awk -F'tokens=' 'NF>1{split($2,a," "); print a[1]; exit}')
    if [ -z "${tokens:-}" ]; then
      tokens=$TOKENS
    fi
    e2e_tokps=$(awk -v t="$tokens" -v e="$e2e_elapsed" 'BEGIN{if (e>0) printf "%.6f", t/e; else print "0"}')

    printf "%s\n" "$e2e_elapsed" >> "$e2e_elapsed_file"
    printf "%s\n" "$e2e_tokps" >> "$e2e_tokps_file"
  fi

  printf "%s\t%s\t%s\t%s\t%s\n" "$i" "${runtime_default:-}" "${runtime_full:-}" "${e2e_elapsed:-}" "${e2e_tokps:-}" >> "$OUT_TSV"
  printf "run %d/%d done\n" "$i" "$RUNS"
  i=$((i + 1))
done

echo ""
echo "Summary"
if [ "$SKIP_RUNTIME" != "1" ]; then
  stats "$runtime_default_file" "runtime.default_prefix" " ns/op"
  stats "$runtime_full_file" "runtime.full_sort" " ns/op"
fi
if [ "$SKIP_E2E" != "1" ]; then
  stats "$e2e_elapsed_file" "e2e.elapsed" " s"
  stats "$e2e_tokps_file" "e2e.tokps" " tok/s"
fi
echo "raw: $OUT_TSV"
