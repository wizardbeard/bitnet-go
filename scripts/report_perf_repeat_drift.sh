#!/usr/bin/env sh
set -eu

CURRENT=${1:-.bench/perf-repeat-summary.tsv}
BASELINE=${2:-testdata/perf-repeat-summary-baseline.tsv}
OUT=${BITNET_REPEAT_DRIFT_OUT:-.bench/perf-repeat-drift.tsv}

if [ ! -f "$CURRENT" ]; then
  echo "current summary not found: $CURRENT" >&2
  exit 1
fi
if [ ! -f "$BASELINE" ]; then
  echo "baseline summary not found: $BASELINE" >&2
  exit 1
fi

./scripts/validate_perf_repeat_summary.sh "$CURRENT" >/dev/null
./scripts/validate_perf_repeat_summary.sh "$BASELINE" >/dev/null

mkdir -p "$(dirname "$OUT")"

tmp_unsorted=$(mktemp)
tmp=$(mktemp)
cleanup() {
  rm -f "$tmp_unsorted" "$tmp"
}
trap cleanup EXIT INT TERM

awk -v baseline="$BASELINE" -v current="$CURRENT" -F'\t' '
  function pct(cur, base) {
    if (base == 0) return ""
    return sprintf("%.3f", ((cur - base) / base) * 100.0)
  }
  FNR == 1 { next }
  FILENAME == baseline {
    b_seen[$1] = 1
    b_runtime_default[$1] = $2 + 0
    b_runtime_full[$1] = $3 + 0
    b_elapsed[$1] = $4 + 0
    b_tokps[$1] = $5 + 0
    next
  }
  FILENAME == current {
    c_seen[$1] = 1
    c_runtime_default[$1] = $2 + 0
    c_runtime_full[$1] = $3 + 0
    c_elapsed[$1] = $4 + 0
    c_tokps[$1] = $5 + 0
    next
  }
  END {
    for (t in b_seen) all[t] = 1
    for (t in c_seen) all[t] = 1

    for (t in all) {
      status = "ok"
      if (!(t in b_seen)) status = "missing_in_baseline"
      if (!(t in c_seen)) status = "missing_in_current"

      btd = (t in b_seen) ? sprintf("%.6f", b_tokps[t]) : ""
      ctd = (t in c_seen) ? sprintf("%.6f", c_tokps[t]) : ""
      bed = (t in b_seen) ? sprintf("%.6f", b_elapsed[t]) : ""
      ced = (t in c_seen) ? sprintf("%.6f", c_elapsed[t]) : ""
      brd = (t in b_seen) ? sprintf("%.6f", b_runtime_default[t]) : ""
      crd = (t in c_seen) ? sprintf("%.6f", c_runtime_default[t]) : ""
      brf = (t in b_seen) ? sprintf("%.6f", b_runtime_full[t]) : ""
      crf = (t in c_seen) ? sprintf("%.6f", c_runtime_full[t]) : ""

      tokps_delta = (status == "ok") ? pct(c_tokps[t], b_tokps[t]) : ""
      elapsed_delta = (status == "ok") ? pct(c_elapsed[t], b_elapsed[t]) : ""
      runtime_default_delta = (status == "ok") ? pct(c_runtime_default[t], b_runtime_default[t]) : ""
      runtime_full_delta = (status == "ok") ? pct(c_runtime_full[t], b_runtime_full[t]) : ""

      printf "%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
        t,
        status,
        btd,
        ctd,
        tokps_delta,
        bed,
        ced,
        elapsed_delta,
        brd,
        crd,
        runtime_default_delta,
        brf,
        crf,
        runtime_full_delta
    }
  }
' "$BASELINE" "$CURRENT" > "$tmp_unsorted"

sort -n "$tmp_unsorted" > "$tmp"

{
  printf "thread\tstatus\ttokps_baseline\ttokps_current\ttokps_delta_pct\telapsed_baseline_s\telapsed_current_s\telapsed_delta_pct\truntime_default_baseline_ns\truntime_default_current_ns\truntime_default_delta_pct\truntime_full_baseline_ns\truntime_full_current_ns\truntime_full_delta_pct\n"
  cat "$tmp"
} > "$OUT"

echo "[perf-repeat-drift] baseline: $BASELINE"
echo "[perf-repeat-drift] current:  $CURRENT"
echo "[perf-repeat-drift] report:   $OUT"
cat "$OUT"
