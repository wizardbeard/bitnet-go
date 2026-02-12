#!/usr/bin/env sh
set -eu

SUMMARY=${1:-.bench/perf-repeat-summary.tsv}

if [ ! -f "$SUMMARY" ]; then
  echo "summary not found: $SUMMARY" >&2
  exit 1
fi

awk -F'\t' '
  NR == 1 {
    expected = "thread\truntime_default_prefix_ns\truntime_full_sort_ns\te2e_elapsed_s\te2e_tokps\traw_tsv"
    if ($0 != expected) {
      printf "invalid header in %s\nexpected: %s\nactual:   %s\n", FILENAME, expected, $0 > "/dev/stderr"
      exit 1
    }
    next
  }
  {
    if (NF != 6) {
      printf "invalid field count at line %d: got %d, want 6\n", NR, NF > "/dev/stderr"
      exit 1
    }
    if ($1 !~ /^[0-9]+$/) {
      printf "invalid thread at line %d: %s\n", NR, $1 > "/dev/stderr"
      exit 1
    }
    if ($2 !~ /^-?[0-9]+(\.[0-9]+)?$/) {
      printf "invalid runtime_default_prefix_ns at line %d: %s\n", NR, $2 > "/dev/stderr"
      exit 1
    }
    if ($3 !~ /^-?[0-9]+(\.[0-9]+)?$/) {
      printf "invalid runtime_full_sort_ns at line %d: %s\n", NR, $3 > "/dev/stderr"
      exit 1
    }
    if ($4 !~ /^-?[0-9]+(\.[0-9]+)?$/) {
      printf "invalid e2e_elapsed_s at line %d: %s\n", NR, $4 > "/dev/stderr"
      exit 1
    }
    if ($5 !~ /^-?[0-9]+(\.[0-9]+)?$/) {
      printf "invalid e2e_tokps at line %d: %s\n", NR, $5 > "/dev/stderr"
      exit 1
    }
    if ($6 == "") {
      printf "empty raw_tsv at line %d\n", NR > "/dev/stderr"
      exit 1
    }
    rows++
  }
  END {
    if (NR < 2 || rows < 1) {
      printf "no data rows in %s\n", FILENAME > "/dev/stderr"
      exit 1
    }
  }
' "$SUMMARY"

echo "[perf-repeat-validate] summary ok: $SUMMARY"
