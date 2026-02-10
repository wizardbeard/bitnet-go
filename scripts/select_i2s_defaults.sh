#!/usr/bin/env sh
set -eu

SUMMARY_FILE=${1:-.bench/i2s-kernels-sweep-summary.tsv}
OUT_FILE=${BITNET_I2S_DEFAULTS_OUT:-.bench/i2s-kernels-defaults.env}

if [ ! -f "$SUMMARY_FILE" ]; then
  echo "missing summary file: $SUMMARY_FILE" >&2
  exit 1
fi

best_line=$(
  awk -F '\t' 'NR>1 && $8 != "NA" {
    if (best == "" || ($8+0) < best+0) {
      best = $8
      line = $0
    }
  } END {print line}' "$SUMMARY_FILE"
)

if [ -z "$best_line" ]; then
  echo "no valid sweep rows in: $SUMMARY_FILE" >&2
  exit 1
fi

label=$(printf '%s\n' "$best_line" | awk -F '\t' '{print $1}')
rows_min=$(printf '%s\n' "$best_line" | awk -F '\t' '{print $2}')
cols_min=$(printf '%s\n' "$best_line" | awk -F '\t' '{print $3}')
chunk_rows=$(printf '%s\n' "$best_line" | awk -F '\t' '{print $4}')
chunk_cols=$(printf '%s\n' "$best_line" | awk -F '\t' '{print $5}')
block_min_rows=$(printf '%s\n' "$best_line" | awk -F '\t' '{print $6}')
avg_ns=$(printf '%s\n' "$best_line" | awk -F '\t' '{print $8}')

mkdir -p "$(dirname "$OUT_FILE")"
cat >"$OUT_FILE" <<EOF
# Suggested i2_s+i8_s defaults from sweep summary: $SUMMARY_FILE
# Winner: $label (avg_dispatch_ns=$avg_ns)
BITNET_ARM64_I2S_I8S_PAR_ROWS_MIN=$rows_min
BITNET_ARM64_I2S_I8S_PAR_COLS_MIN=$cols_min
BITNET_ARM64_I2S_I8S_PAR_CHUNK_ROWS=$chunk_rows
BITNET_ARM64_I2S_I8S_PAR_CHUNK_COLS=$chunk_cols
BITNET_ARM64_I2S_I8S_BLOCK_MIN_ROWS=$block_min_rows
EOF

printf "winner=%s avg_dispatch_ns=%s\n" "$label" "$avg_ns"
cat "$OUT_FILE"

