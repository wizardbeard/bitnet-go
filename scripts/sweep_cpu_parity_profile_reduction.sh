#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"

OUT_DIR=${BITNET_PARITY_REDUCTION_OUT_DIR:-.bench}
SUMMARY=${BITNET_PARITY_REDUCTION_SUMMARY:-$OUT_DIR/cpu-parity-profile-reduction.tsv}
mkdir -p "$OUT_DIR"

run_case() {
  family=$1
  name=$2
  shift 2

  case "$family" in
    i2s)
      enforce_name=BITNET_ENFORCE_I2S
      test_re='TestParityAgainstI2SVectors'
      ;;
    i2s_2b)
      enforce_name=BITNET_ENFORCE_I2S_2B
      test_re='TestParityAgainstI2S2BVectors'
      ;;
    *)
      echo "unknown family: $family" >&2
      exit 1
      ;;
  esac

  log="$OUT_DIR/cpu-parity-reduction-$family-$name.log"
  set +e
  env \
    BITNET_PARITY_PROFILE=cpu_parity_v1 \
    "$enforce_name=1" \
    BITNET_PARITY_FORCE=1 \
    BITNET_PARITY_STRICT=0 \
    BITNET_PARITY_FORCE_RELAX_TOPK=1 \
    "$@" \
    go test ./pkg/bitnet -run "$test_re" -count=1 -v >"$log" 2>&1
  status=$?
  set -e

  mismatch=$(awk '/topk logit mismatch step=/{print; exit}' "$log")
  mismatch_step=$(printf "%s\n" "$mismatch" | sed -n 's/.*step=\([0-9][0-9]*\) token=.*/\1/p')
  mismatch_token=$(printf "%s\n" "$mismatch" | sed -n 's/.*token=\([0-9][0-9]*\): got=.*/\1/p')
  got=$(printf "%s\n" "$mismatch" | sed -n 's/.* got=\([^ ]*\) .*/\1/p')
  want=$(printf "%s\n" "$mismatch" | sed -n 's/.* want=\([^ ]*\) .*/\1/p')

  abs_err=""
  if [ -n "$got" ] && [ -n "$want" ]; then
    abs_err=$(awk -v g="$got" -v w="$want" 'BEGIN{d=g-w; if(d<0)d=-d; printf "%.9g\n", d}')
  fi

  printf "%s\t%s\t%d\t%s\t%s\t%s\t%s\t%s\n" \
    "$family" "$name" "$status" \
    "${mismatch_step:-}" "${mismatch_token:-}" "${got:-}" "${want:-}" "${abs_err:-}"
}

{
  printf "family\tcase\ttest_status\tmismatch_step\tmismatch_token\tmismatch_got\tmismatch_want\tmismatch_abs_err\n"

  for family in i2s i2s_2b; do
    run_case "$family" "baseline"
    run_case "$family" "kq_l13" BITNET_STRICT_KQ_LAYER_MAX=13
    run_case "$family" "kq_l12" BITNET_STRICT_KQ_LAYER_MAX=12
    run_case "$family" "kq_l10" BITNET_STRICT_KQ_LAYER_MAX=10
    run_case "$family" "kq_l13_expf_l0" BITNET_STRICT_KQ_LAYER_MAX=13 BITNET_STRICT_EXPF=1 BITNET_STRICT_EXPF_LAYER_MAX=0
    run_case "$family" "kq_l14_no_expf" BITNET_STRICT_KQ=1 BITNET_STRICT_KQ_LAYER_MAX=14 BITNET_STRICT_EXPF=0
    run_case "$family" "no_expf" BITNET_STRICT_EXPF=0
    run_case "$family" "expf_all_layers" BITNET_STRICT_EXPF=1 BITNET_STRICT_EXPF_LAYER_MAX=255
  done
} > "$SUMMARY"

echo "parity profile reduction summary: $SUMMARY"
cat "$SUMMARY"
