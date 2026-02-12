#!/usr/bin/env sh
set -u

FETCH=${BITNET_AUDIT_FETCH:-0}
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"

summary_tmp=$(mktemp)
cleanup() {
  rm -f "$summary_tmp"
}
trap cleanup EXIT INT TERM

echo "| Stage | Status |" >"$summary_tmp"
echo "| --- | --- |" >>"$summary_tmp"

append_summary() {
  stage=$1
  status=$2
  echo "| $stage | $status |" >>"$summary_tmp"
}

emit_summary() {
  echo ""
  echo "[cpu-parity-audit] summary"
  cat "$summary_tmp"
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    {
      echo "## CPU Parity Audit"
      cat "$summary_tmp"
    } >>"$GITHUB_STEP_SUMMARY"
  fi
}

run_stage() {
  stage=$1
  shift
  echo ""
  echo "[cpu-parity-audit] stage=$stage"
  echo "[cpu-parity-audit] cmd: $*"
  if "$@"; then
    append_summary "$stage" "PASS"
    return 0
  fi
  append_summary "$stage" "FAIL"
  emit_summary
  echo "[cpu-parity-audit] FAIL stage=$stage" >&2
  exit 1
}

if [ "$FETCH" = "1" ]; then
  run_stage "fetch-fixtures" env BITNET_FETCH_YARN=1 BITNET_FETCH_I2S_2B=1 sh ./scripts/fetch_testdata_gguf.sh
fi

run_stage "tokenizer-vectors" go test ./internal/tokenizer -run 'TestTokenizerGPT2FixturePrompt|TestTokenizerFalconFixturePrompt|TestTokenizerQwen2FixturePrompt|TestTokenizerI2SFixturePrompt|TestTokenizerI2S2BFixturePrompt|TestTokenizerYarnFixturePrompt|TestTokenizerKnownPreTypesForFixtures' -count=1

run_stage "gguf-fixture-compat" go test ./internal/gguf -run TestMaintainedFixtureTensorTypesSupported -count=1

run_stage "parity-base" env \
  BITNET_ENFORCE_PARITY=1 \
  BITNET_PARITY_LOGIT_ATOL=1e-1 \
  BITNET_PARITY_LOGIT_RTOL=1e-1 \
  BITNET_PARITY_TOPK_STRICT=1 \
  go test ./pkg/bitnet -run TestParityAgainstFrozenVectors -count=1

run_stage "parity-yarn" env \
  BITNET_ENFORCE_YARN=1 \
  BITNET_PARITY_STRICT=1 \
  BITNET_PARITY_LOGIT_ATOL=1e-3 \
  BITNET_PARITY_LOGIT_RTOL=3e-2 \
  BITNET_PARITY_TOPK_STRICT=1 \
  go test ./pkg/bitnet -run TestParityAgainstYarnVectors -count=1

run_stage "parity-i2s" env \
  BITNET_ENFORCE_I2S=1 \
  BITNET_PARITY_FORCE=1 \
  BITNET_PARITY_STRICT=1 \
  BITNET_I2S_FORCE_LOGIT_ATOL=7e-2 \
  BITNET_I2S_FORCE_LOGIT_RTOL=7e-2 \
  BITNET_I2S_TOPK_STRICT=3 \
  BITNET_PARITY_FORCE_RELAX_TOPK=1 \
  go test ./pkg/bitnet -run TestParityAgainstI2SVectors -count=1

run_stage "parity-i2s-2b" env \
  BITNET_ENFORCE_I2S_2B=1 \
  BITNET_PARITY_FORCE=1 \
  BITNET_PARITY_STRICT=1 \
  BITNET_I2S_FORCE_LOGIT_ATOL=7e-2 \
  BITNET_I2S_FORCE_LOGIT_RTOL=7e-2 \
  BITNET_I2S_TOPK_STRICT=3 \
  BITNET_PARITY_FORCE_RELAX_TOPK=1 \
  go test ./pkg/bitnet -run TestParityAgainstI2S2BVectors -count=1

run_stage "smoke-i2s" env \
  BITNET_ENFORCE_I2S_SMOKE=1 \
  BITNET_PARITY_FORCE=1 \
  BITNET_PARITY_STRICT=1 \
  go test ./pkg/bitnet -run TestParityAgainstI2SSmoke -count=1

run_stage "smoke-i2s-2b" env \
  BITNET_ENFORCE_I2S_2B_SMOKE=1 \
  BITNET_PARITY_FORCE=1 \
  BITNET_PARITY_STRICT=1 \
  go test ./pkg/bitnet -run TestParityAgainstI2S2BSmoke -count=1

run_stage "seed-determinism" env \
  BITNET_ENFORCE_SEED_DETERMINISM=1 \
  BITNET_SEED_DETERMINISM_SEED=7 \
  BITNET_SEED_DETERMINISM_MAX_TOKENS=8 \
  BITNET_SEED_DETERMINISM_TEMP=0.8 \
  BITNET_SEED_DETERMINISM_TOP_P=0.9 \
  BITNET_SEED_DETERMINISM_TOP_K=40 \
  go test ./pkg/bitnet -run TestSeedDeterminismFixtures -count=1

emit_summary
echo ""
echo "[cpu-parity-audit] PASS"
