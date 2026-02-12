#!/usr/bin/env sh
set -eu

FETCH=${BITNET_AUDIT_FETCH:-0}
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"

if [ "$FETCH" = "1" ]; then
  echo "[cpu-parity-audit] fetching fixtures"
  BITNET_FETCH_YARN=1 BITNET_FETCH_I2S_2B=1 sh ./scripts/fetch_testdata_gguf.sh
fi

run() {
  echo ""
  echo "[cpu-parity-audit] $*"
  "$@"
}

run go test ./internal/tokenizer -run 'TestTokenizerGPT2FixturePrompt|TestTokenizerFalconFixturePrompt|TestTokenizerQwen2FixturePrompt|TestTokenizerI2SFixturePrompt|TestTokenizerI2S2BFixturePrompt|TestTokenizerYarnFixturePrompt|TestTokenizerKnownPreTypesForFixtures' -count=1

run go test ./internal/gguf -run TestMaintainedFixtureTensorTypesSupported -count=1

run env \
  BITNET_ENFORCE_PARITY=1 \
  BITNET_PARITY_LOGIT_ATOL=1e-1 \
  BITNET_PARITY_LOGIT_RTOL=1e-1 \
  BITNET_PARITY_TOPK_STRICT=1 \
  go test ./pkg/bitnet -run TestParityAgainstFrozenVectors -count=1

run env \
  BITNET_ENFORCE_YARN=1 \
  BITNET_PARITY_STRICT=1 \
  BITNET_PARITY_LOGIT_ATOL=1e-3 \
  BITNET_PARITY_LOGIT_RTOL=3e-2 \
  BITNET_PARITY_TOPK_STRICT=1 \
  go test ./pkg/bitnet -run TestParityAgainstYarnVectors -count=1

run env \
  BITNET_ENFORCE_I2S=1 \
  BITNET_PARITY_FORCE=1 \
  BITNET_PARITY_STRICT=1 \
  BITNET_I2S_FORCE_LOGIT_ATOL=8e-2 \
  BITNET_I2S_FORCE_LOGIT_RTOL=8e-2 \
  BITNET_I2S_TOPK_STRICT=3 \
  BITNET_PARITY_FORCE_RELAX_TOPK=1 \
  go test ./pkg/bitnet -run TestParityAgainstI2SVectors -count=1

run env \
  BITNET_ENFORCE_I2S_2B=1 \
  BITNET_PARITY_FORCE=1 \
  BITNET_PARITY_STRICT=1 \
  BITNET_I2S_FORCE_LOGIT_ATOL=8e-2 \
  BITNET_I2S_FORCE_LOGIT_RTOL=8e-2 \
  BITNET_I2S_TOPK_STRICT=3 \
  BITNET_PARITY_FORCE_RELAX_TOPK=1 \
  go test ./pkg/bitnet -run TestParityAgainstI2S2BVectors -count=1

run env \
  BITNET_ENFORCE_I2S_SMOKE=1 \
  BITNET_PARITY_FORCE=1 \
  BITNET_PARITY_STRICT=1 \
  go test ./pkg/bitnet -run TestParityAgainstI2SSmoke -count=1

run env \
  BITNET_ENFORCE_I2S_2B_SMOKE=1 \
  BITNET_PARITY_FORCE=1 \
  BITNET_PARITY_STRICT=1 \
  go test ./pkg/bitnet -run TestParityAgainstI2S2BSmoke -count=1

run env \
  BITNET_ENFORCE_SEED_DETERMINISM=1 \
  BITNET_SEED_DETERMINISM_SEED=7 \
  BITNET_SEED_DETERMINISM_MAX_TOKENS=8 \
  BITNET_SEED_DETERMINISM_TEMP=0.8 \
  BITNET_SEED_DETERMINISM_TOP_P=0.9 \
  BITNET_SEED_DETERMINISM_TOP_K=40 \
  go test ./pkg/bitnet -run TestSeedDeterminismFixtures -count=1

echo ""
echo "[cpu-parity-audit] PASS"
