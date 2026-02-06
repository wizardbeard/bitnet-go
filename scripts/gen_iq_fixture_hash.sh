#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TESTDATA_DIR="$ROOT_DIR/testdata"

MODEL_FIXTURE="$TESTDATA_DIR/model_fixture_iq.txt"
if [ ! -f "$MODEL_FIXTURE" ]; then
    echo "Missing $MODEL_FIXTURE. Run scripts/fetch_testdata_gguf.sh with BITNET_FETCH_IQ=1." >&2
    exit 1
fi

MODEL_FILE=$(tr -d '\r\n' < "$MODEL_FIXTURE")
MODEL_PATH="$TESTDATA_DIR/$MODEL_FILE"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Missing IQ model: $MODEL_PATH" >&2
    exit 1
fi

OUT_PATH="$TESTDATA_DIR/expected.iq_hash.json"
COUNT=${BITNET_IQ_COUNT:-4096}
TENSOR=${BITNET_IQ_TENSOR:-}

if [ "$TENSOR" != "" ]; then
    go run ./cmd/ggufhash --model "$MODEL_PATH" --tensor "$TENSOR" --count "$COUNT" --out "$OUT_PATH"
else
    go run ./cmd/ggufhash --model "$MODEL_PATH" --count "$COUNT" --out "$OUT_PATH"
fi

echo "Wrote: $OUT_PATH"
