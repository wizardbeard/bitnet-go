#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TESTDATA_DIR="$ROOT_DIR/testdata"

MODEL_URL=${BITNET_REF_MODEL_URL:-https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q8_0.gguf}
MODEL_FILE=${BITNET_REF_MODEL_FILE:-stories15M-q8_0.gguf}
MODEL_PATH="$TESTDATA_DIR/$MODEL_FILE"
TMP_PATH="$MODEL_PATH.part"
MODEL_SHA256=${BITNET_REF_MODEL_SHA256:-}

mkdir -p "$TESTDATA_DIR"

if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --retry-delay 2 -C - -o "$TMP_PATH" "$MODEL_URL"
elif command -v wget >/dev/null 2>&1; then
    wget -O "$TMP_PATH" "$MODEL_URL"
else
    echo "Neither curl nor wget is available." >&2
    exit 1
fi

size=$(wc -c < "$TMP_PATH" | tr -d ' ')
if [ "$size" -lt 1048576 ]; then
    echo "Downloaded file is unexpectedly small ($size bytes): $TMP_PATH" >&2
    exit 1
fi

if [ "$MODEL_SHA256" != "" ]; then
    if command -v sha256sum >/dev/null 2>&1; then
        got=$(sha256sum "$TMP_PATH" | awk '{print $1}')
    elif command -v shasum >/dev/null 2>&1; then
        got=$(shasum -a 256 "$TMP_PATH" | awk '{print $1}')
    else
        echo "Checksum requested but neither sha256sum nor shasum is available." >&2
        exit 1
    fi
    if [ "$got" != "$MODEL_SHA256" ]; then
        echo "SHA256 mismatch for $TMP_PATH" >&2
        echo "expected: $MODEL_SHA256" >&2
        echo "got:      $got" >&2
        exit 1
    fi
fi

mv "$TMP_PATH" "$MODEL_PATH"
printf '%s\n' "$MODEL_FILE" > "$TESTDATA_DIR/model_fixture.txt"

echo "Downloaded model: $MODEL_PATH"
echo "Updated fixture: $TESTDATA_DIR/model_fixture.txt"
