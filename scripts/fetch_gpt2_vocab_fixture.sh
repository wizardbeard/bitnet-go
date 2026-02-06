#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
REF_DIR="$ROOT_DIR/.ref"
TESTDATA_DIR="$ROOT_DIR/testdata"
DST="$TESTDATA_DIR/ggml-vocab-gpt-2.gguf"

if [ ! -f "$REF_DIR/ref.env" ]; then
    echo "Missing $REF_DIR/ref.env. Run scripts/build_ref.sh first." >&2
    exit 1
fi

# shellcheck disable=SC1090
. "$REF_DIR/ref.env"

SRC=${BITNET_REF_GPT2_MODEL:-$REF_SRC/3rdparty/llama.cpp/models/ggml-vocab-gpt-2.gguf}
if [ ! -f "$SRC" ]; then
    echo "GPT2 vocab GGUF not found: $SRC" >&2
    exit 1
fi

mkdir -p "$TESTDATA_DIR"
cp "$SRC" "$DST"

echo "Copied: $DST"
