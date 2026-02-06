#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
REF_DIR="$ROOT_DIR/.ref"
TESTDATA_DIR="$ROOT_DIR/testdata"

if [ ! -f "$REF_DIR/ref.env" ]; then
    echo "Missing $REF_DIR/ref.env. Run scripts/build_ref.sh first." >&2
    exit 1
fi

# shellcheck disable=SC1090
. "$REF_DIR/ref.env"

mkdir -p "$TESTDATA_DIR"

copy_fixture() {
    src_name=$1
    dst_name=$2
    src="${BITNET_REF_MODELS_DIR:-$REF_SRC/3rdparty/llama.cpp/models}/$src_name"
    dst="$TESTDATA_DIR/$dst_name"
    if [ ! -f "$src" ]; then
        echo "Missing fixture source: $src" >&2
        exit 1
    fi
    cp "$src" "$dst"
    echo "Copied: $dst"
}

copy_fixture "ggml-vocab-gpt-2.gguf" "ggml-vocab-gpt-2.gguf"
copy_fixture "ggml-vocab-falcon.gguf" "ggml-vocab-falcon.gguf"
copy_fixture "ggml-vocab-qwen2.gguf" "ggml-vocab-qwen2.gguf"
