#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
REF_DIR="$ROOT_DIR/.ref"
OUT_BIN="$REF_DIR/bin/ref-tokenize"
SRC_FILE="$ROOT_DIR/scripts/ref_tokenize.cpp"

if [ ! -f "$REF_DIR/ref.env" ]; then
    echo "Missing $REF_DIR/ref.env. Run scripts/build_ref.sh first." >&2
    exit 1
fi

# shellcheck disable=SC1090
. "$REF_DIR/ref.env"

LLAMA_INCLUDE="$REF_SRC/3rdparty/llama.cpp/include"
GGML_INCLUDE="$REF_SRC/3rdparty/llama.cpp/ggml/include"
LLAMA_LIB_DIR="$REF_BUILD_DIR/3rdparty/llama.cpp/src"
GGML_LIB_DIR="$REF_BUILD_DIR/3rdparty/llama.cpp/ggml/src"

mkdir -p "$REF_DIR/bin"

: "${CXX:=c++}"

"$CXX" \
    -std=c++17 \
    -O2 \
    -I"$LLAMA_INCLUDE" \
    -I"$GGML_INCLUDE" \
    "$SRC_FILE" \
    -L"$LLAMA_LIB_DIR" \
    -L"$GGML_LIB_DIR" \
    -Wl,-rpath,"$LLAMA_LIB_DIR" \
    -Wl,-rpath,"$GGML_LIB_DIR" \
    -l:libllama.so \
    -l:libggml.so \
    -pthread \
    -o "$OUT_BIN"

echo "Reference tokenizer ready: $OUT_BIN"
