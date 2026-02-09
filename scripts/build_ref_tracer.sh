#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
REF_DIR="$ROOT_DIR/.ref"
OUT_BIN="$REF_DIR/bin/ref-trace"
SRC_FILE="$ROOT_DIR/scripts/ref_trace.cpp"

if [ ! -f "$REF_DIR/ref.env" ]; then
    echo "Missing $REF_DIR/ref.env. Run scripts/build_ref.sh first." >&2
    exit 1
fi

# shellcheck disable=SC1090
. "$REF_DIR/ref.env"

if [ ! -f "$SRC_FILE" ]; then
    echo "Missing tracer source: $SRC_FILE" >&2
    exit 1
fi

LLAMA_INCLUDE="$REF_SRC/3rdparty/llama.cpp/include"
GGML_INCLUDE="$REF_SRC/3rdparty/llama.cpp/ggml/include"
GGML_SRC_INCLUDE="$REF_SRC/3rdparty/llama.cpp/ggml/src"
LLAMA_LIB_DIR="$REF_BUILD_DIR/3rdparty/llama.cpp/src"
GGML_LIB_DIR="$REF_BUILD_DIR/3rdparty/llama.cpp/ggml/src"

if [ ! -d "$LLAMA_INCLUDE" ] || [ ! -d "$GGML_INCLUDE" ] || [ ! -d "$LLAMA_LIB_DIR" ] || [ ! -d "$GGML_LIB_DIR" ]; then
    echo "Expected llama include/lib directories are missing. Re-run scripts/build_ref.sh." >&2
    exit 1
fi

mkdir -p "$REF_DIR/bin"

: "${CXX:=c++}"

"$CXX" \
    -std=c++17 \
    -O2 \
    -I"$LLAMA_INCLUDE" \
    -I"$GGML_INCLUDE" \
    -I"$GGML_SRC_INCLUDE" \
    "$SRC_FILE" \
    -L"$LLAMA_LIB_DIR" \
    -L"$GGML_LIB_DIR" \
    -Wl,-rpath,"$LLAMA_LIB_DIR" \
    -Wl,-rpath,"$GGML_LIB_DIR" \
    -l:libllama.so \
    -l:libggml.so \
    -pthread \
    -o "$OUT_BIN"

echo "Reference tracer ready: $OUT_BIN"
