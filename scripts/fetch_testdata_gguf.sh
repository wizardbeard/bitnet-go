#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TESTDATA_DIR="$ROOT_DIR/testdata"

mkdir -p "$TESTDATA_DIR"

force_fetch=${BITNET_FORCE_FETCH:-0}

fetch_url() {
    url=$1
    dst=$2
    min_size=$3
    sha256=${4:-}

    if [ -f "$dst" ] && [ "$force_fetch" != "1" ]; then
        size=$(wc -c < "$dst" | tr -d ' ')
        if [ "$size" -ge "$min_size" ]; then
            echo "Exists: $dst"
            return 0
        fi
    fi

    tmp="$dst.part"
    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --retry 3 --retry-delay 2 -C - -o "$tmp" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$tmp" "$url"
    else
        echo "Neither curl nor wget is available." >&2
        exit 1
    fi

    size=$(wc -c < "$tmp" | tr -d ' ')
    if [ "$size" -lt "$min_size" ]; then
        echo "Downloaded file is unexpectedly small ($size bytes): $tmp" >&2
        exit 1
    fi

    if [ "$sha256" != "" ]; then
        if command -v sha256sum >/dev/null 2>&1; then
            got=$(sha256sum "$tmp" | awk '{print $1}')
        elif command -v shasum >/dev/null 2>&1; then
            got=$(shasum -a 256 "$tmp" | awk '{print $1}')
        else
            echo "Checksum requested but neither sha256sum nor shasum is available." >&2
            exit 1
        fi
        if [ "$got" != "$sha256" ]; then
            echo "SHA256 mismatch for $tmp" >&2
            echo "expected: $sha256" >&2
            echo "got:      $got" >&2
            exit 1
        fi
    fi

    mv "$tmp" "$dst"
    echo "Downloaded: $dst"
}

# Core parity model (updates model_fixture.txt).
sh "$ROOT_DIR/scripts/fetch_ref_model.sh"

# Minimal GGUF header fixture.
minimal_path="$TESTDATA_DIR/minimal.gguf"
if [ ! -f "$minimal_path" ] || [ "$force_fetch" = "1" ]; then
    printf '\x47\x47\x55\x46\x03\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00' > "$minimal_path"
    echo "Wrote: $minimal_path"
else
    echo "Exists: $minimal_path"
fi

# Vocab-only GGUF fixtures for tokenizer tests.
fetch_url "${BITNET_GPT2_VOCAB_URL:-https://huggingface.co/spaces/Steven10429/apply_lora_and_quantize/resolve/main/llama.cpp/models/ggml-vocab-gpt-2.gguf}" \
    "$TESTDATA_DIR/ggml-vocab-gpt-2.gguf" 1000000 "${BITNET_GPT2_VOCAB_SHA256:-}"
fetch_url "${BITNET_FALCON_VOCAB_URL:-https://huggingface.co/spaces/Steven10429/apply_lora_and_quantize/resolve/main/llama.cpp/models/ggml-vocab-falcon.gguf}" \
    "$TESTDATA_DIR/ggml-vocab-falcon.gguf" 1000000 "${BITNET_FALCON_VOCAB_SHA256:-}"
fetch_url "${BITNET_QWEN2_VOCAB_URL:-https://huggingface.co/spaces/Steven10429/apply_lora_and_quantize/resolve/main/llama.cpp/models/ggml-vocab-qwen2.gguf}" \
    "$TESTDATA_DIR/ggml-vocab-qwen2.gguf" 1000000 "${BITNET_QWEN2_VOCAB_SHA256:-}"

# Optional YaRN model for parity. Set BITNET_FETCH_YARN=1 to enable.
if [ "${BITNET_FETCH_YARN:-0}" = "1" ]; then
    yarn_file=${BITNET_YARN_MODEL_FILE:-YarnGPT2b.f16.gguf}
    yarn_url=${BITNET_YARN_MODEL_URL:-https://huggingface.co/mradermacher/YarnGPT2b-GGUF/resolve/main/YarnGPT2b.f16.gguf}
    fetch_url "$yarn_url" "$TESTDATA_DIR/$yarn_file" 100000000 "${BITNET_YARN_MODEL_SHA256:-}"
    printf '%s\n' "$yarn_file" > "$TESTDATA_DIR/model_fixture_yarn.txt"
    echo "Updated fixture: $TESTDATA_DIR/model_fixture_yarn.txt"
fi
