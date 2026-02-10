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
    # Use octal escapes for POSIX printf portability.
    printf '\107\107\125\106\003\000\000\000\002\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000' > "$minimal_path"
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

# Optional IQ fixture for decoder validation. Set BITNET_FETCH_IQ=1 to enable.
if [ "${BITNET_FETCH_IQ:-0}" = "1" ]; then
    iq_file=${BITNET_IQ_MODEL_FILE:-smollm2-135m-instruct-iq4_xs-imat.gguf}
    iq_url=${BITNET_IQ_MODEL_URL:-https://huggingface.co/ngxson/SmolLM2-135M-Instruct-IQ4_XS-GGUF/resolve/main/smollm2-135m-instruct-iq4_xs-imat.gguf}
    fetch_url "$iq_url" "$TESTDATA_DIR/$iq_file" 80000000 "${BITNET_IQ_MODEL_SHA256:-}"
    printf '%s\n' "$iq_file" > "$TESTDATA_DIR/model_fixture_iq.txt"
    echo "Updated fixture: $TESTDATA_DIR/model_fixture_iq.txt"
fi

# Optional i2_s fixture for parity. Set BITNET_FETCH_I2S=1 to enable.
if [ "${BITNET_FETCH_I2S:-0}" = "1" ]; then
    i2s_url_default="https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf"
    i2s_sha_default="4221b252fdd5fd25e15847adfeb5ee88886506ba50b8a34548374492884c2162"
    i2s_url=${BITNET_I2S_MODEL_URL:-$i2s_url_default}
    i2s_sha=${BITNET_I2S_MODEL_SHA256:-$i2s_sha_default}
    if [ "${BITNET_I2S_MODEL_FILE:-}" = "" ]; then
        i2s_file=$(basename "$i2s_url")
    else
        i2s_file=$BITNET_I2S_MODEL_FILE
    fi
    i2s_min_size=${BITNET_I2S_MODEL_MIN_SIZE:-1000000}
    fetch_url "$i2s_url" "$TESTDATA_DIR/$i2s_file" "$i2s_min_size" "$i2s_sha"
    printf '%s\n' "$i2s_file" > "$TESTDATA_DIR/model_fixture_i2s.txt"
    echo "Updated fixture: $TESTDATA_DIR/model_fixture_i2s.txt"
fi

# Optional i2_s 2B fixture for parity. Set BITNET_FETCH_I2S_2B=1 to enable.
if [ "${BITNET_FETCH_I2S_2B:-0}" = "1" ]; then
    if [ "${BITNET_I2S_2B_MODEL_URL:-}" = "" ]; then
        echo "BITNET_I2S_2B_MODEL_URL is required when BITNET_FETCH_I2S_2B=1." >&2
        exit 1
    fi
    if [ "${BITNET_I2S_2B_MODEL_FILE:-}" = "" ]; then
        i2s_2b_file=$(basename "$BITNET_I2S_2B_MODEL_URL")
    else
        i2s_2b_file=$BITNET_I2S_2B_MODEL_FILE
    fi
    i2s_2b_min_size=${BITNET_I2S_2B_MODEL_MIN_SIZE:-1000000}
    fetch_url "$BITNET_I2S_2B_MODEL_URL" "$TESTDATA_DIR/$i2s_2b_file" "$i2s_2b_min_size" "${BITNET_I2S_2B_MODEL_SHA256:-}"
    printf '%s\n' "$i2s_2b_file" > "$TESTDATA_DIR/model_fixture_i2s_2b.txt"
    echo "Updated fixture: $TESTDATA_DIR/model_fixture_i2s_2b.txt"
fi
