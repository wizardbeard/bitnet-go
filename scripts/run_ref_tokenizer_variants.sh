#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TESTDATA_DIR="$ROOT_DIR/testdata"
REF_DIR="$ROOT_DIR/.ref"

"$ROOT_DIR/scripts/fetch_bpe_vocab_fixtures.sh"
if [ "${BITNET_SKIP_TOKENIZER_BUILD:-0}" != "1" ]; then
    "$ROOT_DIR/scripts/build_ref_tokenizer.sh" >/dev/null
fi

run_case() {
    model_file=$1
    prompt_file=$2
    out_file=$3
    trace_file=$4

    if [ ! -f "$prompt_file" ]; then
        echo "Missing prompt file: $prompt_file" >&2
        exit 1
    fi

    prompt=$(cat "$prompt_file")
    export BITNET_REF_MODEL="$model_file"
    export BITNET_REF_PROMPT="$prompt"

    "$REF_DIR/bin/ref-tokenize" > "$trace_file"

    awk '
BEGIN { n = 0 }
$1 == "PROMPT_TOKEN" {
    id = ""
    for (i = 1; i <= NF; i++) {
        if ($i ~ /^id=/) {
            split($i, a, "=")
            id = a[2]
        }
    }
    if (id != "") {
        ids[n++] = id
    }
}
END {
    printf "["
    for (i = 0; i < n; i++) {
        if (i > 0) printf ","
        printf "%s", ids[i]
    }
    printf "]\n"
}
' "$trace_file" > "$out_file"

    echo "Wrote: $out_file"
}

run_case "$TESTDATA_DIR/ggml-vocab-gpt-2.gguf" "$TESTDATA_DIR/gpt2.prompt.txt" "$TESTDATA_DIR/expected.gpt2_prompt_tokens.json" "$REF_DIR/tokenizer.gpt2.trace"
run_case "$TESTDATA_DIR/ggml-vocab-falcon.gguf" "$TESTDATA_DIR/falcon.prompt.txt" "$TESTDATA_DIR/expected.falcon_prompt_tokens.json" "$REF_DIR/tokenizer.falcon.trace"
run_case "$TESTDATA_DIR/ggml-vocab-qwen2.gguf" "$TESTDATA_DIR/qwen2.prompt.txt" "$TESTDATA_DIR/expected.qwen2_prompt_tokens.json" "$REF_DIR/tokenizer.qwen2.trace"
