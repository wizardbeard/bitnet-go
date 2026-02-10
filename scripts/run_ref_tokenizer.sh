#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TESTDATA_DIR="$ROOT_DIR/testdata"
REF_DIR="$ROOT_DIR/.ref"
TRACE_FILE="$REF_DIR/tokenizer.trace"
OUT_FILE="$TESTDATA_DIR/expected.gpt2_prompt_tokens.json"
PROMPT_FILE="$TESTDATA_DIR/gpt2.prompt.txt"
MODEL_FILE="$TESTDATA_DIR/ggml-vocab-gpt-2.gguf"

if [ ! -f "$MODEL_FILE" ]; then
    "$ROOT_DIR/scripts/fetch_gpt2_vocab_fixture.sh"
fi

if [ ! -f "$PROMPT_FILE" ]; then
    cat > "$PROMPT_FILE" <<'EOP'
Hello, world! 1234
EOP
fi

if [ "${BITNET_SKIP_TOKENIZER_BUILD:-0}" != "1" ]; then
    "$ROOT_DIR/scripts/build_ref_tokenizer.sh" >/dev/null
fi

PROMPT=$(cat "$PROMPT_FILE")
export BITNET_REF_MODEL="$MODEL_FILE"
export BITNET_REF_PROMPT="$PROMPT"

"$REF_DIR/bin/ref-tokenize" > "$TRACE_FILE"

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
' "$TRACE_FILE" > "$OUT_FILE"

echo "Wrote: $OUT_FILE"
echo "Trace: $TRACE_FILE"
