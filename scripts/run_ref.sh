#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
REF_DIR="$ROOT_DIR/.ref"
TESTDATA_DIR="$ROOT_DIR/testdata"
TRACE_FILE="$REF_DIR/reference.trace"
TOKENS_OUT="$TESTDATA_DIR/expected.tokens.json"
TOPK_OUT="$TESTDATA_DIR/expected.topk_logits.json"
TIMING_OUT="$TESTDATA_DIR/expected.timings.json"
PROMPT_TOKENS_OUT="$TESTDATA_DIR/expected.prompt_tokens.json"

if [ "${BITNET_SKIP_BUILD:-0}" != "1" ]; then
    "$ROOT_DIR/scripts/build_ref.sh"
fi

if [ ! -f "$REF_DIR/ref.env" ]; then
    echo "Missing $REF_DIR/ref.env. Run scripts/build_ref.sh first." >&2
    exit 1
fi

# shellcheck disable=SC1090
. "$REF_DIR/ref.env"

mkdir -p "$REF_DIR" "$TESTDATA_DIR"

MODEL_PATH=${BITNET_REF_MODEL:-}
if [ "$MODEL_PATH" = "" ]; then
    if [ -f "$TESTDATA_DIR/model_fixture.txt" ]; then
        fixture=$(sed -n '1p' "$TESTDATA_DIR/model_fixture.txt")
        case "$fixture" in
            "") ;;
            /*) MODEL_PATH=$fixture ;;
            *) MODEL_PATH="$TESTDATA_DIR/$fixture" ;;
        esac
    fi
fi

if [ "$MODEL_PATH" = "" ]; then
    echo "No model specified. Set BITNET_REF_MODEL or testdata/model_fixture.txt." >&2
    exit 1
fi
if [ ! -f "$MODEL_PATH" ]; then
    echo "Model file not found: $MODEL_PATH" >&2
    exit 1
fi
model_size=$(wc -c < "$MODEL_PATH" | tr -d ' ')
if [ "$model_size" -le 24 ]; then
    cat >&2 <<EOM
Model file appears to be a GGUF header stub ($model_size bytes):
  $MODEL_PATH

Reference inference requires a real GGUF model with tensors.
Run:
  ./scripts/fetch_ref_model.sh
EOM
    exit 1
fi

PROMPT_FILE=${BITNET_REF_PROMPT_FILE:-$TESTDATA_DIR/prompt.txt}
if [ ! -f "$PROMPT_FILE" ]; then
    echo "Prompt file not found: $PROMPT_FILE" >&2
    exit 1
fi

PROMPT=$(cat "$PROMPT_FILE")
SEED=${BITNET_REF_SEED:-1}
MAX_TOKENS=${BITNET_REF_MAX_TOKENS:-32}
TOPK=${BITNET_REF_TOPK:-5}

export BITNET_REF_BIN="${BITNET_REF_BIN:-$REF_BIN}"
export BITNET_REF_MODEL="$MODEL_PATH"
export BITNET_REF_PROMPT="$PROMPT"
export BITNET_REF_SEED="$SEED"
export BITNET_REF_MAX_TOKENS="$MAX_TOKENS"
export BITNET_REF_TOPK="$TOPK"

if [ "${BITNET_REF_RUN_CMD:-}" != "" ]; then
    sh -c "$BITNET_REF_RUN_CMD" > "$TRACE_FILE" 2>&1
else
    TRACE_BIN="$REF_DIR/bin/ref-trace"
    if [ "${BITNET_SKIP_TRACER_BUILD:-0}" != "1" ]; then
        "$ROOT_DIR/scripts/build_ref_tracer.sh" >/dev/null 2>&1 || true
    fi

    if [ -x "$TRACE_BIN" ]; then
        "$TRACE_BIN" > "$TRACE_FILE" 2>&1 || {
            echo "Reference tracer run failed: $TRACE_BIN" >&2
            exit 1
        }
    else
        "$BITNET_REF_BIN" \
            -m "$BITNET_REF_MODEL" \
            -p "$BITNET_REF_PROMPT" \
            -n "$BITNET_REF_MAX_TOKENS" \
            --seed "$BITNET_REF_SEED" \
            --temp 0 \
            --top-k "$BITNET_REF_TOPK" \
            --top-p 1 \
            --repeat-penalty 1 \
            --no-display-prompt \
            --no-warmup \
            -v \
            > "$TRACE_FILE" 2>&1 || {
                echo "Reference run failed. Either set BITNET_REF_RUN_CMD or verify CLI flags for $BITNET_REF_BIN." >&2
                exit 1
            }
    fi
fi

# Mode 1: explicit structured trace from BITNET_REF_RUN_CMD wrapper.
if awk 'BEGIN { found = 0 } $1 == "TOKEN" { found = 1 } END { exit(found ? 0 : 1) }' "$TRACE_FILE"; then
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
' "$TRACE_FILE" > "$PROMPT_TOKENS_OUT"

    awk '
BEGIN { n = 0 }
$1 == "TOKEN" {
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
' "$TRACE_FILE" > "$TOKENS_OUT"

    awk '
BEGIN { n = 0 }
$1 == "TOPK" {
    step = -1
    entries = ""
    for (i = 1; i <= NF; i++) {
        if ($i ~ /^step=/) {
            split($i, a, "=")
            step = a[2]
        }
        if ($i ~ /^entries=/) {
            split($i, a, "=")
            entries = a[2]
        }
    }

    payload = ""
    m = split(entries, pair, ",")
    payload = "["
    first = 1
    for (j = 1; j <= m; j++) {
        if (pair[j] == "") continue
        split(pair[j], kv, ":")
        if (kv[1] == "" || kv[2] == "") continue
        if (!first) payload = payload ","
        payload = payload "{\"token_id\":" kv[1] ",\"logit\":" kv[2] "}"
        first = 0
    }
    payload = payload "]"

    if (step >= 0) {
        steps[n] = step
        entries_json[n] = payload
        n++
    }
}
END {
    printf "["
    for (i = 0; i < n; i++) {
        if (i > 0) printf ","
        printf "{\"step\":%s,\"entries\":%s}", steps[i], entries_json[i]
    }
    printf "]\n"
}
' "$TRACE_FILE" > "$TOPK_OUT"

    awk '
BEGIN { n = 0 }
$1 == "TIME" {
    step = -1
    ms = ""
    for (i = 1; i <= NF; i++) {
        if ($i ~ /^step=/) {
            split($i, a, "=")
            step = a[2]
        }
        if ($i ~ /^ms=/) {
            split($i, a, "=")
            ms = a[2]
        }
    }
    if (step >= 0 && ms != "") {
        steps[n] = step
        timings[n] = ms
        n++
    }
}
END {
    printf "["
    for (i = 0; i < n; i++) {
        if (i > 0) printf ","
        printf "{\"step\":%s,\"ms\":%s}", steps[i], timings[i]
    }
    printf "]\n"
}
' "$TRACE_FILE" > "$TIMING_OUT"

# Mode 2: parse llama-cli verbose output (token IDs and average per-step timing only).
else
    printf "[]\n" > "$PROMPT_TOKENS_OUT"

    awk '
BEGIN {
    n = 0
    in_generate = 0
    skipped_prompt_eval = 0
}
/^generate:/ {
    in_generate = 1
    next
}
in_generate == 1 && /eval: \[/ {
    if (skipped_prompt_eval == 0) {
        skipped_prompt_eval = 1
        next
    }

    line = $0
    gsub(/^.*:/, "", line)
    gsub(/\].*$/, "", line)
    gsub(/^ +/, "", line)
    gsub(/ +$/, "", line)
    if (line ~ /^[0-9]+$/) {
        ids[n++] = line
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
' "$TRACE_FILE" > "$TOKENS_OUT"

    avg_ms=$(sed -n 's/.*( *\([0-9][0-9.]*\) ms per token.*/\1/p' "$TRACE_FILE" | tail -n 1)
    if [ "$avg_ms" = "" ]; then
        avg_ms=0
    fi

    awk -v ms="$avg_ms" '
BEGIN {
    n = 0
    in_generate = 0
    skipped_prompt_eval = 0
}
/^generate:/ {
    in_generate = 1
    next
}
in_generate == 1 && /eval: \[/ {
    if (skipped_prompt_eval == 0) {
        skipped_prompt_eval = 1
        next
    }

    line = $0
    gsub(/^.*:/, "", line)
    gsub(/\].*$/, "", line)
    gsub(/^ +/, "", line)
    gsub(/ +$/, "", line)
    if (line ~ /^[0-9]+$/) {
        ids[n++] = line
    }
}
END {
    printf "["
    for (i = 0; i < n; i++) {
        if (i > 0) printf ","
        printf "{\"step\":%d,\"entries\":[{\"token_id\":%s,\"logit\":0.0}]}", i, ids[i]
    }
    printf "]\n"
}
' "$TRACE_FILE" > "$TOPK_OUT"

    awk -v ms="$avg_ms" '
BEGIN {
    n = 0
    in_generate = 0
    skipped_prompt_eval = 0
}
/^generate:/ {
    in_generate = 1
    next
}
in_generate == 1 && /eval: \[/ {
    if (skipped_prompt_eval == 0) {
        skipped_prompt_eval = 1
        next
    }

    line = $0
    gsub(/^.*:/, "", line)
    gsub(/\].*$/, "", line)
    gsub(/^ +/, "", line)
    gsub(/ +$/, "", line)
    if (line ~ /^[0-9]+$/) {
        ids[n++] = line
    }
}
END {
    printf "["
    for (i = 0; i < n; i++) {
        if (i > 0) printf ","
        printf "{\"step\":%d,\"ms\":%s}", i, ms
    }
    printf "]\n"
}
' "$TRACE_FILE" > "$TIMING_OUT"
fi

token_count=$(awk 'BEGIN { c = 0 } /[0-9]/ { c++ } END { print c }' "$TOKENS_OUT")
if [ "$token_count" -eq 0 ]; then
    cat >&2 <<'EOM'
No generated token IDs were extracted from the reference run.

If your binary output differs, set BITNET_REF_RUN_CMD to emit lines like:
  TOKEN step=0 id=123
  TOPK step=0 entries=123:5.12,77:4.98
  TIME step=0 ms=0.42
EOM
    exit 1
fi

echo "Wrote: $TOKENS_OUT"
echo "Wrote: $TOPK_OUT"
echo "Wrote: $TIMING_OUT"
echo "Wrote: $PROMPT_TOKENS_OUT"
echo "Trace: $TRACE_FILE"
