#!/usr/bin/env sh
set -eu

GO_LOG=${1:-.bench/i2s-drift-trace-i2s.log}
REF_LOG=${2:-.bench/ref-i2s-drift-trace-i2s.log}
OUT=${BITNET_DRIFT_COMPARE_OUT:-.bench/i2s-drift-compare.tsv}

if [ ! -f "$GO_LOG" ]; then
  echo "go log not found: $GO_LOG" >&2
  exit 1
fi
if [ ! -f "$REF_LOG" ]; then
  echo "ref log not found: $REF_LOG" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"

tmp_go=$(mktemp)
tmp_ref=$(mktemp)
cleanup() {
  rm -f "$tmp_go" "$tmp_ref"
}
trap cleanup EXIT INT TERM

awk '
  /^drift_trace layer=/ {
    layer = ""
    type = ""
    for (i = 1; i <= NF; i++) {
      if ($i ~ /^layer=/) { split($i, a, "="); layer = a[2] }
      if ($i == "attn") type = "attn"
      if ($i == "ffn" || $i == "ffn_ref") type = "ffn"
      if ($i ~ /^attn_out_l2=/) { split($i, a, "="); print layer "\tattn_o_out\t" a[2] }
      if ($i ~ /^gate_l2=/) { split($i, a, "="); print layer "\tffn_gate\t" a[2] }
      if ($i ~ /^up_l2=/) { split($i, a, "="); print layer "\tffn_up\t" a[2] }
      if ($i ~ /^act_l2=/) { split($i, a, "="); print layer "\tffn_act\t" a[2] }
      if ($i ~ /^down_l2=/) { split($i, a, "="); print layer "\tffn_down\t" a[2] }
    }
  }
' "$GO_LOG" | sort -t $'\t' -k1,1n -k2,2 > "$tmp_go"

awk '
  /^DEBUG name=/ {
    name = ""
    n = -1
    rms = -1
    for (i = 1; i <= NF; i++) {
      if ($i ~ /^name=/) { split($i, a, "="); name = a[2] }
      if ($i ~ /^n=/) { split($i, a, "="); n = a[2] + 0 }
      if ($i ~ /^rms=/) { split($i, a, "="); rms = a[2] + 0 }
    }
    if (name == "" || n <= 0 || rms < 0) next

    split(name, p, "-")
    if (length(p) != 2) next
    metric = p[1]
    layer = p[2] + 0
    l2 = rms * sqrt(n)
    if (metric == "attn_o_out" || metric == "ffn_gate" || metric == "ffn_up" || metric == "ffn_act" || metric == "ffn_down") {
      printf "%d\t%s\t%.9g\n", layer, metric, l2
    }
  }
' "$REF_LOG" | sort -t $'\t' -k1,1n -k2,2 > "$tmp_ref"

{
  printf "layer\tmetric\tgo_l2\tref_l2\tabs_delta\trel_delta_pct\n"
  awk -F'\t' '
    NR == FNR {
      key = $1 "|" $2
      go[key] = $3
      seen[key] = 1
      next
    }
    {
      key = $1 "|" $2
      ref[key] = $3
      seen[key] = 1
    }
    END {
      for (key in seen) {
        split(key, p, "|")
        layer = p[1]
        metric = p[2]
        g = go[key]
        r = ref[key]
        abs = ""
        rel = ""
        if (g != "" && r != "") {
          gv = g + 0
          rv = r + 0
          d = gv - rv
          if (d < 0) d = -d
          abs = sprintf("%.9g", d)
          if (rv != 0) {
            base = rv
            if (base < 0) base = -base
            rel = sprintf("%.3f", (d / base) * 100)
          }
        }
        printf "%s\t%s\t%s\t%s\t%s\t%s\n", layer, metric, g, r, abs, rel
      }
    }
  ' "$tmp_go" "$tmp_ref" | sort -t $'\t' -k1,1n -k2,2
} > "$OUT"

echo "[drift-compare] go_log:  $GO_LOG"
echo "[drift-compare] ref_log: $REF_LOG"
echo "[drift-compare] out:     $OUT"

go_step=$(awk '/^drift_trace logits step=/{for(i=1;i<=NF;i++){if($i~/^step=/){split($i,a,"=");print a[2]; exit}}}' "$GO_LOG")
go_token=$(awk '/^drift_trace logits step=/{for(i=1;i<=NF;i++){if($i~/^token=/){split($i,a,"=");print a[2]; exit}}}' "$GO_LOG")
go_logit=$(awk '/^drift_trace logits step=/{for(i=1;i<=NF;i++){if($i~/^logit=/){split($i,a,"=");print a[2]; exit}}}' "$GO_LOG")
if [ -n "${go_step:-}" ] && [ -n "${go_token:-}" ]; then
  ref_logit=$(awk -v s="$go_step" -v tok="$go_token" '
    $1=="TOPK" && $2=="step="s {
      split($3, e, "=")
      n=split(e[2], p, ",")
      for (i=1; i<=n; i++) {
        split(p[i], kv, ":")
        if (kv[1] == tok) {
          print kv[2]
          exit
        }
      }
    }
  ' "$REF_LOG")
  if [ -n "${go_logit:-}" ] || [ -n "${ref_logit:-}" ]; then
    echo "[drift-compare] token-logit step=$go_step token=$go_token go=$go_logit ref=$ref_logit"
  fi
fi

cat "$OUT"
