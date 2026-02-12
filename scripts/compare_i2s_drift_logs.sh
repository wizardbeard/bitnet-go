#!/usr/bin/env sh
set -eu

GO_LOG=${1:-.bench/i2s-drift-trace-i2s.log}
REF_LOG=${2:-.bench/ref-i2s-drift-trace-i2s.log}
OUT=${BITNET_DRIFT_COMPARE_OUT:-.bench/i2s-drift-compare.tsv}
TRACE_LAYER=${BITNET_DRIFT_COMPARE_LAYER:-14}
TRACE_NAME=${BITNET_DRIFT_COMPARE_NAME:-ffn_sub_norm}
TRACE_REF_NAME=${BITNET_DRIFT_COMPARE_REF_NAME:-$TRACE_NAME}

case "$TRACE_NAME" in
  attn_softmax_h0)
    TRACE_REF_NAME=${BITNET_DRIFT_COMPARE_REF_NAME:-kq_soft_max_ext}
    ;;
  x_post_attn)
    TRACE_REF_NAME=${BITNET_DRIFT_COMPARE_REF_NAME:-ffn_inp}
    ;;
  ffn_act)
    TRACE_REF_NAME=${BITNET_DRIFT_COMPARE_REF_NAME:-ffn_out}
    ;;
  x_post_ffn)
    TRACE_REF_NAME=${BITNET_DRIFT_COMPARE_REF_NAME:-l_out}
    ;;
esac

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
      if ($i ~ /^subnorm_l2=/) { split($i, a, "="); print layer "\tffn_sub_norm\t" a[2] }
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
    if (metric == "ffn_out") {
      metric = "ffn_act"
    }
    if (metric == "attn_o_out" || metric == "ffn_gate" || metric == "ffn_up" || metric == "ffn_act" || metric == "ffn_sub_norm" || metric == "ffn_down") {
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
go_outnorm_l2=$(awk -F'=' '/^drift_trace output_norm_l2=/{print $2; exit}' "$GO_LOG")
go_outnorm_values=$(awk -F'=' '/^drift_trace output_norm_values=/{print $2; exit}' "$GO_LOG")
ref_result_norm_l2=$(awk '
  /^DEBUG name=result_norm / {
    n = -1
    rms = -1
    for (i = 1; i <= NF; i++) {
      if ($i ~ /^n=/) { split($i, a, "="); n = a[2] + 0 }
      if ($i ~ /^rms=/) { split($i, a, "="); rms = a[2] + 0 }
    }
    if (n > 0 && rms >= 0) {
      printf "%.9g\n", rms * sqrt(n)
      exit
    }
  }
' "$REF_LOG")
ref_result_norm_values=$(awk '
  /^DEBUG_VALUES name=result_norm values=/ {
    sub(/^DEBUG_VALUES name=result_norm values=/, "", $0)
    print
    exit
  }
' "$REF_LOG")
if [ -n "${go_outnorm_l2:-}" ] || [ -n "${ref_result_norm_l2:-}" ]; then
  echo "[drift-compare] output-norm-l2 go=$go_outnorm_l2 ref=$ref_result_norm_l2"
fi
if [ -n "${go_outnorm_values:-}" ] && [ -n "${ref_result_norm_values:-}" ]; then
  norm_values_delta=$(awk -v g="$go_outnorm_values" -v r="$ref_result_norm_values" '
    BEGIN {
      ng = split(g, ga, ",")
      nr = split(r, ra, ",")
      n = ng
      if (nr < n) n = nr
      if (n <= 0) {
        print "n=0"
        exit
      }
      sum = 0
      max = -1
      maxi = -1
      for (i = 1; i <= n; i++) {
        d = ga[i] - ra[i]
        if (d < 0) d = -d
        sum += d
        if (d > max) {
          max = d
          maxi = i - 1
        }
      }
      mean = sum / n
      printf "n=%d mean_abs=%g max_abs=%g max_idx=%d go=%g ref=%g", n, mean, max, maxi, ga[maxi+1], ra[maxi+1]
    }
  ')
  echo "[drift-compare] output-norm-values $norm_values_delta"
fi

go_layer_values=$(awk -v layer="$TRACE_LAYER" -v name="$TRACE_NAME" '
  $1=="drift_trace" && $2=="values" {
    gotLayer = ""
    gotName = ""
    gotValues = ""
    for (i = 1; i <= NF; i++) {
      if ($i ~ /^layer=/) { split($i, a, "="); gotLayer = a[2] }
      if ($i ~ /^name=/) { split($i, a, "="); gotName = a[2] }
      if ($i ~ /^values=/) {
        sub(/^values=/, "", $i)
        gotValues = $i
      }
    }
    if (gotLayer == layer && gotName == name) {
      print gotValues
      exit
    }
  }
' "$GO_LOG")

ref_layer_values=$(awk -v layer="$TRACE_LAYER" -v name="$TRACE_REF_NAME" '
  $1=="DEBUG_VALUES" {
    gotName = ""
    gotValues = ""
    for (i = 1; i <= NF; i++) {
      if ($i ~ /^name=/) { split($i, a, "="); gotName = a[2] }
      if ($i ~ /^values=/) {
        sub(/^values=/, "", $i)
        gotValues = $i
      }
    }
    target = name "-" layer
    if (gotName == target) {
      print gotValues
      exit
    }
  }
' "$REF_LOG")

if [ -n "${go_layer_values:-}" ] && [ -n "${ref_layer_values:-}" ]; then
  layer_values_delta=$(awk -v g="$go_layer_values" -v r="$ref_layer_values" -v layer="$TRACE_LAYER" -v name="$TRACE_NAME" -v ref_name="$TRACE_REF_NAME" '
    BEGIN {
      ng = split(g, ga, ",")
      nr = split(r, ra, ",")
      n = ng
      if (nr < n) n = nr
      if (n <= 0) {
        print "n=0"
        exit
      }
      sum = 0
      max = -1
      maxi = -1
      for (i = 1; i <= n; i++) {
        d = ga[i] - ra[i]
        if (d < 0) d = -d
        sum += d
        if (d > max) {
          max = d
          maxi = i - 1
        }
      }
      mean = sum / n
      printf "layer=%s name=%s ref_name=%s n=%d mean_abs=%g max_abs=%g max_idx=%d go=%g ref=%g", layer, name, ref_name, n, mean, max, maxi, ga[maxi+1], ra[maxi+1]
    }
  ')
  echo "[drift-compare] layer-values $layer_values_delta"

  if [ "$TRACE_NAME" = "Vcur" ]; then
    qkv_dims=$(awk -v layer="$TRACE_LAYER" '
      $1=="drift_trace" && $2=="qkv_dims" {
        gotLayer = ""
        kv = ""
        vlen = ""
        for (i = 1; i <= NF; i++) {
          if ($i ~ /^layer=/) { split($i, a, "="); gotLayer = a[2] }
          if ($i ~ /^kv_heads=/) { split($i, a, "="); kv = a[2] }
          if ($i ~ /^v_len=/) { split($i, a, "="); vlen = a[2] }
        }
        if (gotLayer == layer && kv != "" && vlen != "") {
          print kv "," vlen
          exit
        }
      }
    ' "$GO_LOG")
    if [ -n "${qkv_dims:-}" ]; then
      kv_heads=${qkv_dims%%,*}
      v_len=${qkv_dims##*,}
      if [ -n "${kv_heads:-}" ] && [ "$kv_heads" -gt 0 ] && [ -n "${v_len:-}" ] && [ "$v_len" -gt 0 ]; then
        layout_probe=$(awk -v g="$go_layer_values" -v r="$ref_layer_values" -v kv_heads="$kv_heads" -v v_len="$v_len" '
          function absf(x) { return x < 0 ? -x : x }
          function eval_identity(   i,d,sum,maxd,maxi,mean) {
            sum=0; maxd=-1; maxi=-1
            for (i=1; i<=n; i++) {
              d = absf(ga[i] - ra[i])
              sum += d
              if (d > maxd) { maxd = d; maxi = i-1 }
            }
            mean = sum / n
            if (best_mean < 0 || mean < best_mean) {
              best_mean = mean; best_max = maxd; best_name = "identity"; best_idx = maxi
            }
          }
          function eval_head_shift(shift,   i,h,dim,ri,d,sum,maxd,maxi,mean) {
            sum=0; maxd=-1; maxi=-1
            for (i=0; i<n; i++) {
              h = int(i / head_dim)
              dim = i % head_dim
              ri = ((h + shift) % kv_heads) * head_dim + dim
              d = absf(ga[i+1] - ra[ri+1])
              sum += d
              if (d > maxd) { maxd = d; maxi = i }
            }
            mean = sum / n
            if (best_mean < 0 || mean < best_mean) {
              best_mean = mean; best_max = maxd; best_name = sprintf("head_shift_%d", shift); best_idx = maxi
            }
          }
          function eval_head_dim_transpose(   i,h,dim,ri,d,sum,maxd,maxi,mean) {
            sum=0; maxd=-1; maxi=-1
            for (i=0; i<n; i++) {
              h = int(i / head_dim)
              dim = i % head_dim
              ri = dim * kv_heads + h
              d = absf(ga[i+1] - ra[ri+1])
              sum += d
              if (d > maxd) { maxd = d; maxi = i }
            }
            mean = sum / n
            if (best_mean < 0 || mean < best_mean) {
              best_mean = mean; best_max = maxd; best_name = "head_dim_transpose"; best_idx = maxi
            }
          }
          BEGIN {
            ng = split(g, ga, ",")
            nr = split(r, ra, ",")
            n = ng
            if (nr < n) n = nr
            if (n <= 0 || kv_heads <= 0 || v_len <= 0 || (v_len % kv_heads) != 0) {
              print "probe=unavailable"
              exit
            }
            head_dim = int(v_len / kv_heads)
            if (head_dim <= 0 || n > v_len) {
              print "probe=unavailable"
              exit
            }
            best_mean = -1
            best_max = -1
            best_name = ""
            best_idx = -1
            eval_identity()
            for (s = 1; s < kv_heads; s++) {
              eval_head_shift(s)
            }
            eval_head_dim_transpose()
            printf "probe_best=%s n=%d mean_abs=%g max_abs=%g max_idx=%d kv_heads=%d head_dim=%d", best_name, n, best_mean, best_max, best_idx, kv_heads, head_dim
          }
        ')
        echo "[drift-compare] vcur-layout $layout_probe"
      fi
    fi
  fi
fi

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
