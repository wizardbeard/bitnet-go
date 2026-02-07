package runtime

import (
	"fmt"
	"math"
	"os"
)

func causalAttentionMultiHeadIntoOptimized(dst, scores, q, keys, values []float32, steps, qHeads, kvHeads, kStepDim, vStepDim int, pos int) {
	for i := range dst {
		dst[i] = 0
	}
	if steps <= 0 || len(q) == 0 {
		return
	}

	if qHeads <= 0 {
		qHeads = 1
	}
	if kvHeads <= 0 {
		kvHeads = qHeads
	}
	if len(q)%qHeads != 0 {
		qHeads = 1
		kvHeads = 1
	}
	if kStepDim <= 0 || vStepDim <= 0 || kStepDim%kvHeads != 0 || vStepDim%kvHeads != 0 {
		return
	}
	headDim := len(q) / qHeads
	if headDim == 0 {
		return
	}
	if len(scores) < steps*qHeads {
		return
	}
	if kStepDim/kvHeads != headDim || vStepDim/kvHeads != headDim {
		return
	}
	maxSeq := 0
	if vStepDim > 0 {
		maxSeq = len(values) / vStepDim
	}
	if maxSeq <= 0 {
		return
	}

	for h := 0; h < qHeads; h++ {
		qBase := h * headDim
		qh := q[qBase : qBase+headDim]
		kvHead := h * kvHeads / qHeads
		kBase := kvHead * headDim
		scale := float32(1.0 / math.Sqrt(float64(headDim)))
		maxScore := float32(-math.MaxFloat32)
		scoreBase := h * steps
		for i := 0; i < steps; i++ {
			kb := i*kStepDim + kBase
			var sum float32
			for j := 0; j < headDim; j++ {
				sum += qh[j] * keys[kb+j]
			}
			s := sum * scale
			scores[scoreBase+i] = s
			if s > maxScore {
				maxScore = s
			}
		}

		var sum float32
		for i := 0; i < steps; i++ {
			idx := scoreBase + i
			diff := scores[idx] - maxScore
			var w float32
			if debugStrictExpf {
				w = expf32(diff)
			} else {
				w = float32(math.Exp(float64(diff)))
			}
			scores[idx] = w
			sum += w
		}
		if sum == 0 {
			continue
		}
		inv := 1 / sum
		if debugStrictAttention {
			for i := 0; i < steps; i++ {
				idx := scoreBase + i
				scores[idx] *= inv
			}
		}
		if debugValues && h == 0 && shouldDebug(pos) && !debugSoftmaxPrinted {
			limit := steps
			if limit > debugValuesN {
				limit = debugValuesN
			}
			if limit > 0 {
				fmt.Fprint(os.Stderr, "debug_values kq_softmax values=")
				for i := 0; i < limit; i++ {
					if i > 0 {
						fmt.Fprint(os.Stderr, ",")
					}
					fmt.Fprintf(os.Stderr, "%.9g", scores[i])
				}
				fmt.Fprintln(os.Stderr)
				debugSoftmaxPrinted = true
			}
		}
		if debugStrictAttention {
			vHeadBase := kvHead * headDim * maxSeq
			weights := scores[scoreBase : scoreBase+steps]
			for j := 0; j < headDim; j++ {
				rowBase := vHeadBase + j*maxSeq
				dst[qBase+j] += dotF32GGML(weights, values[rowBase:rowBase+steps])
			}
			continue
		}

		weights := scores[scoreBase : scoreBase+steps]
		for i := 0; i < steps; i++ {
			weights[i] *= inv
		}
		vHeadBase := kvHead * headDim * maxSeq
		for j := 0; j < headDim; j++ {
			rowBase := vHeadBase + j*maxSeq
			row := values[rowBase : rowBase+steps]
			var acc float32
			for i := 0; i < steps; i++ {
				acc += row[i] * weights[i]
			}
			dst[qBase+j] += acc
		}
	}
}
