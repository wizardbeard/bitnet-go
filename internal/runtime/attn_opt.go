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
			if strictKQEnabledForCurrentLayer() {
				sum = dotF32GGML(qh, keys[kb:kb+headDim])
			} else if debugFastKQ {
				sum = dotF32FastN(keys, kb, qh, 0, headDim)
			} else {
				for j := 0; j < headDim; j++ {
					sum += qh[j] * keys[kb+j]
				}
			}
			s := sum * scale
			scores[scoreBase+i] = s
			if s > maxScore {
				maxScore = s
			}
		}

		sum := softmaxInPlace(scores[scoreBase:scoreBase+steps], maxScore)
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
				fmt.Fprint(os.Stderr, "debug_values kq_soft_max_ext values=")
				for i := 0; i < limit; i++ {
					if i > 0 {
						fmt.Fprint(os.Stderr, ",")
					}
					val := scores[scoreBase+i]
					if !debugStrictAttention {
						val *= inv
					}
					fmt.Fprintf(os.Stderr, "%.9g", val)
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
		vHeadBase := kvHead * headDim * maxSeq
		for j := 0; j < headDim; j++ {
			rowBase := vHeadBase + j*maxSeq
			if debugAttnF64 {
				row := values[rowBase : rowBase+steps]
				var sum64 float64
				for i := 0; i < steps; i++ {
					sum64 += float64(row[i]) * float64(weights[i]*inv)
				}
				dst[qBase+j] += float32(sum64)
			} else {
				dst[qBase+j] += dotF32FastNScaled(values, rowBase, weights, 0, steps, inv)
			}
		}
	}
}

func causalAttentionMultiHeadIntoRowMajor(dst, scores, q, keys, values []float32, steps, qHeads, kvHeads, kStepDim, vStepDim int, pos int) {
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
			if strictKQEnabledForCurrentLayer() {
				sum = dotF32GGML(qh, keys[kb:kb+headDim])
			} else if debugFastKQ {
				sum = dotF32FastN(keys, kb, qh, 0, headDim)
			} else {
				for j := 0; j < headDim; j++ {
					sum += qh[j] * keys[kb+j]
				}
			}
			s := sum * scale
			scores[scoreBase+i] = s
			if s > maxScore {
				maxScore = s
			}
		}

		sum := softmaxInPlace(scores[scoreBase:scoreBase+steps], maxScore)
		if sum == 0 {
			continue
		}
		inv := 1 / sum
		if debugStrictAttention {
			weights := scores[scoreBase : scoreBase+steps]
			for j := 0; j < headDim; j++ {
				var acc float32
				for i := 0; i < steps; i++ {
					w := weights[i] * inv
					rowBase := kvHead*maxSeq*headDim + i*headDim
					acc += values[rowBase+j] * w
				}
				dst[qBase+j] += acc
			}
			continue
		}
		if debugValues && h == 0 && shouldDebug(pos) && !debugSoftmaxPrinted {
			limit := steps
			if limit > debugValuesN {
				limit = debugValuesN
			}
			if limit > 0 {
				fmt.Fprint(os.Stderr, "debug_values kq_soft_max_ext values=")
				for i := 0; i < limit; i++ {
					if i > 0 {
						fmt.Fprint(os.Stderr, ",")
					}
					val := scores[scoreBase+i] * inv
					fmt.Fprintf(os.Stderr, "%.9g", val)
				}
				fmt.Fprintln(os.Stderr)
				debugSoftmaxPrinted = true
			}
		}

		weights := scores[scoreBase : scoreBase+steps]
		vHeadBase := kvHead * maxSeq * headDim
		if debugAttnF64 {
			for j := 0; j < headDim; j++ {
				var sum64 float64
				for i := 0; i < steps; i++ {
					rowBase := vHeadBase + i*headDim
					sum64 += float64(values[rowBase+j]) * float64(weights[i]*inv)
				}
				dst[qBase+j] += float32(sum64)
			}
			continue
		}
		dstHead := dst[qBase : qBase+headDim]
		i := 0
		for ; i+1 < steps; i += 2 {
			rowBase := vHeadBase + i*headDim
			row0 := values[rowBase : rowBase+headDim]
			row1 := values[rowBase+headDim : rowBase+2*headDim]
			accumWeightedRow2(dstHead, row0, row1, weights[i]*inv, weights[i+1]*inv)
		}
		if i < steps {
			rowBase := vHeadBase + i*headDim
			accumWeightedRow(dstHead, values[rowBase:rowBase+headDim], weights[i]*inv)
		}
	}
}

func accumWeightedRow(dst, row []float32, w float32) {
	n := len(dst)
	if len(row) < n {
		n = len(row)
	}
	i := 0
	for ; i+7 < n; i += 8 {
		dst[i] += row[i] * w
		dst[i+1] += row[i+1] * w
		dst[i+2] += row[i+2] * w
		dst[i+3] += row[i+3] * w
		dst[i+4] += row[i+4] * w
		dst[i+5] += row[i+5] * w
		dst[i+6] += row[i+6] * w
		dst[i+7] += row[i+7] * w
	}
	for ; i < n; i++ {
		dst[i] += row[i] * w
	}
}

func accumWeightedRow2(dst, row0, row1 []float32, w0, w1 float32) {
	n := len(dst)
	if len(row0) < n {
		n = len(row0)
	}
	if len(row1) < n {
		n = len(row1)
	}
	i := 0
	for ; i+7 < n; i += 8 {
		dst[i] += row0[i]*w0 + row1[i]*w1
		dst[i+1] += row0[i+1]*w0 + row1[i+1]*w1
		dst[i+2] += row0[i+2]*w0 + row1[i+2]*w1
		dst[i+3] += row0[i+3]*w0 + row1[i+3]*w1
		dst[i+4] += row0[i+4]*w0 + row1[i+4]*w1
		dst[i+5] += row0[i+5]*w0 + row1[i+5]*w1
		dst[i+6] += row0[i+6]*w0 + row1[i+6]*w1
		dst[i+7] += row0[i+7]*w0 + row1[i+7]*w1
	}
	for ; i < n; i++ {
		dst[i] += row0[i]*w0 + row1[i]*w1
	}
}

func dotF32Fast(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var sum0, sum1, sum2, sum3 float32
	var sum4, sum5, sum6, sum7 float32
	i := 0
	for ; i+7 < n; i += 8 {
		sum0 += a[i] * b[i]
		sum1 += a[i+1] * b[i+1]
		sum2 += a[i+2] * b[i+2]
		sum3 += a[i+3] * b[i+3]
		sum4 += a[i+4] * b[i+4]
		sum5 += a[i+5] * b[i+5]
		sum6 += a[i+6] * b[i+6]
		sum7 += a[i+7] * b[i+7]
	}
	sum := (sum0 + sum1) + (sum2 + sum3)
	sum += (sum4 + sum5) + (sum6 + sum7)
	for ; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

// dotF32FastN computes dot product for the first n elements starting at aOff/bOff.
// Callers are responsible for bounds; this is a hot-path helper.
func dotF32FastN(a []float32, aOff int, b []float32, bOff int, n int) float32 {
	if n <= 0 {
		return 0
	}
	if aOff < 0 || bOff < 0 {
		return 0
	}
	if aOff+n > len(a) {
		n = len(a) - aOff
	}
	if bOff+n > len(b) {
		n = len(b) - bOff
	}
	if n <= 0 {
		return 0
	}
	var sum0, sum1, sum2, sum3 float32
	var sum4, sum5, sum6, sum7 float32
	i := 0
	for ; i+7 < n; i += 8 {
		sum0 += a[aOff+i] * b[bOff+i]
		sum1 += a[aOff+i+1] * b[bOff+i+1]
		sum2 += a[aOff+i+2] * b[bOff+i+2]
		sum3 += a[aOff+i+3] * b[bOff+i+3]
		sum4 += a[aOff+i+4] * b[bOff+i+4]
		sum5 += a[aOff+i+5] * b[bOff+i+5]
		sum6 += a[aOff+i+6] * b[bOff+i+6]
		sum7 += a[aOff+i+7] * b[bOff+i+7]
	}
	sum := (sum0 + sum1) + (sum2 + sum3)
	sum += (sum4 + sum5) + (sum6 + sum7)
	for ; i < n; i++ {
		sum += a[aOff+i] * b[bOff+i]
	}
	return sum
}

func dotF32FastNScaled(a []float32, aOff int, b []float32, bOff int, n int, scale float32) float32 {
	if n <= 0 {
		return 0
	}
	if aOff < 0 || bOff < 0 {
		return 0
	}
	if aOff+n > len(a) {
		n = len(a) - aOff
	}
	if bOff+n > len(b) {
		n = len(b) - bOff
	}
	if n <= 0 {
		return 0
	}
	var sum0, sum1, sum2, sum3 float32
	var sum4, sum5, sum6, sum7 float32
	i := 0
	for ; i+7 < n; i += 8 {
		sum0 += a[aOff+i] * (b[bOff+i] * scale)
		sum1 += a[aOff+i+1] * (b[bOff+i+1] * scale)
		sum2 += a[aOff+i+2] * (b[bOff+i+2] * scale)
		sum3 += a[aOff+i+3] * (b[bOff+i+3] * scale)
		sum4 += a[aOff+i+4] * (b[bOff+i+4] * scale)
		sum5 += a[aOff+i+5] * (b[bOff+i+5] * scale)
		sum6 += a[aOff+i+6] * (b[bOff+i+6] * scale)
		sum7 += a[aOff+i+7] * (b[bOff+i+7] * scale)
	}
	sum := (sum0 + sum1) + (sum2 + sum3)
	sum += (sum4 + sum5) + (sum6 + sum7)
	for ; i < n; i++ {
		sum += a[aOff+i] * (b[bOff+i] * scale)
	}
	return sum
}

func dotF64(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var sum float64
	for i := 0; i < n; i++ {
		sum += float64(a[i]) * float64(b[i])
	}
	return float32(sum)
}
