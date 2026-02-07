package runtime

import "math"

func softmaxInPlaceOpt(scores []float32, maxScore float32) float32 {
	var sum0, sum1, sum2, sum3 float32
	i := 0
	n := len(scores)
	for ; i+3 < n; i += 4 {
		d0 := scores[i] - maxScore
		d1 := scores[i+1] - maxScore
		d2 := scores[i+2] - maxScore
		d3 := scores[i+3] - maxScore
		var w0, w1, w2, w3 float32
		if debugStrictExpf {
			w0 = expf32(d0)
			w1 = expf32(d1)
			w2 = expf32(d2)
			w3 = expf32(d3)
		} else {
			w0 = float32(math.Exp(float64(d0)))
			w1 = float32(math.Exp(float64(d1)))
			w2 = float32(math.Exp(float64(d2)))
			w3 = float32(math.Exp(float64(d3)))
		}
		scores[i] = w0
		scores[i+1] = w1
		scores[i+2] = w2
		scores[i+3] = w3
		sum0 += w0
		sum1 += w1
		sum2 += w2
		sum3 += w3
	}
	sum := sum0 + sum1 + sum2 + sum3
	for ; i < n; i++ {
		diff := scores[i] - maxScore
		var w float32
		if debugStrictExpf {
			w = expf32(diff)
		} else {
			w = float32(math.Exp(float64(diff)))
		}
		scores[i] = w
		sum += w
	}
	return sum
}
