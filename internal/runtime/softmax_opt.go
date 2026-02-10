package runtime

import "math"

func softmaxInPlaceOpt(scores []float32, maxScore float32) float32 {
	n := len(scores)
	if debugStrictExpf || debugFastExpf {
		var sum0, sum1, sum2, sum3 float32
		var sum4, sum5, sum6, sum7 float32
		i := 0
		for ; i+7 < n; i += 8 {
			d0 := scores[i] - maxScore
			d1 := scores[i+1] - maxScore
			d2 := scores[i+2] - maxScore
			d3 := scores[i+3] - maxScore
			d4 := scores[i+4] - maxScore
			d5 := scores[i+5] - maxScore
			d6 := scores[i+6] - maxScore
			d7 := scores[i+7] - maxScore
			w0 := expf32(d0)
			w1 := expf32(d1)
			w2 := expf32(d2)
			w3 := expf32(d3)
			w4 := expf32(d4)
			w5 := expf32(d5)
			w6 := expf32(d6)
			w7 := expf32(d7)
			scores[i] = w0
			scores[i+1] = w1
			scores[i+2] = w2
			scores[i+3] = w3
			scores[i+4] = w4
			scores[i+5] = w5
			scores[i+6] = w6
			scores[i+7] = w7
			sum0 += w0
			sum1 += w1
			sum2 += w2
			sum3 += w3
			sum4 += w4
			sum5 += w5
			sum6 += w6
			sum7 += w7
		}
		sum := (sum0 + sum1) + (sum2 + sum3)
		sum += (sum4 + sum5) + (sum6 + sum7)
		for ; i < n; i++ {
			diff := scores[i] - maxScore
			w := expf32(diff)
			scores[i] = w
			sum += w
		}
		return sum
	}
	var sum0, sum1, sum2, sum3 float32
	i := 0
	for ; i+3 < n; i += 4 {
		d0 := scores[i] - maxScore
		d1 := scores[i+1] - maxScore
		d2 := scores[i+2] - maxScore
		d3 := scores[i+3] - maxScore
		w0 := float32(math.Exp(float64(d0)))
		w1 := float32(math.Exp(float64(d1)))
		w2 := float32(math.Exp(float64(d2)))
		w3 := float32(math.Exp(float64(d3)))
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
		w := float32(math.Exp(float64(diff)))
		scores[i] = w
		sum += w
	}
	return sum
}
