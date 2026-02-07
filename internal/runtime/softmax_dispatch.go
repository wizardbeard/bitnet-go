package runtime

import "math"

var softmaxInPlaceImpl = softmaxInPlaceGeneric

func softmaxInPlace(scores []float32, maxScore float32) float32 {
	return softmaxInPlaceImpl(scores, maxScore)
}

func softmaxInPlaceGeneric(scores []float32, maxScore float32) float32 {
	var sum float32
	for i := range scores {
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
