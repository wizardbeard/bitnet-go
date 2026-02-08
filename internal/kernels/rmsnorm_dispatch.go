package kernels

import "math"

var rmsNormImpl = rmsNormGeneric

// RMSNormInto computes dst[i] = x[i] * inv_rms * weight[i].
func RMSNormInto(dst, x, weight []float32, eps float32) {
	rmsNormImpl(dst, x, weight, eps)
}

func rmsNormGeneric(dst, x, weight []float32, eps float32) {
	n := len(dst)
	if len(x) < n {
		n = len(x)
	}
	if len(weight) < n {
		n = len(weight)
	}
	if n == 0 {
		return
	}
	var sum float64
	if matchGGML() {
		var sum32 float32
		for i := 0; i < n; i++ {
			v := x[i]
			sum32 += v * v
		}
		sum = float64(sum32)
	} else {
		for i := 0; i < n; i++ {
			v := float64(x[i])
			sum += v * v
		}
	}
	inv := float32(1.0 / math.Sqrt(sum/float64(n)+float64(eps)))
	for i := 0; i < n; i++ {
		dst[i] = x[i] * inv * weight[i]
	}
}
