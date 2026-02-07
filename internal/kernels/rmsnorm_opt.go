package kernels

import "math"

func rmsNormOpt(dst, x, weight []float32, eps float32) {
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
	var sum0, sum1, sum2, sum3 float64
	i := 0
	for ; i+3 < n; i += 4 {
		v0 := float64(x[i])
		v1 := float64(x[i+1])
		v2 := float64(x[i+2])
		v3 := float64(x[i+3])
		sum0 += v0 * v0
		sum1 += v1 * v1
		sum2 += v2 * v2
		sum3 += v3 * v3
	}
	sum := sum0 + sum1 + sum2 + sum3
	for ; i < n; i++ {
		v := float64(x[i])
		sum += v * v
	}
	inv := float32(1.0 / math.Sqrt(sum/float64(n)+float64(eps)))
	i = 0
	for ; i+3 < n; i += 4 {
		dst[i] = x[i] * inv * weight[i]
		dst[i+1] = x[i+1] * inv * weight[i+1]
		dst[i+2] = x[i+2] * inv * weight[i+2]
		dst[i+3] = x[i+3] * inv * weight[i+3]
	}
	for ; i < n; i++ {
		dst[i] = x[i] * inv * weight[i]
	}
}
