package kernels

import (
	"math/rand"
	"testing"
)

func BenchmarkMatVecI2SI8S(b *testing.B) {
	const rows, cols = 2560, 2560
	vec := make([]int8, cols)
	for i := range vec {
		vec[i] = int8(i%255 - 127)
	}
	rng := rand.New(rand.NewSource(1))
	vals := make([]int, rows*cols)
	for i := range vals {
		switch rng.Intn(3) {
		case 0:
			vals[i] = -1
		case 1:
			vals[i] = 0
		default:
			vals[i] = 1
		}
	}
	packed := packI2S(vals)
	dst := make([]float32, rows)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatVecI2SI8S(dst, packed, rows, cols, vec, 1.0, 1.0, 0)
	}
}

func BenchmarkMatVecTI2SI8S(b *testing.B) {
	const rows, cols = 2560, 2560
	vec := make([]int8, rows)
	for i := range vec {
		vec[i] = int8(i%255 - 127)
	}
	rng := rand.New(rand.NewSource(1))
	vals := make([]int, rows*cols)
	for i := range vals {
		switch rng.Intn(3) {
		case 0:
			vals[i] = -1
		case 1:
			vals[i] = 0
		default:
			vals[i] = 1
		}
	}
	packed := packI2S(vals)
	dst := make([]float32, cols)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatVecTI2SI8S(dst, packed, rows, cols, vec, 1.0, 1.0, 0)
	}
}
