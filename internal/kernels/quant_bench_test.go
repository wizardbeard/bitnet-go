package kernels

import (
	"math/rand"
	"strconv"
	"testing"
)

type i2sBenchShape struct {
	rows int
	cols int
}

var i2sBenchShapes = []i2sBenchShape{
	{rows: 512, cols: 512},
	{rows: 1024, cols: 1024},
	{rows: 2560, cols: 2560},
}

func i2sShapeName(s i2sBenchShape) string {
	return "r=" + strconv.Itoa(s.rows) + "/c=" + strconv.Itoa(s.cols)
}

func makeI2SPacked(rows, cols int) []byte {
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
	return packI2S(vals)
}

func BenchmarkMatVecI2SI8S(b *testing.B) {
	const rows, cols = 2560, 2560
	vec := make([]int8, cols)
	for i := range vec {
		vec[i] = int8(i%255 - 127)
	}
	packed := makeI2SPacked(rows, cols)
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
	packed := makeI2SPacked(rows, cols)
	dst := make([]float32, cols)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatVecTI2SI8S(dst, packed, rows, cols, vec, 1.0, 1.0, 0)
	}
}

func BenchmarkMatVecI2SI8SVariants(b *testing.B) {
	for _, s := range i2sBenchShapes {
		b.Run(i2sShapeName(s), func(b *testing.B) {
			packed := makeI2SPacked(s.rows, s.cols)
			dst := make([]float32, s.rows)
			vec := make([]int8, s.cols)
			for i := range vec {
				vec[i] = int8(i%255 - 127)
			}

			b.Run("dispatch", func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64(s.rows*s.cols + s.rows + s.cols))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					MatVecI2SI8S(dst, packed, s.rows, s.cols, vec, 1.0, 1.0, 0)
				}
			})

			b.Run("generic_block", func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64(s.rows*s.cols + s.rows + s.cols))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					matVecI2SI8SRange(dst, packed, s.rows, s.cols, vec, 1.0, 1.0, 0, 0, s.rows)
				}
			})

			b.Run("scalar", func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64(s.rows*s.cols + s.rows + s.cols))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					MatVecI2SI8SScalar(dst, packed, s.rows, s.cols, vec, 1.0, 1.0, 0)
				}
			})
		})
	}
}

func BenchmarkMatVecTI2SI8SVariants(b *testing.B) {
	for _, s := range i2sBenchShapes {
		b.Run(i2sShapeName(s), func(b *testing.B) {
			packed := makeI2SPacked(s.rows, s.cols)
			dst := make([]float32, s.cols)
			vec := make([]int8, s.rows)
			for i := range vec {
				vec[i] = int8(i%255 - 127)
			}

			b.Run("dispatch", func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64(s.rows*s.cols + s.rows + s.cols))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					MatVecTI2SI8S(dst, packed, s.rows, s.cols, vec, 1.0, 1.0, 0)
				}
			})

			b.Run("generic_block", func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64(s.rows*s.cols + s.rows + s.cols))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					matVecTI2SI8SRange(dst, packed, s.rows, s.cols, vec, 1.0, 1.0, 0, 0, s.cols)
				}
			})

			b.Run("scalar", func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64(s.rows*s.cols + s.rows + s.cols))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					MatVecTI2SI8SScalar(dst, packed, s.rows, s.cols, vec, 1.0, 1.0, 0)
				}
			})
		})
	}
}

func BenchmarkQuantizeRowI8S(b *testing.B) {
	const n = 2560
	src := make([]float32, n)
	for i := range src {
		src[i] = float32((i%97)-48) * 0.01
	}
	dst := make([]int8, n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		QuantizeRowI8S(dst, src)
	}
}
