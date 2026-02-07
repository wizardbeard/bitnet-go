//go:build arm64 && cgo

package kernels

import "testing"

func BenchmarkMatVecI2SArm64(b *testing.B) {
	type shape struct {
		rows int
		cols int
	}
	shapes := []shape{
		{128, 256},
		{256, 256},
		{512, 256},
	}
	for _, s := range shapes {
		name := "r=" + itoa(s.rows) + "/c=" + itoa(s.cols)
		b.Run(name, func(b *testing.B) {
			vals := make([]int8, s.rows*s.cols)
			for i := range vals {
				switch i % 3 {
				case 0:
					vals[i] = -1
				case 1:
					vals[i] = 0
				default:
					vals[i] = 1
				}
			}
			packed := make([]byte, (len(vals)+3)/4)
			for i, v := range vals {
				var q byte
				switch v {
				case -1:
					q = 0
				case 0:
					q = 1
				case 1:
					q = 2
				default:
					q = 1
				}
				shift := uint(6 - 2*(i%4))
				packed[i/4] |= q << shift
			}
			vec := make([]float32, s.cols)
			for i := range vec {
				vec[i] = float32(i%31) * 0.01
			}
			dst := make([]float32, s.rows)

			b.Run("dispatch", func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64((s.rows*s.cols + s.cols + s.rows) * 4))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					MatVecI2S(dst, packed, s.rows, s.cols, vec, 1.0)
				}
			})
		})
	}
}
