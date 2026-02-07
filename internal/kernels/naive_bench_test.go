package kernels

import "testing"

func BenchmarkDot(b *testing.B) {
	sizes := []int{256, 1024, 4096}
	for _, n := range sizes {
		b.Run("n="+itoa(n), func(b *testing.B) {
			a := make([]float32, n)
			c := make([]float32, n)
			for i := 0; i < n; i++ {
				a[i] = float32(i%97) * 0.01
				c[i] = float32(i%31) * 0.02
			}
			b.ReportAllocs()
			b.SetBytes(int64(n * 4 * 2))
			b.ResetTimer()
			var sink float32
			for i := 0; i < b.N; i++ {
				sink = Dot(a, c)
			}
			_ = sink
		})
	}
}

func BenchmarkAddScaled(b *testing.B) {
	sizes := []int{256, 1024, 4096}
	for _, n := range sizes {
		b.Run("n="+itoa(n), func(b *testing.B) {
			dst := make([]float32, n)
			src := make([]float32, n)
			for i := 0; i < n; i++ {
				dst[i] = float32(i%17) * 0.03
				src[i] = float32(i%29) * 0.04
			}
			b.ReportAllocs()
			b.SetBytes(int64(n * 4 * 2))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				AddScaled(dst, src, 0.5)
			}
		})
	}
}

func BenchmarkArgmax(b *testing.B) {
	sizes := []int{256, 1024, 4096}
	for _, n := range sizes {
		b.Run("n="+itoa(n), func(b *testing.B) {
			v := make([]float32, n)
			for i := 0; i < n; i++ {
				v[i] = float32((i%101)-50) * 0.01
			}
			b.ReportAllocs()
			b.SetBytes(int64(n * 4))
			b.ResetTimer()
			var sink int
			for i := 0; i < b.N; i++ {
				sink = Argmax(v)
			}
			_ = sink
		})
	}
}

func BenchmarkMatVec(b *testing.B) {
	type shape struct {
		rows int
		cols int
	}
	shapes := []shape{
		{256, 256},
		{512, 512},
		{1024, 1024},
	}
	for _, s := range shapes {
		name := "r=" + itoa(s.rows) + "/c=" + itoa(s.cols)
		b.Run(name, func(b *testing.B) {
			mat := make([]float32, s.rows*s.cols)
			vec := make([]float32, s.cols)
			dst := make([]float32, s.rows)
			for c := 0; c < s.cols; c++ {
				vec[c] = float32(c%31) * 0.01
				for r := 0; r < s.rows; r++ {
					mat[r+s.rows*c] = float32((r+c)%23) * 0.02
				}
			}
			b.ReportAllocs()
			b.SetBytes(int64((s.rows*s.cols + s.cols + s.rows) * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MatVec(dst, mat, s.rows, s.cols, vec)
			}
		})
	}
}

func BenchmarkMatVecT(b *testing.B) {
	type shape struct {
		rows int
		cols int
	}
	shapes := []shape{
		{256, 256},
		{512, 512},
		{1024, 1024},
	}
	for _, s := range shapes {
		name := "r=" + itoa(s.rows) + "/c=" + itoa(s.cols)
		b.Run(name, func(b *testing.B) {
			mat := make([]float32, s.rows*s.cols)
			vec := make([]float32, s.rows)
			dst := make([]float32, s.cols)
			for c := 0; c < s.cols; c++ {
				for r := 0; r < s.rows; r++ {
					mat[r+s.rows*c] = float32((r+c)%23) * 0.02
				}
			}
			for r := 0; r < s.rows; r++ {
				vec[r] = float32(r%29) * 0.01
			}
			b.ReportAllocs()
			b.SetBytes(int64((s.rows*s.cols + s.cols + s.rows) * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				MatVecT(dst, mat, s.rows, s.cols, vec)
			}
		})
	}
}

func BenchmarkMatVecI2SCompare(b *testing.B) {
	type shape struct {
		rows int
		cols int
	}
	shapes := []shape{
		{256, 256},
		{512, 512},
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

			b.Run("generic", func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64((s.rows*s.cols + s.cols + s.rows) * 4))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					matVecI2SGeneric(dst, packed, s.rows, s.cols, vec, 1.0)
				}
			})

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

func BenchmarkMatVecTI2SCompare(b *testing.B) {
	type shape struct {
		rows int
		cols int
	}
	shapes := []shape{
		{256, 256},
		{512, 512},
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
			vec := make([]float32, s.rows)
			for i := range vec {
				vec[i] = float32(i%31) * 0.01
			}
			dst := make([]float32, s.cols)

			b.Run("generic", func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64((s.rows*s.cols + s.cols + s.rows) * 4))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					matVecTI2SGeneric(dst, packed, s.rows, s.cols, vec, 1.0)
				}
			})

			b.Run("dispatch", func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64((s.rows*s.cols + s.cols + s.rows) * 4))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					MatVecTI2S(dst, packed, s.rows, s.cols, vec, 1.0)
				}
			})
		})
	}
}

func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}
	return string(buf[i:])
}
