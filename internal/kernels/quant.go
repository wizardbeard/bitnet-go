package kernels

import "math"

// QuantizeRowI8S quantizes src into dst using i8_s rules and returns the
// dequantization scale and sum of quantized values.
func QuantizeRowI8S(dst []int8, src []float32) (scale float32, sum int32) {
	n := len(src)
	if len(dst) < n {
		n = len(dst)
	}
	if n == 0 {
		return 0, 0
	}

	var maxAbs float32
	for i := 0; i < n; i++ {
		v := src[i]
		if v < 0 {
			v = -v
		}
		if v > maxAbs {
			maxAbs = v
		}
	}
	if maxAbs < 1e-5 {
		for i := 0; i < n; i++ {
			dst[i] = 0
		}
		return 0, 0
	}

	scale = maxAbs / 127.0
	inv := 1.0 / float64(scale)
	for i := 0; i < n; i++ {
		q := math.RoundToEven(float64(src[i]) * inv)
		if q < -128 {
			q = -128
		} else if q > 127 {
			q = 127
		}
		dst[i] = int8(q)
		sum += int32(dst[i])
	}
	return scale, sum
}

// MatVecI2SI8S computes dst = mat * vec where mat is GGML column-major [rows][cols]
// stored in packed i2_s format with a global weight scale, and vec is quantized i8_s.
func MatVecI2SI8S(dst []float32, packed []byte, rows, cols int, vec []int8, weightScale, actScale float32) {
	if rows <= 0 || cols <= 0 {
		return
	}
	if len(dst) < rows || len(vec) < cols {
		return
	}
	if rows*cols == 0 || len(packed) < (rows*cols+3)/4 {
		return
	}
	combined := weightScale * actScale
	for r := 0; r < rows; r++ {
		var sum int32
		for c := 0; c < cols; c++ {
			idx := r + rows*c
			b := packed[idx/4]
			shift := uint(6 - 2*(idx%4))
			q := (b >> shift) & 0x3
			var w int8
			switch q {
			case 0:
				w = -1
			case 1:
				w = 0
			case 2:
				w = 1
			default:
				w = 0
			}
			sum += int32(w) * int32(vec[c])
		}
		dst[r] = float32(sum) * combined
	}
}

// MatVecTI2SI8S computes dst = transpose(mat) * vec where mat is GGML column-major [rows][cols]
// stored in packed i2_s format with a global weight scale, and vec is quantized i8_s.
func MatVecTI2SI8S(dst []float32, packed []byte, rows, cols int, vec []int8, weightScale, actScale float32) {
	if rows <= 0 || cols <= 0 {
		return
	}
	if len(dst) < cols || len(vec) < rows {
		return
	}
	if rows*cols == 0 || len(packed) < (rows*cols+3)/4 {
		return
	}
	combined := weightScale * actScale
	for c := 0; c < cols; c++ {
		var sum int32
		for r := 0; r < rows; r++ {
			idx := r + rows*c
			b := packed[idx/4]
			shift := uint(6 - 2*(idx%4))
			q := (b >> shift) & 0x3
			var w int8
			switch q {
			case 0:
				w = -1
			case 1:
				w = 0
			case 2:
				w = 1
			default:
				w = 0
			}
			sum += int32(w) * int32(vec[r])
		}
		dst[c] = float32(sum) * combined
	}
}
