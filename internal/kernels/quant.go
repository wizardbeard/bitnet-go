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

	var maxAbs float64
	for i := 0; i < n; i++ {
		v := float64(src[i])
		if v < 0 {
			v = -v
		}
		if v > maxAbs {
			maxAbs = v
		}
	}
	if maxAbs < 1e-5 {
		maxAbs = 1e-5
	}

	scale = float32(127.0 / maxAbs)
	for i := 0; i < n; i++ {
		q := nearestInt(src[i] * scale)
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

func nearestInt(fval float32) int {
	const bias = 12582912.0
	val := fval + float32(bias)
	i := math.Float32bits(val)
	return int(i&0x007fffff) - 0x00400000
}

func i2sPackedAt(packed []byte, idx int) byte {
	if idx < 0 {
		return 0
	}
	const block = 128
	const blockBytes = 32
	bi := idx / block
	off := idx % block
	gp := off % 32
	group := off / 32
	p := bi*blockBytes + gp
	if p < 0 || p >= len(packed) {
		return 0
	}
	shift := uint(6 - 2*group)
	return (packed[p] >> shift) & 0x3
}

func i2sPackedLen(count int) int {
	if count <= 0 {
		return 0
	}
	const block = 128
	const blockBytes = 32
	return (count + block - 1) / block * blockBytes
}

// MatVecI2SI8S computes dst = mat * vec where mat is GGML column-major [rows][cols]
// stored in packed i2_s format with a global weight scale, and vec is quantized i8_s.
func MatVecI2SI8S(dst []float32, packed []byte, rows, cols int, vec []int8, weightScale, actScale float32, actSum int32) {
	if rows <= 0 || cols <= 0 {
		return
	}
	if len(dst) < rows || len(vec) < cols {
		return
	}
	if rows*cols == 0 || len(packed) < i2sPackedLen(rows*cols) {
		return
	}
	for r := 0; r < rows; r++ {
		var sum int32
		for c := 0; c < cols; c++ {
			idx := r + rows*c
			q := i2sPackedAt(packed, idx)
			sum += int32(q) * int32(vec[c])
		}
		dst[r] = float32(sum-actSum) * (weightScale / actScale)
	}
}

// MatVecTI2SI8S computes dst = transpose(mat) * vec where mat is GGML column-major [rows][cols]
// stored in packed i2_s format with a global weight scale, and vec is quantized i8_s.
func MatVecTI2SI8S(dst []float32, packed []byte, rows, cols int, vec []int8, weightScale, actScale float32, actSum int32) {
	if rows <= 0 || cols <= 0 {
		return
	}
	if len(dst) < cols || len(vec) < rows {
		return
	}
	if rows*cols == 0 || len(packed) < i2sPackedLen(rows*cols) {
		return
	}
	for c := 0; c < cols; c++ {
		var sum int32
		for r := 0; r < rows; r++ {
			idx := r + rows*c
			q := i2sPackedAt(packed, idx)
			sum += int32(q) * int32(vec[r])
		}
		dst[c] = float32(sum-actSum) * (weightScale / actScale)
	}
}
