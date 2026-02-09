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

	scale = 127.0 / float32(maxAbs)
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

func i2sPackedAtAlt(packed []byte, row, col, rows, cols int) byte {
	if row < 0 || col < 0 || row >= rows || col >= cols {
		return 0
	}
	idx := col + cols*row
	return i2sPackedAt(packed, idx)
}

func i2sPackedLen(count int) int {
	if count <= 0 {
		return 0
	}
	const block = 128
	const blockBytes = 32
	return (count + block - 1) / block * blockBytes
}

func decodeI2SBlock(dst []int8, packed []byte) {
	if len(dst) < 128 || len(packed) < 32 {
		return
	}
	for gp := 0; gp < 32; gp++ {
		b := packed[gp]
		dst[gp] = int8((b >> 6) & 0x3)
		dst[32+gp] = int8((b >> 4) & 0x3)
		dst[64+gp] = int8((b >> 2) & 0x3)
		dst[96+gp] = int8(b & 0x3)
	}
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
	if rows%128 == 0 {
		var block [128]int8
		var sums [128]int32
		for rb := 0; rb < rows; rb += 128 {
			for i := range sums {
				sums[i] = 0
			}
			for c := 0; c < cols; c++ {
				idx := rb + rows*c
				bi := idx / 128
				decodeI2SBlock(block[:], packed[bi*32:bi*32+32])
				v := int32(vec[c])
				for i := 0; i < 128; i++ {
					sums[i] += int32(block[i]) * v
				}
			}
			for i := 0; i < 128; i++ {
				dst[rb+i] = float32(sums[i]-actSum) * (weightScale / actScale)
			}
		}
		return
	}
	for r := 0; r < rows; r++ {
		var sum int32
		c := 0
		for ; c+3 < cols; c += 4 {
			idx0 := r + rows*c
			idx1 := idx0 + rows
			idx2 := idx1 + rows
			idx3 := idx2 + rows
			q0 := i2sPackedAt(packed, idx0)
			q1 := i2sPackedAt(packed, idx1)
			q2 := i2sPackedAt(packed, idx2)
			q3 := i2sPackedAt(packed, idx3)
			sum += int32(q0)*int32(vec[c]) +
				int32(q1)*int32(vec[c+1]) +
				int32(q2)*int32(vec[c+2]) +
				int32(q3)*int32(vec[c+3])
		}
		for ; c < cols; c++ {
			idx := r + rows*c
			q := i2sPackedAt(packed, idx)
			sum += int32(q) * int32(vec[c])
		}
		dst[r] = float32(sum-actSum) * (weightScale / actScale)
	}
}

// MatVecI2SI8SScalar computes dst = mat * vec without block decoding (debug).
func MatVecI2SI8SScalar(dst []float32, packed []byte, rows, cols int, vec []int8, weightScale, actScale float32, actSum int32) {
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

// MatVecI2SI8SAlt treats packed weights as row-major for debug/layout comparison.
func MatVecI2SI8SAlt(dst []float32, packed []byte, rows, cols int, vec []int8, weightScale, actScale float32, actSum int32) {
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
			q := i2sPackedAtAlt(packed, r, c, rows, cols)
			sum += int32(q) * int32(vec[c])
		}
		dst[r] = float32(sum-actSum) * (weightScale / actScale)
	}
}

// MatVecI2SI8SMap computes dst = mat * vec with q=3 mapped to 1 before actSum adjust.
func MatVecI2SI8SMap(dst []float32, packed []byte, rows, cols int, vec []int8, weightScale, actScale float32, actSum int32) {
	if rows <= 0 || cols <= 0 {
		return
	}
	if len(dst) < rows || len(vec) < cols {
		return
	}
	if rows*cols == 0 || len(packed) < i2sPackedLen(rows*cols) {
		return
	}
	if rows%128 == 0 {
		var block [128]int8
		var sums [128]int32
		for rb := 0; rb < rows; rb += 128 {
			for i := range sums {
				sums[i] = 0
			}
			for c := 0; c < cols; c++ {
				idx := rb + rows*c
				bi := idx / 128
				decodeI2SBlock(block[:], packed[bi*32:bi*32+32])
				for i := 0; i < 128; i++ {
					if block[i] == 3 {
						block[i] = 1
					}
				}
				v := int32(vec[c])
				for i := 0; i < 128; i++ {
					sums[i] += int32(block[i]) * v
				}
			}
			for i := 0; i < 128; i++ {
				dst[rb+i] = float32(sums[i]-actSum) * (weightScale / actScale)
			}
		}
		return
	}
	for r := 0; r < rows; r++ {
		var sum int32
		c := 0
		for ; c+3 < cols; c += 4 {
			idx0 := r + rows*c
			idx1 := idx0 + rows
			idx2 := idx1 + rows
			idx3 := idx2 + rows
			q0 := i2sPackedAt(packed, idx0)
			q1 := i2sPackedAt(packed, idx1)
			q2 := i2sPackedAt(packed, idx2)
			q3 := i2sPackedAt(packed, idx3)
			if q0 == 3 {
				q0 = 1
			}
			if q1 == 3 {
				q1 = 1
			}
			if q2 == 3 {
				q2 = 1
			}
			if q3 == 3 {
				q3 = 1
			}
			sum += int32(q0)*int32(vec[c]) +
				int32(q1)*int32(vec[c+1]) +
				int32(q2)*int32(vec[c+2]) +
				int32(q3)*int32(vec[c+3])
		}
		for ; c < cols; c++ {
			idx := r + rows*c
			q := i2sPackedAt(packed, idx)
			if q == 3 {
				q = 1
			}
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
	var block [128]int8
	blockAligned := rows%128 == 0
	for c := 0; c < cols; c++ {
		var sum int32
		r := 0
		if blockAligned && (rows*c)%128 == 0 {
			for ; r+127 < rows; r += 128 {
				idx0 := r + rows*c
				bi := idx0 / 128
				decodeI2SBlock(block[:], packed[bi*32:bi*32+32])
				for i := 0; i < 128; i++ {
					sum += int32(block[i]) * int32(vec[r+i])
				}
			}
		}
		for ; r < rows; r++ {
			idx := r + rows*c
			q := i2sPackedAt(packed, idx)
			sum += int32(q) * int32(vec[r])
		}
		dst[c] = float32(sum-actSum) * (weightScale / actScale)
	}
}

// MatVecTI2SI8SScalar computes dst = transpose(mat) * vec without block decoding (debug).
func MatVecTI2SI8SScalar(dst []float32, packed []byte, rows, cols int, vec []int8, weightScale, actScale float32, actSum int32) {
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

// MatVecTI2SI8SAlt treats packed weights as row-major for debug/layout comparison.
func MatVecTI2SI8SAlt(dst []float32, packed []byte, rows, cols int, vec []int8, weightScale, actScale float32, actSum int32) {
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
			q := i2sPackedAtAlt(packed, r, c, rows, cols)
			sum += int32(q) * int32(vec[r])
		}
		dst[c] = float32(sum-actSum) * (weightScale / actScale)
	}
}

// MatVecTI2SI8SMap computes dst = transpose(mat) * vec with q=3 mapped to 1 before actSum adjust.
func MatVecTI2SI8SMap(dst []float32, packed []byte, rows, cols int, vec []int8, weightScale, actScale float32, actSum int32) {
	if rows <= 0 || cols <= 0 {
		return
	}
	if len(dst) < cols || len(vec) < rows {
		return
	}
	if rows*cols == 0 || len(packed) < i2sPackedLen(rows*cols) {
		return
	}
	var block [128]int8
	blockAligned := rows%128 == 0
	for c := 0; c < cols; c++ {
		var sum int32
		r := 0
		if blockAligned && (rows*c)%128 == 0 {
			for ; r+127 < rows; r += 128 {
				idx0 := r + rows*c
				bi := idx0 / 128
				decodeI2SBlock(block[:], packed[bi*32:bi*32+32])
				for i := 0; i < 128; i++ {
					if block[i] == 3 {
						block[i] = 1
					}
				}
				for i := 0; i < 128; i++ {
					sum += int32(block[i]) * int32(vec[r+i])
				}
			}
		}
		for ; r < rows; r++ {
			idx := r + rows*c
			q := i2sPackedAt(packed, idx)
			if q == 3 {
				q = 1
			}
			sum += int32(q) * int32(vec[r])
		}
		dst[c] = float32(sum-actSum) * (weightScale / actScale)
	}
}

// MatVecI2SI8SRef matches ggml i2_s map2bit semantics (q=3 maps to 0) and ignores actSum.
func MatVecI2SI8SRef(dst []float32, packed []byte, rows, cols int, vec []int8, weightScale, actScale float32) {
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
			var w int32
			switch q {
			case 0:
				w = -1
			case 2:
				w = 1
			default:
				w = 0
			}
			sum += w * int32(vec[c])
		}
		dst[r] = float32(sum) * (weightScale / actScale)
	}
}

// MatVecTI2SI8SRef matches ggml i2_s map2bit semantics (q=3 maps to 0) and ignores actSum.
func MatVecTI2SI8SRef(dst []float32, packed []byte, rows, cols int, vec []int8, weightScale, actScale float32) {
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
			var w int32
			switch q {
			case 0:
				w = -1
			case 2:
				w = 1
			default:
				w = 0
			}
			sum += w * int32(vec[r])
		}
		dst[c] = float32(sum) * (weightScale / actScale)
	}
}
