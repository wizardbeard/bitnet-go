package kernels

func Dot(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	if matchGGML() {
		var sum float32
		for i := 0; i < n; i++ {
			sum += a[i] * b[i]
		}
		return sum
	}
	var sum float64
	for i := 0; i < n; i++ {
		sum += float64(a[i]) * float64(b[i])
	}
	return float32(sum)
}

func AddScaled(dst, src []float32, scale float32) {
	n := len(dst)
	if len(src) < n {
		n = len(src)
	}
	for i := 0; i < n; i++ {
		dst[i] += src[i] * scale
	}
}

func Argmax(v []float32) int {
	if len(v) == 0 {
		return -1
	}
	best := 0
	bestVal := v[0]
	for i := 1; i < len(v); i++ {
		if v[i] > bestVal {
			bestVal = v[i]
			best = i
		}
	}
	return best
}

// MulRelu2Into computes dst[i] = max(gate[i], 0)^2 * up[i].
func MulRelu2Into(dst, gate, up []float32) {
	mulReluImpl(dst, gate, up)
}

// MatVec computes dst = mat * vec where mat is GGML column-major [rows][cols]
// with contiguous dimension ne0=rows.
func MatVec(dst, mat []float32, rows, cols int, vec []float32) {
	matVecImpl(dst, mat, rows, cols, vec)
}

// MatVecT computes dst = transpose(mat) * vec where mat is GGML column-major [rows][cols]
// with contiguous dimension ne0=rows.
func MatVecT(dst, mat []float32, rows, cols int, vec []float32) {
	matVecTImpl(dst, mat, rows, cols, vec)
}

func matVecGeneric(dst, mat []float32, rows, cols int, vec []float32) {
	if rows <= 0 || cols <= 0 {
		return
	}
	if len(dst) < rows || len(vec) < cols || len(mat) < rows*cols {
		return
	}
	for r := 0; r < rows; r++ {
		var sum float64
		for c := 0; c < cols; c++ {
			sum += float64(mat[r+rows*c]) * float64(vec[c])
		}
		dst[r] = float32(sum)
	}
}

func matVecTGeneric(dst, mat []float32, rows, cols int, vec []float32) {
	if rows <= 0 || cols <= 0 {
		return
	}
	if len(dst) < cols || len(vec) < rows || len(mat) < rows*cols {
		return
	}
	for c := 0; c < cols; c++ {
		var sum float64
		for r := 0; r < rows; r++ {
			sum += float64(mat[r+rows*c]) * float64(vec[r])
		}
		dst[c] = float32(sum)
	}
}

// matVecI2SGeneric computes dst = mat * vec where mat is GGML column-major [rows][cols]
// stored in packed i2_s format with 2-bit values and a global scale.
func matVecI2SGeneric(dst []float32, packed []byte, rows, cols int, vec []float32, scale float32) {
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
		var sum float32
		for c := 0; c < cols; c++ {
			idx := r + rows*c
			q := i2sPackedAt(packed, idx)
			var w float32
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
			sum += w * scale * vec[c]
		}
		dst[r] = sum
	}
}

// matVecTI2SGeneric computes dst = transpose(mat) * vec where mat is GGML column-major [rows][cols]
// stored in packed i2_s format with 2-bit values and a global scale.
func matVecTI2SGeneric(dst []float32, packed []byte, rows, cols int, vec []float32, scale float32) {
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
		var sum float32
		for r := 0; r < rows; r++ {
			idx := r + rows*c
			q := i2sPackedAt(packed, idx)
			var w float32
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
			sum += w * scale * vec[r]
		}
		dst[c] = sum
	}
}
