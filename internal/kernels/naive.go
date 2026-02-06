package kernels

func Dot(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var sum float32
	for i := 0; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
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

// MatVec computes dst = mat * vec where mat is GGML column-major [rows][cols]
// with contiguous dimension ne0=rows.
func MatVec(dst, mat []float32, rows, cols int, vec []float32) {
	if rows <= 0 || cols <= 0 {
		return
	}
	if len(dst) < rows || len(vec) < cols || len(mat) < rows*cols {
		return
	}
	for r := 0; r < rows; r++ {
		var sum float32
		for c := 0; c < cols; c++ {
			sum += mat[r+rows*c] * vec[c]
		}
		dst[r] = sum
	}
}

// MatVecT computes dst = transpose(mat) * vec where mat is GGML column-major [rows][cols]
// with contiguous dimension ne0=rows.
func MatVecT(dst, mat []float32, rows, cols int, vec []float32) {
	if rows <= 0 || cols <= 0 {
		return
	}
	if len(dst) < cols || len(vec) < rows || len(mat) < rows*cols {
		return
	}
	for c := 0; c < cols; c++ {
		var sum float32
		for r := 0; r < rows; r++ {
			sum += mat[r+rows*c] * vec[r]
		}
		dst[c] = sum
	}
}
