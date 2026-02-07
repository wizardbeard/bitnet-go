package kernels

func matVecOpt(dst, mat []float32, rows, cols int, vec []float32) {
	if rows <= 0 || cols <= 0 {
		return
	}
	if len(dst) < rows || len(vec) < cols || len(mat) < rows*cols {
		return
	}
	for r := 0; r < rows; r++ {
		var sum0, sum1, sum2, sum3 float64
		c := 0
		for ; c+3 < cols; c += 4 {
			base0 := r + rows*c
			base1 := base0 + rows
			base2 := base1 + rows
			base3 := base2 + rows
			sum0 += float64(mat[base0]) * float64(vec[c])
			sum1 += float64(mat[base1]) * float64(vec[c+1])
			sum2 += float64(mat[base2]) * float64(vec[c+2])
			sum3 += float64(mat[base3]) * float64(vec[c+3])
		}
		sum := sum0 + sum1 + sum2 + sum3
		for ; c < cols; c++ {
			sum += float64(mat[r+rows*c]) * float64(vec[c])
		}
		dst[r] = float32(sum)
	}
}

func matVecTOpt(dst, mat []float32, rows, cols int, vec []float32) {
	if rows <= 0 || cols <= 0 {
		return
	}
	if len(dst) < cols || len(vec) < rows || len(mat) < rows*cols {
		return
	}
	for c := 0; c < cols; c++ {
		var sum0, sum1, sum2, sum3 float64
		r := 0
		base := rows * c
		for ; r+3 < rows; r += 4 {
			sum0 += float64(mat[base+r]) * float64(vec[r])
			sum1 += float64(mat[base+r+1]) * float64(vec[r+1])
			sum2 += float64(mat[base+r+2]) * float64(vec[r+2])
			sum3 += float64(mat[base+r+3]) * float64(vec[r+3])
		}
		sum := sum0 + sum1 + sum2 + sum3
		for ; r < rows; r++ {
			sum += float64(mat[base+r]) * float64(vec[r])
		}
		dst[c] = float32(sum)
	}
}
