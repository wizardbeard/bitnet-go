package kernels

import (
	"runtime"
	"sync"
)

var (
	matVecTParMinRows = envIntArch("BITNET_MATVECT_PAR_MIN_ROWS", 512)
	matVecTParMinCols = envIntArch("BITNET_MATVECT_PAR_MIN_COLS", 8192)
	matVecTParWorkers = envIntArch("BITNET_MATVECT_PAR_WORKERS", 0)
)

func matVecOpt(dst, mat []float32, rows, cols int, vec []float32) {
	if rows <= 0 || cols <= 0 {
		return
	}
	if len(dst) < rows || len(vec) < cols || len(mat) < rows*cols {
		return
	}
	if fastColMatVec() && !matchGGML() {
		for r := 0; r < rows; r++ {
			dst[r] = 0
		}
		for c := 0; c < cols; c++ {
			scale := vec[c]
			base := rows * c
			r := 0
			for ; r+7 < rows; r += 8 {
				dst[r] += mat[base+r] * scale
				dst[r+1] += mat[base+r+1] * scale
				dst[r+2] += mat[base+r+2] * scale
				dst[r+3] += mat[base+r+3] * scale
				dst[r+4] += mat[base+r+4] * scale
				dst[r+5] += mat[base+r+5] * scale
				dst[r+6] += mat[base+r+6] * scale
				dst[r+7] += mat[base+r+7] * scale
			}
			for ; r < rows; r++ {
				dst[r] += mat[base+r] * scale
			}
		}
		return
	}
	for r := 0; r < rows; r++ {
		if matchGGML() {
			var sum float32
			for c := 0; c < cols; c++ {
				sum += mat[r+rows*c] * vec[c]
			}
			dst[r] = sum
			continue
		}
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
	if !matchGGML() && rows >= matVecTParMinRows && cols >= matVecTParMinCols {
		workers := matVecTParWorkers
		if workers <= 0 {
			workers = runtime.GOMAXPROCS(0)
		}
		if workers > cols {
			workers = cols
		}
		if workers > 1 {
			chunk := (cols + workers - 1) / workers
			var wg sync.WaitGroup
			wg.Add(workers)
			for w := 0; w < workers; w++ {
				start := w * chunk
				end := start + chunk
				if end > cols {
					end = cols
				}
				go func(start, end int) {
					defer wg.Done()
					for c := start; c < end; c++ {
						base := rows * c
						var sum0, sum1, sum2, sum3 float64
						r := 0
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
				}(start, end)
			}
			wg.Wait()
			return
		}
	}
	for c := 0; c < cols; c++ {
		if matchGGML() {
			var sum float32
			base := rows * c
			for r := 0; r < rows; r++ {
				sum += mat[base+r] * vec[r]
			}
			dst[c] = sum
			continue
		}
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
