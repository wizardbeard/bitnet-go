package kernels

import (
	"math"
	"sync"
)

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

var i2sDecodeTable = func() [256][4]int8 {
	var table [256][4]int8
	for b := 0; b < 256; b++ {
		table[b][0] = int8((b >> 6) & 0x3)
		table[b][1] = int8((b >> 4) & 0x3)
		table[b][2] = int8((b >> 2) & 0x3)
		table[b][3] = int8(b & 0x3)
	}
	return table
}()

var matVecI2SI8SFast func(dst []float32, packed []byte, rows, cols int, vec []int8, weightScale, actScale float32, actSum int32)
var matVecI2SI8SFastRange func(dst []float32, packed []byte, rows, cols int, vec []int8, weightScale, actScale float32, actSum int32, cStart, cEnd int) bool
var matVecTI2SI8SFast func(dst []float32, packed []byte, rows, cols int, vec []int8, weightScale, actScale float32, actSum int32)
var matVecTI2SI8SFastRange func(dst []float32, packed []byte, rows, cols int, vec []int8, weightScale, actScale float32, actSum int32, cStart, cEnd int) bool
var i2sI8SParallelRowsMin = envIntArch("BITNET_I2S_I8S_PAR_ROWS_MIN", 512)
var i2sI8SParallelColsMin = envIntArch("BITNET_I2S_I8S_PAR_COLS_MIN", 512)
var i2sI8SParallelChunkRows = envIntArch("BITNET_I2S_I8S_PAR_CHUNK_ROWS", 0)
var i2sI8SParallelChunkCols = envIntArch("BITNET_I2S_I8S_PAR_CHUNK_COLS", 0)
var i2sI8SFastMinElems = envIntArch("BITNET_I2S_I8S_FAST_MIN_ELEMS", 0)
var i2sI8SBlockMinRows = envIntArch("BITNET_I2S_I8S_BLOCK_MIN_ROWS", 256)
var i2sI8SFastParallelNTColsMin = envIntArch("BITNET_I2S_I8S_FAST_PAR_NT_COLS_MIN", 0)
var i2sI8SFastParallelColsMin = envIntArch("BITNET_I2S_I8S_FAST_PAR_COLS_MIN", 0)
var matVecI2SPartialPool = sync.Pool{
	New: func() any {
		return make([]float32, 0)
	},
}

func acquireI2SPartialBuf(n int) []float32 {
	buf := matVecI2SPartialPool.Get().([]float32)
	if cap(buf) < n {
		return make([]float32, n)
	}
	return buf[:n]
}

func releaseI2SPartialBuf(buf []float32) {
	matVecI2SPartialPool.Put(buf[:0])
}

func useI2SI8SFast(rows, cols int) bool {
	if i2sI8SFastMinElems <= 0 {
		return true
	}
	return rows*cols >= i2sI8SFastMinElems
}

func useI2SBlockPath(rows int) bool {
	blockMin := i2sI8SBlockMinRows
	if blockMin < 128 {
		blockMin = 128
	}
	return rows%128 == 0 && rows >= blockMin
}

func decodeI2SBlock(dst []int8, packed []byte) {
	if len(dst) < 128 || len(packed) < 32 {
		return
	}
	for gp := 0; gp < 32; gp++ {
		vals := i2sDecodeTable[packed[gp]]
		dst[gp] = vals[0]
		dst[32+gp] = vals[1]
		dst[64+gp] = vals[2]
		dst[96+gp] = vals[3]
	}
}

func accumI2SBlock128(sums *[128]int32, block *[128]int8, v int32) {
	i := 0
	for ; i+7 < 128; i += 8 {
		sums[i] += int32(block[i]) * v
		sums[i+1] += int32(block[i+1]) * v
		sums[i+2] += int32(block[i+2]) * v
		sums[i+3] += int32(block[i+3]) * v
		sums[i+4] += int32(block[i+4]) * v
		sums[i+5] += int32(block[i+5]) * v
		sums[i+6] += int32(block[i+6]) * v
		sums[i+7] += int32(block[i+7]) * v
	}
	for ; i < 128; i++ {
		sums[i] += int32(block[i]) * v
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
	outScale := weightScale / actScale
	corr := float32(actSum) * outScale
	if matVecI2SI8SFast != nil && useI2SI8SFast(rows, cols) {
		if matVecThreads() > 1 && i2sI8SFastParallelNTColsMin > 0 &&
			cols >= i2sI8SFastParallelNTColsMin && matVecI2SI8SFastRange != nil {
			matVecI2SI8SFastParallel(dst, packed, rows, cols, vec, weightScale, actScale, actSum)
		} else {
			matVecI2SI8SFast(dst, packed, rows, cols, vec, weightScale, actScale, actSum)
		}
		return
	}
	if matVecThreads() > 1 && rows >= i2sI8SParallelRowsMin {
		matVecI2SI8SParallel(dst, packed, rows, cols, vec, weightScale, actScale, actSum)
		return
	}
	if useI2SBlockPath(rows) {
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
				accumI2SBlock128(&sums, &block, v)
			}
			for i := 0; i < 128; i++ {
				dst[rb+i] = float32(sums[i])*outScale - corr
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
		dst[r] = float32(sum)*outScale - corr
	}
}

func matVecI2SI8SFastParallel(dst []float32, packed []byte, rows, cols int, vec []int8, weightScale, actScale float32, actSum int32) {
	threads := matVecThreads()
	if threads <= 1 {
		matVecI2SI8SFast(dst, packed, rows, cols, vec, weightScale, actScale, actSum)
		return
	}
	if threads > cols {
		threads = cols
	}
	if threads < 1 {
		threads = 1
	}
	chunk := cols / threads
	if i2sI8SParallelChunkCols > 0 {
		chunk = i2sI8SParallelChunkCols
	}
	if chunk < 1 {
		chunk = 1
	}
	numParts := (cols + chunk - 1) / chunk
	if numParts <= 0 {
		return
	}
	partials := acquireI2SPartialBuf(rows * numParts)
	defer releaseI2SPartialBuf(partials)
	outs := make([][]float32, numParts)
	starts := make([]int, numParts)
	ends := make([]int, numParts)
	for i := 0; i < numParts; i++ {
		start := i * chunk
		end := start + chunk
		if end > cols {
			end = cols
		}
		starts[i] = start
		ends[i] = end
		base := i * rows
		outs[i] = partials[base : base+rows]
	}
	var wg sync.WaitGroup
	wg.Add(numParts)
	for i := 0; i < numParts; i++ {
		start := starts[i]
		end := ends[i]
		out := outs[i]
		go func(start, end int, out []float32) {
			defer wg.Done()
			if !matVecI2SI8SFastRange(out, packed, rows, cols, vec, weightScale, actScale, 0, start, end) {
				matVecI2SI8SRangeCols(out, packed, rows, cols, vec, weightScale, actScale, 0, start, end)
			}
		}(start, end, out)
	}
	wg.Wait()

	copy(dst, outs[0])
	for i := 1; i < numParts; i++ {
		p := outs[i]
		for r := 0; r < rows; r++ {
			dst[r] += p[r]
		}
	}
	if actSum != 0 {
		corr := float32(actSum) * (weightScale / actScale)
		for r := 0; r < rows; r++ {
			dst[r] -= corr
		}
	}
}

func matVecI2SI8SRangeCols(dst []float32, packed []byte, rows, cols int, vec []int8, weightScale, actScale float32, actSum int32, cStart, cEnd int) {
	if cStart < 0 {
		cStart = 0
	}
	if cEnd > cols {
		cEnd = cols
	}
	if cStart >= cEnd {
		return
	}
	outScale := weightScale / actScale
	corr := float32(actSum) * outScale
	if useI2SBlockPath(rows) {
		var block [128]int8
		var sums [128]int32
		for rb := 0; rb < rows; rb += 128 {
			for i := range sums {
				sums[i] = 0
			}
			for c := cStart; c < cEnd; c++ {
				idx := rb + rows*c
				bi := idx / 128
				decodeI2SBlock(block[:], packed[bi*32:bi*32+32])
				v := int32(vec[c])
				accumI2SBlock128(&sums, &block, v)
			}
			for i := 0; i < 128; i++ {
				dst[rb+i] = float32(sums[i])*outScale - corr
			}
		}
		return
	}
	for r := 0; r < rows; r++ {
		var sum int32
		for c := cStart; c < cEnd; c++ {
			idx := r + rows*c
			q := i2sPackedAt(packed, idx)
			sum += int32(q) * int32(vec[c])
		}
		dst[r] = float32(sum)*outScale - corr
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
	if useI2SBlockPath(rows) {
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
	outScale := weightScale / actScale
	corr := float32(actSum) * outScale
	if matVecTI2SI8SFast != nil && useI2SI8SFast(rows, cols) {
		if matVecThreads() > 1 && i2sI8SFastParallelColsMin > 0 &&
			cols >= i2sI8SFastParallelColsMin && matVecTI2SI8SFastRange != nil {
			matVecTI2SI8SParallel(dst, packed, rows, cols, vec, weightScale, actScale, actSum)
		} else {
			matVecTI2SI8SFast(dst, packed, rows, cols, vec, weightScale, actScale, actSum)
		}
		return
	}
	if matVecThreads() > 1 && cols >= i2sI8SParallelColsMin {
		matVecTI2SI8SParallel(dst, packed, rows, cols, vec, weightScale, actScale, actSum)
		return
	}
	var block [128]int8
	blockAligned := useI2SBlockPath(rows)
	for c := 0; c < cols; c++ {
		var sum int32
		r := 0
		if blockAligned && (rows*c)%128 == 0 {
			for ; r+127 < rows; r += 128 {
				idx0 := r + rows*c
				bi := idx0 / 128
				decodeI2SBlock(block[:], packed[bi*32:bi*32+32])
				i := 0
				for ; i+7 < 128; i += 8 {
					sum += int32(block[i]) * int32(vec[r+i])
					sum += int32(block[i+1]) * int32(vec[r+i+1])
					sum += int32(block[i+2]) * int32(vec[r+i+2])
					sum += int32(block[i+3]) * int32(vec[r+i+3])
					sum += int32(block[i+4]) * int32(vec[r+i+4])
					sum += int32(block[i+5]) * int32(vec[r+i+5])
					sum += int32(block[i+6]) * int32(vec[r+i+6])
					sum += int32(block[i+7]) * int32(vec[r+i+7])
				}
				for ; i < 128; i++ {
					sum += int32(block[i]) * int32(vec[r+i])
				}
			}
		}
		for ; r < rows; r++ {
			idx := r + rows*c
			q := i2sPackedAt(packed, idx)
			sum += int32(q) * int32(vec[r])
		}
		dst[c] = float32(sum)*outScale - corr
	}
}

func matVecI2SI8SParallel(dst []float32, packed []byte, rows, cols int, vec []int8, weightScale, actScale float32, actSum int32) {
	threads := matVecThreads()
	if threads <= 1 {
		return
	}
	if threads > rows/128 {
		threads = rows / 128
	}
	if threads < 1 {
		threads = 1
	}
	blockAligned := useI2SBlockPath(rows)
	chunk := rows / threads
	if i2sI8SParallelChunkRows > 0 {
		chunk = i2sI8SParallelChunkRows
	}
	if blockAligned {
		chunk = (chunk / 128) * 128
		if chunk == 0 {
			chunk = 128
		}
	}
	if chunk < 1 {
		chunk = 1
	}
	wg := acquireI2SI8SWG()
	for start := 0; start < rows; start += chunk {
		end := start + chunk
		if end > rows {
			end = rows
		}
		wg.Add(1)
		submitI2SI8STask(i2sI8STask{
			transposed: false,
			dst:        dst,
			packed:     packed,
			rows:       rows,
			cols:       cols,
			vec:        vec,
			weight:     weightScale,
			act:        actScale,
			actSum:     actSum,
			start:      start,
			end:        end,
			wg:         wg,
		})
	}
	wg.Wait()
	releaseI2SI8SWG(wg)
}

func matVecI2SI8SRange(dst []float32, packed []byte, rows, cols int, vec []int8, weightScale, actScale float32, actSum int32, rStart, rEnd int) {
	if rStart < 0 {
		rStart = 0
	}
	if rEnd > rows {
		rEnd = rows
	}
	if rStart >= rEnd {
		return
	}
	outScale := weightScale / actScale
	corr := float32(actSum) * outScale
	if useI2SBlockPath(rows) {
		var block [128]int8
		var sums [128]int32
		for rb := rStart; rb < rEnd; rb += 128 {
			for i := range sums {
				sums[i] = 0
			}
			for c := 0; c < cols; c++ {
				idx := rb + rows*c
				bi := idx / 128
				decodeI2SBlock(block[:], packed[bi*32:bi*32+32])
				v := int32(vec[c])
				accumI2SBlock128(&sums, &block, v)
			}
			for i := 0; i < 128; i++ {
				dst[rb+i] = float32(sums[i])*outScale - corr
			}
		}
		return
	}
	for r := rStart; r < rEnd; r++ {
		var sum int32
		for c := 0; c < cols; c++ {
			idx := r + rows*c
			q := i2sPackedAt(packed, idx)
			sum += int32(q) * int32(vec[c])
		}
		dst[r] = float32(sum)*outScale - corr
	}
}

func matVecTI2SI8SParallel(dst []float32, packed []byte, rows, cols int, vec []int8, weightScale, actScale float32, actSum int32) {
	threads := matVecThreads()
	if threads <= 1 {
		return
	}
	if threads > cols {
		threads = cols
	}
	if threads < 1 {
		threads = 1
	}
	chunk := cols / threads
	if i2sI8SParallelChunkCols > 0 {
		chunk = i2sI8SParallelChunkCols
	}
	if chunk < 1 {
		chunk = 1
	}
	wg := acquireI2SI8SWG()
	for start := 0; start < cols; start += chunk {
		end := start + chunk
		if end > cols {
			end = cols
		}
		wg.Add(1)
		submitI2SI8STask(i2sI8STask{
			transposed: true,
			dst:        dst,
			packed:     packed,
			rows:       rows,
			cols:       cols,
			vec:        vec,
			weight:     weightScale,
			act:        actScale,
			actSum:     actSum,
			start:      start,
			end:        end,
			wg:         wg,
		})
	}
	wg.Wait()
	releaseI2SI8SWG(wg)
}

func matVecTI2SI8SRange(dst []float32, packed []byte, rows, cols int, vec []int8, weightScale, actScale float32, actSum int32, cStart, cEnd int) {
	if cStart < 0 {
		cStart = 0
	}
	if cEnd > cols {
		cEnd = cols
	}
	if cStart >= cEnd {
		return
	}
	outScale := weightScale / actScale
	corr := float32(actSum) * outScale
	if matVecTI2SI8SFastRange != nil && useI2SI8SFast(rows, cEnd-cStart) &&
		matVecTI2SI8SFastRange(dst, packed, rows, cols, vec, weightScale, actScale, actSum, cStart, cEnd) {
		return
	}
	var block [128]int8
	blockAligned := useI2SBlockPath(rows)
	for c := cStart; c < cEnd; c++ {
		var sum int32
		r := 0
		if blockAligned && (rows*c)%128 == 0 {
			for ; r+127 < rows; r += 128 {
				idx0 := r + rows*c
				bi := idx0 / 128
				decodeI2SBlock(block[:], packed[bi*32:bi*32+32])
				i := 0
				for ; i+7 < 128; i += 8 {
					sum += int32(block[i]) * int32(vec[r+i])
					sum += int32(block[i+1]) * int32(vec[r+i+1])
					sum += int32(block[i+2]) * int32(vec[r+i+2])
					sum += int32(block[i+3]) * int32(vec[r+i+3])
					sum += int32(block[i+4]) * int32(vec[r+i+4])
					sum += int32(block[i+5]) * int32(vec[r+i+5])
					sum += int32(block[i+6]) * int32(vec[r+i+6])
					sum += int32(block[i+7]) * int32(vec[r+i+7])
				}
				for ; i < 128; i++ {
					sum += int32(block[i]) * int32(vec[r+i])
				}
			}
		}
		for ; r < rows; r++ {
			idx := r + rows*c
			q := i2sPackedAt(packed, idx)
			sum += int32(q) * int32(vec[r])
		}
		dst[c] = float32(sum)*outScale - corr
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
	blockAligned := useI2SBlockPath(rows)
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
