//go:build amd64

package kernels

func init() {
	matVecI2SImpl = matVecI2SAMD64
	matVecTI2SImpl = matVecTI2SAMD64
}

func matVecI2SAMD64(dst []float32, packed []byte, rows, cols int, vec []float32, scale float32) {
	if rows <= 0 || cols <= 0 {
		return
	}
	if len(dst) < rows || len(vec) < cols {
		return
	}
	if rows*cols == 0 || len(packed) < i2sPackedLen(rows*cols) {
		return
	}
	if rows%128 != 0 {
		matVecI2SGeneric(dst, packed, rows, cols, vec, scale)
		return
	}

	for i := 0; i < rows; i++ {
		dst[i] = 0
	}

	lut := [4]float32{-1, 0, 1, 0}
	blocks := rows / 128
	for c := 0; c < cols; c++ {
		v := vec[c] * scale
		if v == 0 {
			continue
		}
		basePacked := (c * rows / 128) * 32
		rowBase := 0
		for b := 0; b < blocks; b++ {
			p := packed[basePacked : basePacked+32]
			for gp := 0; gp < 32; gp++ {
				val := p[gp]
				r0 := rowBase + gp
				r1 := rowBase + 32 + gp
				r2 := rowBase + 64 + gp
				r3 := rowBase + 96 + gp
				dst[r0] += lut[val>>6] * v
				dst[r1] += lut[(val>>4)&0x3] * v
				dst[r2] += lut[(val>>2)&0x3] * v
				dst[r3] += lut[val&0x3] * v
			}
			basePacked += 32
			rowBase += 128
		}
	}
}

func matVecTI2SAMD64(dst []float32, packed []byte, rows, cols int, vec []float32, scale float32) {
	if rows <= 0 || cols <= 0 {
		return
	}
	if len(dst) < cols || len(vec) < rows {
		return
	}
	if rows*cols == 0 || len(packed) < i2sPackedLen(rows*cols) {
		return
	}
	if rows%128 != 0 {
		matVecTI2SGeneric(dst, packed, rows, cols, vec, scale)
		return
	}

	lut := [4]float32{-1, 0, 1, 0}
	blocks := rows / 128
	for c := 0; c < cols; c++ {
		var sum float32
		basePacked := (c * rows / 128) * 32
		rowBase := 0
		for b := 0; b < blocks; b++ {
			p := packed[basePacked : basePacked+32]
			for gp := 0; gp < 32; gp++ {
				val := p[gp]
				r0 := rowBase + gp
				r1 := rowBase + 32 + gp
				r2 := rowBase + 64 + gp
				r3 := rowBase + 96 + gp
				sum += lut[val>>6]*vec[r0] +
					lut[(val>>4)&0x3]*vec[r1] +
					lut[(val>>2)&0x3]*vec[r2] +
					lut[val&0x3]*vec[r3]
			}
			basePacked += 32
			rowBase += 128
		}
		dst[c] = sum * scale
	}
}
