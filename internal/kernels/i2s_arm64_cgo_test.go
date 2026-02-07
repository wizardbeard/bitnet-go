//go:build arm64 && cgo

package kernels

import "testing"

func TestMatVecI2SArm64CgoSmoke(t *testing.T) {
	rows, cols := 128, 2
	vals := make([]int, rows*cols)
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
	packed := make([]byte, i2sPackedLen(rows*cols))
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
		blk := i / 128
		off := i % 128
		gp := off % 32
		group := off / 32
		shift := uint(6 - 2*group)
		packed[blk*32+gp] |= q << shift
	}
	vec := make([]float32, cols)
	for i := range vec {
		vec[i] = float32(i+1) * 0.25
	}
	dst := make([]float32, rows)
	MatVecI2S(dst, packed, rows, cols, vec, 1.0)
	for i := range dst {
		if dst[i] == 0 {
			t.Fatalf("dst[%d] unexpectedly zero", i)
		}
	}
}
