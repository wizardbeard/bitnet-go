//go:build amd64 && cgo

package kernels

import "testing"

func TestMatVecI2SAVX2MatchesGeneric(t *testing.T) {
	rows, cols := 5, 7
	vals := make([]int8, rows*cols)
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
	intVals := make([]int, len(vals))
	for i, v := range vals {
		intVals[i] = int(v)
	}
	packed := packI2S(intVals)
	vec := make([]float32, cols)
	for i := range vec {
		vec[i] = float32(i%5) * 0.1
	}
	want := make([]float32, rows)
	got := make([]float32, rows)
	matVecI2SGeneric(want, packed, rows, cols, vec, 1.0)
	matVecI2SAVX2(got, packed, rows, cols, vec, 1.0)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("i2s avx2 mismatch at %d: got=%f want=%f", i, got[i], want[i])
		}
	}
}

func TestMatVecTI2SAVX2MatchesGeneric(t *testing.T) {
	rows, cols := 5, 7
	vals := make([]int8, rows*cols)
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
	intVals := make([]int, len(vals))
	for i, v := range vals {
		intVals[i] = int(v)
	}
	packed := packI2S(intVals)
	vec := make([]float32, rows)
	for i := range vec {
		vec[i] = float32(i%5) * 0.1
	}
	want := make([]float32, cols)
	got := make([]float32, cols)
	matVecTI2SGeneric(want, packed, rows, cols, vec, 1.0)
	matVecTI2SAVX2(got, packed, rows, cols, vec, 1.0)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("i2s avx2 T mismatch at %d: got=%f want=%f", i, got[i], want[i])
		}
	}
}
