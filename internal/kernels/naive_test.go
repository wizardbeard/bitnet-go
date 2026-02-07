package kernels

import "testing"

func TestDot(t *testing.T) {
	got := Dot([]float32{1, 2, 3}, []float32{4, 5, 6})
	if got != 32 {
		t.Fatalf("Dot = %v, want 32", got)
	}
}

func TestAddScaled(t *testing.T) {
	dst := []float32{1, 2, 3}
	AddScaled(dst, []float32{2, 1, -1}, 0.5)
	want := []float32{2, 2.5, 2.5}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("dst[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestArgmax(t *testing.T) {
	if got := Argmax([]float32{-2, 5, 3}); got != 1 {
		t.Fatalf("Argmax = %d, want 1", got)
	}
	if got := Argmax(nil); got != -1 {
		t.Fatalf("Argmax(nil) = %d, want -1", got)
	}
}

func TestMulRelu2Into(t *testing.T) {
	dst := make([]float32, 2)
	gate := []float32{-1, 2}
	up := []float32{3, 4}
	MulRelu2Into(dst, gate, up)
	if dst[0] != 0 {
		t.Fatalf("dst[0] = %f, want 0", dst[0])
	}
	if dst[1] != 16 {
		t.Fatalf("dst[1] = %f, want 16", dst[1])
	}
}

func TestMatVec(t *testing.T) {
	// matrix:
	// [1 2 3]
	// [4 5 6]
	// stored in GGML column-major (ne0=rows).
	mat := []float32{
		1, 4,
		2, 5,
		3, 6,
	}
	vec := []float32{2, -1, 0.5}
	dst := make([]float32, 2)
	MatVec(dst, mat, 2, 3, vec)

	if dst[0] != 1.5 {
		t.Fatalf("dst[0] = %f, want 1.5", dst[0])
	}
	if dst[1] != 6.0 {
		t.Fatalf("dst[1] = %f, want 6", dst[1])
	}
}

func TestMatVecT(t *testing.T) {
	// matrix:
	// [1 2 3]
	// [4 5 6]
	// stored in GGML column-major (ne0=rows).
	mat := []float32{
		1, 4,
		2, 5,
		3, 6,
	}
	vec := []float32{2, -1}
	dst := make([]float32, 3)
	MatVecT(dst, mat, 2, 3, vec)

	if dst[0] != -2 {
		t.Fatalf("dst[0] = %f, want -2", dst[0])
	}
	if dst[1] != -1 {
		t.Fatalf("dst[1] = %f, want -1", dst[1])
	}
	if dst[2] != 0 {
		t.Fatalf("dst[2] = %f, want 0", dst[2])
	}
}

func TestMatVecI2S(t *testing.T) {
	// matrix:
	// [1 0  1]
	// [-1 1 0]
	// stored in GGML column-major (ne0=rows).
	rows, cols := 2, 3
	vals := []int{1, -1, 0, 1, 1, 0}
	packed := packI2S(vals)
	vec := []float32{2, -1, 0.5}
	dst := make([]float32, rows)
	MatVecI2S(dst, packed, rows, cols, vec, 1.0)

	if dst[0] != 2.5 {
		t.Fatalf("dst[0] = %f, want 2.5", dst[0])
	}
	if dst[1] != -3.0 {
		t.Fatalf("dst[1] = %f, want -3", dst[1])
	}
}

func TestMatVecTI2S(t *testing.T) {
	rows, cols := 2, 3
	vals := []int{1, -1, 0, 1, 1, 0}
	packed := packI2S(vals)
	vec := []float32{2, -1}
	dst := make([]float32, cols)
	MatVecTI2S(dst, packed, rows, cols, vec, 1.0)

	if dst[0] != 3 {
		t.Fatalf("dst[0] = %f, want 3", dst[0])
	}
	if dst[1] != -1 {
		t.Fatalf("dst[1] = %f, want -1", dst[1])
	}
	if dst[2] != 2 {
		t.Fatalf("dst[2] = %f, want 2", dst[2])
	}
}

func packI2S(vals []int) []byte {
	const block = 128
	const blockBytes = 32
	n := len(vals)
	packed := make([]byte, (n+block-1)/block*blockBytes)
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
		blk := i / block
		off := i % block
		gp := off % 32
		group := off / 32
		shift := uint(6 - 2*group)
		packed[blk*blockBytes+gp] |= q << shift
	}
	return packed
}
