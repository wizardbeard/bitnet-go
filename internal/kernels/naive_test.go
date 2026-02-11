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

func TestMulRelu2IntoMatchesOpt(t *testing.T) {
	dstA := make([]float32, 8)
	dstB := make([]float32, 8)
	gate := make([]float32, 8)
	up := make([]float32, 8)
	for i := range gate {
		gate[i] = float32((i%7)-3) * 0.2
		up[i] = float32(i%5) * 0.3
	}
	mulReluGeneric(dstA, gate, up)
	mulReluOpt(dstB, gate, up)
	for i := range dstA {
		diff := dstA[i] - dstB[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > 1e-6 {
			t.Fatalf("mismatch at %d: got=%f want=%f", i, dstB[i], dstA[i])
		}
	}
}

func TestMulSiluInto(t *testing.T) {
	dst := make([]float32, 3)
	gate := []float32{-2, 0, 2}
	up := []float32{3, 4, 5}
	MulSiluInto(dst, gate, up)

	if !(dst[0] < 0 && dst[2] > 0) {
		t.Fatalf("unexpected signs for silu output: %v", dst)
	}
	if dst[1] != 0 {
		t.Fatalf("dst[1] = %f, want 0", dst[1])
	}
}

func TestRMSNormInto(t *testing.T) {
	dst := make([]float32, 2)
	x := []float32{3, 4}
	w := []float32{1, 2}
	RMSNormInto(dst, x, w, 1e-6)
	if dst[0] <= 0 || dst[1] <= 0 {
		t.Fatalf("unexpected RMSNorm output: %v", dst)
	}
	if dst[1] <= dst[0] {
		t.Fatalf("expected weighted output to scale second element: %v", dst)
	}
}

func TestRMSNormIntoMatchesOpt(t *testing.T) {
	dstA := make([]float32, 16)
	dstB := make([]float32, 16)
	x := make([]float32, 16)
	w := make([]float32, 16)
	for i := range x {
		x[i] = float32((i%7)-3) * 0.1
		w[i] = 1.0 + float32(i%5)*0.01
	}
	rmsNormGeneric(dstA, x, w, 1e-5)
	rmsNormOpt(dstB, x, w, 1e-5)
	for i := range dstA {
		diff := dstA[i] - dstB[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > 1e-6 {
			t.Fatalf("mismatch at %d: got=%f want=%f", i, dstB[i], dstA[i])
		}
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

func TestMatVecMatchesOpt(t *testing.T) {
	mat := make([]float32, 6)
	// matrix:
	// [1 2 3]
	// [4 5 6]
	mat[0] = 1
	mat[1] = 4
	mat[2] = 2
	mat[3] = 5
	mat[4] = 3
	mat[5] = 6
	vec := []float32{2, -1, 0.5}
	dstA := make([]float32, 2)
	dstB := make([]float32, 2)
	matVecGeneric(dstA, mat, 2, 3, vec)
	matVecOpt(dstB, mat, 2, 3, vec)
	for i := range dstA {
		diff := dstA[i] - dstB[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > 1e-6 {
			t.Fatalf("mismatch at %d: got=%f want=%f", i, dstB[i], dstA[i])
		}
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

func TestMatVecTMatchesOpt(t *testing.T) {
	mat := make([]float32, 6)
	mat[0] = 1
	mat[1] = 4
	mat[2] = 2
	mat[3] = 5
	mat[4] = 3
	mat[5] = 6
	vec := []float32{2, -1}
	dstA := make([]float32, 3)
	dstB := make([]float32, 3)
	matVecTGeneric(dstA, mat, 2, 3, vec)
	matVecTOpt(dstB, mat, 2, 3, vec)
	for i := range dstA {
		diff := dstA[i] - dstB[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > 1e-6 {
			t.Fatalf("mismatch at %d: got=%f want=%f", i, dstB[i], dstA[i])
		}
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

func TestMatVecI2SMatchesGeneric(t *testing.T) {
	rows, cols := 4, 4
	vals := []int{1, -1, 0, 1, 1, 0, -1, 1, 0, 1, -1, 0, 1, 0, 1, -1}
	packed := packI2S(vals)
	vec := []float32{0.2, -0.3, 0.4, -0.5}
	dstA := make([]float32, rows)
	dstB := make([]float32, rows)
	matVecI2SGeneric(dstA, packed, rows, cols, vec, 1.0)
	MatVecI2S(dstB, packed, rows, cols, vec, 1.0)
	for i := range dstA {
		diff := dstA[i] - dstB[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > 1e-6 {
			t.Fatalf("mismatch at %d: got=%f want=%f", i, dstB[i], dstA[i])
		}
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

func TestMatVecTI2SMatchesGeneric(t *testing.T) {
	rows, cols := 4, 4
	vals := []int{1, -1, 0, 1, 1, 0, -1, 1, 0, 1, -1, 0, 1, 0, 1, -1}
	packed := packI2S(vals)
	vec := []float32{0.2, -0.3, 0.4, -0.5}
	dstA := make([]float32, cols)
	dstB := make([]float32, cols)
	matVecTI2SGeneric(dstA, packed, rows, cols, vec, 1.0)
	MatVecTI2S(dstB, packed, rows, cols, vec, 1.0)
	for i := range dstA {
		diff := dstA[i] - dstB[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > 1e-6 {
			t.Fatalf("mismatch at %d: got=%f want=%f", i, dstB[i], dstA[i])
		}
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
