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
