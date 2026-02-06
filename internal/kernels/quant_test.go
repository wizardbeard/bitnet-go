package kernels

import "testing"

func TestQuantizeRowI8S(t *testing.T) {
	src := []float32{0, 1, -1, 0.25}
	dst := make([]int8, len(src))
	scale, sum := QuantizeRowI8S(dst, src)

	if scale == 0 {
		t.Fatalf("scale = %v, want non-zero", scale)
	}
	wantScale := float32(1.0 / 127.0)
	if diff := scale - wantScale; diff < -1e-7 || diff > 1e-7 {
		t.Fatalf("scale = %v, want %v", scale, wantScale)
	}
	want := []int8{0, 127, -127, 32}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("dst[%d] = %d, want %d", i, dst[i], want[i])
		}
	}
	if sum != 32 {
		t.Fatalf("sum = %d, want 32", sum)
	}
}

func TestQuantizeRowI8SZero(t *testing.T) {
	src := []float32{0, 0, 0}
	dst := make([]int8, len(src))
	scale, sum := QuantizeRowI8S(dst, src)
	if scale != 0 {
		t.Fatalf("scale = %v, want 0", scale)
	}
	if sum != 0 {
		t.Fatalf("sum = %d, want 0", sum)
	}
	for i := range dst {
		if dst[i] != 0 {
			t.Fatalf("dst[%d] = %d, want 0", i, dst[i])
		}
	}
}

func TestMatVecI2SI8S(t *testing.T) {
	rows, cols := 2, 3
	vals := []int{1, -1, 0, 1, 1, 0}
	packed := make([]byte, (rows*cols+3)/4)
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
		shift := uint(6 - 2*(i%4))
		packed[i/4] |= q << shift
	}
	vec := []int8{10, -5, 2}
	dst := make([]float32, rows)
	MatVecI2SI8S(dst, packed, rows, cols, vec, 2.0, 0.1)

	if dst[0] != 2.4 {
		t.Fatalf("dst[0] = %f, want 2.4", dst[0])
	}
	if dst[1] != -3.0 {
		t.Fatalf("dst[1] = %f, want -3", dst[1])
	}
}

func TestMatVecTI2SI8S(t *testing.T) {
	rows, cols := 2, 3
	vals := []int{1, -1, 0, 1, 1, 0}
	packed := make([]byte, (rows*cols+3)/4)
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
		shift := uint(6 - 2*(i%4))
		packed[i/4] |= q << shift
	}
	vec := []int8{10, -5}
	dst := make([]float32, cols)
	MatVecTI2SI8S(dst, packed, rows, cols, vec, 2.0, 0.1)

	if dst[0] != 3.0 {
		t.Fatalf("dst[0] = %f, want 3", dst[0])
	}
	if dst[1] != -1.0 {
		t.Fatalf("dst[1] = %f, want -1", dst[1])
	}
	if dst[2] != 2.0 {
		t.Fatalf("dst[2] = %f, want 2", dst[2])
	}
}
