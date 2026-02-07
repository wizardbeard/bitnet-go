//go:build arm64 && cgo

package kernels

/*
#cgo CFLAGS: -O3
#include <stdint.h>

void matvec_i2s_cgo(float* dst, const uint8_t* packed, int rows, int cols, const float* vec, float scale);
void matvec_t_i2s_cgo(float* dst, const uint8_t* packed, int rows, int cols, const float* vec, float scale);
*/
import "C"
import "unsafe"

func init() {
	matVecI2SImpl = matVecI2SCgo
	matVecTI2SImpl = matVecTI2SCgo
}

func matVecI2SCgo(dst []float32, packed []byte, rows, cols int, vec []float32, scale float32) {
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
	C.matvec_i2s_cgo(
		(*C.float)(unsafe.Pointer(&dst[0])),
		(*C.uint8_t)(unsafe.Pointer(&packed[0])),
		C.int(rows),
		C.int(cols),
		(*C.float)(unsafe.Pointer(&vec[0])),
		C.float(scale),
	)
}

func matVecTI2SCgo(dst []float32, packed []byte, rows, cols int, vec []float32, scale float32) {
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
	C.matvec_t_i2s_cgo(
		(*C.float)(unsafe.Pointer(&dst[0])),
		(*C.uint8_t)(unsafe.Pointer(&packed[0])),
		C.int(rows),
		C.int(cols),
		(*C.float)(unsafe.Pointer(&vec[0])),
		C.float(scale),
	)
}
