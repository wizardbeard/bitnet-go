//go:build amd64 && cgo

package kernels

/*
#cgo CFLAGS: -mavx2
void matvec_i2s_avx2(float *dst, const unsigned char *packed, int rows, int cols, const float *vec, float scale);
void matvec_t_i2s_avx2(float *dst, const unsigned char *packed, int rows, int cols, const float *vec, float scale);
*/
import "C"
import "unsafe"

func matVecI2SAVX2(dst []float32, packed []byte, rows, cols int, vec []float32, scale float32) {
	if rows <= 0 || cols <= 0 {
		return
	}
	if len(dst) < rows || len(vec) < cols {
		return
	}
	if rows*cols == 0 || len(packed) < i2sPackedLen(rows*cols) {
		return
	}
	C.matvec_i2s_avx2((*C.float)(unsafe.Pointer(&dst[0])), (*C.uchar)(unsafe.Pointer(&packed[0])), C.int(rows), C.int(cols), (*C.float)(unsafe.Pointer(&vec[0])), C.float(scale))
}

func matVecTI2SAVX2(dst []float32, packed []byte, rows, cols int, vec []float32, scale float32) {
	if rows <= 0 || cols <= 0 {
		return
	}
	if len(dst) < cols || len(vec) < rows {
		return
	}
	if rows*cols == 0 || len(packed) < i2sPackedLen(rows*cols) {
		return
	}
	C.matvec_t_i2s_avx2((*C.float)(unsafe.Pointer(&dst[0])), (*C.uchar)(unsafe.Pointer(&packed[0])), C.int(rows), C.int(cols), (*C.float)(unsafe.Pointer(&vec[0])), C.float(scale))
}
