//go:build amd64 && cgo

package kernels

/*
#cgo CFLAGS: -mavx2
void matvec_t_i2s_i8s_avx2(float *dst, const unsigned char *packed, int rows, int cols, const signed char *vec, float weight_scale, float act_scale, int act_sum);
*/
import "C"
import (
	"os"
	"unsafe"
)

func init() {
	if os.Getenv("BITNET_FORCE_AVX2") == "1" {
		matVecTI2SI8SFast = matVecTI2SI8SAVX2
	}
}

func matVecTI2SI8SAVX2(dst []float32, packed []byte, rows, cols int, vec []int8, weightScale, actScale float32, actSum int32) {
	if rows <= 0 || cols <= 0 {
		return
	}
	if len(dst) < cols || len(vec) < rows {
		return
	}
	if rows*cols == 0 || len(packed) < i2sPackedLen(rows*cols) {
		return
	}
	C.matvec_t_i2s_i8s_avx2(
		(*C.float)(unsafe.Pointer(&dst[0])),
		(*C.uchar)(unsafe.Pointer(&packed[0])),
		C.int(rows),
		C.int(cols),
		(*C.schar)(unsafe.Pointer(&vec[0])),
		C.float(weightScale),
		C.float(actScale),
		C.int(actSum),
	)
}
