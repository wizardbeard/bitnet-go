//go:build amd64 && cgo

package kernels

/*
#cgo CFLAGS: -mavx2
#include <immintrin.h>
int bitnet_has_avx2() {
#if defined(__GNUC__)
    return __builtin_cpu_supports("avx2");
#else
    return 0;
#endif
}
void matvec_t_i2s_i8s_avx2(float *dst, const unsigned char *packed, int rows, int cols, const signed char *vec, float weight_scale, float act_scale, int act_sum);
void matvec_i2s_i8s_avx2(float *dst, const unsigned char *packed, int rows, int cols, const signed char *vec, float weight_scale, float act_scale, int act_sum);
*/
import "C"
import (
	"os"
	"unsafe"
)

func init() {
	if os.Getenv("BITNET_I2S_I8S_DISABLE_FAST") == "1" {
		return
	}
	if os.Getenv("BITNET_FORCE_AVX2") == "1" || C.bitnet_has_avx2() != 0 {
		matVecI2SI8SFast = matVecI2SI8SAVX2
		matVecTI2SI8SFast = matVecTI2SI8SAVX2
		matVecTI2SI8SFastRange = matVecTI2SI8SAVX2Range
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

func matVecI2SI8SAVX2(dst []float32, packed []byte, rows, cols int, vec []int8, weightScale, actScale float32, actSum int32) {
	if rows <= 0 || cols <= 0 {
		return
	}
	if len(dst) < rows || len(vec) < cols {
		return
	}
	if rows*cols == 0 || len(packed) < i2sPackedLen(rows*cols) {
		return
	}
	C.matvec_i2s_i8s_avx2(
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

func matVecTI2SI8SAVX2Range(dst []float32, packed []byte, rows, cols int, vec []int8, weightScale, actScale float32, actSum int32, cStart, cEnd int) bool {
	if rows <= 0 || cols <= 0 || cStart < 0 || cEnd > cols || cStart >= cEnd {
		return false
	}
	if len(dst) < cols || len(vec) < rows {
		return false
	}
	if rows*cols == 0 || len(packed) < i2sPackedLen(rows*cols) {
		return false
	}
	// The AVX2 kernel expects the packed matrix pointer to be column-aligned.
	// This holds for column chunks when rows are multiples of 128.
	if rows%128 != 0 {
		return false
	}
	elemOffset := rows * cStart
	if elemOffset%128 != 0 {
		return false
	}
	byteOffset := (elemOffset / 128) * 32
	if byteOffset < 0 || byteOffset >= len(packed) {
		return false
	}
	chunkCols := cEnd - cStart
	C.matvec_t_i2s_i8s_avx2(
		(*C.float)(unsafe.Pointer(&dst[cStart])),
		(*C.uchar)(unsafe.Pointer(&packed[byteOffset])),
		C.int(rows),
		C.int(chunkCols),
		(*C.schar)(unsafe.Pointer(&vec[0])),
		C.float(weightScale),
		C.float(actScale),
		C.int(actSum),
	)
	return true
}
