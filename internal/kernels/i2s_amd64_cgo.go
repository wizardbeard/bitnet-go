//go:build amd64 && cgo

package kernels

import "os"

func init() {
	if os.Getenv("BITNET_FORCE_AVX2") == "1" {
		matVecI2SImpl = matVecI2SAVX2
		matVecTI2SImpl = matVecTI2SAVX2
	}
}
