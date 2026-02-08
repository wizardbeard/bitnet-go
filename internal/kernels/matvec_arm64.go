//go:build arm64

package kernels

func init() {
	if parityStrict() {
		return
	}
	matVecImpl = matVecOpt
	matVecTImpl = matVecTOpt
	mulReluImpl = mulReluOpt
	rmsNormImpl = rmsNormOpt
}
