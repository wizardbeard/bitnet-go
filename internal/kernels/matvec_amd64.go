//go:build amd64

package kernels

func init() {
	matVecImpl = matVecOpt
	matVecTImpl = matVecTOpt
	mulReluImpl = mulReluOpt
	rmsNormImpl = rmsNormOpt
}
