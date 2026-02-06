package kernels

var (
	matVecI2SImpl  = matVecI2SGeneric
	matVecTI2SImpl = matVecTI2SGeneric
)

// MatVecI2S computes dst = mat * vec where mat is GGML column-major [rows][cols]
// stored in packed i2_s format with 2-bit values and a global scale.
func MatVecI2S(dst []float32, packed []byte, rows, cols int, vec []float32, scale float32) {
	matVecI2SImpl(dst, packed, rows, cols, vec, scale)
}

// MatVecTI2S computes dst = transpose(mat) * vec where mat is GGML column-major [rows][cols]
// stored in packed i2_s format with 2-bit values and a global scale.
func MatVecTI2S(dst []float32, packed []byte, rows, cols int, vec []float32, scale float32) {
	matVecTI2SImpl(dst, packed, rows, cols, vec, scale)
}
