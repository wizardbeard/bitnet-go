package kernels

var mulReluImpl = mulReluGeneric

func mulReluGeneric(dst, gate, up []float32) {
	n := len(dst)
	if len(gate) < n {
		n = len(gate)
	}
	if len(up) < n {
		n = len(up)
	}
	for i := 0; i < n; i++ {
		g := gate[i]
		if g < 0 {
			g = 0
		}
		g = g * g
		dst[i] = g * up[i]
	}
}
