package kernels

func mulReluOpt(dst, gate, up []float32) {
	n := len(dst)
	if len(gate) < n {
		n = len(gate)
	}
	if len(up) < n {
		n = len(up)
	}
	i := 0
	for ; i+3 < n; i += 4 {
		g0 := gate[i]
		if g0 < 0 {
			g0 = 0
		}
		g0 = g0 * g0
		g1 := gate[i+1]
		if g1 < 0 {
			g1 = 0
		}
		g1 = g1 * g1
		g2 := gate[i+2]
		if g2 < 0 {
			g2 = 0
		}
		g2 = g2 * g2
		g3 := gate[i+3]
		if g3 < 0 {
			g3 = 0
		}
		g3 = g3 * g3
		dst[i] = g0 * up[i]
		dst[i+1] = g1 * up[i+1]
		dst[i+2] = g2 * up[i+2]
		dst[i+3] = g3 * up[i+3]
	}
	for ; i < n; i++ {
		g := gate[i]
		if g < 0 {
			g = 0
		}
		g = g * g
		dst[i] = g * up[i]
	}
}
