package runtime

func storeCacheVectorOpt(cache []float32, pos int, vec []float32) {
	base := pos * len(vec)
	copy(cache[base:base+len(vec)], vec)
}

func storeCacheVectorVOpt(cache []float32, pos int, vec []float32, kvHeads int) {
	if kvHeads <= 0 || len(vec) == 0 {
		return
	}
	if len(vec)%kvHeads != 0 {
		kvHeads = 1
	}
	headDim := len(vec) / kvHeads
	if headDim == 0 {
		return
	}
	maxSeq := len(cache) / len(vec)
	if maxSeq <= 0 || pos < 0 || pos >= maxSeq {
		return
	}
	basePos := pos
	stride := maxSeq
	for h := 0; h < kvHeads; h++ {
		src := vec[h*headDim : (h+1)*headDim]
		dstBase := h * headDim * maxSeq
		for d := 0; d < headDim; d++ {
			cache[dstBase+d*stride+basePos] = src[d]
		}
	}
}
