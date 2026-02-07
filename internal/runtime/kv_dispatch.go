package runtime

var (
	storeCacheVectorImpl  = storeCacheVectorGeneric
	storeCacheVectorVImpl = storeCacheVectorVGeneric
)

func storeCacheVectorGeneric(cache []float32, pos int, vec []float32) {
	base := pos * len(vec)
	copy(cache[base:base+len(vec)], vec)
}

func storeCacheVectorVGeneric(cache []float32, pos int, vec []float32, kvHeads int) {
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
	for h := 0; h < kvHeads; h++ {
		baseHead := h * headDim
		for d := 0; d < headDim; d++ {
			cache[h*headDim*maxSeq+d*maxSeq+pos] = vec[baseHead+d]
		}
	}
}
