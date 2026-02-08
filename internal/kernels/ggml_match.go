package kernels

import "os"

var matchGGMLFlag = os.Getenv("BITNET_MATCH_GGML") == "1" || os.Getenv("BITNET_PARITY_STRICT") == "1"
var parityStrictFlag = os.Getenv("BITNET_PARITY_STRICT") == "1"
var fastColMatVecFlag = os.Getenv("BITNET_FAST_COL_MATVEC") == "1"
var fastColMatVecAutoFlag = os.Getenv("BITNET_FAST_COL_MATVEC_AUTO") != "0"

func matchGGML() bool {
	return matchGGMLFlag
}

func parityStrict() bool {
	return parityStrictFlag
}

func fastColMatVec() bool {
	if fastColMatVecFlag {
		return true
	}
	return fastColMatVecAutoFlag && !parityStrictFlag
}
