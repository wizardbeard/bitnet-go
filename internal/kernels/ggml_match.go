package kernels

import "os"

var matchGGMLFlag = os.Getenv("BITNET_MATCH_GGML") == "1" || os.Getenv("BITNET_PARITY_STRICT") == "1"
var parityStrictFlag = os.Getenv("BITNET_PARITY_STRICT") == "1"

func matchGGML() bool {
	return matchGGMLFlag
}

func parityStrict() bool {
	return parityStrictFlag
}
