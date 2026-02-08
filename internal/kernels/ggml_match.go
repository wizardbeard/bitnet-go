package kernels

import "os"

var matchGGMLFlag = os.Getenv("BITNET_MATCH_GGML") == "1"

func matchGGML() bool {
	return matchGGMLFlag
}
