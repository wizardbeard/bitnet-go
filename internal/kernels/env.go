package kernels

import (
	"os"
	"runtime"
	"strconv"
	"strings"
)

func envInt(name string, fallback int) int {
	raw := os.Getenv(name)
	if raw == "" {
		return fallback
	}
	v, err := strconv.Atoi(raw)
	if err != nil {
		return fallback
	}
	return v
}

func envIntArch(name string, fallback int) int {
	if runtime.GOARCH == "arm64" {
		suffix := name
		if strings.HasPrefix(name, "BITNET_") {
			suffix = strings.TrimPrefix(name, "BITNET_")
		}
		armName := "BITNET_ARM64_" + suffix
		raw := os.Getenv(armName)
		if raw != "" {
			v, err := strconv.Atoi(raw)
			if err == nil {
				return v
			}
		}
	}
	return envInt(name, fallback)
}
