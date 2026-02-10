//go:build !unix

package runtime

import (
	"errors"
	"os"
)

func mmapReadOnly(_ *os.File) ([]byte, error) {
	return nil, errors.New("mmap not supported on this platform")
}
