//go:build unix

package runtime

import (
	"os"
	"syscall"
)

func mmapReadOnly(f *os.File) ([]byte, error) {
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	size := int(st.Size())
	if size <= 0 {
		return nil, nil
	}
	return syscall.Mmap(int(f.Fd()), 0, size, syscall.PROT_READ, syscall.MAP_SHARED)
}
