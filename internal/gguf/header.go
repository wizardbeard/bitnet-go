package gguf

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
)

var ErrInvalidMagic = errors.New("invalid gguf magic")

type Header struct {
	Version     uint32
	TensorCount uint64
	KVCount     uint64
}

func ReadHeader(path string) (Header, error) {
	f, err := os.Open(path)
	if err != nil {
		return Header{}, err
	}
	defer f.Close()

	return DecodeHeader(f)
}

func DecodeHeader(r io.Reader) (Header, error) {
	var magic [4]byte
	if _, err := io.ReadFull(r, magic[:]); err != nil {
		return Header{}, fmt.Errorf("read magic: %w", err)
	}
	if string(magic[:]) != "GGUF" {
		return Header{}, ErrInvalidMagic
	}

	var h Header
	if err := binary.Read(r, binary.LittleEndian, &h.Version); err != nil {
		return Header{}, fmt.Errorf("read version: %w", err)
	}
	if err := binary.Read(r, binary.LittleEndian, &h.TensorCount); err != nil {
		return Header{}, fmt.Errorf("read tensor count: %w", err)
	}
	if err := binary.Read(r, binary.LittleEndian, &h.KVCount); err != nil {
		return Header{}, fmt.Errorf("read kv count: %w", err)
	}

	return h, nil
}
