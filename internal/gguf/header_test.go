package gguf

import (
	"bytes"
	"testing"
)

func TestDecodeHeader(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	buf.WriteString("GGUF")
	buf.Write([]byte{0x03, 0x00, 0x00, 0x00})
	buf.Write([]byte{0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00})
	buf.Write([]byte{0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00})

	h, err := DecodeHeader(buf)
	if err != nil {
		t.Fatalf("DecodeHeader() error = %v", err)
	}
	if h.Version != 3 {
		t.Fatalf("Version = %d, want 3", h.Version)
	}
	if h.TensorCount != 2 {
		t.Fatalf("TensorCount = %d, want 2", h.TensorCount)
	}
	if h.KVCount != 5 {
		t.Fatalf("KVCount = %d, want 5", h.KVCount)
	}
}

func TestDecodeHeaderInvalidMagic(t *testing.T) {
	_, err := DecodeHeader(bytes.NewReader([]byte("BAD!")))
	if err == nil {
		t.Fatal("expected error")
	}
	if err != ErrInvalidMagic {
		t.Fatalf("err = %v, want %v", err, ErrInvalidMagic)
	}
}
