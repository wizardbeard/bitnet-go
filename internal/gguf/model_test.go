package gguf

import (
	"bytes"
	"encoding/binary"
	"testing"
)

func TestDecodeModelInfo(t *testing.T) {
	buf := bytes.NewBuffer(nil)

	writeString(t, buf, "GGUF")
	writeU32(t, buf, 3) // version
	writeU64(t, buf, 4) // tensor count
	writeU64(t, buf, 6) // kv count

	// general.architecture = "llama"
	writeGGUFString(t, buf, "general.architecture")
	writeU32(t, buf, valueTypeString)
	writeGGUFString(t, buf, "llama")

	// llama.context_length = u32(128)
	writeGGUFString(t, buf, "llama.context_length")
	writeU32(t, buf, valueTypeUint32)
	writeU32(t, buf, 128)

	// tokenizer.ggml.bos_token_id = u32(1)
	writeGGUFString(t, buf, "tokenizer.ggml.bos_token_id")
	writeU32(t, buf, valueTypeUint32)
	writeU32(t, buf, 1)

	// tokenizer.ggml.tokens = array[str] (skip only)
	writeGGUFString(t, buf, "tokenizer.ggml.tokens")
	writeU32(t, buf, valueTypeArray)
	writeU32(t, buf, valueTypeString)
	writeU64(t, buf, 2)
	writeGGUFString(t, buf, "<s>")
	writeGGUFString(t, buf, "</s>")

	// general.quantization_version = u32(2)
	writeGGUFString(t, buf, "general.quantization_version")
	writeU32(t, buf, valueTypeUint32)
	writeU32(t, buf, 2)

	// general.file_type = u32(7)
	writeGGUFString(t, buf, "general.file_type")
	writeU32(t, buf, valueTypeUint32)
	writeU32(t, buf, 7)

	for i := 0; i < 4; i++ {
		writeGGUFString(t, buf, "tensor."+string(rune('a'+i)))
		writeU32(t, buf, 2)                // n_dims
		writeU64(t, buf, uint64(64+i*16))  // dim0
		writeU64(t, buf, uint64(128+i*16)) // dim1
		writeU32(t, buf, 0)                // type
		writeU64(t, buf, uint64(i*1024))   // offset
	}

	info, err := DecodeModelInfo(buf)
	if err != nil {
		t.Fatalf("DecodeModelInfo() error = %v", err)
	}

	if info.Version != 3 {
		t.Fatalf("Version = %d, want 3", info.Version)
	}
	if info.TensorCount != 4 {
		t.Fatalf("TensorCount = %d, want 4", info.TensorCount)
	}
	if info.KVCount != 6 {
		t.Fatalf("KVCount = %d, want 6", info.KVCount)
	}
	if got := info.KeyValues["general.architecture"]; got != "llama" {
		t.Fatalf("general.architecture = %v, want llama", got)
	}
	if got := info.KeyValues["llama.context_length"]; got != uint32(128) {
		t.Fatalf("llama.context_length = %v, want 128", got)
	}
	if got := info.KeyValues["tokenizer.ggml.tokens.count"]; got != uint64(2) {
		t.Fatalf("tokenizer.ggml.tokens.count = %v, want 2", got)
	}
	if got := info.KeyValues["tokenizer.ggml.tokens"]; got == nil {
		t.Fatalf("tokenizer.ggml.tokens should be captured for tokenizer plumbing")
	}
	if len(info.Tensors) != 4 {
		t.Fatalf("len(Tensors) = %d, want 4", len(info.Tensors))
	}
	if info.Tensors[0].Name != "tensor.a" {
		t.Fatalf("Tensors[0].Name = %q, want tensor.a", info.Tensors[0].Name)
	}
	if len(info.Tensors[0].Dimensions) != 2 || info.Tensors[0].Dimensions[0] != 64 || info.Tensors[0].Dimensions[1] != 128 {
		t.Fatalf("Tensors[0].Dimensions = %v, want [64 128]", info.Tensors[0].Dimensions)
	}
}

func TestDecodeModelInfoUnsupportedType(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	writeString(t, buf, "GGUF")
	writeU32(t, buf, 3)
	writeU64(t, buf, 0)
	writeU64(t, buf, 1)

	writeGGUFString(t, buf, "bad.type")
	writeU32(t, buf, 999)

	_, err := DecodeModelInfo(buf)
	if err == nil {
		t.Fatal("expected error")
	}
}

func writeGGUFString(t *testing.T, buf *bytes.Buffer, s string) {
	t.Helper()
	writeU64(t, buf, uint64(len(s)))
	writeString(t, buf, s)
}

func writeString(t *testing.T, buf *bytes.Buffer, s string) {
	t.Helper()
	if _, err := buf.WriteString(s); err != nil {
		t.Fatalf("WriteString() error = %v", err)
	}
}

func writeU32(t *testing.T, buf *bytes.Buffer, v uint32) {
	t.Helper()
	if err := binary.Write(buf, binary.LittleEndian, v); err != nil {
		t.Fatalf("binary.Write(u32) error = %v", err)
	}
}

func writeU64(t *testing.T, buf *bytes.Buffer, v uint64) {
	t.Helper()
	if err := binary.Write(buf, binary.LittleEndian, v); err != nil {
		t.Fatalf("binary.Write(u64) error = %v", err)
	}
}
