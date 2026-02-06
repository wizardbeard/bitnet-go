package gguf

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestDecodeModelInfoTensorDataOffsetUsesAlignment(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	writeString(t, buf, "GGUF")
	writeU32(t, buf, 3)
	writeU64(t, buf, 1) // tensor count
	writeU64(t, buf, 1) // kv count

	writeGGUFString(t, buf, "general.alignment")
	writeU32(t, buf, valueTypeUint32)
	writeU32(t, buf, 64)

	writeGGUFString(t, buf, "w")
	writeU32(t, buf, 2)
	writeU64(t, buf, 2)
	writeU64(t, buf, 2)
	writeU32(t, buf, GGMLTypeF32)
	writeU64(t, buf, 0)

	info, err := DecodeModelInfo(buf)
	if err != nil {
		t.Fatalf("DecodeModelInfo() error = %v", err)
	}
	if info.Alignment != 64 {
		t.Fatalf("Alignment = %d, want 64", info.Alignment)
	}
	if info.TensorDataOffset%64 != 0 {
		t.Fatalf("TensorDataOffset = %d, want 64-byte alignment", info.TensorDataOffset)
	}
}

func TestReadTensorF32(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	writeString(t, buf, "GGUF")
	writeU32(t, buf, 3)
	writeU64(t, buf, 1) // tensor count
	writeU64(t, buf, 1) // kv count

	writeGGUFString(t, buf, "general.alignment")
	writeU32(t, buf, valueTypeUint32)
	writeU32(t, buf, 32)

	writeGGUFString(t, buf, "w")
	writeU32(t, buf, 2)
	writeU64(t, buf, 2)
	writeU64(t, buf, 2)
	writeU32(t, buf, GGMLTypeF32)
	writeU64(t, buf, 0)

	padTo(t, buf, 32)
	writeF32(t, buf, 1.5)
	writeF32(t, buf, -2.0)
	writeF32(t, buf, 0.25)
	writeF32(t, buf, 3.75)

	path := filepath.Join(t.TempDir(), "tensor.gguf")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	info, err := ReadModelInfo(path)
	if err != nil {
		t.Fatalf("ReadModelInfo() error = %v", err)
	}
	got, err := ReadTensorF32(path, info, "w")
	if err != nil {
		t.Fatalf("ReadTensorF32() error = %v", err)
	}

	want := []float32{1.5, -2.0, 0.25, 3.75}
	if len(got) != len(want) {
		t.Fatalf("len(got) = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestReadTensorF16(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	writeString(t, buf, "GGUF")
	writeU32(t, buf, 3)
	writeU64(t, buf, 1) // tensor count
	writeU64(t, buf, 1) // kv count

	writeGGUFString(t, buf, "general.alignment")
	writeU32(t, buf, valueTypeUint32)
	writeU32(t, buf, 32)

	writeGGUFString(t, buf, "w")
	writeU32(t, buf, 1)
	writeU64(t, buf, 4)
	writeU32(t, buf, GGMLTypeF16)
	writeU64(t, buf, 0)

	padTo(t, buf, 32)
	writeU16(t, buf, 0x3c00) // 1.0
	writeU16(t, buf, 0xbc00) // -1.0
	writeU16(t, buf, 0x3400) // 0.25
	writeU16(t, buf, 0xc000) // -2.0

	path := filepath.Join(t.TempDir(), "tensor_f16.gguf")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	info, err := ReadModelInfo(path)
	if err != nil {
		t.Fatalf("ReadModelInfo() error = %v", err)
	}
	got, err := ReadTensorAsF32(path, info, "w")
	if err != nil {
		t.Fatalf("ReadTensorAsF32() error = %v", err)
	}

	want := []float32{1, -1, 0.25, -2}
	if len(got) != len(want) {
		t.Fatalf("len(got) = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}

func TestReadTensorQ80AsF32(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	writeString(t, buf, "GGUF")
	writeU32(t, buf, 3)
	writeU64(t, buf, 1) // tensor count
	writeU64(t, buf, 1) // kv count

	writeGGUFString(t, buf, "general.alignment")
	writeU32(t, buf, valueTypeUint32)
	writeU32(t, buf, 32)

	writeGGUFString(t, buf, "wq")
	writeU32(t, buf, 1)
	writeU64(t, buf, 32)
	writeU32(t, buf, GGMLTypeQ8_0)
	writeU64(t, buf, 0)

	padTo(t, buf, 32)

	// one q8_0 block: d=0.5, qs=[-16..15]
	writeU16(t, buf, 0x3800) // float16(0.5)
	for i := -16; i <= 15; i++ {
		if err := binary.Write(buf, binary.LittleEndian, int8(i)); err != nil {
			t.Fatalf("binary.Write(i8) error = %v", err)
		}
	}

	path := filepath.Join(t.TempDir(), "tensor_q80.gguf")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	info, err := ReadModelInfo(path)
	if err != nil {
		t.Fatalf("ReadModelInfo() error = %v", err)
	}
	got, err := ReadTensorAsF32(path, info, "wq")
	if err != nil {
		t.Fatalf("ReadTensorAsF32() error = %v", err)
	}
	if len(got) != 32 {
		t.Fatalf("len(got) = %d, want 32", len(got))
	}
	for i := 0; i < 32; i++ {
		want := 0.5 * float32(i-16)
		if got[i] != want {
			t.Fatalf("got[%d] = %f, want %f", i, got[i], want)
		}
	}
}

func TestReadTensorQ40AsF32(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	writeString(t, buf, "GGUF")
	writeU32(t, buf, 3)
	writeU64(t, buf, 1) // tensor count
	writeU64(t, buf, 1) // kv count

	writeGGUFString(t, buf, "general.alignment")
	writeU32(t, buf, valueTypeUint32)
	writeU32(t, buf, 32)

	writeGGUFString(t, buf, "wq")
	writeU32(t, buf, 1)
	writeU64(t, buf, 32)
	writeU32(t, buf, GGMLTypeQ4_0)
	writeU64(t, buf, 0)

	padTo(t, buf, 32)
	writeU16(t, buf, 0x3800) // float16(0.5)
	for i := 0; i < 32; i += 2 {
		low := uint8(i % 16)
		high := uint8((i + 1) % 16)
		writeU8(t, buf, (high<<4)|low)
	}

	path := filepath.Join(t.TempDir(), "tensor_q40.gguf")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	info, err := ReadModelInfo(path)
	if err != nil {
		t.Fatalf("ReadModelInfo() error = %v", err)
	}
	got, err := ReadTensorAsF32(path, info, "wq")
	if err != nil {
		t.Fatalf("ReadTensorAsF32() error = %v", err)
	}
	if len(got) != 32 {
		t.Fatalf("len(got) = %d, want 32", len(got))
	}
	for i := 0; i < 32; i++ {
		q := float32(i % 16)
		want := 0.5 * (q - 8)
		if got[i] != want {
			t.Fatalf("got[%d] = %f, want %f", i, got[i], want)
		}
	}
}

func TestReadTensorQ41AsF32(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	writeString(t, buf, "GGUF")
	writeU32(t, buf, 3)
	writeU64(t, buf, 1) // tensor count
	writeU64(t, buf, 1) // kv count

	writeGGUFString(t, buf, "general.alignment")
	writeU32(t, buf, valueTypeUint32)
	writeU32(t, buf, 32)

	writeGGUFString(t, buf, "wq")
	writeU32(t, buf, 1)
	writeU64(t, buf, 32)
	writeU32(t, buf, GGMLTypeQ4_1)
	writeU64(t, buf, 0)

	padTo(t, buf, 32)
	writeU16(t, buf, 0x3400) // float16(0.25)
	writeU16(t, buf, 0x3c00) // float16(1.0)
	for i := 0; i < 32; i += 2 {
		low := uint8(i % 16)
		high := uint8((i + 1) % 16)
		writeU8(t, buf, (high<<4)|low)
	}

	path := filepath.Join(t.TempDir(), "tensor_q41.gguf")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	info, err := ReadModelInfo(path)
	if err != nil {
		t.Fatalf("ReadModelInfo() error = %v", err)
	}
	got, err := ReadTensorAsF32(path, info, "wq")
	if err != nil {
		t.Fatalf("ReadTensorAsF32() error = %v", err)
	}
	if len(got) != 32 {
		t.Fatalf("len(got) = %d, want 32", len(got))
	}
	for i := 0; i < 32; i++ {
		q := float32(i % 16)
		want := 0.25*q + 1.0
		if got[i] != want {
			t.Fatalf("got[%d] = %f, want %f", i, got[i], want)
		}
	}
}

func TestReadTensorQ50AsF32(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	writeString(t, buf, "GGUF")
	writeU32(t, buf, 3)
	writeU64(t, buf, 1) // tensor count
	writeU64(t, buf, 1) // kv count

	writeGGUFString(t, buf, "general.alignment")
	writeU32(t, buf, valueTypeUint32)
	writeU32(t, buf, 32)

	writeGGUFString(t, buf, "wq")
	writeU32(t, buf, 1)
	writeU64(t, buf, 32)
	writeU32(t, buf, GGMLTypeQ5_0)
	writeU64(t, buf, 0)

	padTo(t, buf, 32)
	writeU16(t, buf, 0x3800) // float16(0.5)
	var qh [4]uint8
	for i := 0; i < 32; i++ {
		if i >= 16 {
			qh[i/8] |= 1 << (uint(i) % 8)
		}
	}
	for i := 0; i < 4; i++ {
		writeU8(t, buf, qh[i])
	}
	for i := 0; i < 32; i += 2 {
		low := uint8(i % 16)
		high := uint8((i + 1) % 16)
		writeU8(t, buf, (high<<4)|low)
	}

	path := filepath.Join(t.TempDir(), "tensor_q50.gguf")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	info, err := ReadModelInfo(path)
	if err != nil {
		t.Fatalf("ReadModelInfo() error = %v", err)
	}
	got, err := ReadTensorAsF32(path, info, "wq")
	if err != nil {
		t.Fatalf("ReadTensorAsF32() error = %v", err)
	}
	if len(got) != 32 {
		t.Fatalf("len(got) = %d, want 32", len(got))
	}
	for i := 0; i < 32; i++ {
		q := float32(i)
		want := 0.5 * (q - 16)
		if got[i] != want {
			t.Fatalf("got[%d] = %f, want %f", i, got[i], want)
		}
	}
}

func TestReadTensorQ51AsF32(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	writeString(t, buf, "GGUF")
	writeU32(t, buf, 3)
	writeU64(t, buf, 1) // tensor count
	writeU64(t, buf, 1) // kv count

	writeGGUFString(t, buf, "general.alignment")
	writeU32(t, buf, valueTypeUint32)
	writeU32(t, buf, 32)

	writeGGUFString(t, buf, "wq")
	writeU32(t, buf, 1)
	writeU64(t, buf, 32)
	writeU32(t, buf, GGMLTypeQ5_1)
	writeU64(t, buf, 0)

	padTo(t, buf, 32)
	writeU16(t, buf, 0x3400) // float16(0.25)
	writeU16(t, buf, 0xbc00) // float16(-1.0)
	var qh [4]uint8
	for i := 0; i < 32; i++ {
		if i >= 16 {
			qh[i/8] |= 1 << (uint(i) % 8)
		}
	}
	for i := 0; i < 4; i++ {
		writeU8(t, buf, qh[i])
	}
	for i := 0; i < 32; i += 2 {
		low := uint8(i % 16)
		high := uint8((i + 1) % 16)
		writeU8(t, buf, (high<<4)|low)
	}

	path := filepath.Join(t.TempDir(), "tensor_q51.gguf")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	info, err := ReadModelInfo(path)
	if err != nil {
		t.Fatalf("ReadModelInfo() error = %v", err)
	}
	got, err := ReadTensorAsF32(path, info, "wq")
	if err != nil {
		t.Fatalf("ReadTensorAsF32() error = %v", err)
	}
	if len(got) != 32 {
		t.Fatalf("len(got) = %d, want 32", len(got))
	}
	for i := 0; i < 32; i++ {
		q := float32(i)
		want := 0.25*q - 1.0
		if got[i] != want {
			t.Fatalf("got[%d] = %f, want %f", i, got[i], want)
		}
	}
}

func TestReadTensorQ2KAsF32(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	writeString(t, buf, "GGUF")
	writeU32(t, buf, 3)
	writeU64(t, buf, 1) // tensor count
	writeU64(t, buf, 1) // kv count

	writeGGUFString(t, buf, "general.alignment")
	writeU32(t, buf, valueTypeUint32)
	writeU32(t, buf, 32)

	writeGGUFString(t, buf, "wq")
	writeU32(t, buf, 1)
	writeU64(t, buf, 256)
	writeU32(t, buf, GGMLTypeQ2_K)
	writeU64(t, buf, 0)

	padTo(t, buf, 32)
	writeU16(t, buf, 0x3c00) // d = 1.0
	writeU16(t, buf, 0x0000) // dmin = 0.0
	for i := 0; i < 16; i++ {
		writeU8(t, buf, 0x01)
	}
	for i := 0; i < 64; i++ {
		writeU8(t, buf, 0xFF)
	}

	path := filepath.Join(t.TempDir(), "tensor_q2k.gguf")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	info, err := ReadModelInfo(path)
	if err != nil {
		t.Fatalf("ReadModelInfo() error = %v", err)
	}
	got, err := ReadTensorAsF32(path, info, "wq")
	if err != nil {
		t.Fatalf("ReadTensorAsF32() error = %v", err)
	}
	if len(got) != 256 {
		t.Fatalf("len(got) = %d, want 256", len(got))
	}
	for i := 0; i < 256; i++ {
		if got[i] != 3 {
			t.Fatalf("got[%d] = %f, want 3", i, got[i])
		}
	}
}

func TestReadTensorQ3KAsF32(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	writeString(t, buf, "GGUF")
	writeU32(t, buf, 3)
	writeU64(t, buf, 1) // tensor count
	writeU64(t, buf, 1) // kv count

	writeGGUFString(t, buf, "general.alignment")
	writeU32(t, buf, valueTypeUint32)
	writeU32(t, buf, 32)

	writeGGUFString(t, buf, "wq")
	writeU32(t, buf, 1)
	writeU64(t, buf, 256)
	writeU32(t, buf, GGMLTypeQ3_K)
	writeU64(t, buf, 0)

	padTo(t, buf, 32)
	for i := 0; i < 32; i++ {
		writeU8(t, buf, 0xFF)
	}
	for i := 0; i < 64; i++ {
		writeU8(t, buf, 0xFF)
	}
	for i := 0; i < 12; i++ {
		writeU8(t, buf, 0x00)
	}
	writeU16(t, buf, 0x0000) // d = 0.0

	path := filepath.Join(t.TempDir(), "tensor_q3k.gguf")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	info, err := ReadModelInfo(path)
	if err != nil {
		t.Fatalf("ReadModelInfo() error = %v", err)
	}
	got, err := ReadTensorAsF32(path, info, "wq")
	if err != nil {
		t.Fatalf("ReadTensorAsF32() error = %v", err)
	}
	if len(got) != 256 {
		t.Fatalf("len(got) = %d, want 256", len(got))
	}
	for i := 0; i < 256; i++ {
		if got[i] != 0 {
			t.Fatalf("got[%d] = %f, want 0", i, got[i])
		}
	}
}

func TestReadTensorQ4KAsF32(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	writeString(t, buf, "GGUF")
	writeU32(t, buf, 3)
	writeU64(t, buf, 1) // tensor count
	writeU64(t, buf, 1) // kv count

	writeGGUFString(t, buf, "general.alignment")
	writeU32(t, buf, valueTypeUint32)
	writeU32(t, buf, 32)

	writeGGUFString(t, buf, "wq")
	writeU32(t, buf, 1)
	writeU64(t, buf, 256)
	writeU32(t, buf, GGMLTypeQ4_K)
	writeU64(t, buf, 0)

	padTo(t, buf, 32)
	writeU16(t, buf, 0x3c00) // d = 1.0
	writeU16(t, buf, 0x0000) // dmin = 0.0
	for i := 0; i < 12; i++ {
		switch {
		case i < 4:
			writeU8(t, buf, 0x01)
		case i < 8:
			writeU8(t, buf, 0x00)
		default:
			writeU8(t, buf, 0x01)
		}
	}
	for i := 0; i < 128; i++ {
		writeU8(t, buf, 0x21)
	}

	path := filepath.Join(t.TempDir(), "tensor_q4k.gguf")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	info, err := ReadModelInfo(path)
	if err != nil {
		t.Fatalf("ReadModelInfo() error = %v", err)
	}
	got, err := ReadTensorAsF32(path, info, "wq")
	if err != nil {
		t.Fatalf("ReadTensorAsF32() error = %v", err)
	}
	if len(got) != 256 {
		t.Fatalf("len(got) = %d, want 256", len(got))
	}
	for i := 0; i < 256; i++ {
		want := float32(1)
		if i%64 >= 32 {
			want = 2
		}
		if got[i] != want {
			t.Fatalf("got[%d] = %f, want %f", i, got[i], want)
		}
	}
}

func TestReadTensorQ5KAsF32(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	writeString(t, buf, "GGUF")
	writeU32(t, buf, 3)
	writeU64(t, buf, 1) // tensor count
	writeU64(t, buf, 1) // kv count

	writeGGUFString(t, buf, "general.alignment")
	writeU32(t, buf, valueTypeUint32)
	writeU32(t, buf, 32)

	writeGGUFString(t, buf, "wq")
	writeU32(t, buf, 1)
	writeU64(t, buf, 256)
	writeU32(t, buf, GGMLTypeQ5_K)
	writeU64(t, buf, 0)

	padTo(t, buf, 32)
	writeU16(t, buf, 0x3c00) // d = 1.0
	writeU16(t, buf, 0x0000) // dmin = 0.0
	for i := 0; i < 12; i++ {
		switch {
		case i < 4:
			writeU8(t, buf, 0x01)
		case i < 8:
			writeU8(t, buf, 0x00)
		default:
			writeU8(t, buf, 0x01)
		}
	}
	for i := 0; i < 32; i++ {
		writeU8(t, buf, 0x00)
	}
	for i := 0; i < 128; i++ {
		writeU8(t, buf, 0x21)
	}

	path := filepath.Join(t.TempDir(), "tensor_q5k.gguf")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	info, err := ReadModelInfo(path)
	if err != nil {
		t.Fatalf("ReadModelInfo() error = %v", err)
	}
	got, err := ReadTensorAsF32(path, info, "wq")
	if err != nil {
		t.Fatalf("ReadTensorAsF32() error = %v", err)
	}
	if len(got) != 256 {
		t.Fatalf("len(got) = %d, want 256", len(got))
	}
	for i := 0; i < 256; i++ {
		want := float32(1)
		if i%64 >= 32 {
			want = 2
		}
		if got[i] != want {
			t.Fatalf("got[%d] = %f, want %f", i, got[i], want)
		}
	}
}

func TestReadTensorQ6KAsF32(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	writeString(t, buf, "GGUF")
	writeU32(t, buf, 3)
	writeU64(t, buf, 1) // tensor count
	writeU64(t, buf, 1) // kv count

	writeGGUFString(t, buf, "general.alignment")
	writeU32(t, buf, valueTypeUint32)
	writeU32(t, buf, 32)

	writeGGUFString(t, buf, "wq")
	writeU32(t, buf, 1)
	writeU64(t, buf, 256)
	writeU32(t, buf, GGMLTypeQ6_K)
	writeU64(t, buf, 0)

	padTo(t, buf, 32)
	for i := 0; i < 128; i++ {
		writeU8(t, buf, 0x00)
	}
	for i := 0; i < 64; i++ {
		writeU8(t, buf, 0x00)
	}
	for i := 0; i < 16; i++ {
		writeU8(t, buf, 0x01)
	}
	writeU16(t, buf, 0x0000) // d = 0.0

	path := filepath.Join(t.TempDir(), "tensor_q6k.gguf")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	info, err := ReadModelInfo(path)
	if err != nil {
		t.Fatalf("ReadModelInfo() error = %v", err)
	}
	got, err := ReadTensorAsF32(path, info, "wq")
	if err != nil {
		t.Fatalf("ReadTensorAsF32() error = %v", err)
	}
	if len(got) != 256 {
		t.Fatalf("len(got) = %d, want 256", len(got))
	}
	for i := 0; i < 256; i++ {
		if got[i] != 0 {
			t.Fatalf("got[%d] = %f, want 0", i, got[i])
		}
	}
}

func TestReadTensorQ8KAsF32(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	writeString(t, buf, "GGUF")
	writeU32(t, buf, 3)
	writeU64(t, buf, 1) // tensor count
	writeU64(t, buf, 1) // kv count

	writeGGUFString(t, buf, "general.alignment")
	writeU32(t, buf, valueTypeUint32)
	writeU32(t, buf, 32)

	writeGGUFString(t, buf, "wq")
	writeU32(t, buf, 1)
	writeU64(t, buf, 256)
	writeU32(t, buf, GGMLTypeQ8_K)
	writeU64(t, buf, 0)

	padTo(t, buf, 32)
	writeF32(t, buf, 0.5)
	for i := 0; i < 256; i++ {
		writeI8(t, buf, int8(i%4-2))
	}
	for i := 0; i < 16; i++ {
		writeI16(t, buf, 0)
	}

	path := filepath.Join(t.TempDir(), "tensor_q8k.gguf")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	info, err := ReadModelInfo(path)
	if err != nil {
		t.Fatalf("ReadModelInfo() error = %v", err)
	}
	got, err := ReadTensorAsF32(path, info, "wq")
	if err != nil {
		t.Fatalf("ReadTensorAsF32() error = %v", err)
	}
	if len(got) != 256 {
		t.Fatalf("len(got) = %d, want 256", len(got))
	}
	for i := 0; i < 256; i++ {
		want := 0.5 * float32(int8(i%4-2))
		if got[i] != want {
			t.Fatalf("got[%d] = %f, want %f", i, got[i], want)
		}
	}
}

func TestFloat16ToFloat32(t *testing.T) {
	tests := []struct {
		in   uint16
		want float32
	}{
		{0x0000, 0},
		{0x8000, float32(math.Copysign(0, -1))},
		{0x3c00, 1},
		{0x3800, 0.5},
		{0xc000, -2},
	}
	for _, tc := range tests {
		got := float16ToFloat32(tc.in)
		if got != tc.want {
			t.Fatalf("float16ToFloat32(0x%04x) = %f, want %f", tc.in, got, tc.want)
		}
	}
}

func padTo(t *testing.T, buf *bytes.Buffer, align int) {
	t.Helper()
	rem := buf.Len() % align
	if rem == 0 {
		return
	}
	n := align - rem
	padding := make([]byte, n)
	if _, err := buf.Write(padding); err != nil {
		t.Fatalf("Write(padding) error = %v", err)
	}
}

func writeF32(t *testing.T, buf *bytes.Buffer, v float32) {
	t.Helper()
	if err := binary.Write(buf, binary.LittleEndian, v); err != nil {
		t.Fatalf("binary.Write(f32) error = %v", err)
	}
}

func writeU16(t *testing.T, buf *bytes.Buffer, v uint16) {
	t.Helper()
	if err := binary.Write(buf, binary.LittleEndian, v); err != nil {
		t.Fatalf("binary.Write(u16) error = %v", err)
	}
}

func writeU8(t *testing.T, buf *bytes.Buffer, v uint8) {
	t.Helper()
	if err := buf.WriteByte(v); err != nil {
		t.Fatalf("WriteByte() error = %v", err)
	}
}

func writeI8(t *testing.T, buf *bytes.Buffer, v int8) {
	t.Helper()
	if err := binary.Write(buf, binary.LittleEndian, v); err != nil {
		t.Fatalf("binary.Write(i8) error = %v", err)
	}
}

func writeI16(t *testing.T, buf *bytes.Buffer, v int16) {
	t.Helper()
	if err := binary.Write(buf, binary.LittleEndian, v); err != nil {
		t.Fatalf("binary.Write(i16) error = %v", err)
	}
}
