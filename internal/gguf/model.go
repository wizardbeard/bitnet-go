package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

const (
	valueTypeUint8   = 0
	valueTypeInt8    = 1
	valueTypeUint16  = 2
	valueTypeInt16   = 3
	valueTypeUint32  = 4
	valueTypeInt32   = 5
	valueTypeFloat32 = 6
	valueTypeBool    = 7
	valueTypeString  = 8
	valueTypeArray   = 9
	valueTypeUint64  = 10
	valueTypeInt64   = 11
	valueTypeFloat64 = 12
)

type TensorInfo struct {
	Name       string
	Dimensions []uint64
	Type       uint32
	Offset     uint64
}

type ModelInfo struct {
	Header
	KeyValues        map[string]any
	Tensors          []TensorInfo
	Alignment        uint32
	TensorDataOffset uint64
}

func ReadModelInfo(path string) (ModelInfo, error) {
	f, err := os.Open(path)
	if err != nil {
		return ModelInfo{}, err
	}
	defer f.Close()
	return DecodeModelInfo(f)
}

func DecodeModelInfo(r io.Reader) (ModelInfo, error) {
	cr := &countingReader{r: r}
	h, err := DecodeHeader(cr)
	if err != nil {
		return ModelInfo{}, err
	}

	info := ModelInfo{
		Header:    h,
		KeyValues: make(map[string]any, h.KVCount),
		Tensors:   make([]TensorInfo, 0, h.TensorCount),
	}

	for i := uint64(0); i < h.KVCount; i++ {
		key, err := readGGUFString(cr)
		if err != nil {
			return ModelInfo{}, fmt.Errorf("read kv key[%d]: %w", i, err)
		}

		t, err := readUint32(cr)
		if err != nil {
			return ModelInfo{}, fmt.Errorf("read kv type[%d]: %w", i, err)
		}

		if t == valueTypeArray {
			parsed, n, captured, err := readArrayValue(cr, key)
			if err != nil {
				return ModelInfo{}, fmt.Errorf("read kv array[%d]: %w", i, err)
			}
			info.KeyValues[key+".count"] = n
			if captured {
				info.KeyValues[key] = parsed
			}
			continue
		}

		v, capture, err := readValueByType(cr, t)
		if err != nil {
			return ModelInfo{}, fmt.Errorf("read kv value[%d] type=%d: %w", i, t, err)
		}
		if capture {
			info.KeyValues[key] = v
		}
	}

	for i := uint64(0); i < h.TensorCount; i++ {
		name, err := readGGUFString(cr)
		if err != nil {
			return ModelInfo{}, fmt.Errorf("read tensor name[%d]: %w", i, err)
		}
		nDims, err := readUint32(cr)
		if err != nil {
			return ModelInfo{}, fmt.Errorf("read tensor n_dims[%d]: %w", i, err)
		}
		dims := make([]uint64, nDims)
		for j := uint32(0); j < nDims; j++ {
			dims[j], err = readUint64(cr)
			if err != nil {
				return ModelInfo{}, fmt.Errorf("read tensor dim[%d][%d]: %w", i, j, err)
			}
		}
		tType, err := readUint32(cr)
		if err != nil {
			return ModelInfo{}, fmt.Errorf("read tensor type[%d]: %w", i, err)
		}
		offset, err := readUint64(cr)
		if err != nil {
			return ModelInfo{}, fmt.Errorf("read tensor offset[%d]: %w", i, err)
		}

		info.Tensors = append(info.Tensors, TensorInfo{
			Name:       name,
			Dimensions: dims,
			Type:       tType,
			Offset:     offset,
		})
	}

	info.Alignment = modelAlignment(info.KeyValues)
	info.TensorDataOffset = alignUpUint64(cr.n, uint64(info.Alignment))

	return info, nil
}

type countingReader struct {
	r io.Reader
	n uint64
}

func (r *countingReader) Read(p []byte) (int, error) {
	n, err := r.r.Read(p)
	r.n += uint64(n)
	return n, err
}

func modelAlignment(kv map[string]any) uint32 {
	if v, ok := kv["general.alignment"]; ok {
		switch x := v.(type) {
		case uint32:
			if x > 0 {
				return x
			}
		case uint64:
			if x > 0 && x <= uint64(^uint32(0)) {
				return uint32(x)
			}
		case int32:
			if x > 0 {
				return uint32(x)
			}
		case int64:
			if x > 0 && x <= int64(^uint32(0)) {
				return uint32(x)
			}
		}
	}
	return 32
}

func alignUpUint64(v, align uint64) uint64 {
	if align == 0 {
		return v
	}
	rem := v % align
	if rem == 0 {
		return v
	}
	return v + (align - rem)
}

func readValueByType(r io.Reader, valueType uint32) (any, bool, error) {
	switch valueType {
	case valueTypeUint8:
		v, err := readUint8(r)
		return v, true, err
	case valueTypeInt8:
		v, err := readInt8(r)
		return v, true, err
	case valueTypeUint16:
		v, err := readUint16(r)
		return v, true, err
	case valueTypeInt16:
		v, err := readInt16(r)
		return v, true, err
	case valueTypeUint32:
		v, err := readUint32(r)
		return v, true, err
	case valueTypeInt32:
		v, err := readInt32(r)
		return v, true, err
	case valueTypeFloat32:
		v, err := readFloat32(r)
		return v, true, err
	case valueTypeBool:
		v, err := readBool(r)
		return v, true, err
	case valueTypeString:
		v, err := readGGUFString(r)
		return v, true, err
	case valueTypeUint64:
		v, err := readUint64(r)
		return v, true, err
	case valueTypeInt64:
		v, err := readInt64(r)
		return v, true, err
	case valueTypeFloat64:
		v, err := readFloat64(r)
		return v, true, err
	default:
		return nil, false, fmt.Errorf("unsupported gguf value type: %d", valueType)
	}
}

func readArrayValue(r io.Reader, key string) (any, uint64, bool, error) {
	elemType, err := readUint32(r)
	if err != nil {
		return nil, 0, false, err
	}
	n, err := readUint64(r)
	if err != nil {
		return nil, 0, false, err
	}

	if elemType == valueTypeString {
		if key == "tokenizer.ggml.tokens" || key == "tokenizer.ggml.merges" {
			tokens := make([]string, 0, n)
			for i := uint64(0); i < n; i++ {
				s, err := readGGUFString(r)
				if err != nil {
					return nil, 0, false, err
				}
				tokens = append(tokens, s)
			}
			return tokens, n, true, nil
		}
		for i := uint64(0); i < n; i++ {
			if _, err := readGGUFString(r); err != nil {
				return nil, 0, false, err
			}
		}
		return nil, n, false, nil
	}

	if elemType == valueTypeFloat32 && key == "tokenizer.ggml.scores" {
		out := make([]float32, n)
		for i := uint64(0); i < n; i++ {
			v, err := readFloat32(r)
			if err != nil {
				return nil, 0, false, err
			}
			out[i] = v
		}
		return out, n, true, nil
	}

	if elemType == valueTypeInt32 && key == "tokenizer.ggml.token_type" {
		out := make([]int32, n)
		for i := uint64(0); i < n; i++ {
			v, err := readInt32(r)
			if err != nil {
				return nil, 0, false, err
			}
			out[i] = v
		}
		return out, n, true, nil
	}

	size := valueTypeSize(elemType)
	if size == 0 {
		return nil, 0, false, fmt.Errorf("unsupported array element type: %d", elemType)
	}

	total := n * uint64(size)
	_, err = io.CopyN(io.Discard, r, int64(total))
	if err != nil {
		return nil, 0, false, err
	}
	return nil, n, false, nil
}

func valueTypeSize(t uint32) int {
	switch t {
	case valueTypeUint8, valueTypeInt8, valueTypeBool:
		return 1
	case valueTypeUint16, valueTypeInt16:
		return 2
	case valueTypeUint32, valueTypeInt32, valueTypeFloat32:
		return 4
	case valueTypeUint64, valueTypeInt64, valueTypeFloat64:
		return 8
	default:
		return 0
	}
}

func readGGUFString(r io.Reader) (string, error) {
	n, err := readUint64(r)
	if err != nil {
		return "", err
	}
	if n > math.MaxInt32 {
		return "", fmt.Errorf("string too large: %d", n)
	}
	buf := make([]byte, int(n))
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

func readUint8(r io.Reader) (uint8, error) {
	var v uint8
	if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
		return 0, err
	}
	return v, nil
}

func readInt8(r io.Reader) (int8, error) {
	var v int8
	if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
		return 0, err
	}
	return v, nil
}

func readUint16(r io.Reader) (uint16, error) {
	var v uint16
	if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
		return 0, err
	}
	return v, nil
}

func readInt16(r io.Reader) (int16, error) {
	var v int16
	if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
		return 0, err
	}
	return v, nil
}

func readUint32(r io.Reader) (uint32, error) {
	var v uint32
	if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
		return 0, err
	}
	return v, nil
}

func readInt32(r io.Reader) (int32, error) {
	var v int32
	if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
		return 0, err
	}
	return v, nil
}

func readFloat32(r io.Reader) (float32, error) {
	var v float32
	if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
		return 0, err
	}
	return v, nil
}

func readBool(r io.Reader) (bool, error) {
	var b uint8
	if err := binary.Read(r, binary.LittleEndian, &b); err != nil {
		return false, err
	}
	return b != 0, nil
}

func readUint64(r io.Reader) (uint64, error) {
	var v uint64
	if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
		return 0, err
	}
	return v, nil
}

func readInt64(r io.Reader) (int64, error) {
	var v int64
	if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
		return 0, err
	}
	return v, nil
}

func readFloat64(r io.Reader) (float64, error) {
	var v float64
	if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
		return 0, err
	}
	return v, nil
}
