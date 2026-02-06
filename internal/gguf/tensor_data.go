package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

const (
	GGMLTypeF32  = 0
	GGMLTypeF16  = 1
	GGMLTypeQ4_0 = 2
	GGMLTypeQ4_1 = 3
	GGMLTypeQ5_0 = 6
	GGMLTypeQ5_1 = 7
	GGMLTypeQ8_0 = 8
	GGMLTypeQ2_K = 10
	GGMLTypeQ3_K = 11
	GGMLTypeQ4_K = 12
	GGMLTypeQ5_K = 13
	GGMLTypeQ6_K = 14
	GGMLTypeQ8_K = 15
)

func (m ModelInfo) TensorByName(name string) (TensorInfo, bool) {
	for i := range m.Tensors {
		if m.Tensors[i].Name == name {
			return m.Tensors[i], true
		}
	}
	return TensorInfo{}, false
}

func TensorElementCount(t TensorInfo) (uint64, error) {
	if len(t.Dimensions) == 0 {
		return 0, fmt.Errorf("tensor %q has no dimensions", t.Name)
	}
	n := uint64(1)
	for _, d := range t.Dimensions {
		if d == 0 {
			return 0, fmt.Errorf("tensor %q has zero-sized dimension", t.Name)
		}
		if n > math.MaxUint64/d {
			return 0, fmt.Errorf("tensor %q element count overflow", t.Name)
		}
		n *= d
	}
	return n, nil
}

func ReadTensorF32(path string, info ModelInfo, name string) ([]float32, error) {
	return ReadTensorAsF32(path, info, name)
}

func ReadTensorAsF32(path string, info ModelInfo, name string) ([]float32, error) {
	t, ok := info.TensorByName(name)
	if !ok {
		return nil, fmt.Errorf("tensor not found: %s", name)
	}

	count, err := TensorElementCount(t)
	if err != nil {
		return nil, err
	}
	if count > uint64(math.MaxInt/4) {
		return nil, fmt.Errorf("tensor %q too large to load", name)
	}

	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	start := info.TensorDataOffset + t.Offset
	if _, err := f.Seek(int64(start), io.SeekStart); err != nil {
		return nil, fmt.Errorf("seek tensor %q: %w", name, err)
	}

	switch t.Type {
	case GGMLTypeF32:
		out := make([]float32, count)
		if err := binary.Read(f, binary.LittleEndian, out); err != nil {
			return nil, fmt.Errorf("read tensor %q f32: %w", name, err)
		}
		return out, nil
	case GGMLTypeF16:
		return readTensorF16AsF32(f, name, count)
	case GGMLTypeQ4_0:
		return readTensorQ40AsF32(f, name, count)
	case GGMLTypeQ4_1:
		return readTensorQ41AsF32(f, name, count)
	case GGMLTypeQ5_0:
		return readTensorQ50AsF32(f, name, count)
	case GGMLTypeQ5_1:
		return readTensorQ51AsF32(f, name, count)
	case GGMLTypeQ8_0:
		return readTensorQ80AsF32(f, name, count)
	case GGMLTypeQ2_K:
		return readTensorQ2KAsF32(f, name, count)
	case GGMLTypeQ3_K:
		return readTensorQ3KAsF32(f, name, count)
	case GGMLTypeQ4_K:
		return readTensorQ4KAsF32(f, name, count)
	case GGMLTypeQ5_K:
		return readTensorQ5KAsF32(f, name, count)
	case GGMLTypeQ6_K:
		return readTensorQ6KAsF32(f, name, count)
	case GGMLTypeQ8_K:
		return readTensorQ8KAsF32(f, name, count)
	default:
		return nil, fmt.Errorf("tensor %q type=%d not supported", name, t.Type)
	}
}

func readTensorF16AsF32(r io.Reader, name string, count uint64) ([]float32, error) {
	if count > uint64(math.MaxInt) {
		return nil, fmt.Errorf("tensor %q has too many f16 elements", name)
	}
	out := make([]float32, count)
	buf := make([]uint16, count)
	if err := binary.Read(r, binary.LittleEndian, buf); err != nil {
		return nil, fmt.Errorf("read tensor %q f16: %w", name, err)
	}
	for i := range buf {
		out[i] = float16ToFloat32(buf[i])
	}
	return out, nil
}

func readTensorQ80AsF32(r io.Reader, name string, count uint64) ([]float32, error) {
	const qk = 32
	if count%qk != 0 {
		return nil, fmt.Errorf("tensor %q q8_0 element count=%d not divisible by %d", name, count, qk)
	}
	blocks := count / qk
	if blocks > uint64(math.MaxInt) {
		return nil, fmt.Errorf("tensor %q has too many q8_0 blocks", name)
	}
	out := make([]float32, count)

	type q80Block struct {
		D  uint16
		Qs [qk]int8
	}
	var blk q80Block
	for b := uint64(0); b < blocks; b++ {
		if err := binary.Read(r, binary.LittleEndian, &blk); err != nil {
			return nil, fmt.Errorf("read tensor %q q8_0 block %d: %w", name, b, err)
		}
		scale := float16ToFloat32(blk.D)
		base := int(b * qk)
		for i := 0; i < qk; i++ {
			out[base+i] = scale * float32(blk.Qs[i])
		}
	}
	return out, nil
}

func readTensorQ40AsF32(r io.Reader, name string, count uint64) ([]float32, error) {
	const qk = 32
	if count%qk != 0 {
		return nil, fmt.Errorf("tensor %q q4_0 element count=%d not divisible by %d", name, count, qk)
	}
	blocks := count / qk
	if blocks > uint64(math.MaxInt) {
		return nil, fmt.Errorf("tensor %q has too many q4_0 blocks", name)
	}
	out := make([]float32, count)

	type q40Block struct {
		D  uint16
		Qs [qk / 2]uint8
	}
	var blk q40Block
	for b := uint64(0); b < blocks; b++ {
		if err := binary.Read(r, binary.LittleEndian, &blk); err != nil {
			return nil, fmt.Errorf("read tensor %q q4_0 block %d: %w", name, b, err)
		}
		scale := float16ToFloat32(blk.D)
		base := int(b * qk)
		for i := 0; i < qk; i++ {
			q := nibbleAt(blk.Qs[:], i)
			out[base+i] = scale * float32(int(q)-8)
		}
	}
	return out, nil
}

func readTensorQ41AsF32(r io.Reader, name string, count uint64) ([]float32, error) {
	const qk = 32
	if count%qk != 0 {
		return nil, fmt.Errorf("tensor %q q4_1 element count=%d not divisible by %d", name, count, qk)
	}
	blocks := count / qk
	if blocks > uint64(math.MaxInt) {
		return nil, fmt.Errorf("tensor %q has too many q4_1 blocks", name)
	}
	out := make([]float32, count)

	type q41Block struct {
		D  uint16
		M  uint16
		Qs [qk / 2]uint8
	}
	var blk q41Block
	for b := uint64(0); b < blocks; b++ {
		if err := binary.Read(r, binary.LittleEndian, &blk); err != nil {
			return nil, fmt.Errorf("read tensor %q q4_1 block %d: %w", name, b, err)
		}
		scale := float16ToFloat32(blk.D)
		min := float16ToFloat32(blk.M)
		base := int(b * qk)
		for i := 0; i < qk; i++ {
			q := nibbleAt(blk.Qs[:], i)
			out[base+i] = scale*float32(q) + min
		}
	}
	return out, nil
}

func readTensorQ50AsF32(r io.Reader, name string, count uint64) ([]float32, error) {
	const qk = 32
	if count%qk != 0 {
		return nil, fmt.Errorf("tensor %q q5_0 element count=%d not divisible by %d", name, count, qk)
	}
	blocks := count / qk
	if blocks > uint64(math.MaxInt) {
		return nil, fmt.Errorf("tensor %q has too many q5_0 blocks", name)
	}
	out := make([]float32, count)

	type q50Block struct {
		D  uint16
		Qh [4]uint8
		Qs [qk / 2]uint8
	}
	var blk q50Block
	for b := uint64(0); b < blocks; b++ {
		if err := binary.Read(r, binary.LittleEndian, &blk); err != nil {
			return nil, fmt.Errorf("read tensor %q q5_0 block %d: %w", name, b, err)
		}
		scale := float16ToFloat32(blk.D)
		base := int(b * qk)
		for i := 0; i < qk; i++ {
			low := nibbleAt(blk.Qs[:], i)
			high := (blk.Qh[i/8] >> (uint(i) % 8)) & 0x1
			q := (high << 4) | low
			out[base+i] = scale * float32(int(q)-16)
		}
	}
	return out, nil
}

func readTensorQ51AsF32(r io.Reader, name string, count uint64) ([]float32, error) {
	const qk = 32
	if count%qk != 0 {
		return nil, fmt.Errorf("tensor %q q5_1 element count=%d not divisible by %d", name, count, qk)
	}
	blocks := count / qk
	if blocks > uint64(math.MaxInt) {
		return nil, fmt.Errorf("tensor %q has too many q5_1 blocks", name)
	}
	out := make([]float32, count)

	type q51Block struct {
		D  uint16
		M  uint16
		Qh [4]uint8
		Qs [qk / 2]uint8
	}
	var blk q51Block
	for b := uint64(0); b < blocks; b++ {
		if err := binary.Read(r, binary.LittleEndian, &blk); err != nil {
			return nil, fmt.Errorf("read tensor %q q5_1 block %d: %w", name, b, err)
		}
		scale := float16ToFloat32(blk.D)
		min := float16ToFloat32(blk.M)
		base := int(b * qk)
		for i := 0; i < qk; i++ {
			low := nibbleAt(blk.Qs[:], i)
			high := (blk.Qh[i/8] >> (uint(i) % 8)) & 0x1
			q := (high << 4) | low
			out[base+i] = scale*float32(q) + min
		}
	}
	return out, nil
}

func readTensorQ2KAsF32(r io.Reader, name string, count uint64) ([]float32, error) {
	const qk = 256
	if count%qk != 0 {
		return nil, fmt.Errorf("tensor %q q2_k element count=%d not divisible by %d", name, count, qk)
	}
	blocks := count / qk
	if blocks > uint64(math.MaxInt) {
		return nil, fmt.Errorf("tensor %q has too many q2_k blocks", name)
	}
	out := make([]float32, count)

	const blockSize = 2 + 2 + qk/16 + qk/4
	buf := make([]byte, blockSize)
	for b := uint64(0); b < blocks; b++ {
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, fmt.Errorf("read tensor %q q2_k block %d: %w", name, b, err)
		}
		idx := 0
		d := float16ToFloat32(binary.LittleEndian.Uint16(buf[idx:]))
		idx += 2
		dmin := float16ToFloat32(binary.LittleEndian.Uint16(buf[idx:]))
		idx += 2
		scales := buf[idx : idx+qk/16]
		idx += qk / 16
		qs := buf[idx:]

		base := int(b * qk)
		q := qs
		is := 0
		outIdx := base
		for n := 0; n < qk; n += 128 {
			shift := 0
			for j := 0; j < 4; j++ {
				sc := scales[is]
				is++
				dl := d * float32(sc&0xF)
				ml := dmin * float32(sc>>4)
				for l := 0; l < 16; l++ {
					out[outIdx] = dl*float32((q[l]>>uint(shift))&3) - ml
					outIdx++
				}

				sc = scales[is]
				is++
				dl = d * float32(sc&0xF)
				ml = dmin * float32(sc>>4)
				for l := 0; l < 16; l++ {
					out[outIdx] = dl*float32((q[l+16]>>uint(shift))&3) - ml
					outIdx++
				}
				shift += 2
			}
			q = q[32:]
		}
	}
	return out, nil
}

func readTensorQ3KAsF32(r io.Reader, name string, count uint64) ([]float32, error) {
	const qk = 256
	if count%qk != 0 {
		return nil, fmt.Errorf("tensor %q q3_k element count=%d not divisible by %d", name, count, qk)
	}
	blocks := count / qk
	if blocks > uint64(math.MaxInt) {
		return nil, fmt.Errorf("tensor %q has too many q3_k blocks", name)
	}
	out := make([]float32, count)

	const blockSize = qk/8 + qk/4 + 12 + 2
	buf := make([]byte, blockSize)
	for b := uint64(0); b < blocks; b++ {
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, fmt.Errorf("read tensor %q q3_k block %d: %w", name, b, err)
		}
		idx := 0
		hmask := buf[idx : idx+qk/8]
		idx += qk / 8
		qs := buf[idx : idx+qk/4]
		idx += qk / 4
		scaleBytes := buf[idx : idx+12]
		idx += 12
		d := float16ToFloat32(binary.LittleEndian.Uint16(buf[idx:]))

		scales := q3KScales(scaleBytes)
		base := int(b * qk)
		q := qs
		hm := hmask
		outIdx := base
		m := uint8(1)
		is := 0
		for n := 0; n < qk; n += 128 {
			shift := 0
			for j := 0; j < 4; j++ {
				dl := d * float32(scales[is]-32)
				is++
				for l := 0; l < 16; l++ {
					v := int8((q[l] >> uint(shift)) & 3)
					if hm[l]&m == 0 {
						v -= 4
					}
					out[outIdx] = dl * float32(v)
					outIdx++
				}

				dl = d * float32(scales[is]-32)
				is++
				for l := 0; l < 16; l++ {
					v := int8((q[l+16] >> uint(shift)) & 3)
					if hm[l+16]&m == 0 {
						v -= 4
					}
					out[outIdx] = dl * float32(v)
					outIdx++
				}
				shift += 2
				m <<= 1
			}
			q = q[32:]
		}
	}
	return out, nil
}

func readTensorQ4KAsF32(r io.Reader, name string, count uint64) ([]float32, error) {
	const qk = 256
	if count%qk != 0 {
		return nil, fmt.Errorf("tensor %q q4_k element count=%d not divisible by %d", name, count, qk)
	}
	blocks := count / qk
	if blocks > uint64(math.MaxInt) {
		return nil, fmt.Errorf("tensor %q has too many q4_k blocks", name)
	}
	out := make([]float32, count)

	const blockSize = 2 + 2 + 12 + qk/2
	buf := make([]byte, blockSize)
	for b := uint64(0); b < blocks; b++ {
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, fmt.Errorf("read tensor %q q4_k block %d: %w", name, b, err)
		}
		idx := 0
		d := float16ToFloat32(binary.LittleEndian.Uint16(buf[idx:]))
		idx += 2
		dmin := float16ToFloat32(binary.LittleEndian.Uint16(buf[idx:]))
		idx += 2
		scales := buf[idx : idx+12]
		idx += 12
		qs := buf[idx:]

		base := int(b * qk)
		q := qs
		outIdx := base
		is := 0
		for j := 0; j < qk; j += 64 {
			sc, m := getScaleMinK4(is, scales)
			d1 := d * float32(sc)
			m1 := dmin * float32(m)
			sc, m = getScaleMinK4(is+1, scales)
			d2 := d * float32(sc)
			m2 := dmin * float32(m)
			for l := 0; l < 32; l++ {
				out[outIdx] = d1*float32(q[l]&0xF) - m1
				outIdx++
			}
			for l := 0; l < 32; l++ {
				out[outIdx] = d2*float32(q[l]>>4) - m2
				outIdx++
			}
			q = q[32:]
			is += 2
		}
	}
	return out, nil
}

func readTensorQ5KAsF32(r io.Reader, name string, count uint64) ([]float32, error) {
	const qk = 256
	if count%qk != 0 {
		return nil, fmt.Errorf("tensor %q q5_k element count=%d not divisible by %d", name, count, qk)
	}
	blocks := count / qk
	if blocks > uint64(math.MaxInt) {
		return nil, fmt.Errorf("tensor %q has too many q5_k blocks", name)
	}
	out := make([]float32, count)

	const blockSize = 2 + 2 + 12 + qk/8 + qk/2
	buf := make([]byte, blockSize)
	for b := uint64(0); b < blocks; b++ {
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, fmt.Errorf("read tensor %q q5_k block %d: %w", name, b, err)
		}
		idx := 0
		d := float16ToFloat32(binary.LittleEndian.Uint16(buf[idx:]))
		idx += 2
		dmin := float16ToFloat32(binary.LittleEndian.Uint16(buf[idx:]))
		idx += 2
		scales := buf[idx : idx+12]
		idx += 12
		qh := buf[idx : idx+qk/8]
		idx += qk / 8
		qs := buf[idx:]

		base := int(b * qk)
		ql := qs
		outIdx := base
		is := 0
		u1 := uint8(1)
		u2 := uint8(2)
		for j := 0; j < qk; j += 64 {
			sc, m := getScaleMinK4(is, scales)
			d1 := d * float32(sc)
			m1 := dmin * float32(m)
			sc, m = getScaleMinK4(is+1, scales)
			d2 := d * float32(sc)
			m2 := dmin * float32(m)
			for l := 0; l < 32; l++ {
				val := int(ql[l] & 0xF)
				if qh[l]&u1 != 0 {
					val += 16
				}
				out[outIdx] = d1*float32(val) - m1
				outIdx++
			}
			for l := 0; l < 32; l++ {
				val := int(ql[l] >> 4)
				if qh[l]&u2 != 0 {
					val += 16
				}
				out[outIdx] = d2*float32(val) - m2
				outIdx++
			}
			ql = ql[32:]
			is += 2
			u1 <<= 2
			u2 <<= 2
		}
	}
	return out, nil
}

func readTensorQ6KAsF32(r io.Reader, name string, count uint64) ([]float32, error) {
	const qk = 256
	if count%qk != 0 {
		return nil, fmt.Errorf("tensor %q q6_k element count=%d not divisible by %d", name, count, qk)
	}
	blocks := count / qk
	if blocks > uint64(math.MaxInt) {
		return nil, fmt.Errorf("tensor %q has too many q6_k blocks", name)
	}
	out := make([]float32, count)

	const blockSize = qk/2 + qk/4 + qk/16 + 2
	buf := make([]byte, blockSize)
	for b := uint64(0); b < blocks; b++ {
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, fmt.Errorf("read tensor %q q6_k block %d: %w", name, b, err)
		}
		idx := 0
		ql := buf[idx : idx+qk/2]
		idx += qk / 2
		qh := buf[idx : idx+qk/4]
		idx += qk / 4
		scales := buf[idx : idx+qk/16]
		idx += qk / 16
		d := float16ToFloat32(binary.LittleEndian.Uint16(buf[idx:]))

		base := int(b * qk)
		outIdx := base
		qlo := ql
		qhi := qh
		sc := scales
		for n := 0; n < qk; n += 128 {
			for l := 0; l < 32; l++ {
				is := l / 16
				q1 := int8((qlo[l+0] & 0xF) | (((qhi[l] >> 0) & 3) << 4))
				q2 := int8((qlo[l+32] & 0xF) | (((qhi[l] >> 2) & 3) << 4))
				q3 := int8((qlo[l+0] >> 4) | (((qhi[l] >> 4) & 3) << 4))
				q4 := int8((qlo[l+32] >> 4) | (((qhi[l] >> 6) & 3) << 4))
				q1 -= 32
				q2 -= 32
				q3 -= 32
				q4 -= 32
				out[outIdx+0] = d * float32(int8(sc[is+0])) * float32(q1)
				out[outIdx+32] = d * float32(int8(sc[is+2])) * float32(q2)
				out[outIdx+64] = d * float32(int8(sc[is+4])) * float32(q3)
				out[outIdx+96] = d * float32(int8(sc[is+6])) * float32(q4)
				outIdx++
			}
			outIdx += 96
			qlo = qlo[64:]
			qhi = qhi[32:]
			sc = sc[8:]
		}
	}
	return out, nil
}

func readTensorQ8KAsF32(r io.Reader, name string, count uint64) ([]float32, error) {
	const qk = 256
	if count%qk != 0 {
		return nil, fmt.Errorf("tensor %q q8_k element count=%d not divisible by %d", name, count, qk)
	}
	blocks := count / qk
	if blocks > uint64(math.MaxInt) {
		return nil, fmt.Errorf("tensor %q has too many q8_k blocks", name)
	}
	out := make([]float32, count)

	const blockSize = 4 + qk + (qk/16)*2
	buf := make([]byte, blockSize)
	for b := uint64(0); b < blocks; b++ {
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, fmt.Errorf("read tensor %q q8_k block %d: %w", name, b, err)
		}
		idx := 0
		d := math.Float32frombits(binary.LittleEndian.Uint32(buf[idx:]))
		idx += 4
		qs := buf[idx : idx+qk]
		idx += qk
		_ = buf[idx:] // bsums

		base := int(b * qk)
		for i := 0; i < qk; i++ {
			out[base+i] = d * float32(int8(qs[i]))
		}
	}
	return out, nil
}

func nibbleAt(qs []uint8, idx int) uint8 {
	b := qs[idx/2]
	if idx%2 == 0 {
		return b & 0x0f
	}
	return (b >> 4) & 0x0f
}

func getScaleMinK4(j int, q []uint8) (uint8, uint8) {
	if j < 4 {
		return q[j] & 63, q[j+4] & 63
	}
	d := (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
	m := (q[j+4] >> 4) | ((q[j-0] >> 6) << 4)
	return d, m
}

func q3KScales(src []byte) [16]int8 {
	const kmask1 = 0x03030303
	const kmask2 = 0x0f0f0f0f
	var aux [4]uint32
	aux[0] = binary.LittleEndian.Uint32(src[0:4])
	aux[1] = binary.LittleEndian.Uint32(src[4:8])
	aux[2] = binary.LittleEndian.Uint32(src[8:12])
	tmp := aux[2]
	aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
	aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
	aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4)
	aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4)

	var scales [16]int8
	for i := 0; i < 4; i++ {
		v := aux[i]
		scales[i*4+0] = int8(v & 0xFF)
		scales[i*4+1] = int8((v >> 8) & 0xFF)
		scales[i*4+2] = int8((v >> 16) & 0xFF)
		scales[i*4+3] = int8((v >> 24) & 0xFF)
	}
	return scales
}

func float16ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) & 0x1
	exp := int32((h >> 10) & 0x1f)
	mant := uint32(h & 0x03ff)

	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign << 31)
		}
		// Subnormal.
		for (mant & 0x0400) == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x03ff
	case 0x1f:
		bits := (sign << 31) | 0x7f800000 | (mant << 13)
		return math.Float32frombits(bits)
	}

	exp = exp + (127 - 15)
	bits := (sign << 31) | (uint32(exp) << 23) | (mant << 13)
	return math.Float32frombits(bits)
}
