package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

const (
	GGMLTypeF32      = 0
	GGMLTypeF16      = 1
	GGMLTypeQ4_0     = 2
	GGMLTypeQ4_1     = 3
	GGMLTypeQ5_0     = 6
	GGMLTypeQ5_1     = 7
	GGMLTypeQ8_0     = 8
	GGMLTypeQ8_1     = 9
	GGMLTypeQ2_K     = 10
	GGMLTypeQ3_K     = 11
	GGMLTypeQ4_K     = 12
	GGMLTypeQ5_K     = 13
	GGMLTypeQ6_K     = 14
	GGMLTypeQ8_K     = 15
	GGMLTypeIQ2_XXS  = 16
	GGMLTypeIQ2_XS   = 17
	GGMLTypeIQ3_XXS  = 18
	GGMLTypeIQ1_S    = 19
	GGMLTypeIQ4_NL   = 20
	GGMLTypeIQ3_S    = 21
	GGMLTypeIQ2_S    = 22
	GGMLTypeIQ4_XS   = 23
	GGMLTypeI8       = 24
	GGMLTypeI16      = 25
	GGMLTypeI32      = 26
	GGMLTypeI64      = 27
	GGMLTypeF64      = 28
	GGMLTypeIQ1_M    = 29
	GGMLTypeBF16     = 30
	GGMLTypeQ4_0_4_4 = 31
	GGMLTypeQ4_0_4_8 = 32
	GGMLTypeQ4_0_8_8 = 33
	GGMLTypeTQ1_0    = 34
	GGMLTypeTQ2_0    = 35
	GGMLTypeI2_S     = 36
	GGMLTypeI8_S     = 37
	GGMLTypeTL1      = 38
	GGMLTypeTL2      = 39
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
	case GGMLTypeTQ1_0:
		return readTensorTQ10AsF32(f, name, count)
	case GGMLTypeTQ2_0:
		return readTensorTQ20AsF32(f, name, count)
	case GGMLTypeI2_S:
		return readTensorI2SAsF32(f, name, count)
	case GGMLTypeIQ2_XXS:
		return readTensorIQ2XXSAsF32(f, name, count)
	case GGMLTypeIQ2_XS:
		return readTensorIQ2XSAsF32(f, name, count)
	case GGMLTypeIQ2_S:
		return readTensorIQ2SAsF32(f, name, count)
	case GGMLTypeIQ3_XXS:
		return readTensorIQ3XXSAsF32(f, name, count)
	case GGMLTypeIQ3_S:
		return readTensorIQ3SAsF32(f, name, count)
	case GGMLTypeIQ1_S:
		return readTensorIQ1SAsF32(f, name, count)
	case GGMLTypeIQ1_M:
		return readTensorIQ1MAsF32(f, name, count)
	case GGMLTypeIQ4_NL:
		return readTensorIQ4NLAsF32(f, name, count)
	case GGMLTypeIQ4_XS:
		return readTensorIQ4XSAsF32(f, name, count)
	default:
		return nil, fmt.Errorf("tensor %q type=%d not supported", name, t.Type)
	}
}

func ReadTensorI2SPacked(path string, info ModelInfo, name string) ([]byte, float32, uint64, error) {
	t, ok := info.TensorByName(name)
	if !ok {
		return nil, 0, 0, fmt.Errorf("tensor not found: %s", name)
	}
	if t.Type != GGMLTypeI2_S {
		return nil, 0, 0, fmt.Errorf("tensor %q type=%d is not i2_s", name, t.Type)
	}
	count, err := TensorElementCount(t)
	if err != nil {
		return nil, 0, 0, err
	}
	if count > uint64(math.MaxInt) {
		return nil, 0, 0, fmt.Errorf("tensor %q has too many i2_s elements", name)
	}

	f, err := os.Open(path)
	if err != nil {
		return nil, 0, 0, err
	}
	defer f.Close()

	start := info.TensorDataOffset + t.Offset
	if _, err := f.Seek(int64(start), io.SeekStart); err != nil {
		return nil, 0, 0, fmt.Errorf("seek tensor %q: %w", name, err)
	}

	const block = 128
	const blockBytes = 32
	packed := make([]byte, (count+block-1)/block*blockBytes)
	if _, err := io.ReadFull(f, packed); err != nil {
		return nil, 0, 0, fmt.Errorf("read tensor %q i2_s packed: %w", name, err)
	}
	var scale float32
	if err := binary.Read(f, binary.LittleEndian, &scale); err != nil {
		return nil, 0, 0, fmt.Errorf("read tensor %q i2_s scale: %w", name, err)
	}
	return packed, scale, count, nil
}

func readTensorTQ10AsF32(r io.Reader, name string, count uint64) ([]float32, error) {
	const qk = 256
	const qh = qk / 64
	const qs = (qk - 4*qk/64) / 5
	const blockSize = qs + qh + 2
	if count%qk != 0 {
		return nil, fmt.Errorf("tensor %q tq1_0 element count=%d not divisible by %d", name, count, qk)
	}
	blocks := count / qk
	if blocks > uint64(math.MaxInt) {
		return nil, fmt.Errorf("tensor %q has too many tq1_0 blocks", name)
	}
	out := make([]float32, count)
	buf := make([]byte, blockSize)
	pow3 := [6]uint16{1, 3, 9, 27, 81, 243}

	outIdx := 0
	for b := uint64(0); b < blocks; b++ {
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, fmt.Errorf("read tensor %q tq1_0 block %d: %w", name, b, err)
		}
		qsBytes := buf[:qs]
		qhBytes := buf[qs : qs+qh]
		d := binary.LittleEndian.Uint16(buf[qs+qh:])
		scale := float16ToFloat32(d)

		mainLen := len(qsBytes) - (len(qsBytes) % 32)
		for j := 0; j < mainLen; j += 32 {
			for n := 0; n < 5; n++ {
				p := pow3[n]
				for m := 0; m < 32; m++ {
					q := uint16(qsBytes[j+m]) * p
					xi := (q * 3) >> 8
					out[outIdx] = float32(int16(xi)-1) * scale
					outIdx++
				}
			}
		}
		for j := mainLen; j < len(qsBytes); j += 16 {
			for n := 0; n < 5; n++ {
				p := pow3[n]
				for m := 0; m < 16 && j+m < len(qsBytes); m++ {
					q := uint16(qsBytes[j+m]) * p
					xi := (q * 3) >> 8
					out[outIdx] = float32(int16(xi)-1) * scale
					outIdx++
				}
			}
		}

		for n := 0; n < 4; n++ {
			p := pow3[n]
			for j := 0; j < len(qhBytes); j++ {
				q := uint16(qhBytes[j]) * p
				xi := (q * 3) >> 8
				out[outIdx] = float32(int16(xi)-1) * scale
				outIdx++
			}
		}
	}
	return out, nil
}

func readTensorTQ20AsF32(r io.Reader, name string, count uint64) ([]float32, error) {
	const qk = 256
	const qs = qk / 4
	const blockSize = qs + 2
	if count%qk != 0 {
		return nil, fmt.Errorf("tensor %q tq2_0 element count=%d not divisible by %d", name, count, qk)
	}
	blocks := count / qk
	if blocks > uint64(math.MaxInt) {
		return nil, fmt.Errorf("tensor %q has too many tq2_0 blocks", name)
	}
	out := make([]float32, count)
	buf := make([]byte, blockSize)
	outIdx := 0
	for b := uint64(0); b < blocks; b++ {
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, fmt.Errorf("read tensor %q tq2_0 block %d: %w", name, b, err)
		}
		qsBytes := buf[:qs]
		d := binary.LittleEndian.Uint16(buf[qs:])
		scale := float16ToFloat32(d)
		for j := 0; j < len(qsBytes); j += 32 {
			for l := 0; l < 4; l++ {
				shift := uint(l * 2)
				for m := 0; m < 32; m++ {
					q := (qsBytes[j+m] >> shift) & 0x3
					out[outIdx] = float32(int8(q)-1) * scale
					outIdx++
				}
			}
		}
	}
	return out, nil
}

func readTensorI2SAsF32(r io.Reader, name string, count uint64) ([]float32, error) {
	if count > uint64(math.MaxInt) {
		return nil, fmt.Errorf("tensor %q has too many i2_s elements", name)
	}
	const block = 128
	const blockBytes = 32
	packed := make([]byte, (count+block-1)/block*blockBytes)
	if _, err := io.ReadFull(r, packed); err != nil {
		return nil, fmt.Errorf("read tensor %q i2_s packed: %w", name, err)
	}
	var scale float32
	if err := binary.Read(r, binary.LittleEndian, &scale); err != nil {
		return nil, fmt.Errorf("read tensor %q i2_s scale: %w", name, err)
	}

	out := make([]float32, count)
	const (
		v0 = -1.0
		v1 = 0.0
		v2 = 1.0
		v3 = 0.0
	)
	var done uint64
	for done < count {
		blkE := uint64(block)
		if count-done < blkE {
			blkE = count - done
		}
		cols0 := blkE
		if cols0 > 32 {
			cols0 = 32
		}
		cols1 := blkE
		if cols1 > 64 {
			cols1 = 32
		} else if cols1 > 32 {
			cols1 -= 32
		} else {
			cols1 = 0
		}
		cols2 := blkE
		if cols2 > 96 {
			cols2 = 32
		} else if cols2 > 64 {
			cols2 -= 64
		} else {
			cols2 = 0
		}
		cols3 := blkE
		if cols3 > 128 {
			cols3 = 32
		} else if cols3 > 96 {
			cols3 -= 96
		} else {
			cols3 = 0
		}
		base := int(done / block * blockBytes)
		for gp := uint64(0); gp < 32; gp++ {
			b := packed[base+int(gp)]
			c0 := (b >> 6) & 0x3
			c1 := (b >> 4) & 0x3
			c2 := (b >> 2) & 0x3
			c3 := b & 0x3
			if gp < cols0 {
				out[done+0*32+gp] = float32(mapI2S(c0, v0, v1, v2, v3)) * scale
			}
			if gp < cols1 {
				out[done+1*32+gp] = float32(mapI2S(c1, v0, v1, v2, v3)) * scale
			}
			if gp < cols2 {
				out[done+2*32+gp] = float32(mapI2S(c2, v0, v1, v2, v3)) * scale
			}
			if gp < cols3 {
				out[done+3*32+gp] = float32(mapI2S(c3, v0, v1, v2, v3)) * scale
			}
		}
		done += blkE
	}
	return out, nil
}

func mapI2S(v uint8, v0, v1, v2, v3 float32) float32 {
	switch v {
	case 0:
		return v0
	case 1:
		return v1
	case 2:
		return v2
	default:
		return v3
	}
}

func readTensorIQ2XXSAsF32(r io.Reader, name string, count uint64) ([]float32, error) {
	const qk = 256
	const qsBytes = qk / 8 * 2
	const blockSize = 2 + qsBytes
	if count%qk != 0 {
		return nil, fmt.Errorf("tensor %q iq2_xxs element count=%d not divisible by %d", name, count, qk)
	}
	blocks := count / qk
	if blocks > uint64(math.MaxInt) {
		return nil, fmt.Errorf("tensor %q has too many iq2_xxs blocks", name)
	}
	out := make([]float32, count)
	buf := make([]byte, blockSize)
	outIdx := 0
	for b := uint64(0); b < blocks; b++ {
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, fmt.Errorf("read tensor %q iq2_xxs block %d: %w", name, b, err)
		}
		d := binary.LittleEndian.Uint16(buf[:2])
		scale := float16ToFloat32(d)
		qs := buf[2:]
		for ib32 := 0; ib32 < qk/32; ib32++ {
			off := 4 * ib32
			aux1 := binary.LittleEndian.Uint32(qs[off+4 : off+8])
			db := scale * (0.5 + float32(aux1>>28)) * 0.25
			for l := 0; l < 4; l++ {
				gridIdx := int(qs[off+l])
				signs := ksigns_iq2xs[(aux1>>(7*l))&127]
				grid := iq2xxs_grid[gridIdx]
				for j := 0; j < 8; j++ {
					g := float32(uint8(grid >> (8 * j)))
					if signs&kmask_iq2xs[j] != 0 {
						g = -g
					}
					out[outIdx] = db * g
					outIdx++
				}
			}
		}
	}
	return out, nil
}

func readTensorIQ2XSAsF32(r io.Reader, name string, count uint64) ([]float32, error) {
	const qk = 256
	const qsBytes = qk / 8 * 2
	const scalesBytes = qk / 32
	const blockSize = 2 + qsBytes + scalesBytes
	if count%qk != 0 {
		return nil, fmt.Errorf("tensor %q iq2_xs element count=%d not divisible by %d", name, count, qk)
	}
	blocks := count / qk
	if blocks > uint64(math.MaxInt) {
		return nil, fmt.Errorf("tensor %q has too many iq2_xs blocks", name)
	}
	out := make([]float32, count)
	buf := make([]byte, blockSize)
	outIdx := 0
	for b := uint64(0); b < blocks; b++ {
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, fmt.Errorf("read tensor %q iq2_xs block %d: %w", name, b, err)
		}
		d := binary.LittleEndian.Uint16(buf[:2])
		scale := float16ToFloat32(d)
		qsRaw := buf[2 : 2+qsBytes]
		scales := buf[2+qsBytes:]
		qs := make([]uint16, qk/8)
		for i := range qs {
			qs[i] = binary.LittleEndian.Uint16(qsRaw[i*2:])
		}
		for ib32 := 0; ib32 < qk/32; ib32++ {
			db0 := scale * (0.5 + float32(scales[ib32]&0x0f)) * 0.25
			db1 := scale * (0.5 + float32(scales[ib32]>>4)) * 0.25
			for l := 0; l < 4; l++ {
				q := qs[4*ib32+l]
				gridIdx := int(q & 0x1ff)
				signs := ksigns_iq2xs[q>>9]
				grid := iq2xs_grid[gridIdx]
				db := db0
				if l >= 2 {
					db = db1
				}
				for j := 0; j < 8; j++ {
					g := float32(uint8(grid >> (8 * j)))
					if signs&kmask_iq2xs[j] != 0 {
						g = -g
					}
					out[outIdx] = db * g
					outIdx++
				}
			}
		}
	}
	return out, nil
}

func readTensorIQ2SAsF32(r io.Reader, name string, count uint64) ([]float32, error) {
	const qk = 256
	const qsBytes = qk / 4
	const qhBytes = qk / 32
	const scalesBytes = qk / 32
	const blockSize = 2 + qsBytes + qhBytes + scalesBytes
	if count%qk != 0 {
		return nil, fmt.Errorf("tensor %q iq2_s element count=%d not divisible by %d", name, count, qk)
	}
	blocks := count / qk
	if blocks > uint64(math.MaxInt) {
		return nil, fmt.Errorf("tensor %q has too many iq2_s blocks", name)
	}
	out := make([]float32, count)
	buf := make([]byte, blockSize)
	outIdx := 0
	for b := uint64(0); b < blocks; b++ {
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, fmt.Errorf("read tensor %q iq2_s block %d: %w", name, b, err)
		}
		d := binary.LittleEndian.Uint16(buf[:2])
		scale := float16ToFloat32(d)
		qs := buf[2 : 2+qsBytes]
		qh := buf[2+qsBytes : 2+qsBytes+qhBytes]
		scales := buf[2+qsBytes+qhBytes:]
		signs := qs[qk/8:]
		qsIdx := 0
		signIdx := 0
		for ib32 := 0; ib32 < qk/32; ib32++ {
			db0 := scale * (0.5 + float32(scales[ib32]&0x0f)) * 0.25
			db1 := scale * (0.5 + float32(scales[ib32]>>4)) * 0.25
			for l := 0; l < 4; l++ {
				dl := db0
				if l >= 2 {
					dl = db1
				}
				gridIdx := int(qs[qsIdx+l]) | (int(qh[ib32]) << (8 - 2*l) & 0x300)
				grid := iq2s_grid[gridIdx]
				sign := signs[signIdx+l]
				for j := 0; j < 8; j++ {
					g := float32(uint8(grid >> (8 * j)))
					if sign&kmask_iq2xs[j] != 0 {
						g = -g
					}
					out[outIdx] = dl * g
					outIdx++
				}
			}
			qsIdx += 4
			signIdx += 4
		}
	}
	return out, nil
}

func readTensorIQ3XXSAsF32(r io.Reader, name string, count uint64) ([]float32, error) {
	const qk = 256
	const qsBytes = 3 * qk / 8
	const blockSize = 2 + qsBytes
	if count%qk != 0 {
		return nil, fmt.Errorf("tensor %q iq3_xxs element count=%d not divisible by %d", name, count, qk)
	}
	blocks := count / qk
	if blocks > uint64(math.MaxInt) {
		return nil, fmt.Errorf("tensor %q has too many iq3_xxs blocks", name)
	}
	out := make([]float32, count)
	buf := make([]byte, blockSize)
	outIdx := 0
	for b := uint64(0); b < blocks; b++ {
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, fmt.Errorf("read tensor %q iq3_xxs block %d: %w", name, b, err)
		}
		d := binary.LittleEndian.Uint16(buf[:2])
		scale := float16ToFloat32(d)
		qs := buf[2:]
		scales := qs[qk/4:]
		qsIdx := 0
		for ib32 := 0; ib32 < qk/32; ib32++ {
			aux := binary.LittleEndian.Uint32(scales[4*ib32:])
			db := scale * (0.5 + float32(aux>>28)) * 0.5
			for l := 0; l < 4; l++ {
				signs := ksigns_iq2xs[(aux>>(7*l))&127]
				grid1 := iq3xxs_grid[int(qs[qsIdx+2*l])]
				grid2 := iq3xxs_grid[int(qs[qsIdx+2*l+1])]
				base := outIdx
				for j := 0; j < 4; j++ {
					g0 := float32(uint8(grid1 >> (8 * j)))
					if signs&kmask_iq2xs[j] != 0 {
						g0 = -g0
					}
					out[base+j] = db * g0
					g1 := float32(uint8(grid2 >> (8 * j)))
					if signs&kmask_iq2xs[j+4] != 0 {
						g1 = -g1
					}
					out[base+4+j] = db * g1
				}
				outIdx += 8
			}
			qsIdx += 8
		}
	}
	return out, nil
}

func readTensorIQ3SAsF32(r io.Reader, name string, count uint64) ([]float32, error) {
	const qk = 256
	const qsBytes = qk / 4
	const qhBytes = qk / 32
	const signsBytes = qk / 8
	const scalesBytes = qk / 64
	const blockSize = 2 + qsBytes + qhBytes + signsBytes + scalesBytes
	if count%qk != 0 {
		return nil, fmt.Errorf("tensor %q iq3_s element count=%d not divisible by %d", name, count, qk)
	}
	blocks := count / qk
	if blocks > uint64(math.MaxInt) {
		return nil, fmt.Errorf("tensor %q has too many iq3_s blocks", name)
	}
	out := make([]float32, count)
	buf := make([]byte, blockSize)
	outIdx := 0
	for b := uint64(0); b < blocks; b++ {
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, fmt.Errorf("read tensor %q iq3_s block %d: %w", name, b, err)
		}
		d := binary.LittleEndian.Uint16(buf[:2])
		scale := float16ToFloat32(d)
		qs := buf[2 : 2+qsBytes]
		qh := buf[2+qsBytes : 2+qsBytes+qhBytes]
		signs := buf[2+qsBytes+qhBytes : 2+qsBytes+qhBytes+signsBytes]
		scales := buf[2+qsBytes+qhBytes+signsBytes:]
		qsIdx := 0
		signIdx := 0
		qhIdx := 0
		for ib32 := 0; ib32 < qk/32; ib32 += 2 {
			db1 := scale * float32(1+2*int(scales[ib32/2]&0x0f))
			db2 := scale * float32(1+2*int(scales[ib32/2]>>4))
			for l := 0; l < 4; l++ {
				sign := signs[signIdx+l]
				grid1 := iq3s_grid[int(qs[qsIdx+2*l])|((int(qh[qhIdx+0])<<(8-2*l))&0x100)]
				grid2 := iq3s_grid[int(qs[qsIdx+2*l+1])|((int(qh[qhIdx+0])<<(7-2*l))&0x100)]
				base := outIdx
				for j := 0; j < 4; j++ {
					g0 := float32(uint8(grid1 >> (8 * j)))
					if sign&kmask_iq2xs[j] != 0 {
						g0 = -g0
					}
					out[base+j] = db1 * g0
					g1 := float32(uint8(grid2 >> (8 * j)))
					if sign&kmask_iq2xs[j+4] != 0 {
						g1 = -g1
					}
					out[base+4+j] = db1 * g1
				}
				outIdx += 8
			}
			qsIdx += 8
			signIdx += 4
			for l := 0; l < 4; l++ {
				sign := signs[signIdx+l]
				grid1 := iq3s_grid[int(qs[qsIdx+2*l])|((int(qh[qhIdx+1])<<(8-2*l))&0x100)]
				grid2 := iq3s_grid[int(qs[qsIdx+2*l+1])|((int(qh[qhIdx+1])<<(7-2*l))&0x100)]
				base := outIdx
				for j := 0; j < 4; j++ {
					g0 := float32(uint8(grid1 >> (8 * j)))
					if sign&kmask_iq2xs[j] != 0 {
						g0 = -g0
					}
					out[base+j] = db2 * g0
					g1 := float32(uint8(grid2 >> (8 * j)))
					if sign&kmask_iq2xs[j+4] != 0 {
						g1 = -g1
					}
					out[base+4+j] = db2 * g1
				}
				outIdx += 8
			}
			qhIdx += 2
			qsIdx += 8
			signIdx += 4
		}
	}
	return out, nil
}

func readTensorIQ1SAsF32(r io.Reader, name string, count uint64) ([]float32, error) {
	const qk = 256
	const qsBytes = qk / 8
	const qhBytes = qk / 32 * 2
	const blockSize = 2 + qsBytes + qhBytes
	if count%qk != 0 {
		return nil, fmt.Errorf("tensor %q iq1_s element count=%d not divisible by %d", name, count, qk)
	}
	blocks := count / qk
	if blocks > uint64(math.MaxInt) {
		return nil, fmt.Errorf("tensor %q has too many iq1_s blocks", name)
	}
	out := make([]float32, count)
	buf := make([]byte, blockSize)
	outIdx := 0
	for b := uint64(0); b < blocks; b++ {
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, fmt.Errorf("read tensor %q iq1_s block %d: %w", name, b, err)
		}
		d := binary.LittleEndian.Uint16(buf[:2])
		scale := float16ToFloat32(d)
		qs := buf[2 : 2+qsBytes]
		qhRaw := buf[2+qsBytes:]
		qh := make([]uint16, qk/32)
		for i := range qh {
			qh[i] = binary.LittleEndian.Uint16(qhRaw[i*2:])
		}
		qsIdx := 0
		for ib := 0; ib < qk/32; ib++ {
			dl := scale * float32(2*int((qh[ib]>>12)&7)+1)
			delta := iq1sDelta
			if qh[ib]&0x8000 != 0 {
				delta = -iq1sDelta
			}
			for l := 0; l < 4; l++ {
				gridIdx := int(qs[qsIdx+l]) | (int((qh[ib]>>(3*l))&7) << 8)
				grid := iq1s_grid[gridIdx]
				for j := 0; j < 8; j++ {
					g := int8(grid >> (8 * j))
					out[outIdx] = dl * (float32(g) + delta)
					outIdx++
				}
			}
			qsIdx += 4
		}
	}
	return out, nil
}

func readTensorIQ1MAsF32(r io.Reader, name string, count uint64) ([]float32, error) {
	const qk = 256
	const qsBytes = qk / 8
	const qhBytes = qk / 16
	const scalesBytes = qk / 32
	const blockSize = qsBytes + qhBytes + scalesBytes
	if count%qk != 0 {
		return nil, fmt.Errorf("tensor %q iq1_m element count=%d not divisible by %d", name, count, qk)
	}
	blocks := count / qk
	if blocks > uint64(math.MaxInt) {
		return nil, fmt.Errorf("tensor %q has too many iq1_m blocks", name)
	}
	out := make([]float32, count)
	buf := make([]byte, blockSize)
	outIdx := 0
	for b := uint64(0); b < blocks; b++ {
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, fmt.Errorf("read tensor %q iq1_m block %d: %w", name, b, err)
		}
		qs := buf[:qsBytes]
		qh := buf[qsBytes : qsBytes+qhBytes]
		scalesRaw := buf[qsBytes+qhBytes:]
		sc := make([]uint16, qk/64)
		for i := range sc {
			sc[i] = binary.LittleEndian.Uint16(scalesRaw[i*2:])
		}
		scaleU16 := uint16((sc[0]>>12)&0x000f | (sc[1]>>8)&0x00f0 | (sc[2]>>4)&0x0f00 | (sc[3] & 0xf000))
		scale := float16ToFloat32(scaleU16)
		qsIdx := 0
		qhIdx := 0
		for ib := 0; ib < qk/32; ib++ {
			dl1 := scale * float32(2*int((sc[ib/2]>>(6*(ib%2)+0))&0x7)+1)
			dl2 := scale * float32(2*int((sc[ib/2]>>(6*(ib%2)+3))&0x7)+1)
			idx0 := int(qs[qsIdx+0]) | (int(qh[qhIdx+0])<<8)&0x700
			idx1 := int(qs[qsIdx+1]) | (int(qh[qhIdx+0])<<4)&0x700
			idx2 := int(qs[qsIdx+2]) | (int(qh[qhIdx+1])<<8)&0x700
			idx3 := int(qs[qsIdx+3]) | (int(qh[qhIdx+1])<<4)&0x700
			delta0 := iq1sDelta
			if qh[qhIdx+0]&0x08 != 0 {
				delta0 = -iq1sDelta
			}
			delta1 := iq1sDelta
			if qh[qhIdx+0]&0x80 != 0 {
				delta1 = -iq1sDelta
			}
			delta2 := iq1sDelta
			if qh[qhIdx+1]&0x08 != 0 {
				delta2 = -iq1sDelta
			}
			delta3 := iq1sDelta
			if qh[qhIdx+1]&0x80 != 0 {
				delta3 = -iq1sDelta
			}
			grid := iq1s_grid[idx0]
			for j := 0; j < 8; j++ {
				g := int8(grid >> (8 * j))
				out[outIdx] = dl1 * (float32(g) + delta0)
				outIdx++
			}
			grid = iq1s_grid[idx1]
			for j := 0; j < 8; j++ {
				g := int8(grid >> (8 * j))
				out[outIdx] = dl1 * (float32(g) + delta1)
				outIdx++
			}
			grid = iq1s_grid[idx2]
			for j := 0; j < 8; j++ {
				g := int8(grid >> (8 * j))
				out[outIdx] = dl2 * (float32(g) + delta2)
				outIdx++
			}
			grid = iq1s_grid[idx3]
			for j := 0; j < 8; j++ {
				g := int8(grid >> (8 * j))
				out[outIdx] = dl2 * (float32(g) + delta3)
				outIdx++
			}
			qsIdx += 4
			qhIdx += 2
		}
	}
	return out, nil
}

func readTensorIQ4NLAsF32(r io.Reader, name string, count uint64) ([]float32, error) {
	const qk = 32
	const qsBytes = qk / 2
	const blockSize = 2 + qsBytes
	if count%qk != 0 {
		return nil, fmt.Errorf("tensor %q iq4_nl element count=%d not divisible by %d", name, count, qk)
	}
	blocks := count / qk
	if blocks > uint64(math.MaxInt) {
		return nil, fmt.Errorf("tensor %q has too many iq4_nl blocks", name)
	}
	out := make([]float32, count)
	buf := make([]byte, blockSize)
	outIdx := 0
	for b := uint64(0); b < blocks; b++ {
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, fmt.Errorf("read tensor %q iq4_nl block %d: %w", name, b, err)
		}
		d := binary.LittleEndian.Uint16(buf[:2])
		scale := float16ToFloat32(d)
		qs := buf[2:]
		for j := 0; j < qsBytes; j++ {
			q := qs[j]
			out[outIdx+j] = scale * float32(kvaluesIQ4NL[q&0x0f])
			out[outIdx+j+qsBytes] = scale * float32(kvaluesIQ4NL[q>>4])
		}
		outIdx += qk
	}
	return out, nil
}

func readTensorIQ4XSAsF32(r io.Reader, name string, count uint64) ([]float32, error) {
	const qk = 256
	const qsBytes = qk / 2
	const scalesLBytes = qk / 64
	const blockSize = 2 + 2 + scalesLBytes + qsBytes
	if count%qk != 0 {
		return nil, fmt.Errorf("tensor %q iq4_xs element count=%d not divisible by %d", name, count, qk)
	}
	blocks := count / qk
	if blocks > uint64(math.MaxInt) {
		return nil, fmt.Errorf("tensor %q has too many iq4_xs blocks", name)
	}
	out := make([]float32, count)
	buf := make([]byte, blockSize)
	outIdx := 0
	for b := uint64(0); b < blocks; b++ {
		if _, err := io.ReadFull(r, buf); err != nil {
			return nil, fmt.Errorf("read tensor %q iq4_xs block %d: %w", name, b, err)
		}
		d := binary.LittleEndian.Uint16(buf[:2])
		scale := float16ToFloat32(d)
		scalesH := binary.LittleEndian.Uint16(buf[2:4])
		scalesL := buf[4 : 4+scalesLBytes]
		qs := buf[4+scalesLBytes:]
		qsIdx := 0
		for ib := 0; ib < qk/32; ib++ {
			ls := int((scalesL[ib/2]>>(4*(ib%2)))&0x0f) | int(((scalesH>>(2*ib))&0x3)<<4)
			dl := scale * float32(ls-32)
			for j := 0; j < 16; j++ {
				q := qs[qsIdx+j]
				out[outIdx+j] = dl * float32(kvaluesIQ4NL[q&0x0f])
				out[outIdx+j+16] = dl * float32(kvaluesIQ4NL[q>>4])
			}
			outIdx += 32
			qsIdx += 16
		}
	}
	return out, nil
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
