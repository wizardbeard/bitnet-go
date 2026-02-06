package runtime

import (
	"bytes"
	"context"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func TestRunForwardStubDeterministic(t *testing.T) {
	a := make([]int32, 8)
	b := make([]int32, 8)
	runForwardStub(32000, 42, []int32{1, 2, 3}, a, nil)
	runForwardStub(32000, 42, []int32{1, 2, 3}, b, nil)
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("token[%d] mismatch: %d vs %d", i, a[i], b[i])
		}
	}
}

func TestRunForwardStubPromptAffectsOutput(t *testing.T) {
	a := make([]int32, 8)
	b := make([]int32, 8)
	runForwardStub(32000, 42, []int32{1, 2, 3}, a, nil)
	runForwardStub(32000, 42, []int32{1, 2, 4}, b, nil)
	same := true
	for i := range a {
		if a[i] != b[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatalf("different prompts should change output, got same tokens: %v", a)
	}
}

func TestGenerateUsesTensorBlockWhenPresent(t *testing.T) {
	modelPath := buildTensorBlockModel(t)

	rt, err := New(context.Background(), modelPath)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if rt.block == nil {
		t.Fatal("expected tensor block to be loaded")
	}

	got, err := rt.Generate(context.Background(), GenerateRequest{
		Prompt:    "hello",
		Seed:      7,
		MaxTokens: 8,
	})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if len(got.TokenIDs) != 8 {
		t.Fatalf("len(TokenIDs) = %d, want 8", len(got.TokenIDs))
	}
	for i, tok := range got.TokenIDs {
		if tok < 0 || tok >= int32(rt.block.vocabDim) {
			t.Fatalf("token[%d] = %d out of tensor-block vocab range [0,%d)", i, tok, rt.block.vocabDim)
		}
	}
}

func TestRunForwardTensorBlockDeterministic(t *testing.T) {
	block := &tensorBlock{
		mode:      tensorBlockModeProjection,
		hiddenDim: 4,
		vocabDim:  6,
		stateProj: []float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		},
		logitsProj: []float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
			0.5, 0.5, 0.5, 0.5,
			-1, -1, -1, -1,
		},
	}
	a := make([]int32, 8)
	b := make([]int32, 8)
	runForwardTensorBlock(block, 123, []int32{9, 10, 11}, a, nil)
	runForwardTensorBlock(block, 123, []int32{9, 10, 11}, b, nil)
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("token[%d] mismatch: %d vs %d", i, a[i], b[i])
		}
	}
}

func TestGenerateUsesLlamaEmbeddingOutputBlock(t *testing.T) {
	modelPath := buildLlamaEmbeddingOutputModel(t)

	rt, err := New(context.Background(), modelPath)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if rt.block == nil {
		t.Fatal("expected tensor block to be loaded")
	}
	if rt.block.mode != tensorBlockModeEmbeddingOutput {
		t.Fatalf("block mode = %d, want embedding/output", rt.block.mode)
	}

	a, err := rt.Generate(context.Background(), GenerateRequest{
		Prompt:    "hello",
		Seed:      5,
		MaxTokens: 6,
	})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	b, err := rt.Generate(context.Background(), GenerateRequest{
		Prompt:    "hello",
		Seed:      5,
		MaxTokens: 6,
	})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	for i := range a.TokenIDs {
		if a.TokenIDs[i] != b.TokenIDs[i] {
			t.Fatalf("token[%d] mismatch: %d vs %d", i, a.TokenIDs[i], b.TokenIDs[i])
		}
	}
}

func TestGenerateUsesLlamaStackWhenPresent(t *testing.T) {
	modelPath := buildLlamaBlock0Model(t)

	rt, err := New(context.Background(), modelPath)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if rt.block == nil {
		t.Fatal("expected tensor block to be loaded")
	}
	if rt.block.mode != tensorBlockModeLlamaStack {
		t.Fatalf("block mode = %d, want llama stack", rt.block.mode)
	}
	if rt.block.attnHeads != 2 {
		t.Fatalf("attnHeads = %d, want 2", rt.block.attnHeads)
	}
	if rt.block.kvHeads != 1 {
		t.Fatalf("kvHeads = %d, want 1", rt.block.kvHeads)
	}
	if rt.block.ropeScalingType != "linear" {
		t.Fatalf("ropeScalingType = %q, want linear", rt.block.ropeScalingType)
	}
	if rt.block.ropeScale != 2 {
		t.Fatalf("ropeScale = %f, want 2", rt.block.ropeScale)
	}
	if rt.block.ropeDim != 2 {
		t.Fatalf("ropeDim = %d, want 2", rt.block.ropeDim)
	}
	if len(rt.block.layers) != 2 {
		t.Fatalf("len(layers) = %d, want 2", len(rt.block.layers))
	}

	a, err := rt.Generate(context.Background(), GenerateRequest{
		Prompt:    "hello",
		Seed:      5,
		MaxTokens: 4,
	})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	b, err := rt.Generate(context.Background(), GenerateRequest{
		Prompt:    "hello",
		Seed:      5,
		MaxTokens: 4,
	})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	for i := range a.TokenIDs {
		if a.TokenIDs[i] != b.TokenIDs[i] {
			t.Fatalf("token[%d] mismatch: %d vs %d", i, a.TokenIDs[i], b.TokenIDs[i])
		}
	}

	pa := make([]int32, 4)
	pb := make([]int32, 4)
	runForwardLlamaStack(rt.block, 5, []int32{1, 2, 3}, pa, nil)
	runForwardLlamaStack(rt.block, 5, []int32{1, 2, 4}, pb, nil)
	same := true
	for i := range pa {
		if pa[i] != pb[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatalf("different prompt token streams should affect llama block0 output: %v vs %v", pa, pb)
	}
}

func buildTensorBlockModel(t *testing.T) string {
	t.Helper()

	const (
		valueTypeUint32 = 4
		ggmlTypeF32     = 0
		alignBytes      = 32
	)

	buf := bytes.NewBuffer(nil)
	rtWriteString(t, buf, "GGUF")
	rtWriteU32(t, buf, 3) // version
	rtWriteU64(t, buf, 2) // tensor count
	rtWriteU64(t, buf, 1) // kv count

	rtWriteGGUFString(t, buf, "general.alignment")
	rtWriteU32(t, buf, valueTypeUint32)
	rtWriteU32(t, buf, alignBytes)

	rtWriteGGUFString(t, buf, "bitnet_go.state_proj")
	rtWriteU32(t, buf, 2) // n_dims
	rtWriteU64(t, buf, 4)
	rtWriteU64(t, buf, 4)
	rtWriteU32(t, buf, ggmlTypeF32)
	rtWriteU64(t, buf, 0) // tensor offset in data region

	rtWriteGGUFString(t, buf, "bitnet_go.logits_proj")
	rtWriteU32(t, buf, 2) // n_dims
	rtWriteU64(t, buf, 4)
	rtWriteU64(t, buf, 8)
	rtWriteU32(t, buf, ggmlTypeF32)
	rtWriteU64(t, buf, 4*4*4) // after state_proj payload

	rtPadTo(t, buf, alignBytes)

	// state_proj (4x4 identity), GGML column-major (ne0=rows).
	for c := 0; c < 4; c++ {
		for r := 0; r < 4; r++ {
			v := float32(0)
			if r == c {
				v = 1
			}
			rtWriteF32(t, buf, v)
		}
	}

	// logits_proj (4x8): each output column is weighted by one hidden channel.
	// Stored in GGML column-major (ne0=rows).
	for c := 0; c < 8; c++ {
		for r := 0; r < 4; r++ {
			v := float32(0)
			if c%4 == r {
				v = float32(c + 1)
			}
			rtWriteF32(t, buf, v)
		}
	}

	path := filepath.Join(t.TempDir(), "tensor_block.gguf")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}
	return path
}

func buildLlamaEmbeddingOutputModel(t *testing.T) string {
	t.Helper()

	const (
		valueTypeUint32 = 4
		ggmlTypeF32     = 0
		alignBytes      = 32
		hidden          = 4
		vocab           = 8
	)

	buf := bytes.NewBuffer(nil)
	rtWriteString(t, buf, "GGUF")
	rtWriteU32(t, buf, 3) // version
	rtWriteU64(t, buf, 2) // tensor count
	rtWriteU64(t, buf, 1) // kv count

	rtWriteGGUFString(t, buf, "general.alignment")
	rtWriteU32(t, buf, valueTypeUint32)
	rtWriteU32(t, buf, alignBytes)

	rtWriteGGUFString(t, buf, "token_embd.weight")
	rtWriteU32(t, buf, 2) // n_dims
	rtWriteU64(t, buf, hidden)
	rtWriteU64(t, buf, vocab)
	rtWriteU32(t, buf, ggmlTypeF32)
	rtWriteU64(t, buf, 0)

	rtWriteGGUFString(t, buf, "output.weight")
	rtWriteU32(t, buf, 2) // n_dims
	rtWriteU64(t, buf, hidden)
	rtWriteU64(t, buf, vocab)
	rtWriteU32(t, buf, ggmlTypeF32)
	rtWriteU64(t, buf, hidden*vocab*4)

	rtPadTo(t, buf, alignBytes)

	// token_embd: each token column has a dominant hidden channel.
	// Stored in GGML column-major (ne0=rows).
	for c := 0; c < vocab; c++ {
		for r := 0; r < hidden; r++ {
			v := float32(0.1)
			if c%hidden == r {
				v = 1.0 + float32(c)/10.0
			}
			rtWriteF32(t, buf, v)
		}
	}

	// output.weight mirrors the same orientation for easy deterministic argmax.
	// Stored in GGML column-major (ne0=rows).
	for c := 0; c < vocab; c++ {
		for r := 0; r < hidden; r++ {
			v := float32(0.05)
			if c%hidden == r {
				v = float32(c + 1)
			}
			rtWriteF32(t, buf, v)
		}
	}

	path := filepath.Join(t.TempDir(), "llama_embed_output.gguf")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}
	return path
}

func buildLlamaBlock0Model(t *testing.T) string {
	t.Helper()

	const (
		valueTypeUint32  = 4
		valueTypeFloat32 = 6
		ggmlTypeF32      = 0
		alignBytes       = 32
		hidden           = 4
		vocab            = 8
		ffn              = 6
	)

	type tensorSpec struct {
		name string
		dims []uint64
		data []float32
	}

	makeIdentity := func(n int, scale float32) []float32 {
		out := make([]float32, n*n)
		for i := 0; i < n; i++ {
			out[i+n*i] = scale
		}
		return out
	}

	tensors := []tensorSpec{
		{
			name: "token_embd.weight",
			dims: []uint64{hidden, vocab},
			data: func() []float32 {
				out := make([]float32, hidden*vocab)
				for c := 0; c < vocab; c++ {
					for r := 0; r < hidden; r++ {
						v := float32(0.1)
						if c%hidden == r {
							v = 1 + float32(c)/8.0
						}
						out[r+hidden*c] = v
					}
				}
				return out
			}(),
		},
		{
			name: "output_norm.weight",
			dims: []uint64{hidden},
			data: []float32{1, 1, 1, 1},
		},
		{
			name: "output.weight",
			dims: []uint64{hidden, vocab},
			data: func() []float32 {
				out := make([]float32, hidden*vocab)
				for c := 0; c < vocab; c++ {
					for r := 0; r < hidden; r++ {
						v := float32(0.05)
						if c%hidden == r {
							v = float32(c + 1)
						}
						out[r+hidden*c] = v
					}
				}
				return out
			}(),
		},
		{name: "blk.0.attn_q.weight", dims: []uint64{hidden, hidden}, data: makeIdentity(hidden, 1)},
		{
			name: "blk.0.attn_k.weight",
			dims: []uint64{2, hidden},
			data: []float32{
				1, 0,
				0, 1,
				0, 0,
				0, 0,
			},
		},
		{
			name: "blk.0.attn_v.weight",
			dims: []uint64{2, hidden},
			data: []float32{
				1, 0,
				0, 1,
				0, 0,
				0, 0,
			},
		},
		{name: "blk.0.attn_output.weight", dims: []uint64{hidden, hidden}, data: makeIdentity(hidden, 0.5)},
		{name: "blk.0.attn_norm.weight", dims: []uint64{hidden}, data: []float32{1, 1, 1, 1}},
		{
			name: "blk.0.ffn_gate.weight",
			dims: []uint64{ffn, hidden},
			data: func() []float32 {
				out := make([]float32, ffn*hidden)
				for i := 0; i < ffn; i++ {
					for j := 0; j < hidden; j++ {
						out[i+ffn*j] = float32((i+j)%3+1) * 0.1
					}
				}
				return out
			}(),
		},
		{
			name: "blk.0.ffn_up.weight",
			dims: []uint64{ffn, hidden},
			data: func() []float32 {
				out := make([]float32, ffn*hidden)
				for i := 0; i < ffn; i++ {
					for j := 0; j < hidden; j++ {
						out[i+ffn*j] = float32((i*j)%4+1) * 0.08
					}
				}
				return out
			}(),
		},
		{
			name: "blk.0.ffn_down.weight",
			dims: []uint64{hidden, ffn},
			data: func() []float32 {
				out := make([]float32, hidden*ffn)
				for i := 0; i < hidden; i++ {
					for j := 0; j < ffn; j++ {
						out[i+hidden*j] = float32((i+j)%2+1) * 0.07
					}
				}
				return out
			}(),
		},
		{name: "blk.0.ffn_norm.weight", dims: []uint64{hidden}, data: []float32{1, 1, 1, 1}},
		{name: "blk.1.attn_q.weight", dims: []uint64{hidden, hidden}, data: makeIdentity(hidden, 0.9)},
		{
			name: "blk.1.attn_k.weight",
			dims: []uint64{2, hidden},
			data: []float32{
				0.9, 0,
				0, 0.9,
				0, 0,
				0, 0,
			},
		},
		{
			name: "blk.1.attn_v.weight",
			dims: []uint64{2, hidden},
			data: []float32{
				0.9, 0,
				0, 0.9,
				0, 0,
				0, 0,
			},
		},
		{name: "blk.1.attn_output.weight", dims: []uint64{hidden, hidden}, data: makeIdentity(hidden, 0.4)},
		{name: "blk.1.attn_norm.weight", dims: []uint64{hidden}, data: []float32{1, 1, 1, 1}},
		{
			name: "blk.1.ffn_gate.weight",
			dims: []uint64{ffn, hidden},
			data: func() []float32 {
				out := make([]float32, ffn*hidden)
				for i := 0; i < ffn; i++ {
					for j := 0; j < hidden; j++ {
						out[i+ffn*j] = float32((i+j)%4+1) * 0.09
					}
				}
				return out
			}(),
		},
		{
			name: "blk.1.ffn_up.weight",
			dims: []uint64{ffn, hidden},
			data: func() []float32 {
				out := make([]float32, ffn*hidden)
				for i := 0; i < ffn; i++ {
					for j := 0; j < hidden; j++ {
						out[i+ffn*j] = float32((i*j)%5+1) * 0.06
					}
				}
				return out
			}(),
		},
		{
			name: "blk.1.ffn_down.weight",
			dims: []uint64{hidden, ffn},
			data: func() []float32 {
				out := make([]float32, hidden*ffn)
				for i := 0; i < hidden; i++ {
					for j := 0; j < ffn; j++ {
						out[i+hidden*j] = float32((i+j)%3+1) * 0.05
					}
				}
				return out
			}(),
		},
		{name: "blk.1.ffn_norm.weight", dims: []uint64{hidden}, data: []float32{1, 1, 1, 1}},
	}

	buf := bytes.NewBuffer(nil)
	rtWriteString(t, buf, "GGUF")
	rtWriteU32(t, buf, 3)
	rtWriteU64(t, buf, uint64(len(tensors)))
	rtWriteU64(t, buf, 8) // alignment + rms eps + head_count + kv_head_count + rope base + rope scaling + rope dim

	rtWriteGGUFString(t, buf, "general.alignment")
	rtWriteU32(t, buf, valueTypeUint32)
	rtWriteU32(t, buf, alignBytes)

	rtWriteGGUFString(t, buf, "llama.attention.layer_norm_rms_epsilon")
	rtWriteU32(t, buf, valueTypeFloat32)
	rtWriteF32(t, buf, 1e-5)

	rtWriteGGUFString(t, buf, "llama.attention.head_count")
	rtWriteU32(t, buf, valueTypeUint32)
	rtWriteU32(t, buf, 2)

	rtWriteGGUFString(t, buf, "llama.attention.head_count_kv")
	rtWriteU32(t, buf, valueTypeUint32)
	rtWriteU32(t, buf, 1)

	rtWriteGGUFString(t, buf, "llama.rope.freq_base")
	rtWriteU32(t, buf, valueTypeFloat32)
	rtWriteF32(t, buf, 10000)

	rtWriteGGUFString(t, buf, "llama.rope.scaling.type")
	rtWriteU32(t, buf, 8) // string
	rtWriteGGUFString(t, buf, "linear")

	rtWriteGGUFString(t, buf, "llama.rope.scaling.factor")
	rtWriteU32(t, buf, valueTypeFloat32)
	rtWriteF32(t, buf, 2.0)

	rtWriteGGUFString(t, buf, "llama.rope.dimension_count")
	rtWriteU32(t, buf, valueTypeUint32)
	rtWriteU32(t, buf, 2)

	offset := uint64(0)
	for _, ts := range tensors {
		rtWriteGGUFString(t, buf, ts.name)
		rtWriteU32(t, buf, uint32(len(ts.dims)))
		for _, d := range ts.dims {
			rtWriteU64(t, buf, d)
		}
		rtWriteU32(t, buf, ggmlTypeF32)
		rtWriteU64(t, buf, offset)
		offset += uint64(len(ts.data) * 4)
	}

	rtPadTo(t, buf, alignBytes)
	for _, ts := range tensors {
		for _, v := range ts.data {
			rtWriteF32(t, buf, v)
		}
	}

	path := filepath.Join(t.TempDir(), "llama_block0.gguf")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}
	return path
}

func TestRoPEScaledPosition(t *testing.T) {
	if got := ropeScaledPosition(10, 2, "linear"); got != 5 {
		t.Fatalf("ropeScaledPosition linear = %v, want 5", got)
	}
	if got := ropeScaledPosition(10, 2, ""); got != 5 {
		t.Fatalf("ropeScaledPosition default = %v, want 5", got)
	}
	if got := ropeScaledPosition(10, 2, "yarn"); got != 5 {
		t.Fatalf("ropeScaledPosition yarn = %v, want 5", got)
	}
}

func rtWriteGGUFString(t *testing.T, buf *bytes.Buffer, s string) {
	t.Helper()
	rtWriteU64(t, buf, uint64(len(s)))
	rtWriteString(t, buf, s)
}

func rtWriteString(t *testing.T, buf *bytes.Buffer, s string) {
	t.Helper()
	if _, err := buf.WriteString(s); err != nil {
		t.Fatalf("WriteString() error = %v", err)
	}
}

func rtWriteU32(t *testing.T, buf *bytes.Buffer, v uint32) {
	t.Helper()
	if err := binary.Write(buf, binary.LittleEndian, v); err != nil {
		t.Fatalf("binary.Write(u32) error = %v", err)
	}
}

func rtWriteU64(t *testing.T, buf *bytes.Buffer, v uint64) {
	t.Helper()
	if err := binary.Write(buf, binary.LittleEndian, v); err != nil {
		t.Fatalf("binary.Write(u64) error = %v", err)
	}
}

func rtWriteF32(t *testing.T, buf *bytes.Buffer, v float32) {
	t.Helper()
	if err := binary.Write(buf, binary.LittleEndian, v); err != nil {
		t.Fatalf("binary.Write(f32) error = %v", err)
	}
}

func rtPadTo(t *testing.T, buf *bytes.Buffer, align int) {
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
