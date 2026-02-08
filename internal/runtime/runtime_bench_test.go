package runtime

import (
	"strconv"
	"testing"

	"bitnet-go/internal/gguf"
)

func BenchmarkRMSNormInto(b *testing.B) {
	sizes := []int{256, 1024, 4096}
	for _, n := range sizes {
		b.Run("n="+strconv.Itoa(n), func(b *testing.B) {
			dst := make([]float32, n)
			x := make([]float32, n)
			weight := make([]float32, n)
			for i := 0; i < n; i++ {
				x[i] = float32(i%97) * 0.01
				weight[i] = 1.0 + float32(i%13)*0.001
			}
			b.ReportAllocs()
			b.SetBytes(int64(n * 4 * 3))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				rmsNormInto(dst, x, weight, 1e-5)
			}
		})
	}
}

func BenchmarkApplyRoPEInPlace(b *testing.B) {
	type cfg struct {
		heads int
		dim   int
		scale float32
		mode  string
	}
	cases := []cfg{
		{heads: 8, dim: 64, scale: 1.0, mode: "linear"},
		{heads: 8, dim: 64, scale: 4.0, mode: "linear"},
		{heads: 8, dim: 64, scale: 4.0, mode: "yarn"},
		{heads: 16, dim: 128, scale: 1.0, mode: "linear"},
	}
	for _, c := range cases {
		name := "h=" + strconv.Itoa(c.heads) + "/d=" + strconv.Itoa(c.dim) + "/mode=" + c.mode
		b.Run(name, func(b *testing.B) {
			n := c.heads * c.dim
			v := make([]float32, n)
			for i := 0; i < n; i++ {
				v[i] = float32(i%29) * 0.01
			}
			b.ReportAllocs()
			b.SetBytes(int64(n * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				applyRoPEInPlace(v, 64, c.heads, 10000, c.scale, c.mode, c.dim, true, 1, 1, 1, 1)
			}
		})
	}
}

func BenchmarkCausalAttentionMultiHeadInto(b *testing.B) {
	type cfg struct {
		steps int
		heads int
		dim   int
	}
	cases := []cfg{
		{steps: 64, heads: 8, dim: 64},
		{steps: 128, heads: 8, dim: 64},
		{steps: 256, heads: 16, dim: 64},
	}
	for _, c := range cases {
		name := "steps=" + strconv.Itoa(c.steps) + "/h=" + strconv.Itoa(c.heads) + "/d=" + strconv.Itoa(c.dim)
		b.Run(name, func(b *testing.B) {
			q := make([]float32, c.heads*c.dim)
			for i := range q {
				q[i] = float32(i%31) * 0.01
			}
			kStepDim := c.heads * c.dim
			vStepDim := c.heads * c.dim
			keys := make([]float32, c.steps*kStepDim)
			values := make([]float32, c.steps*vStepDim)
			for i := range keys {
				keys[i] = float32(i%37) * 0.01
				values[i] = float32(i%41) * 0.01
			}
			dst := make([]float32, len(q))
			scores := make([]float32, c.steps*c.heads)

			b.ReportAllocs()
			b.SetBytes(int64((len(q) + len(keys) + len(values) + len(scores) + len(dst)) * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				causalAttentionMultiHeadInto(dst, scores, q, keys, values, c.steps, c.heads, c.heads, kStepDim, vStepDim, 0)
			}
		})
	}
}

func BenchmarkCausalAttentionMultiHeadIntoCompare(b *testing.B) {
	type cfg struct {
		steps int
		heads int
		dim   int
	}
	cases := []cfg{
		{steps: 64, heads: 8, dim: 64},
		{steps: 128, heads: 8, dim: 64},
		{steps: 256, heads: 16, dim: 64},
	}
	for _, c := range cases {
		name := "steps=" + strconv.Itoa(c.steps) + "/h=" + strconv.Itoa(c.heads) + "/d=" + strconv.Itoa(c.dim)
		b.Run(name, func(b *testing.B) {
			q := make([]float32, c.heads*c.dim)
			for i := range q {
				q[i] = float32(i%31) * 0.01
			}
			kStepDim := c.heads * c.dim
			vStepDim := c.heads * c.dim
			keys := make([]float32, c.steps*kStepDim)
			values := make([]float32, c.steps*vStepDim)
			for i := range keys {
				keys[i] = float32(i%37) * 0.01
				values[i] = float32(i%41) * 0.01
			}
			dst := make([]float32, len(q))
			scores := make([]float32, c.steps*c.heads)

			b.Run("generic", func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64((len(q) + len(keys) + len(values) + len(scores) + len(dst)) * 4))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					causalAttentionMultiHeadIntoGeneric(dst, scores, q, keys, values, c.steps, c.heads, c.heads, kStepDim, vStepDim, 0)
				}
			})

			b.Run("dispatch", func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64((len(q) + len(keys) + len(values) + len(scores) + len(dst)) * 4))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					causalAttentionMultiHeadInto(dst, scores, q, keys, values, c.steps, c.heads, c.heads, kStepDim, vStepDim, 0)
				}
			})
		})
	}
}

func BenchmarkKVCacheStore(b *testing.B) {
	steps := 256
	heads := 8
	dim := 64
	kvHeads := 4
	kStepDim := heads * dim
	vStepDim := heads * dim
	keys := make([]float32, steps*kStepDim)
	values := make([]float32, steps*vStepDim)
	vecK := make([]float32, kStepDim)
	vecV := make([]float32, vStepDim)
	for i := range vecK {
		vecK[i] = float32(i%31) * 0.01
	}
	for i := range vecV {
		vecV[i] = float32(i%37) * 0.01
	}

	b.Run("generic", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64((len(vecK) + len(vecV)) * 4))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			pos := i % steps
			storeCacheVectorGeneric(keys, pos, vecK)
			storeCacheVectorVGeneric(values, pos, vecV, kvHeads)
		}
	})

	b.Run("dispatch", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64((len(vecK) + len(vecV)) * 4))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			pos := i % steps
			storeCacheVector(keys, pos, vecK)
			storeCacheVectorV(values, pos, vecV, kvHeads)
		}
	})
}

func BenchmarkKQVAccumulation(b *testing.B) {
	type cfg struct {
		steps   int
		headDim int
	}
	cases := []cfg{
		{steps: 64, headDim: 64},
		{steps: 128, headDim: 64},
		{steps: 256, headDim: 64},
	}
	for _, c := range cases {
		name := "steps=" + strconv.Itoa(c.steps) + "/d=" + strconv.Itoa(c.headDim)
		b.Run(name, func(b *testing.B) {
			maxSeq := c.steps
			values := make([]float32, maxSeq*c.headDim)
			weights := make([]float32, maxSeq)
			for i := range values {
				values[i] = float32(i%41) * 0.01
			}
			for i := range weights {
				weights[i] = float32((i%31)-15) * 0.001
			}
			dst := make([]float32, c.headDim)

			b.Run("fast", func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64((len(values) + len(weights) + len(dst)) * 4))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					for j := 0; j < c.headDim; j++ {
						rowBase := j * maxSeq
						dst[j] = dotF32Fast(values[rowBase:rowBase+maxSeq], weights)
					}
				}
			})

			b.Run("ggml", func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64((len(values) + len(weights) + len(dst)) * 4))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					for j := 0; j < c.headDim; j++ {
						rowBase := j * maxSeq
						dst[j] = dotF32GGML(values[rowBase:rowBase+maxSeq], weights)
					}
				}
			})
		})
	}
}

func BenchmarkSoftmaxInPlace(b *testing.B) {
	steps := 256
	scores := make([]float32, steps)
	for i := range scores {
		scores[i] = float32((i%17)-8) * 0.01
	}
	maxScore := float32(0.05)

	tmp := make([]float32, len(scores))

	b.Run("generic", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(scores) * 4))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			copy(tmp, scores)
			softmaxInPlaceGeneric(tmp, maxScore)
		}
	})

	b.Run("dispatch", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64(len(scores) * 4))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			copy(tmp, scores)
			softmaxInPlace(tmp, maxScore)
		}
	})
}

func BenchmarkQKVMatVecCompare(b *testing.B) {
	rows := 256
	cols := 256
	matA := make([]float32, rows*cols)
	matB := make([]float32, rows*cols)
	matC := make([]float32, rows*cols)
	vec := make([]float32, cols)
	for c := 0; c < cols; c++ {
		vec[c] = float32(c%31) * 0.01
		for r := 0; r < rows; r++ {
			idx := r + rows*c
			matA[idx] = float32((r+c)%23) * 0.01
			matB[idx] = float32((r+c)%19) * 0.02
			matC[idx] = float32((r+c)%17) * 0.03
		}
	}
	dstA := make([]float32, rows)
	dstB := make([]float32, rows)
	dstC := make([]float32, rows)

	b.Run("separate", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64((rows*cols*3 + cols + rows*3) * 4))
		b.ResetTimer()
		wA := linearWeight{data: matA, rows: rows, cols: cols, qtype: gguf.GGMLTypeF32}
		wB := linearWeight{data: matB, rows: rows, cols: cols, qtype: gguf.GGMLTypeF32}
		wC := linearWeight{data: matC, rows: rows, cols: cols, qtype: gguf.GGMLTypeF32}
		for i := 0; i < b.N; i++ {
			linearApplyIntoWeight(dstA, wA, vec)
			linearApplyIntoWeight(dstB, wB, vec)
			linearApplyIntoWeight(dstC, wC, vec)
		}
	})

	b.Run("fused", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(int64((rows*cols*3 + cols + rows*3) * 4))
		b.ResetTimer()
		wA := linearWeight{data: matA, rows: rows, cols: cols, qtype: gguf.GGMLTypeF32}
		wB := linearWeight{data: matB, rows: rows, cols: cols, qtype: gguf.GGMLTypeF32}
		wC := linearWeight{data: matC, rows: rows, cols: cols, qtype: gguf.GGMLTypeF32}
		for i := 0; i < b.N; i++ {
			linearApplyQKV(dstA, dstB, dstC, wA, wB, wC, vec)
		}
	})
}

func BenchmarkLinearApplyInto(b *testing.B) {
	type cfg struct {
		rows int
		cols int
		tr   bool
	}
	cases := []cfg{
		{rows: 512, cols: 512, tr: false},
		{rows: 1024, cols: 1024, tr: false},
		{rows: 512, cols: 512, tr: true},
		{rows: 1024, cols: 1024, tr: true},
	}
	for _, c := range cases {
		name := "r=" + strconv.Itoa(c.rows) + "/c=" + strconv.Itoa(c.cols)
		if c.tr {
			name += "/T"
		}
		b.Run(name, func(b *testing.B) {
			data := make([]float32, c.rows*c.cols)
			x := make([]float32, c.cols)
			dst := make([]float32, c.rows)
			if c.tr {
				x = make([]float32, c.rows)
				dst = make([]float32, c.cols)
			}
			for i := range data {
				data[i] = float32(i%19) * 0.01
			}
			for i := range x {
				x[i] = float32(i%23) * 0.02
			}
			b.ReportAllocs()
			b.SetBytes(int64((len(data) + len(x) + len(dst)) * 4))
			b.ResetTimer()
			w := linearWeight{data: data, rows: c.rows, cols: c.cols, transposed: c.tr, qtype: gguf.GGMLTypeF32}
			for i := 0; i < b.N; i++ {
				linearApplyIntoWeight(dst, w, x)
			}
		})
	}
}

func BenchmarkLlamaLayerStep(b *testing.B) {
	type cfg struct {
		hidden   int
		ffn      int
		heads    int
		kvHeads  int
		steps    int
		ropeBase float32
	}
	cases := []cfg{
		{hidden: 512, ffn: 2048, heads: 8, kvHeads: 8, steps: 64, ropeBase: 10000},
		{hidden: 1024, ffn: 4096, heads: 16, kvHeads: 8, steps: 128, ropeBase: 10000},
	}
	for _, c := range cases {
		name := "h=" + strconv.Itoa(c.hidden) + "/ffn=" + strconv.Itoa(c.ffn) + "/heads=" + strconv.Itoa(c.heads) + "/steps=" + strconv.Itoa(c.steps)
		b.Run(name, func(b *testing.B) {
			block := &tensorBlock{
				hiddenDim:    c.hidden,
				attnHeads:    c.heads,
				kvHeads:      c.kvHeads,
				ropeFreqBase: c.ropeBase,
				ropeScale:    1,
				ropeDim:      c.hidden / c.heads,
				rmsEps:       1e-5,
				outputNorm:   make([]float32, c.hidden),
				outputWeight: make([]float32, c.hidden*c.hidden),
				outputRows:   c.hidden,
				outputCols:   c.hidden,
			}
			for i := range block.outputNorm {
				block.outputNorm[i] = 1
			}
			ffnLen := c.hidden * c.ffn
			layer := llamaLayer{
				attnNorm: make([]float32, c.hidden),
				ffnNorm:  make([]float32, c.hidden),
				attnQ:    linearWeight{data: make([]float32, c.hidden*c.hidden), rows: c.hidden, cols: c.hidden, qtype: gguf.GGMLTypeF32},
				attnK:    linearWeight{data: make([]float32, c.hidden*c.hidden), rows: c.hidden, cols: c.hidden, qtype: gguf.GGMLTypeF32},
				attnV:    linearWeight{data: make([]float32, c.hidden*c.hidden), rows: c.hidden, cols: c.hidden, qtype: gguf.GGMLTypeF32},
				attnOut:  linearWeight{data: make([]float32, c.hidden*c.hidden), rows: c.hidden, cols: c.hidden, qtype: gguf.GGMLTypeF32},
				ffnGate:  linearWeight{data: make([]float32, ffnLen), rows: c.ffn, cols: c.hidden, qtype: gguf.GGMLTypeF32},
				ffnUp:    linearWeight{data: make([]float32, ffnLen), rows: c.ffn, cols: c.hidden, qtype: gguf.GGMLTypeF32},
				ffnDown:  linearWeight{data: make([]float32, ffnLen), rows: c.hidden, cols: c.ffn, qtype: gguf.GGMLTypeF32},
			}
			block.layers = []llamaLayer{layer}

			st := makeLlamaLayerState(layer, c.hidden, c.steps, c.heads)
			states := []llamaLayerState{st}

			x := make([]float32, c.hidden)
			n1 := make([]float32, c.hidden)
			n2 := make([]float32, c.hidden)
			logits := make([]float32, c.hidden)
			for i := range x {
				x[i] = float32(i%31) * 0.01
			}

			b.ReportAllocs()
			b.SetBytes(int64((len(x) + len(n1) + len(n2) + len(logits)) * 4))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				runLlamaStackStep(block, states, 0, i%c.steps, x, n1, n2, logits)
			}
		})
	}
}
