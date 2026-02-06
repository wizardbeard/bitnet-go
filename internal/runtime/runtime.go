package runtime

import (
	"context"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"

	"bitnet-go/internal/gguf"
	"bitnet-go/internal/kernels"
	"bitnet-go/internal/tokenizer"
)

type GenerateRequest struct {
	Prompt    string
	Seed      int64
	MaxTokens int
}

type Metadata struct {
	Path          string
	Version       uint32
	TensorCount   uint64
	KVCount       uint64
	Architecture  string
	ContextLength uint32
	VocabSize     uint32
}

type TopKEntry struct {
	TokenID int32
	Logit   float32
}

type TopKStep struct {
	Step    int
	Entries []TopKEntry
}

type Runtime struct {
	meta      Metadata
	tokenizer *tokenizer.Tokenizer
	block     *tensorBlock
}

var debugStep0 = os.Getenv("BITNET_DEBUG_STEP0") == "1"
var disableFFN = os.Getenv("BITNET_DISABLE_FFN") == "1"
var disableAttn = os.Getenv("BITNET_DISABLE_ATTN") == "1"
var disableLayers = os.Getenv("BITNET_DISABLE_LAYERS") == "1"
var debugOutput = os.Getenv("BITNET_DEBUG_OUTPUT") == "1"
var debugOutputOnly = os.Getenv("BITNET_DEBUG_OUTPUT_ONLY") == "1"
var debugStages = os.Getenv("BITNET_DEBUG_STAGES") == "1"
var debugFFNTranspose = os.Getenv("BITNET_DEBUG_FFN_TRANSPOSE") == "1"
var debugFFNLoad = os.Getenv("BITNET_DEBUG_FFN_LOAD") == "1"
var debugAttnMeta = os.Getenv("BITNET_DEBUG_ATTN_META") == "1"
var debugValues = os.Getenv("BITNET_DEBUG_VALUES") == "1"
var debugValuesN = parseDebugValuesN(os.Getenv("BITNET_DEBUG_VALUES_N"))
var debugPos = parseDebugPos(os.Getenv("BITNET_DEBUG_POS"))
var debugTokens = parseDebugTokens(os.Getenv("BITNET_DEBUG_TOKENS"))
var debugStep0Printed bool
var debugI2SDisableActSum = os.Getenv("BITNET_I2S_DISABLE_ACTSUM") == "1"
var debugI2SInvertActScale = os.Getenv("BITNET_I2S_INVERT_ACT_SCALE") == "1"
var debugI2SFloat = os.Getenv("BITNET_I2S_F32") == "1"
var i8ScratchPool = sync.Pool{
	New: func() any {
		return make([]int8, 0)
	},
}

func parseDebugPos(v string) int {
	if v == "" {
		return -1
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return -1
	}
	return n
}

func parseDebugTokens(v string) []int {
	if v == "" {
		return []int{0, 1, 2, 3, 4}
	}
	parts := strings.Split(v, ",")
	out := make([]int, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		n, err := strconv.Atoi(part)
		if err != nil {
			continue
		}
		out = append(out, n)
	}
	return out
}

func parseDebugValuesN(v string) int {
	if v == "" {
		return 8
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		return 8
	}
	return n
}

func shouldDebug(pos int) bool {
	if !debugStep0 || debugStep0Printed {
		return false
	}
	if debugPos >= 0 {
		return pos == debugPos
	}
	return pos == 0
}

func New(_ context.Context, modelPath string) (*Runtime, error) {
	info, err := gguf.ReadModelInfo(modelPath)
	if err != nil {
		h, hErr := gguf.ReadHeader(modelPath)
		if hErr != nil {
			return nil, fmt.Errorf("read gguf model info: %w", err)
		}
		return &Runtime{meta: Metadata{
			Path:        modelPath,
			Version:     h.Version,
			TensorCount: h.TensorCount,
			KVCount:     h.KVCount,
		}}, nil
	}

	arch, _ := info.KeyValues["general.architecture"].(string)
	ctxLen := firstUint32(
		info.KeyValues["bitnet-b1.58.context_length"],
		info.KeyValues["llama.context_length"],
		info.KeyValues["falcon.context_length"],
		info.KeyValues["gpt2.context_length"],
	)
	vocab := firstUint32(
		info.KeyValues["bitnet-b1.58.vocab_size"],
		info.KeyValues["llama.vocab_size"],
		info.KeyValues["gpt2.vocab_size"],
		info.KeyValues["tokenizer.ggml.tokens_count"],
		info.KeyValues["tokenizer.ggml.tokens.count"],
	)

	tok, _ := tokenizer.NewFromModelInfo(info)
	block, err := loadTensorBlock(modelPath, info)
	if err != nil {
		return nil, err
	}

	return &Runtime{
		meta: Metadata{
			Path:          modelPath,
			Version:       info.Version,
			TensorCount:   info.TensorCount,
			KVCount:       info.KVCount,
			Architecture:  arch,
			ContextLength: ctxLen,
			VocabSize:     vocab,
		},
		tokenizer: tok,
		block:     block,
	}, nil
}

func (r *Runtime) Metadata() Metadata {
	return r.meta
}

func firstUint32(values ...any) uint32 {
	for _, v := range values {
		switch x := v.(type) {
		case uint32:
			return x
		case uint64:
			if x <= uint64(^uint32(0)) {
				return uint32(x)
			}
		case int32:
			if x >= 0 {
				return uint32(x)
			}
		case int64:
			if x >= 0 && x <= int64(^uint32(0)) {
				return uint32(x)
			}
		}
	}
	return 0
}

func (r *Runtime) Generate(_ context.Context, req GenerateRequest) (struct {
	TokenIDs []int32
	Text     string
	TopK     []TopKStep
}, error) {
	if req.MaxTokens == 0 {
		return struct {
			TokenIDs []int32
			Text     string
			TopK     []TopKStep
		}{}, nil
	}

	promptTokens := []int32{}
	if r.tokenizer != nil {
		promptTokens = r.tokenizer.Tokenize(req.Prompt)
	}

	// Phase-2 stepping stone: minimal forward loop with naive kernels and
	// procedural weights. If model carries bitnet_go.* f32 tensors, use a first
	// tensor-backed block path instead.
	tokens := make([]int32, req.MaxTokens)
	topk := make([]TopKStep, 0, req.MaxTokens)
	if r.block != nil {
		runForwardTensorBlock(r.block, req.Seed, promptTokens, tokens, &topk)
	} else {
		runForwardStub(r.meta.VocabSize, req.Seed, promptTokens, tokens, &topk)
	}

	return struct {
		TokenIDs []int32
		Text     string
		TopK     []TopKStep
	}{
		TokenIDs: tokens,
		Text:     req.Prompt,
		TopK:     topk,
	}, nil
}

type tensorBlock struct {
	hiddenDim            int
	vocabDim             int
	mode                 tensorBlockMode
	attnHeads            int
	kvHeads              int
	ropeFreqBase         float32
	ropeScale            float32
	ropeScalingType      string
	ropeDim              int
	ropeYarnBetaFast     float32
	ropeYarnBetaSlow     float32
	ropeYarnOrigCtx      float32
	ropeYarnExtFactor    float32
	ropeYarnAttnFactor   float32
	stateProj            []float32
	logitsProj           []float32
	logitsProjTransposed bool
	tokenEmbd            []float32
	tokenEmbdRows        int
	tokenEmbdCols        int
	outputWeight         []float32
	outputRows           int
	outputCols           int
	outputTransposed     bool
	outputWeightPacked   []byte
	outputWeightScale    float32
	outputWeightType     uint32
	outputNorm           []float32
	rmsEps               float32
	layers               []llamaLayer
}

type tensorBlockMode int

const (
	tensorBlockModeProjection tensorBlockMode = iota + 1
	tensorBlockModeEmbeddingOutput
	tensorBlockModeLlamaStack
)

type llamaLayer struct {
	attnNorm []float32
	ffnNorm  []float32
	attnSubNorm []float32
	ffnSubNorm  []float32
	attnQ    linearWeight
	attnK    linearWeight
	attnV    linearWeight
	attnOut  linearWeight
	ffnGate  linearWeight
	ffnUp    linearWeight
	ffnDown  linearWeight
	debugFFNGateF32 []float32
	debugFFNUpF32   []float32
	debugFFNDownF32 []float32
}

type linearWeight struct {
	data       []float32
	rows       int
	cols       int
	transposed bool
	qtype      uint32
	i2sPacked  []byte
	i2sScale   float32
}

func loadTensorBlock(path string, info gguf.ModelInfo) (*tensorBlock, error) {
	block, found, err := loadLlamaStack(path, info)
	if err != nil {
		return nil, err
	}
	if found {
		return block, nil
	}

	block, found, err = loadProjectionBlock(path, info)
	if err != nil {
		return nil, err
	}
	if found {
		return block, nil
	}
	return loadEmbeddingOutputBlock(path, info)
}

func loadProjectionBlock(path string, info gguf.ModelInfo) (*tensorBlock, bool, error) {
	const (
		stateProjName  = "bitnet_go.state_proj"
		logitsProjName = "bitnet_go.logits_proj"
	)

	_, hasState := info.TensorByName(stateProjName)
	_, hasLogits := info.TensorByName(logitsProjName)
	if !hasState && !hasLogits {
		return nil, false, nil
	}
	if !hasState || !hasLogits {
		return nil, false, fmt.Errorf("incomplete tensor block: need %q and %q", stateProjName, logitsProjName)
	}

	stateInfo, _ := info.TensorByName(stateProjName)
	if len(stateInfo.Dimensions) != 2 {
		return nil, false, fmt.Errorf("%s: expected 2 dims, got %d", stateProjName, len(stateInfo.Dimensions))
	}
	if stateInfo.Dimensions[0] != stateInfo.Dimensions[1] {
		return nil, false, fmt.Errorf("%s: expected square matrix, got %v", stateProjName, stateInfo.Dimensions)
	}
	hidden := int(stateInfo.Dimensions[0])
	if hidden <= 0 {
		return nil, false, fmt.Errorf("%s: invalid hidden dim %d", stateProjName, hidden)
	}

	stateProj, err := gguf.ReadTensorAsF32(path, info, stateProjName)
	if err != nil {
		return nil, false, err
	}

	logitsInfo, _ := info.TensorByName(logitsProjName)
	if len(logitsInfo.Dimensions) != 2 {
		return nil, false, fmt.Errorf("%s: expected 2 dims, got %d", logitsProjName, len(logitsInfo.Dimensions))
	}
	logitsProj, err := gguf.ReadTensorAsF32(path, info, logitsProjName)
	if err != nil {
		return nil, false, err
	}

	block := &tensorBlock{
		mode:       tensorBlockModeProjection,
		hiddenDim:  hidden,
		stateProj:  stateProj,
		logitsProj: logitsProj,
	}

	switch {
	case int(logitsInfo.Dimensions[1]) == hidden:
		block.vocabDim = int(logitsInfo.Dimensions[0])
	case int(logitsInfo.Dimensions[0]) == hidden:
		block.vocabDim = int(logitsInfo.Dimensions[1])
		block.logitsProjTransposed = true
	default:
		return nil, false, fmt.Errorf("%s dims %v incompatible with hidden=%d", logitsProjName, logitsInfo.Dimensions, hidden)
	}
	if block.vocabDim <= 0 {
		return nil, false, fmt.Errorf("%s has invalid vocab dim %d", logitsProjName, block.vocabDim)
	}

	return block, true, nil
}

func loadEmbeddingOutputBlock(path string, info gguf.ModelInfo) (*tensorBlock, error) {
	const (
		tokenEmbdName = "token_embd.weight"
		outputName    = "output.weight"
	)

	embInfo, hasEmb := info.TensorByName(tokenEmbdName)
	outInfo, hasOut := info.TensorByName(outputName)
	if !hasEmb && !hasOut {
		return nil, nil
	}
	if !hasEmb || !hasOut {
		return nil, fmt.Errorf("incomplete embedding/output block: need %q and %q", tokenEmbdName, outputName)
	}
	if len(embInfo.Dimensions) != 2 {
		return nil, fmt.Errorf("%s: expected 2 dims, got %d", tokenEmbdName, len(embInfo.Dimensions))
	}
	if len(outInfo.Dimensions) != 2 {
		return nil, fmt.Errorf("%s: expected 2 dims, got %d", outputName, len(outInfo.Dimensions))
	}

	emb := &tensorBlock{
		mode:          tensorBlockModeEmbeddingOutput,
		tokenEmbdRows: int(embInfo.Dimensions[0]),
		tokenEmbdCols: int(embInfo.Dimensions[1]),
	}
	if emb.tokenEmbdRows <= 0 || emb.tokenEmbdCols <= 0 {
		return nil, fmt.Errorf("%s has invalid dims %v", tokenEmbdName, embInfo.Dimensions)
	}

	tokenEmbd, err := gguf.ReadTensorAsF32(path, info, tokenEmbdName)
	if err != nil {
		return nil, err
	}
	emb.tokenEmbd = tokenEmbd

	var outputWeight []float32
	var outputPacked []byte
	var outputScale float32
	if outInfo.Type == gguf.GGMLTypeI2_S {
		packed, scale, _, err := gguf.ReadTensorI2SPacked(path, info, outputName)
		if err != nil {
			return nil, err
		}
		outputPacked = packed
		outputScale = scale
	} else {
		outputWeight, err = gguf.ReadTensorAsF32(path, info, outputName)
		if err != nil {
			return nil, err
		}
	}
	emb.outputWeight = outputWeight
	emb.outputWeightPacked = outputPacked
	emb.outputWeightScale = outputScale
	emb.outputWeightType = outInfo.Type

	emb.hiddenDim = emb.tokenEmbdRows
	emb.vocabDim = emb.tokenEmbdCols

	emb.outputRows = int(outInfo.Dimensions[0])
	emb.outputCols = int(outInfo.Dimensions[1])
	switch {
	case emb.outputRows == emb.hiddenDim && emb.outputCols == emb.vocabDim:
		emb.outputTransposed = true
	case emb.outputRows == emb.vocabDim && emb.outputCols == emb.hiddenDim:
		emb.outputTransposed = false
	default:
		return nil, fmt.Errorf("%s dims %v incompatible with emb dims %v", outputName, outInfo.Dimensions, embInfo.Dimensions)
	}

	return emb, nil
}

func loadLlamaStack(path string, info gguf.ModelInfo) (*tensorBlock, bool, error) {
	if _, ok := info.TensorByName("blk.0.attn_q.weight"); !ok {
		return nil, false, nil
	}
	arch, _ := info.KeyValues["general.architecture"].(string)
	useTokEmbOut := arch == "bitnet-b1.58" || arch == "bitnet" || arch == "bitnet-25"

	embInfo, ok := info.TensorByName("token_embd.weight")
	if !ok {
		return nil, false, fmt.Errorf("missing tensor: token_embd.weight")
	}
	if len(embInfo.Dimensions) != 2 {
		return nil, false, fmt.Errorf("token_embd.weight: expected 2 dims, got %d", len(embInfo.Dimensions))
	}
	hidden := int(embInfo.Dimensions[0])
	vocab := int(embInfo.Dimensions[1])
	if hidden <= 0 || vocab <= 0 {
		return nil, false, fmt.Errorf("token_embd.weight invalid dims: %v", embInfo.Dimensions)
	}

	b := &tensorBlock{
		mode:          tensorBlockModeLlamaStack,
		hiddenDim:     hidden,
		vocabDim:      vocab,
		tokenEmbdRows: hidden,
		tokenEmbdCols: vocab,
		rmsEps: firstFloat32From(
			float32(1e-5),
			info.KeyValues["llama.attention.layer_norm_rms_epsilon"],
			info.KeyValues["bitnet-b1.58.attention.layer_norm_rms_epsilon"],
		),
		attnHeads: int(firstUint32(
			info.KeyValues["llama.attention.head_count"],
			info.KeyValues["bitnet-b1.58.attention.head_count"],
		)),
		kvHeads: int(firstUint32(
			info.KeyValues["llama.attention.head_count_kv"],
			info.KeyValues["bitnet-b1.58.attention.head_count_kv"],
		)),
		ropeFreqBase: firstFloat32From(
			float32(10000),
			info.KeyValues["llama.rope.freq_base"],
			info.KeyValues["bitnet-b1.58.rope.freq_base"],
		),
		ropeScale:     firstFloat32(info.KeyValues["llama.rope.scaling.factor"], 1.0),
		ropeScalingType: firstString(
			info.KeyValues["llama.rope.scaling.type"],
			info.KeyValues["llama.rope.scaling_type"],
		),
		ropeDim: int(firstUint32(
			info.KeyValues["llama.rope.dimension_count"],
			info.KeyValues["bitnet-b1.58.rope.dimension_count"],
		)),
		ropeYarnBetaFast:   firstFloat32(info.KeyValues["llama.rope.scaling.beta_fast"], 0),
		ropeYarnBetaSlow:   firstFloat32(info.KeyValues["llama.rope.scaling.beta_slow"], 0),
		ropeYarnOrigCtx:    firstFloat32(info.KeyValues["llama.rope.scaling.original_context_length"], 0),
		ropeYarnExtFactor:  firstFloat32(info.KeyValues["llama.rope.scaling.ext_factor"], 0),
		ropeYarnAttnFactor: firstFloat32(info.KeyValues["llama.rope.scaling.attn_factor"], 1.0),
	}
	if b.attnHeads <= 0 {
		b.attnHeads = 1
	}
	if b.kvHeads <= 0 {
		b.kvHeads = b.attnHeads
	}
	if b.ropeFreqBase <= 0 {
		b.ropeFreqBase = 10000
	}
	if b.ropeScale <= 0 {
		b.ropeScale = 1
	}
	if b.ropeDim <= 0 {
		b.ropeDim = 0
	}
	if b.ropeYarnAttnFactor == 0 {
		b.ropeYarnAttnFactor = 1
	}

	var err error
	if b.tokenEmbd, err = gguf.ReadTensorAsF32(path, info, "token_embd.weight"); err != nil {
		return nil, false, err
	}
	if b.outputNorm, err = gguf.ReadTensorAsF32(path, info, "output_norm.weight"); err != nil {
		return nil, false, err
	}
	if len(b.outputNorm) != hidden {
		return nil, false, fmt.Errorf("output_norm.weight len=%d want=%d", len(b.outputNorm), hidden)
	}
	if useTokEmbOut {
		b.outputWeight = b.tokenEmbd
		b.outputRows = hidden
		b.outputCols = b.vocabDim
		b.outputTransposed = true
		b.outputWeightType = gguf.GGMLTypeF32
	} else if outInfo, ok := info.TensorByName("output.weight"); ok {
		if b.outputWeight, b.outputRows, b.outputCols, b.outputTransposed, err = loadLinearTensor(path, info, "output.weight", hidden); err != nil {
			return nil, false, err
		}
		b.outputWeightType = outInfo.Type
		if outInfo.Type == gguf.GGMLTypeI2_S {
			packed, scale, _, err := gguf.ReadTensorI2SPacked(path, info, "output.weight")
			if err != nil {
				return nil, false, err
			}
			b.outputWeightPacked = packed
			b.outputWeightScale = scale
		}
	} else {
		// Some models tie output weights to token embeddings and omit output.weight.
		b.outputWeight = b.tokenEmbd
		b.outputRows = hidden
		b.outputCols = b.vocabDim
		b.outputTransposed = true
		b.outputWeightType = gguf.GGMLTypeF32
	}

	for idx := 0; ; idx++ {
		prefix := fmt.Sprintf("blk.%d.", idx)
		if _, ok := info.TensorByName(prefix + "attn_q.weight"); !ok {
			if idx == 0 {
				return nil, false, fmt.Errorf("missing tensor: %sattn_q.weight", prefix)
			}
			break
		}
		layer, err := loadLlamaLayer(path, info, idx, prefix, hidden, b.attnHeads, b.kvHeads)
		if err != nil {
			return nil, false, err
		}
		b.layers = append(b.layers, layer)
	}
	if len(b.layers) == 0 {
		return nil, false, nil
	}
	return b, true, nil
}

func loadLlamaLayer(path string, info gguf.ModelInfo, layerIdx int, prefix string, hidden, attnHeads, kvHeads int) (llamaLayer, error) {
	var l llamaLayer
	var err error

	if l.attnNorm, err = gguf.ReadTensorAsF32(path, info, prefix+"attn_norm.weight"); err != nil {
		return llamaLayer{}, err
	}
	if len(l.attnNorm) != hidden {
		return llamaLayer{}, fmt.Errorf("%sattn_norm.weight len=%d want=%d", prefix, len(l.attnNorm), hidden)
	}
	if l.attnSubNorm, err = gguf.ReadTensorAsF32(path, info, prefix+"attn_sub_norm.weight"); err != nil {
		return llamaLayer{}, err
	}
	if len(l.attnSubNorm) != hidden {
		return llamaLayer{}, fmt.Errorf("%sattn_sub_norm.weight len=%d want=%d", prefix, len(l.attnSubNorm), hidden)
	}
	if l.ffnNorm, err = gguf.ReadTensorAsF32(path, info, prefix+"ffn_norm.weight"); err != nil {
		return llamaLayer{}, err
	}
	if len(l.ffnNorm) != hidden {
		return llamaLayer{}, fmt.Errorf("%sffn_norm.weight len=%d want=%d", prefix, len(l.ffnNorm), hidden)
	}
	if l.attnQ, err = loadLinearWeight(path, info, prefix+"attn_q.weight", hidden); err != nil {
		return llamaLayer{}, err
	}
	if l.attnK, err = loadLinearWeight(path, info, prefix+"attn_k.weight", hidden); err != nil {
		return llamaLayer{}, err
	}
	if l.attnV, err = loadLinearWeight(path, info, prefix+"attn_v.weight", hidden); err != nil {
		return llamaLayer{}, err
	}
	qDim := linearOutputLen(l.attnQ)
	kDim := linearOutputLen(l.attnK)
	vDim := linearOutputLen(l.attnV)
	if qDim%attnHeads != 0 {
		return llamaLayer{}, fmt.Errorf("%sattn_q.weight dim=%d not divisible by head_count=%d", prefix, qDim, attnHeads)
	}
	if kDim%kvHeads != 0 || vDim%kvHeads != 0 {
		return llamaLayer{}, fmt.Errorf("%sattn_k/v dims=%d/%d not divisible by kv_head_count=%d", prefix, kDim, vDim, kvHeads)
	}
	qHeadDim := qDim / attnHeads
	kHeadDim := kDim / kvHeads
	vHeadDim := vDim / kvHeads
	if qHeadDim != kHeadDim || qHeadDim != vHeadDim {
		return llamaLayer{}, fmt.Errorf("%shead dims mismatch q=%d k=%d v=%d", prefix, qHeadDim, kHeadDim, vHeadDim)
	}
	if l.attnOut, err = loadLinearWeight(path, info, prefix+"attn_output.weight", qDim); err != nil {
		return llamaLayer{}, err
	}
	if linearOutputLen(l.attnOut) != hidden {
		return llamaLayer{}, fmt.Errorf("%sattn_output.weight output dim=%d want=%d", prefix, linearOutputLen(l.attnOut), hidden)
	}
	if l.ffnGate, err = loadLinearWeight(path, info, prefix+"ffn_gate.weight", hidden); err != nil {
		return llamaLayer{}, err
	}
	if l.ffnUp, err = loadLinearWeight(path, info, prefix+"ffn_up.weight", hidden); err != nil {
		return llamaLayer{}, err
	}
	if l.ffnSubNorm, err = gguf.ReadTensorAsF32(path, info, prefix+"ffn_sub_norm.weight"); err != nil {
		return llamaLayer{}, err
	}
	if len(l.ffnSubNorm) != linearOutputLen(l.ffnGate) {
		return llamaLayer{}, fmt.Errorf("%sffn_sub_norm.weight len=%d want=%d", prefix, len(l.ffnSubNorm), linearOutputLen(l.ffnGate))
	}
	if linearOutputLen(l.ffnGate) != linearOutputLen(l.ffnUp) {
		return llamaLayer{}, fmt.Errorf("%sincompatible ffn gate/up dims", prefix)
	}
	if l.ffnDown, err = loadLinearWeight(path, info, prefix+"ffn_down.weight", linearOutputLen(l.ffnGate)); err != nil {
		return llamaLayer{}, err
	}
	if linearOutputLen(l.ffnDown) != hidden {
		return llamaLayer{}, fmt.Errorf("%sffn_down.weight output dim=%d want=%d", prefix, linearOutputLen(l.ffnDown), hidden)
	}
	if debugFFNLoad && layerIdx == 0 {
		if l.debugFFNGateF32, err = gguf.ReadTensorAsF32(path, info, prefix+"ffn_gate.weight"); err != nil {
			return llamaLayer{}, err
		}
		if l.debugFFNUpF32, err = gguf.ReadTensorAsF32(path, info, prefix+"ffn_up.weight"); err != nil {
			return llamaLayer{}, err
		}
		if l.debugFFNDownF32, err = gguf.ReadTensorAsF32(path, info, prefix+"ffn_down.weight"); err != nil {
			return llamaLayer{}, err
		}
		fmt.Fprintf(os.Stderr, "debug ffn load: layer=%d gate=%d up=%d down=%d\n", layerIdx, len(l.debugFFNGateF32), len(l.debugFFNUpF32), len(l.debugFFNDownF32))
		fmt.Fprintf(os.Stderr, "debug ffn meta: gate rows=%d cols=%d transposed=%v qtype=%d\n", l.ffnGate.rows, l.ffnGate.cols, l.ffnGate.transposed, l.ffnGate.qtype)
		fmt.Fprintf(os.Stderr, "debug ffn meta: up rows=%d cols=%d transposed=%v qtype=%d\n", l.ffnUp.rows, l.ffnUp.cols, l.ffnUp.transposed, l.ffnUp.qtype)
		fmt.Fprintf(os.Stderr, "debug ffn meta: down rows=%d cols=%d transposed=%v qtype=%d\n", l.ffnDown.rows, l.ffnDown.cols, l.ffnDown.transposed, l.ffnDown.qtype)
		debugVecStats("debug attn_norm.weight", l.attnNorm)
		debugVecStats("debug attn_sub_norm.weight", l.attnSubNorm)
		debugVecStats("debug ffn_norm.weight", l.ffnNorm)
		debugVecStats("debug ffn_sub_norm.weight", l.ffnSubNorm)
	}
	return l, nil
}

func runForwardTensorBlock(block *tensorBlock, seed int64, promptTokens []int32, out []int32, topk *[]TopKStep) {
	switch block.mode {
	case tensorBlockModeProjection:
		runForwardProjectionBlock(block, seed, promptTokens, out, topk)
	case tensorBlockModeEmbeddingOutput:
		runForwardEmbeddingOutputBlock(block, seed, promptTokens, out, topk)
	case tensorBlockModeLlamaStack:
		runForwardLlamaStack(block, seed, promptTokens, out, topk)
	default:
		runForwardStub(uint32(block.vocabDim), seed, promptTokens, out, topk)
	}
}

func runForwardProjectionBlock(block *tensorBlock, seed int64, promptTokens []int32, out []int32, topk *[]TopKStep) {
	state := make([]float32, block.hiddenDim)
	tokenVec := make([]float32, block.hiddenDim)
	nextState := make([]float32, block.hiddenDim)
	logits := make([]float32, block.vocabDim)

	mixState(state, tokenVec, int32(seed))
	for _, tok := range promptTokens {
		mixState(state, tokenVec, tok)
	}

	for i := range out {
		kernels.MatVec(nextState, block.stateProj, block.hiddenDim, block.hiddenDim, state)
		copy(state, nextState)

		if block.logitsProjTransposed {
			kernels.MatVecT(logits, block.logitsProj, block.hiddenDim, block.vocabDim, state)
		} else {
			kernels.MatVec(logits, block.logitsProj, block.vocabDim, block.hiddenDim, state)
		}
		if topk != nil {
			*topk = appendTopKStep(*topk, i, logits, 5)
		}

		next := kernels.Argmax(logits)
		if next < 0 {
			out[i] = 0
			continue
		}
		out[i] = int32(next)

		fillTokenVector(tokenVec, out[i])
		kernels.AddScaled(state, tokenVec, 0.05)
	}
}

func runForwardEmbeddingOutputBlock(block *tensorBlock, seed int64, promptTokens []int32, out []int32, topk *[]TopKStep) {
	state := make([]float32, block.hiddenDim)
	logits := make([]float32, block.vocabDim)
	scratch := make([]float32, block.hiddenDim)

	fillTokenVector(state, int32(seed))
	for _, tok := range promptTokens {
		if !embedToken(state, block, tok) {
			fillTokenVector(scratch, tok)
			kernels.AddScaled(state, scratch, 1.0)
		}
	}

	for i := range out {
		linearApplyIntoWeight(logits, linearWeight{
			data:       block.outputWeight,
			rows:       block.outputRows,
			cols:       block.outputCols,
			transposed: block.outputTransposed,
			qtype:      block.outputWeightType,
			i2sPacked:  block.outputWeightPacked,
			i2sScale:   block.outputWeightScale,
		}, state)
		if topk != nil {
			*topk = appendTopKStep(*topk, i, logits, 5)
		}

		next := kernels.Argmax(logits)
		if next < 0 {
			out[i] = 0
			continue
		}
		out[i] = int32(next)

		if !embedToken(state, block, out[i]) {
			fillTokenVector(state, out[i])
		}
	}
}

func embedToken(dst []float32, block *tensorBlock, token int32) bool {
	if token < 0 || int(token) >= block.vocabDim {
		return false
	}
	idx := int(token)
	for r := 0; r < block.hiddenDim; r++ {
		dst[r] = block.tokenEmbd[r+block.tokenEmbdRows*idx]
	}
	return true
}

func runForwardLlamaStack(block *tensorBlock, seed int64, promptTokens []int32, out []int32, topk *[]TopKStep) {
	if len(out) == 0 {
		return
	}
	if debugAttnMeta {
		fmt.Fprintf(os.Stderr, "debug attn meta: heads=%d kv_heads=%d rope_dim=%d rope_freq_base=%g rope_scale=%g rope_type=%q rope_yarn_beta_fast=%g rope_yarn_beta_slow=%g rope_yarn_ext=%g rope_yarn_attn=%g\n",
			block.attnHeads, block.kvHeads, block.ropeDim, block.ropeFreqBase, block.ropeScale, block.ropeScalingType, block.ropeYarnBetaFast, block.ropeYarnBetaSlow, block.ropeYarnExtFactor, block.ropeYarnAttnFactor)
	}
	maxSeq := len(promptTokens) + len(out)
	if maxSeq < 1 {
		maxSeq = 1
	}

	x := make([]float32, block.hiddenDim)
	n1 := make([]float32, block.hiddenDim)
	n2 := make([]float32, block.hiddenDim)
	logits := make([]float32, block.vocabDim)
	layerStates := make([]llamaLayerState, len(block.layers))
	for i := range block.layers {
		layerStates[i] = makeLlamaLayerState(block.layers[i], block.hiddenDim, maxSeq, block.attnHeads)
	}

	currentToken := seedToken(seed, block.vocabDim)
	startPos := 0
	if len(promptTokens) > 0 {
		currentToken = promptTokens[len(promptTokens)-1]
		for pos := 0; pos < len(promptTokens)-1; pos++ {
			runLlamaStackStep(block, layerStates, promptTokens[pos], pos, x, n1, n2, logits)
		}
		startPos = len(promptTokens) - 1
	}

	for i := range out {
		runLlamaStackStep(block, layerStates, currentToken, startPos+i, x, n1, n2, logits)
		if topk != nil {
			*topk = appendTopKStep(*topk, i, logits, 5)
		}
		next := kernels.Argmax(logits)
		if next < 0 {
			out[i] = 0
			currentToken = 0
			continue
		}
		out[i] = int32(next)
		currentToken = out[i]
	}
}

type llamaLayerState struct {
	q       []float32
	k       []float32
	v       []float32
	attnAcc []float32
	attnOut []float32
	gate    []float32
	up      []float32
	ffnAct  []float32
	ffnDown []float32
	scores  []float32
	keys    []float32
	values  []float32
}

func makeLlamaLayerState(layer llamaLayer, hiddenDim, maxSeq, heads int) llamaLayerState {
	kdim := linearOutputLen(layer.attnK)
	vdim := linearOutputLen(layer.attnV)
	qdim := linearOutputLen(layer.attnQ)
	ffnDim := linearOutputLen(layer.ffnGate)
	if heads <= 0 {
		heads = 1
	}
	return llamaLayerState{
		q:       make([]float32, qdim),
		k:       make([]float32, kdim),
		v:       make([]float32, vdim),
		attnAcc: make([]float32, qdim),
		attnOut: make([]float32, hiddenDim),
		gate:    make([]float32, ffnDim),
		up:      make([]float32, linearOutputLen(layer.ffnUp)),
		ffnAct:  make([]float32, ffnDim),
		ffnDown: make([]float32, hiddenDim),
		scores:  make([]float32, maxSeq*heads),
		keys:    make([]float32, maxSeq*kdim),
		values:  make([]float32, maxSeq*vdim),
	}
}

func runLlamaStackStep(block *tensorBlock, layerStates []llamaLayerState, token int32, pos int, x, n1, n2, logits []float32) {
	embedded := embedToken(x, block, token)
	if !embedded {
		fillTokenVector(x, token)
	}

	var stageNormBuf []float32
	if debugStages && shouldDebug(pos) {
		stageNormBuf = make([]float32, block.hiddenDim)
		debugStage("stage.embed", x, block, stageNormBuf, false)
	}

	if debugOutput && shouldDebug(pos) && embedded {
		fmt.Fprintf(os.Stderr, "debug emb dims rows=%d cols=%d\n", block.tokenEmbdRows, block.tokenEmbdCols)
		fmt.Fprintf(os.Stderr, "debug output dims rows=%d cols=%d transposed=%v qtype=%d\n", block.outputRows, block.outputCols, block.outputTransposed, block.outputWeightType)

		xCol := make([]float32, block.hiddenDim)
		copy(xCol, x)
		xRow := make([]float32, block.hiddenDim)
		for r := 0; r < block.hiddenDim; r++ {
			xRow[r] = block.tokenEmbd[int(token)+block.tokenEmbdCols*r]
		}
		debugVecStats("embed.col", xCol)
		debugVecStats("embed.row", xRow)
		debugVecDiff("embed.col_vs_row", xCol, xRow)

		nCol := make([]float32, block.hiddenDim)
		nRow := make([]float32, block.hiddenDim)
		rmsNormInto(nCol, xCol, block.outputNorm, block.rmsEps)
		rmsNormInto(nRow, xRow, block.outputNorm, block.rmsEps)
		debugVecStats("output_norm.col", nCol)
		debugVecStats("output_norm.row", nRow)
		debugVecDiff("output_norm.col_vs_row", nCol, nRow)

		w := linearWeight{
			data:       block.outputWeight,
			rows:       block.outputRows,
			cols:       block.outputCols,
			transposed: block.outputTransposed,
			qtype:      block.outputWeightType,
			i2sPacked:  block.outputWeightPacked,
			i2sScale:   block.outputWeightScale,
		}
		debugLogitsForTokens("col", w, nCol)
		debugLogitsForTokens("row", w, nRow)
		debugStep0Printed = true
	}
	if debugOutputOnly && debugStep0Printed {
		return
	}

	if disableLayers {
		if shouldDebug(pos) {
			fmt.Fprintln(os.Stderr, "debug layers: disabled")
		}
		rmsNormInto(n1, x, block.outputNorm, block.rmsEps)
		linearApplyIntoWeight(logits, linearWeight{
			data:       block.outputWeight,
			rows:       block.outputRows,
			cols:       block.outputCols,
			transposed: block.outputTransposed,
			qtype:      block.outputWeightType,
			i2sPacked:  block.outputWeightPacked,
			i2sScale:   block.outputWeightScale,
		}, n1)
		if shouldDebug(pos) {
			debugVecStats("output_norm", n1)
			debugVecStats("logits", logits)
			debugStep0Printed = true
		}
		return
	}

	for i := range block.layers {
		layer := block.layers[i]
		st := &layerStates[i]

		if !disableAttn {
			rmsNormInto(n1, x, layer.attnNorm, block.rmsEps)
			if debugStages && shouldDebug(pos) && i == 0 {
				debugStage("stage.attn_norm", n1, block, stageNormBuf, false)
			}
			if shouldDebug(pos) && i == 0 {
				debugVecStats("x.embed", x)
				debugVecStats("attn_norm", n1)
			}
			linearApplyIntoWeight(st.q, layer.attnQ, n1)
			linearApplyIntoWeight(st.k, layer.attnK, n1)
			linearApplyIntoWeight(st.v, layer.attnV, n1)
			if debugAttnMeta && shouldDebug(pos) && i == 0 {
				qHead := 0
				kHead := 0
				if block.attnHeads > 0 {
					qHead = len(st.q) / block.attnHeads
				}
				if block.kvHeads > 0 {
					kHead = len(st.k) / block.kvHeads
				}
				fmt.Fprintf(os.Stderr, "debug attn dims: q=%d k=%d v=%d qHeadDim=%d kHeadDim=%d\n", len(st.q), len(st.k), len(st.v), qHead, kHead)
				debugVecSlice("q.pre", st.q, 8)
				debugVecSlice("k.pre", st.k, 8)
			}
			if shouldDebug(pos) && i == 0 {
				debugVecStats("q", st.q)
				debugVecStats("k", st.k)
				debugVecStats("v", st.v)
			}
			applyRoPEInPlace(st.q, pos, block.attnHeads, block.ropeFreqBase, block.ropeScale, block.ropeScalingType, block.ropeDim, block.ropeYarnBetaFast, block.ropeYarnBetaSlow, block.ropeYarnExtFactor, block.ropeYarnAttnFactor)
			applyRoPEInPlace(st.k, pos, block.kvHeads, block.ropeFreqBase, block.ropeScale, block.ropeScalingType, block.ropeDim, block.ropeYarnBetaFast, block.ropeYarnBetaSlow, block.ropeYarnExtFactor, block.ropeYarnAttnFactor)
			if debugAttnMeta && shouldDebug(pos) && i == 0 {
				debugVecSlice("q.post", st.q, 8)
				debugVecSlice("k.post", st.k, 8)
			}

			storeCacheVector(st.keys, pos, st.k)
			storeCacheVector(st.values, pos, st.v)
			causalAttentionMultiHeadInto(st.attnAcc, st.scores, st.q, st.keys, st.values, pos+1, block.attnHeads, block.kvHeads, len(st.k), len(st.v))

			rmsNormInto(n2, st.attnAcc, layer.attnSubNorm, block.rmsEps)
			if debugStages && shouldDebug(pos) && i == 0 {
				debugStage("stage.attn_sub_norm", n2, block, stageNormBuf, false)
			}
			if debugValues && shouldDebug(pos) && i == 0 {
				debugVecValues("attn_sub_norm", n2, debugValuesN)
			}
			linearApplyIntoWeight(st.attnOut, layer.attnOut, n2)
			kernels.AddScaled(x, st.attnOut, 1.0)
			if debugStages && shouldDebug(pos) && i == 0 {
				debugStage("stage.post_attn", x, block, stageNormBuf, false)
			}
			if shouldDebug(pos) && i == 0 {
				debugVecStats("attn_acc", st.attnAcc)
				debugVecStats("attn_out", st.attnOut)
				debugVecStats("x.post_attn", x)
			}
		} else if shouldDebug(pos) && i == 0 {
			fmt.Fprintln(os.Stderr, "debug attn: disabled")
		}

		if !disableFFN {
			rmsNormInto(n2, x, layer.ffnNorm, block.rmsEps)
			if debugStages && shouldDebug(pos) && i == 0 {
				debugStage("stage.ffn_norm", n2, block, stageNormBuf, false)
			}
			if debugValues && shouldDebug(pos) && i == 0 {
				debugVecValues("ffn_norm", n2, debugValuesN)
			}
			if debugFFNTranspose {
				linearApplyIntoWeightTransposed(st.gate, layer.ffnGate, n2, !layer.ffnGate.transposed)
				linearApplyIntoWeightTransposed(st.up, layer.ffnUp, n2, !layer.ffnUp.transposed)
			} else {
				linearApplyIntoWeight(st.gate, layer.ffnGate, n2)
				linearApplyIntoWeight(st.up, layer.ffnUp, n2)
			}
			if debugFFNLoad && shouldDebug(pos) && i == 0 {
				debugFfnCompare("ffn_gate", st.gate, layer.ffnGate, layer.debugFFNGateF32, n2)
				debugFfnCompare("ffn_up", st.up, layer.ffnUp, layer.debugFFNUpF32, n2)
			}
			if shouldDebug(pos) && i == 0 {
				debugVecStats("ffn_norm", n2)
				debugVecStats("ffn_gate", st.gate)
				debugVecStats("ffn_up", st.up)
			}
			n := len(st.gate)
			if len(st.up) < n {
				n = len(st.up)
			}
			for j := 0; j < n; j++ {
				gg := st.gate[j]
				if gg < 0 {
					gg = 0
				}
				gg = gg * gg
				st.ffnAct[j] = gg * st.up[j]
			}
			if debugValues && shouldDebug(pos) && i == 0 {
				debugVecValues("ffn_out", st.ffnAct, debugValuesN)
			}
			rmsNormInto(st.up, st.ffnAct, layer.ffnSubNorm, block.rmsEps)
			if debugStages && shouldDebug(pos) && i == 0 {
				debugStage("stage.ffn_sub_norm", st.up, block, stageNormBuf, false)
			}
			if debugValues && shouldDebug(pos) && i == 0 {
				debugVecValues("ffn_sub_norm", st.up, debugValuesN)
			}
			if debugFFNTranspose {
				linearApplyIntoWeightTransposed(st.ffnDown, layer.ffnDown, st.up, !layer.ffnDown.transposed)
			} else {
				linearApplyIntoWeight(st.ffnDown, layer.ffnDown, st.up)
			}
			if debugFFNLoad && shouldDebug(pos) && i == 0 {
				debugFfnCompare("ffn_down", st.ffnDown, layer.ffnDown, layer.debugFFNDownF32, st.up)
			}
			kernels.AddScaled(x, st.ffnDown, 1.0)
			if debugStages && shouldDebug(pos) && i == 0 {
				debugStage("stage.post_ffn", x, block, stageNormBuf, false)
			}
			if shouldDebug(pos) && i == 0 {
				debugVecStats("ffn_act", st.ffnAct)
				debugVecStats("ffn_down", st.ffnDown)
				debugVecStats("x.post_ffn", x)
			}
		} else if shouldDebug(pos) && i == 0 {
			fmt.Fprintln(os.Stderr, "debug ffn: disabled")
		}
	}

	rmsNormInto(n1, x, block.outputNorm, block.rmsEps)
	if debugStages && shouldDebug(pos) {
		debugStage("stage.output_norm", n1, block, stageNormBuf, true)
	}
	if debugValues && shouldDebug(pos) {
		debugVecValues("result_norm", n1, debugValuesN)
	}
	linearApplyIntoWeight(logits, linearWeight{
		data:       block.outputWeight,
		rows:       block.outputRows,
		cols:       block.outputCols,
		transposed: block.outputTransposed,
		qtype:      block.outputWeightType,
		i2sPacked:  block.outputWeightPacked,
		i2sScale:   block.outputWeightScale,
	}, n1)
	if shouldDebug(pos) {
		debugVecStats("output_norm", n1)
		debugVecStats("logits", logits)
		debugStep0Printed = true
	}
}

func causalAttentionMultiHeadInto(dst, scores, q, keys, values []float32, steps, qHeads, kvHeads, kStepDim, vStepDim int) {
	for i := range dst {
		dst[i] = 0
	}
	if steps <= 0 || len(q) == 0 {
		return
	}

	if qHeads <= 0 {
		qHeads = 1
	}
	if kvHeads <= 0 {
		kvHeads = qHeads
	}
	if len(q)%qHeads != 0 {
		qHeads = 1
		kvHeads = 1
	}
	if kStepDim <= 0 || vStepDim <= 0 || kStepDim%kvHeads != 0 || vStepDim%kvHeads != 0 {
		return
	}
	headDim := len(q) / qHeads
	if headDim == 0 {
		return
	}
	if len(scores) < steps*qHeads {
		return
	}
	if kStepDim/kvHeads != headDim || vStepDim/kvHeads != headDim {
		return
	}

	for h := 0; h < qHeads; h++ {
		qBase := h * headDim
		qh := q[qBase : qBase+headDim]
		kvHead := h * kvHeads / qHeads
		kBase := kvHead * headDim
		scale := 1.0 / math.Sqrt(float64(headDim))
		maxScore := -math.MaxFloat64
		for i := 0; i < steps; i++ {
			kb := i*kStepDim + kBase
			var sum float64
			for j := 0; j < headDim; j++ {
				sum += float64(qh[j]) * float64(keys[kb+j])
			}
			s := sum * scale
			scores[h*steps+i] = float32(s)
			if s > maxScore {
				maxScore = s
			}
		}

		var sum float64
		for i := 0; i < steps; i++ {
			idx := h*steps + i
			w := math.Exp(float64(scores[idx]) - maxScore)
			scores[idx] = float32(w)
			sum += w
		}
		if sum == 0 {
			continue
		}
		inv := 1.0 / sum
		for i := 0; i < steps; i++ {
			w := float32(float64(scores[h*steps+i]) * inv)
			vb := i*vStepDim + kBase
			for j := 0; j < headDim; j++ {
				dst[qBase+j] += values[vb+j] * w
			}
		}
	}
}

func storeCacheVector(cache []float32, pos int, vec []float32) {
	base := pos * len(vec)
	copy(cache[base:base+len(vec)], vec)
}

func seedToken(seed int64, vocab int) int32 {
	if vocab <= 0 {
		return int32(seed)
	}
	x := seed % int64(vocab)
	if x < 0 {
		x += int64(vocab)
	}
	return int32(x)
}

func applyRoPEInPlace(v []float32, pos, heads int, base, scale float32, scalingType string, ropeDim int, betaFast, betaSlow, extFactor, attnFactor float32) {
	if heads <= 0 || len(v) == 0 {
		return
	}
	if len(v)%heads != 0 {
		heads = 1
	}
	headDim := len(v) / heads
	if headDim < 2 || base <= 0 {
		return
	}
	if ropeDim <= 0 || ropeDim > headDim {
		ropeDim = headDim
	}
	half := ropeDim / 2
	if half == 0 {
		return
	}
	posf := ropeScaledPosition(pos, scale, scalingType)
	basef := float64(base)

	for h := 0; h < heads; h++ {
		offset := h * headDim
		for i := 0; i+1 < ropeDim; i += 2 {
			pair := i / 2
			exponent := float64(pair) / float64(half)
			thetaBase := posf / math.Pow(basef, exponent)
			cosT, sinT := ropeCosSin(thetaBase, scale, scalingType, betaFast, betaSlow, extFactor, attnFactor, i)
			x0 := v[offset+i]
			x1 := v[offset+i+1]
			v[offset+i] = x0*cosT - x1*sinT
			v[offset+i+1] = x0*sinT + x1*cosT
		}
	}
}

func ropeScaledPosition(pos int, scale float32, scalingType string) float64 {
	p := float64(pos)
	if scale <= 0 {
		return p
	}
	switch scalingType {
	case "linear":
		if scale != 1 {
			return p / float64(scale)
		}
	case "yarn":
		if scale != 1 {
			return p / float64(scale)
		}
	default:
		if scale != 1 {
			return p / float64(scale)
		}
	}
	return p
}

func ropeCosSin(thetaBase float64, scale float32, scalingType string, betaFast, betaSlow, extFactor, attnFactor float32, i0 int) (float32, float32) {
	switch scalingType {
	case "yarn":
		return ropeYarnCosSin(thetaBase, scale, betaFast, betaSlow, extFactor, attnFactor, i0)
	default:
		theta := thetaBase
		cosT := float32(math.Cos(theta))
		sinT := float32(math.Sin(theta))
		return cosT, sinT
	}
}

func ropeYarnCosSin(thetaExtrap float64, scale, betaFast, betaSlow, extFactor, attnFactor float32, i0 int) (float32, float32) {
	freqScale := float32(1.0)
	if scale != 0 {
		freqScale = 1 / scale
	}
	corrLow, corrHigh := ropeYarnCorrDims(betaFast, betaSlow)
	thetaInterp := float32(freqScale) * float32(thetaExtrap)
	theta := thetaInterp
	mscale := attnFactor
	if extFactor != 0 {
		rampMix := ropeYarnRamp(corrLow, corrHigh, i0) * extFactor
		theta = thetaInterp*(1-rampMix) + float32(thetaExtrap)*rampMix
		mscale *= 1.0 + 0.1*float32(math.Log(float64(1.0/freqScale)))
	}
	cosT := float32(math.Cos(float64(theta))) * mscale
	sinT := float32(math.Sin(float64(theta))) * mscale
	return cosT, sinT
}

func ropeYarnRamp(low, high float32, i0 int) float32 {
	denom := high - low
	if denom < 0.001 {
		denom = 0.001
	}
	y := (float32(i0)/2 - low) / denom
	if y < 0 {
		y = 0
	} else if y > 1 {
		y = 1
	}
	return 1 - y
}

func ropeYarnCorrDims(betaFast, betaSlow float32) (float32, float32) {
	low := betaSlow
	high := betaFast
	if low < 0 {
		low = 0
	}
	if high < low {
		high = low
	}
	return low, high
}

func rmsNormInto(dst, x, weight []float32, eps float32) {
	n := len(dst)
	if len(x) < n {
		n = len(x)
	}
	if len(weight) < n {
		n = len(weight)
	}
	if n == 0 {
		return
	}
	var sum float64
	for i := 0; i < n; i++ {
		v := float64(x[i])
		sum += v * v
	}
	inv := float32(1.0 / math.Sqrt(sum/float64(n)+float64(eps)))
	for i := 0; i < n; i++ {
		dst[i] = x[i] * inv * weight[i]
	}
}

func debugVecStats(label string, v []float32) {
	if len(v) == 0 {
		fmt.Fprintf(os.Stderr, "debug %s: empty\n", label)
		return
	}
	min := v[0]
	max := v[0]
	var sum float32
	var sumSq float64
	for _, val := range v {
		if val < min {
			min = val
		}
		if val > max {
			max = val
		}
		sum += val
		sumSq += float64(val) * float64(val)
	}
	mean := sum / float32(len(v))
	rms := math.Sqrt(sumSq / float64(len(v)))
	fmt.Fprintf(os.Stderr, "debug %s: n=%d min=%g max=%g mean=%g rms=%g\n", label, len(v), min, max, mean, rms)
}

func debugVecDiff(label string, a, b []float32) {
	if len(a) == 0 || len(b) == 0 {
		fmt.Fprintf(os.Stderr, "debug %s: empty\n", label)
		return
	}
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var sum float32
	var maxAbs float32
	for i := 0; i < n; i++ {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		sum += d
		if d > maxAbs {
			maxAbs = d
		}
	}
	mean := sum / float32(n)
	fmt.Fprintf(os.Stderr, "debug %s: n=%d mean_abs=%g max_abs=%g\n", label, n, mean, maxAbs)
}

func debugVecSlice(label string, v []float32, n int) {
	if len(v) == 0 {
		fmt.Fprintf(os.Stderr, "debug %s: empty\n", label)
		return
	}
	if n <= 0 {
		return
	}
	if len(v) < n {
		n = len(v)
	}
	fmt.Fprintf(os.Stderr, "debug %s:", label)
	for i := 0; i < n; i++ {
		fmt.Fprintf(os.Stderr, " %g", v[i])
	}
	fmt.Fprintln(os.Stderr)
}

func debugVecValues(label string, v []float32, n int) {
	if n <= 0 {
		return
	}
	if len(v) < n {
		n = len(v)
	}
	if n == 0 {
		return
	}
	fmt.Fprintf(os.Stderr, "debug_values %s values=", label)
	for i := 0; i < n; i++ {
		if i > 0 {
			fmt.Fprint(os.Stderr, ",")
		}
		fmt.Fprintf(os.Stderr, "%g", v[i])
	}
	fmt.Fprintln(os.Stderr)
}

func debugLogitsForTokens(label string, w linearWeight, x []float32) {
	if len(debugTokens) == 0 {
		return
	}
	if w.qtype == gguf.GGMLTypeI2_S && len(w.i2sPacked) > 0 {
		vec := make([]int8, len(x))
		actScale, actSum := kernels.QuantizeRowI8S(vec, x)
		for _, tok := range debugTokens {
			v := i2sLogitForToken(w.i2sPacked, w.rows, w.cols, tok, vec, w.i2sScale, actScale, actSum, w.transposed)
			alt := i2sLogitForToken(w.i2sPacked, w.rows, w.cols, tok, vec, w.i2sScale, actScale, actSum, !w.transposed)
			fmt.Fprintf(os.Stderr, "debug logits.%s token=%d logit=%g altT=%g\n", label, tok, v, alt)
		}
		return
	}
	for _, tok := range debugTokens {
		v := f32LogitForToken(w.data, w.rows, w.cols, tok, x, w.transposed)
		alt := f32LogitForToken(w.data, w.rows, w.cols, tok, x, !w.transposed)
		fmt.Fprintf(os.Stderr, "debug logits.%s token=%d logit=%g altT=%g\n", label, tok, v, alt)
	}
}

func debugStage(label string, vec []float32, block *tensorBlock, normBuf []float32, alreadyNorm bool) {
	debugVecStats(label, vec)
	w := linearWeight{
		data:       block.outputWeight,
		rows:       block.outputRows,
		cols:       block.outputCols,
		transposed: block.outputTransposed,
		qtype:      block.outputWeightType,
		i2sPacked:  block.outputWeightPacked,
		i2sScale:   block.outputWeightScale,
	}
	if alreadyNorm {
		debugLogitsForTokens(label, w, vec)
		return
	}
	if len(normBuf) < len(vec) {
		return
	}
	rmsNormInto(normBuf[:len(vec)], vec, block.outputNorm, block.rmsEps)
	debugVecStats(label+".outnorm", normBuf[:len(vec)])
	debugLogitsForTokens(label+".outnorm", w, normBuf[:len(vec)])
}

func debugFfnCompare(label string, got []float32, w linearWeight, wF32 []float32, x []float32) {
	if len(wF32) == 0 {
		return
	}
	out := make([]float32, len(got))
	if w.transposed {
		kernels.MatVecT(out, wF32, w.rows, w.cols, x)
	} else {
		kernels.MatVec(out, wF32, w.rows, w.cols, x)
	}
	debugVecStats(label+".f32", out)
	debugVecDiff(label+".diff", got, out)
}

func f32LogitForToken(mat []float32, rows, cols, token int, x []float32, transposed bool) float32 {
	if transposed {
		if token < 0 || token >= cols || len(x) < rows {
			return 0
		}
		var sum float32
		for r := 0; r < rows; r++ {
			sum += mat[r+rows*token] * x[r]
		}
		return sum
	}
	if token < 0 || token >= rows || len(x) < cols {
		return 0
	}
	var sum float32
	for c := 0; c < cols; c++ {
		sum += mat[token+rows*c] * x[c]
	}
	return sum
}

func i2sLogitForToken(packed []byte, rows, cols, token int, vec []int8, weightScale, actScale float32, actSum int32, transposed bool) float32 {
	if actScale == 0 {
		return 0
	}
	if transposed {
		if token < 0 || token >= cols || len(vec) < rows {
			return 0
		}
		var sum int32
		for r := 0; r < rows; r++ {
			idx := r + rows*token
			q := i2sPackedAtLocal(packed, idx)
			if q == 3 {
				q = 1
			}
			sum += int32(q) * int32(vec[r])
		}
		return float32(sum-actSum) * (weightScale / actScale)
	}
	if token < 0 || token >= rows || len(vec) < cols {
		return 0
	}
	var sum int32
	for c := 0; c < cols; c++ {
		idx := token + rows*c
		q := i2sPackedAtLocal(packed, idx)
		if q == 3 {
			q = 1
		}
		sum += int32(q) * int32(vec[c])
	}
	return float32(sum-actSum) * (weightScale / actScale)
}

func i2sPackedAtLocal(packed []byte, idx int) byte {
	if idx < 0 {
		return 0
	}
	const block = 128
	const blockBytes = 32
	bi := idx / block
	off := idx % block
	gp := off % 32
	group := off / 32
	p := bi*blockBytes + gp
	if p < 0 || p >= len(packed) {
		return 0
	}
	shift := uint(6 - 2*group)
	return (packed[p] >> shift) & 0x3
}

func linearOutputLen(w linearWeight) int {
	if w.transposed {
		return w.cols
	}
	return w.rows
}

func linearApplyIntoWeight(dst []float32, w linearWeight, x []float32) {
	if w.qtype == gguf.GGMLTypeI2_S && len(w.i2sPacked) > 0 {
		if debugI2SFloat {
			if w.transposed {
				kernels.MatVecTI2S(dst, w.i2sPacked, w.rows, w.cols, x, w.i2sScale)
			} else {
				kernels.MatVecI2S(dst, w.i2sPacked, w.rows, w.cols, x, w.i2sScale)
			}
			return
		}
		scratch := i8ScratchPool.Get().([]int8)
		if cap(scratch) < len(x) {
			scratch = make([]int8, len(x))
		} else {
			scratch = scratch[:len(x)]
		}
		actScale, actSum := kernels.QuantizeRowI8S(scratch, x)
		if debugI2SDisableActSum {
			actSum = 0
		}
		if debugI2SInvertActScale && actScale != 0 {
			actScale = 1 / actScale
		}
		if w.transposed {
			kernels.MatVecTI2SI8S(dst, w.i2sPacked, w.rows, w.cols, scratch, w.i2sScale, actScale, actSum)
		} else {
			kernels.MatVecI2SI8S(dst, w.i2sPacked, w.rows, w.cols, scratch, w.i2sScale, actScale, actSum)
		}
		i8ScratchPool.Put(scratch[:0])
		return
	}
	if w.transposed {
		kernels.MatVecT(dst, w.data, w.rows, w.cols, x)
		return
	}
	kernels.MatVec(dst, w.data, w.rows, w.cols, x)
}

func linearApplyIntoWeightTransposed(dst []float32, w linearWeight, x []float32, transposed bool) {
	w.transposed = transposed
	linearApplyIntoWeight(dst, w, x)
}

func loadLinearWeight(path string, info gguf.ModelInfo, name string, inDim int) (linearWeight, error) {
	data, rows, cols, transposed, err := loadLinearTensor(path, info, name, inDim)
	if err != nil {
		return linearWeight{}, err
	}
	ti, _ := info.TensorByName(name)
	w := linearWeight{
		data:       data,
		rows:       rows,
		cols:       cols,
		transposed: transposed,
		qtype:      ti.Type,
	}
	if ti.Type == gguf.GGMLTypeI2_S {
		packed, scale, _, err := gguf.ReadTensorI2SPacked(path, info, name)
		if err != nil {
			return linearWeight{}, err
		}
		w.i2sPacked = packed
		w.i2sScale = scale
	}
	return w, nil
}

func loadLinearTensor(path string, info gguf.ModelInfo, name string, inDim int) ([]float32, int, int, bool, error) {
	ti, ok := info.TensorByName(name)
	if !ok {
		return nil, 0, 0, false, fmt.Errorf("missing tensor: %s", name)
	}
	if len(ti.Dimensions) != 2 {
		return nil, 0, 0, false, fmt.Errorf("%s: expected 2 dims, got %d", name, len(ti.Dimensions))
	}
	rows := int(ti.Dimensions[0])
	cols := int(ti.Dimensions[1])
	if rows <= 0 || cols <= 0 {
		return nil, 0, 0, false, fmt.Errorf("%s: invalid dims %v", name, ti.Dimensions)
	}
	if ti.Type == gguf.GGMLTypeI2_S {
		return nil, rows, cols, rows == inDim, nil
	}
	data, err := gguf.ReadTensorAsF32(path, info, name)
	if err != nil {
		return nil, 0, 0, false, err
	}
	if rows == inDim {
		return data, rows, cols, true, nil
	}
	if cols == inDim {
		return data, rows, cols, false, nil
	}
	return nil, 0, 0, false, fmt.Errorf("%s dims %v incompatible with inDim=%d", name, ti.Dimensions, inDim)
}

func firstFloat32(v any, fallback float32) float32 {
	switch x := v.(type) {
	case float32:
		return x
	case float64:
		return float32(x)
	case uint32:
		return float32(x)
	case uint64:
		return float32(x)
	case int32:
		return float32(x)
	case int64:
		return float32(x)
	default:
		return fallback
	}
}

func firstFloat32From(fallback float32, values ...any) float32 {
	for _, v := range values {
		switch x := v.(type) {
		case float32:
			return x
		case float64:
			return float32(x)
		case uint32:
			return float32(x)
		case uint64:
			return float32(x)
		case int32:
			return float32(x)
		case int64:
			return float32(x)
		}
	}
	return fallback
}

func firstString(values ...any) string {
	for _, v := range values {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

func runForwardStub(vocabSize uint32, seed int64, promptTokens []int32, out []int32, topk *[]TopKStep) {
	const hiddenDim = 32
	state := make([]float32, hiddenDim)
	tokenVec := make([]float32, hiddenDim)
	logitsCap := int(vocabSize)
	if logitsCap <= 0 {
		logitsCap = 32000
	}
	if logitsCap > 4096 {
		logitsCap = 4096
	}
	logits := make([]float32, logitsCap)

	mixState(state, tokenVec, int32(seed))
	for _, tok := range promptTokens {
		mixState(state, tokenVec, tok)
	}

	for i := range out {
		for id := 0; id < logitsCap; id++ {
			fillTokenVector(tokenVec, int32(id))
			logits[id] = kernels.Dot(state, tokenVec)
		}
		if topk != nil {
			*topk = appendTopKStep(*topk, i, logits, 5)
		}

		next := kernels.Argmax(logits)
		if next < 0 {
			out[i] = 0
			continue
		}
		out[i] = int32(next)

		fillTokenVector(tokenVec, out[i])
		kernels.AddScaled(state, tokenVec, 0.05)
	}
}

func appendTopKStep(dst []TopKStep, step int, logits []float32, k int) []TopKStep {
	if k <= 0 || len(logits) == 0 {
		return append(dst, TopKStep{Step: step})
	}
	if k > len(logits) {
		k = len(logits)
	}
	entries := make([]TopKEntry, 0, k)
	for id, logit := range logits {
		entry := TopKEntry{TokenID: int32(id), Logit: logit}
		if len(entries) == 0 {
			entries = append(entries, entry)
			continue
		}
		insert := len(entries)
		for i := 0; i < len(entries); i++ {
			if entry.Logit > entries[i].Logit {
				insert = i
				break
			}
		}
		if len(entries) < k {
			entries = append(entries, TopKEntry{})
			copy(entries[insert+1:], entries[insert:])
			entries[insert] = entry
			continue
		}
		if insert < k {
			copy(entries[insert+1:], entries[insert:k-1])
			entries[insert] = entry
		}
	}
	return append(dst, TopKStep{Step: step, Entries: entries})
}

func mixState(state, scratch []float32, token int32) {
	fillTokenVector(scratch, token)
	kernels.AddScaled(state, scratch, 1.0)
}

func fillTokenVector(dst []float32, token int32) {
	x := uint32(token) + 0x9e3779b9
	for i := 0; i < len(dst); i++ {
		x ^= x << 13
		x ^= x >> 17
		x ^= x << 5
		dst[i] = float32(int32(x&0xffff)-32768) / 32768.0
	}
}
