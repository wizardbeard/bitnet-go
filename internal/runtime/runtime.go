package runtime

import (
	"context"
	"fmt"
	"math"
	"os"
	"strconv"

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
var debugPos = parseDebugPos(os.Getenv("BITNET_DEBUG_POS"))
var debugStep0Printed bool

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
		info.KeyValues["llama.context_length"],
		info.KeyValues["falcon.context_length"],
		info.KeyValues["gpt2.context_length"],
	)
	vocab := firstUint32(
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
	attnQ    linearWeight
	attnK    linearWeight
	attnV    linearWeight
	attnOut  linearWeight
	ffnGate  linearWeight
	ffnUp    linearWeight
	ffnDown  linearWeight
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
		rmsEps:        firstFloat32(info.KeyValues["llama.attention.layer_norm_rms_epsilon"], float32(1e-5)),
		attnHeads:     int(firstUint32(info.KeyValues["llama.attention.head_count"])),
		kvHeads:       int(firstUint32(info.KeyValues["llama.attention.head_count_kv"])),
		ropeFreqBase:  firstFloat32(info.KeyValues["llama.rope.freq_base"], float32(10000)),
		ropeScale:     firstFloat32(info.KeyValues["llama.rope.scaling.factor"], 1.0),
		ropeScalingType: firstString(
			info.KeyValues["llama.rope.scaling.type"],
			info.KeyValues["llama.rope.scaling_type"],
		),
		ropeDim:            int(firstUint32(info.KeyValues["llama.rope.dimension_count"])),
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
	if outInfo, ok := info.TensorByName("output.weight"); ok {
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
	}

	for idx := 0; ; idx++ {
		prefix := fmt.Sprintf("blk.%d.", idx)
		if _, ok := info.TensorByName(prefix + "attn_q.weight"); !ok {
			if idx == 0 {
				return nil, false, fmt.Errorf("missing tensor: %sattn_q.weight", prefix)
			}
			break
		}
		layer, err := loadLlamaLayer(path, info, prefix, hidden, b.attnHeads, b.kvHeads)
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

func loadLlamaLayer(path string, info gguf.ModelInfo, prefix string, hidden, attnHeads, kvHeads int) (llamaLayer, error) {
	var l llamaLayer
	var err error

	if l.attnNorm, err = gguf.ReadTensorAsF32(path, info, prefix+"attn_norm.weight"); err != nil {
		return llamaLayer{}, err
	}
	if len(l.attnNorm) != hidden {
		return llamaLayer{}, fmt.Errorf("%sattn_norm.weight len=%d want=%d", prefix, len(l.attnNorm), hidden)
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
	if linearOutputLen(l.ffnGate) != linearOutputLen(l.ffnUp) {
		return llamaLayer{}, fmt.Errorf("%sincompatible ffn gate/up dims", prefix)
	}
	if l.ffnDown, err = loadLinearWeight(path, info, prefix+"ffn_down.weight", linearOutputLen(l.ffnGate)); err != nil {
		return llamaLayer{}, err
	}
	if linearOutputLen(l.ffnDown) != hidden {
		return llamaLayer{}, fmt.Errorf("%sffn_down.weight output dim=%d want=%d", prefix, linearOutputLen(l.ffnDown), hidden)
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
	if !embedToken(x, block, token) {
		fillTokenVector(x, token)
	}

	for i := range block.layers {
		layer := block.layers[i]
		st := &layerStates[i]

		rmsNormInto(n1, x, layer.attnNorm, block.rmsEps)
		if shouldDebug(pos) && i == 0 {
			debugVecStats("x.embed", x)
			debugVecStats("attn_norm", n1)
		}
		linearApplyIntoWeight(st.q, layer.attnQ, n1)
		linearApplyIntoWeight(st.k, layer.attnK, n1)
		linearApplyIntoWeight(st.v, layer.attnV, n1)
		if shouldDebug(pos) && i == 0 {
			debugVecStats("q", st.q)
			debugVecStats("k", st.k)
			debugVecStats("v", st.v)
		}
		applyRoPEInPlace(st.q, pos, block.attnHeads, block.ropeFreqBase, block.ropeScale, block.ropeScalingType, block.ropeDim, block.ropeYarnBetaFast, block.ropeYarnBetaSlow, block.ropeYarnExtFactor, block.ropeYarnAttnFactor)
		applyRoPEInPlace(st.k, pos, block.kvHeads, block.ropeFreqBase, block.ropeScale, block.ropeScalingType, block.ropeDim, block.ropeYarnBetaFast, block.ropeYarnBetaSlow, block.ropeYarnExtFactor, block.ropeYarnAttnFactor)

		storeCacheVector(st.keys, pos, st.k)
		storeCacheVector(st.values, pos, st.v)
		causalAttentionMultiHeadInto(st.attnAcc, st.scores, st.q, st.keys, st.values, pos+1, block.attnHeads, block.kvHeads, len(st.k), len(st.v))

		linearApplyIntoWeight(st.attnOut, layer.attnOut, st.attnAcc)
		kernels.AddScaled(x, st.attnOut, 1.0)
		if shouldDebug(pos) && i == 0 {
			debugVecStats("attn_acc", st.attnAcc)
			debugVecStats("attn_out", st.attnOut)
			debugVecStats("x.post_attn", x)
		}

		if !disableFFN {
			rmsNormInto(n2, x, layer.ffnNorm, block.rmsEps)
			linearApplyIntoWeight(st.gate, layer.ffnGate, n2)
			linearApplyIntoWeight(st.up, layer.ffnUp, n2)
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
				st.ffnAct[j] = (gg / (1 + float32(math.Exp(float64(-gg))))) * st.up[j]
			}
			linearApplyIntoWeight(st.ffnDown, layer.ffnDown, st.ffnAct)
			kernels.AddScaled(x, st.ffnDown, 1.0)
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
	var sum float32
	for i := 0; i < n; i++ {
		sum += x[i] * x[i]
	}
	inv := float32(1.0 / math.Sqrt(float64(sum/float32(n)+eps)))
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

func linearOutputLen(w linearWeight) int {
	if w.transposed {
		return w.cols
	}
	return w.rows
}

func linearApplyIntoWeight(dst []float32, w linearWeight, x []float32) {
	if w.qtype == gguf.GGMLTypeI2_S && len(w.i2sPacked) > 0 {
		if w.transposed {
			kernels.MatVecTI2S(dst, w.i2sPacked, w.rows, w.cols, x, w.i2sScale)
			return
		}
		kernels.MatVecI2S(dst, w.i2sPacked, w.rows, w.cols, x, w.i2sScale)
		return
	}
	if w.transposed {
		kernels.MatVecT(dst, w.data, w.rows, w.cols, x)
		return
	}
	kernels.MatVec(dst, w.data, w.rows, w.cols, x)
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
