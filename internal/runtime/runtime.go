package runtime

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"bitnet-go/internal/gguf"
	"bitnet-go/internal/kernels"
	"bitnet-go/internal/tokenizer"
)

type GenerateRequest struct {
	Prompt             string
	Seed               int64
	MaxTokens          int
	Temp               float32
	TopP               float32
	TopK               int
	DisableTopKCapture bool
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

type topKWriter struct {
	steps   []TopKStep
	entries []TopKEntry
	next    int
	k       int
}

func newTopKWriter(steps, k int) *topKWriter {
	if steps <= 0 || k <= 0 {
		return &topKWriter{steps: nil, entries: nil, next: 0, k: k}
	}
	return &topKWriter{
		steps:   make([]TopKStep, 0, steps),
		entries: make([]TopKEntry, steps*k),
		next:    0,
		k:       k,
	}
}

func (w *topKWriter) append(step int, logits []float32) {
	if w == nil || w.k <= 0 {
		return
	}
	if w.next+w.k > len(w.entries) {
		// Fallback to allocating if the buffer is exhausted.
		w.steps = appendTopKStep(w.steps, step, logits, w.k)
		return
	}
	slice := w.entries[w.next : w.next+w.k]
	n := fillTopK(slice, logits, w.k)
	w.steps = append(w.steps, TopKStep{Step: step, Entries: slice[:n]})
	w.next += w.k
}

func (w *topKWriter) result() []TopKStep {
	if w == nil {
		return nil
	}
	return w.steps
}

type Runtime struct {
	meta             Metadata
	tokenizer        *tokenizer.Tokenizer
	block            *tensorBlock
	promptCacheMu    sync.RWMutex
	promptTokenCache map[string][]int32
	promptCacheOrder []string
	promptCacheCap   int
	decodeCacheMu    sync.RWMutex
	decodeTextCache  map[decodeCacheKey][]decodeCacheEntry
	decodeCacheOrder []decodeCacheKey
	decodeCacheCap   int
	decodeCacheMax   int
}

type decodeCacheKey struct {
	h1 uint64
	h2 uint64
	n  int
}

type decodeCacheEntry struct {
	tokens []int32
	text   string
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
var debugPosOffset = parseDebugPosOffset(os.Getenv("BITNET_DEBUG_POS_OFFSET"))
var debugTokens = parseDebugTokens(os.Getenv("BITNET_DEBUG_TOKENS"))
var debugSoftmaxPrinted bool
var debugParityStrict = os.Getenv("BITNET_PARITY_STRICT") == "1"
var debugStrictAttention = os.Getenv("BITNET_STRICT_ATTENTION") == "1"
var debugStrictExpf = os.Getenv("BITNET_STRICT_EXPF") == "1"
var debugFastExpf = os.Getenv("BITNET_FAST_EXPF") == "1" && !debugParityStrict
var debugAttnF64 = os.Getenv("BITNET_ATTN_F64") == "1"
var debugStrictKQ = os.Getenv("BITNET_STRICT_KQ") == "1" || debugParityStrict
var debugFastKQ = os.Getenv("BITNET_FAST_KQ_DOT") != "0" && !debugParityStrict
var debugFastV = os.Getenv("BITNET_FAST_V_DOT") != "0" && !debugParityStrict
var debugKVRowMajor = os.Getenv("BITNET_KV_ROWMAJOR") != "0"
var debugFastQKVCol = os.Getenv("BITNET_FAST_QKV_COL") == "1" && !debugParityStrict
var debugQKVFusedMax = parseEnvInt("BITNET_QKV_FUSED_MAX", 512*512)
var debugStrictAttnRef = os.Getenv("BITNET_STRICT_ATTENTION_REF") == "1"
var debugStrictFFNRef = os.Getenv("BITNET_STRICT_FFN_REF") == "1"
var debugMatchGGML = os.Getenv("BITNET_MATCH_GGML") == "1" || debugParityStrict
var debugAttnRef = os.Getenv("BITNET_DEBUG_ATTN_REF") == "1"
var debugFFNRef = os.Getenv("BITNET_DEBUG_FFN_REF") == "1"
var debugFfnActRef = os.Getenv("BITNET_DEBUG_FFN_ACT_REF") == "1"
var debugFFNRefF32 = os.Getenv("BITNET_DEBUG_FFN_REF_F32") == "1"
var debugEmbedRowMajor = os.Getenv("BITNET_DEBUG_EMBD_ROW_MAJOR") == "1"
var debugStep0Printed bool
var debugI2SDisableActSum = os.Getenv("BITNET_I2S_DISABLE_ACTSUM") == "1"
var debugI2SInvertActScale = os.Getenv("BITNET_I2S_INVERT_ACT_SCALE") == "1"
var debugI2SFloat = os.Getenv("BITNET_I2S_F32") == "1" || debugParityStrict
var debugI2SForceQuant = os.Getenv("BITNET_I2S_FORCE_Q") == "1"
var debugI2SPretransposeMax = parseEnvInt("BITNET_I2S_PRETRANSPOSE_MAX", 0)
var debugI2SRefDot = os.Getenv("BITNET_I2S_REF_DOT") == "1"
var debugI2SMatvecRef = os.Getenv("BITNET_DEBUG_I2S_MATVEC_REF") == "1"
var debugI2SMatvecPrinted bool
var debugI2SRefOnce = os.Getenv("BITNET_I2S_REF_ONCE") == "1"
var debugI2SRefOncePrinted bool
var debugI2SMap3To1 = os.Getenv("BITNET_I2S_MAP3_TO1") == "1"
var debugI2SAltLayout = os.Getenv("BITNET_I2S_ALT_LAYOUT") == "1"
var debugI2SScalar = os.Getenv("BITNET_I2S_SCALAR") == "1"
var disableTopK = os.Getenv("BITNET_DISABLE_TOPK") == "1"
var topPHeapCap = parseEnvInt("BITNET_TOPP_HEAP_CAP", 0)
var topPSortPrefix = parseEnvInt("BITNET_TOPP_SORT_PREFIX", 0)
var topPPrefilterK = parseEnvInt("BITNET_TOPP_PREFILTER_K", 0)
var promptCacheCapDefault = parseEnvInt("BITNET_PROMPT_CACHE_CAP", 128)
var decodeCacheCapDefault = parseEnvInt("BITNET_DECODE_CACHE_CAP", 256)
var decodeCacheMaxTokensDefault = parseEnvInt("BITNET_DECODE_CACHE_MAX_TOKENS", 64)
var profileLoad = os.Getenv("BITNET_PROFILE_LOAD") == "1"
var profileStep = os.Getenv("BITNET_PROFILE_STEP") == "1"
var ffnShareI2SQuant = os.Getenv("BITNET_FFN_SHARE_I2S_QUANT") == "1"
var ffnParGateUp = os.Getenv("BITNET_FFN_PAR_GATE_UP") == "1"
var useF16TokenEmbd = os.Getenv("BITNET_USE_F16_TOKEN_EMBD") == "1"
var fastGreedyArgmax = os.Getenv("BITNET_FAST_GREEDY_ARGMAX") == "1"
var useMmapI2S = os.Getenv("BITNET_MMAP_I2S") == "1"
var i8ScratchPool = sync.Pool{
	New: func() any {
		return make([]int8, 0)
	},
}

type samplingConfig struct {
	temp float32
	topP float32
	topK int
}

func (c *samplingConfig) normalize() {
	if c.temp < 0 {
		c.temp = 0
	}
	if c.topP <= 0 || c.topP > 1 {
		c.topP = 1
	}
	if c.topK < 0 {
		c.topK = 0
	}
}

type sampler struct {
	state uint64
}

func newSampler(seed int64) *sampler {
	state := uint64(seed) ^ 0x9e3779b97f4a7c15
	if state == 0 {
		state = 1
	}
	return &sampler{state: state}
}

func (s *sampler) nextU64() uint64 {
	x := s.state
	x ^= x >> 12
	x ^= x << 25
	x ^= x >> 27
	s.state = x
	return x * 2685821657736338717
}

func (s *sampler) nextFloat() float32 {
	const denom = 1.0 / (1 << 53)
	return float32(float64(s.nextU64()>>11) * denom)
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

func parseDebugPosOffset(v string) int {
	if v == "" {
		return 0
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return 0
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

func parseForceTokens(v string) []int32 {
	if v == "" {
		return nil
	}
	parts := strings.Split(v, ",")
	out := make([]int32, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		n, err := strconv.Atoi(part)
		if err != nil {
			continue
		}
		out = append(out, int32(n))
	}
	return out
}

func forceTokensFromEnv() []int32 {
	if v := os.Getenv("BITNET_PARITY_FORCE_TOKENS"); v != "" {
		return parseForceTokens(v)
	}
	return parseForceTokens(os.Getenv("BITNET_FORCE_TOKENS"))
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

func parseEnvInt(key string, fallback int) int {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return fallback
	}
	return n
}

func setTopPSortPrefixForTest(v int) func() {
	old := topPSortPrefix
	topPSortPrefix = v
	return func() {
		topPSortPrefix = old
	}
}

func shouldDebug(pos int) bool {
	if !debugStep0 || debugStep0Printed {
		return false
	}
	if debugPos >= 0 {
		return pos+debugPosOffset == debugPos
	}
	return pos == 0
}

func New(_ context.Context, modelPath string) (*Runtime, error) {
	t0 := time.Now()
	info, err := gguf.ReadModelInfo(modelPath)
	tInfo := time.Since(t0)
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

	tTokStart := time.Now()
	tok, _ := tokenizer.NewFromModelInfo(info)
	tTok := time.Since(tTokStart)
	tBlockStart := time.Now()
	block, err := loadTensorBlock(modelPath, info)
	if err != nil {
		return nil, err
	}
	tBlock := time.Since(tBlockStart)
	if profileLoad {
		fmt.Fprintf(os.Stderr, "load_profile model=%s read_model_info=%s tokenizer=%s tensor_block=%s total=%s\n",
			modelPath, tInfo, tTok, tBlock, time.Since(t0))
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
		tokenizer:        tok,
		block:            block,
		promptTokenCache: make(map[string][]int32),
		promptCacheCap:   promptCacheCapDefault,
		decodeTextCache:  make(map[decodeCacheKey][]decodeCacheEntry),
		decodeCacheCap:   decodeCacheCapDefault,
		decodeCacheMax:   decodeCacheMaxTokensDefault,
	}, nil
}

func (r *Runtime) Metadata() Metadata {
	return r.meta
}

func (r *Runtime) promptTokens(prompt string) []int32 {
	if r.tokenizer == nil {
		return nil
	}
	if r.promptCacheCap > 0 {
		r.promptCacheMu.RLock()
		if tok, ok := r.promptTokenCache[prompt]; ok {
			r.promptCacheMu.RUnlock()
			return tok
		}
		r.promptCacheMu.RUnlock()
	}
	tok := r.tokenizer.Tokenize(prompt)
	if r.promptCacheCap <= 0 {
		return tok
	}
	r.promptCacheMu.Lock()
	if existing, ok := r.promptTokenCache[prompt]; ok {
		r.promptCacheMu.Unlock()
		return existing
	}
	for len(r.promptCacheOrder) >= r.promptCacheCap {
		evict := r.promptCacheOrder[0]
		r.promptCacheOrder = r.promptCacheOrder[1:]
		delete(r.promptTokenCache, evict)
	}
	r.promptTokenCache[prompt] = tok
	r.promptCacheOrder = append(r.promptCacheOrder, prompt)
	r.promptCacheMu.Unlock()
	return tok
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

	promptTokens := r.promptTokens(req.Prompt)

	// Phase-2 stepping stone: minimal forward loop with naive kernels and
	// procedural weights. If model carries bitnet_go.* f32 tensors, use a first
	// tensor-backed block path instead.
	tokens := make([]int32, req.MaxTokens)
	var topkWriter *topKWriter
	if !disableTopK && !req.DisableTopKCapture {
		topkWriter = newTopKWriter(req.MaxTokens, 5)
	}
	cfg := samplingConfig{
		temp: req.Temp,
		topP: req.TopP,
		topK: req.TopK,
	}
	cfg.normalize()
	forceTokens := forceTokensFromEnv()
	if r.block != nil {
		runForwardTensorBlock(r.block, req.Seed, promptTokens, tokens, topkWriter, forceTokens, cfg)
	} else {
		runForwardStub(r.meta.VocabSize, req.Seed, promptTokens, tokens, topkWriter, cfg)
	}

	return struct {
		TokenIDs []int32
		Text     string
		TopK     []TopKStep
	}{
		TokenIDs: tokens,
		Text:     req.Prompt + r.decodeTokens(tokens),
		TopK:     topkWriter.result(),
	}, nil
}

func (r *Runtime) decodeTokens(tokens []int32) string {
	if r.tokenizer == nil {
		return ""
	}
	if len(tokens) == 0 || r.decodeCacheCap <= 0 || r.decodeCacheMax <= 0 || len(tokens) > r.decodeCacheMax {
		return r.tokenizer.Decode(tokens)
	}
	key := makeDecodeCacheKey(tokens)
	r.decodeCacheMu.RLock()
	if bucket, ok := r.decodeTextCache[key]; ok {
		for i := range bucket {
			if slices.Equal(bucket[i].tokens, tokens) {
				text := bucket[i].text
				r.decodeCacheMu.RUnlock()
				return text
			}
		}
	}
	r.decodeCacheMu.RUnlock()

	text := r.tokenizer.Decode(tokens)

	r.decodeCacheMu.Lock()
	if bucket, ok := r.decodeTextCache[key]; ok {
		for i := range bucket {
			if slices.Equal(bucket[i].tokens, tokens) {
				text = bucket[i].text
				r.decodeCacheMu.Unlock()
				return text
			}
		}
	}
	for len(r.decodeCacheOrder) >= r.decodeCacheCap {
		evict := r.decodeCacheOrder[0]
		r.decodeCacheOrder = r.decodeCacheOrder[1:]
		delete(r.decodeTextCache, evict)
	}
	copied := append([]int32(nil), tokens...)
	r.decodeTextCache[key] = append(r.decodeTextCache[key], decodeCacheEntry{
		tokens: copied,
		text:   text,
	})
	r.decodeCacheOrder = append(r.decodeCacheOrder, key)
	r.decodeCacheMu.Unlock()
	return text
}

func makeDecodeCacheKey(tokens []int32) decodeCacheKey {
	// Two independent 64-bit mixes to keep collision probability negligible.
	h1 := uint64(1469598103934665603)
	h2 := uint64(1099511628211)
	for _, tok := range tokens {
		v := uint64(uint32(tok))
		h1 ^= v + 0x9e3779b97f4a7c15 + (h1 << 6) + (h1 >> 2)
		h1 *= 1099511628211
		h2 += v*0x517cc1b727220a95 + (h2 << 7) + (h2 >> 3)
		h2 ^= h2 >> 33
		h2 *= 0xff51afd7ed558ccd
	}
	return decodeCacheKey{h1: h1, h2: h2, n: len(tokens)}
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
	ropeNeox             bool
	ropeYarnBetaFast     float32
	ropeYarnBetaSlow     float32
	ropeYarnOrigCtx      float32
	ropeYarnExtFactor    float32
	ropeYarnAttnFactor   float32
	stateProj            []float32
	logitsProj           []float32
	logitsProjTransposed bool
	tokenEmbd            []float32
	tokenEmbdF16         []uint16
	tokenEmbdType        uint32
	tokenEmbdRows        int
	tokenEmbdCols        int
	outputWeight         []float32
	outputWeightF16      []uint16
	outputRows           int
	outputCols           int
	outputTransposed     bool
	outputWeightPacked   []byte
	outputWeightScale    float32
	outputWeightType     uint32
	outputNorm           []float32
	rmsEps               float32
	ffnUseSilu           bool
	layers               []llamaLayer
}

type tensorBlockMode int

const (
	tensorBlockModeProjection tensorBlockMode = iota + 1
	tensorBlockModeEmbeddingOutput
	tensorBlockModeLlamaStack
)

type llamaLayer struct {
	attnNorm        []float32
	ffnNorm         []float32
	attnSubNorm     []float32
	ffnSubNorm      []float32
	attnQ           linearWeight
	attnK           linearWeight
	attnV           linearWeight
	attnOut         linearWeight
	ffnGate         linearWeight
	ffnUp           linearWeight
	ffnDown         linearWeight
	debugFFNGateF32 []float32
	debugFFNUpF32   []float32
	debugFFNDownF32 []float32
}

type linearWeight struct {
	data       []float32
	dataF16    []uint16
	rows       int
	cols       int
	transposed bool
	qtype      uint32
	i2sPacked  []byte
	i2sScale   float32
}

type modelTensorLoader struct {
	info     gguf.ModelInfo
	f        *os.File
	mmapData []byte
}

func newModelTensorLoader(path string, info gguf.ModelInfo) (*modelTensorLoader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	l := &modelTensorLoader{info: info, f: f}
	if useMmapI2S {
		if data, err := mmapReadOnly(f); err == nil {
			l.mmapData = data
		}
	}
	return l, nil
}

func (l *modelTensorLoader) close() error {
	if l == nil || l.f == nil {
		return nil
	}
	return l.f.Close()
}

func (l *modelTensorLoader) readTensorAsF32(name string) ([]float32, error) {
	return gguf.ReadTensorAsF32FromFile(l.f, l.info, name)
}

func (l *modelTensorLoader) readTensorI2SPacked(name string) ([]byte, float32, uint64, error) {
	if len(l.mmapData) > 0 {
		t, ok := l.info.TensorByName(name)
		if !ok {
			return nil, 0, 0, fmt.Errorf("tensor not found: %s", name)
		}
		if t.Type != gguf.GGMLTypeI2_S {
			return nil, 0, 0, fmt.Errorf("tensor %q type=%d is not i2_s", name, t.Type)
		}
		count, err := gguf.TensorElementCount(t)
		if err != nil {
			return nil, 0, 0, err
		}
		packedLen := int((count + 127) / 128 * 32)
		start := int(l.info.TensorDataOffset + t.Offset)
		end := start + packedLen + 4
		if start < 0 || end < start || end > len(l.mmapData) {
			return nil, 0, 0, fmt.Errorf("tensor %q mmap bounds out of range", name)
		}
		packed := l.mmapData[start : start+packedLen]
		scale := math.Float32frombits(binary.LittleEndian.Uint32(l.mmapData[start+packedLen : end]))
		return packed, scale, count, nil
	}
	return gguf.ReadTensorI2SPackedFromFile(l.f, l.info, name)
}

func (l *modelTensorLoader) readTensorF16Raw(name string) ([]uint16, error) {
	return gguf.ReadTensorF16RawFromFile(l.f, l.info, name)
}

func loadTensorBlock(path string, info gguf.ModelInfo) (*tensorBlock, error) {
	loader, err := newModelTensorLoader(path, info)
	if err != nil {
		return nil, err
	}
	defer loader.close()

	block, found, err := loadLlamaStack(info, loader)
	if err != nil {
		return nil, err
	}
	if found {
		return block, nil
	}

	block, found, err = loadProjectionBlock(info, loader)
	if err != nil {
		return nil, err
	}
	if found {
		return block, nil
	}
	return loadEmbeddingOutputBlock(info, loader)
}

func loadProjectionBlock(info gguf.ModelInfo, loader *modelTensorLoader) (*tensorBlock, bool, error) {
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

	stateProj, err := loader.readTensorAsF32(stateProjName)
	if err != nil {
		return nil, false, err
	}

	logitsInfo, _ := info.TensorByName(logitsProjName)
	if len(logitsInfo.Dimensions) != 2 {
		return nil, false, fmt.Errorf("%s: expected 2 dims, got %d", logitsProjName, len(logitsInfo.Dimensions))
	}
	logitsProj, err := loader.readTensorAsF32(logitsProjName)
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

func loadEmbeddingOutputBlock(info gguf.ModelInfo, loader *modelTensorLoader) (*tensorBlock, error) {
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
		tokenEmbdType: embInfo.Type,
	}
	if emb.tokenEmbdRows <= 0 || emb.tokenEmbdCols <= 0 {
		return nil, fmt.Errorf("%s has invalid dims %v", tokenEmbdName, embInfo.Dimensions)
	}

	if embInfo.Type == gguf.GGMLTypeF16 && useF16TokenEmbd {
		tokenEmbdF16, err := loader.readTensorF16Raw(tokenEmbdName)
		if err != nil {
			return nil, err
		}
		emb.tokenEmbdF16 = tokenEmbdF16
	} else {
		tokenEmbd, err := loader.readTensorAsF32(tokenEmbdName)
		if err != nil {
			return nil, err
		}
		emb.tokenEmbd = tokenEmbd
	}

	var outputWeight []float32
	var outputWeightF16 []uint16
	var outputPacked []byte
	var outputScale float32
	var err error
	if outInfo.Type == gguf.GGMLTypeI2_S {
		packed, scale, _, err := loader.readTensorI2SPacked(outputName)
		if err != nil {
			return nil, err
		}
		outputPacked = packed
		outputScale = scale
	} else if outInfo.Type == gguf.GGMLTypeF16 {
		outputWeightF16, err = loader.readTensorF16Raw(outputName)
		if err != nil {
			return nil, err
		}
	} else {
		outputWeight, err = loader.readTensorAsF32(outputName)
		if err != nil {
			return nil, err
		}
	}
	emb.outputWeight = outputWeight
	emb.outputWeightF16 = outputWeightF16
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

func loadLlamaStack(info gguf.ModelInfo, loader *modelTensorLoader) (*tensorBlock, bool, error) {
	if _, ok := info.TensorByName("blk.0.attn_q.weight"); !ok {
		return nil, false, nil
	}
	arch, _ := info.KeyValues["general.architecture"].(string)
	useTokEmbOut := arch == "bitnet-b1.58" || arch == "bitnet" || arch == "bitnet-25"
	ropeNeox := arch == "bitnet-b1.58" || arch == "bitnet" || arch == "bitnet-25"

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
		tokenEmbdType: embInfo.Type,
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
		ropeScale: firstFloat32(info.KeyValues["llama.rope.scaling.factor"], 1.0),
		ropeScalingType: firstString(
			info.KeyValues["llama.rope.scaling.type"],
			info.KeyValues["llama.rope.scaling_type"],
		),
		ropeDim: int(firstUint32(
			info.KeyValues["llama.rope.dimension_count"],
			info.KeyValues["bitnet-b1.58.rope.dimension_count"],
		)),
		ropeNeox:           ropeNeox,
		ropeYarnBetaFast:   firstFloat32(info.KeyValues["llama.rope.scaling.beta_fast"], 0),
		ropeYarnBetaSlow:   firstFloat32(info.KeyValues["llama.rope.scaling.beta_slow"], 0),
		ropeYarnOrigCtx:    firstFloat32(info.KeyValues["llama.rope.scaling.original_context_length"], 0),
		ropeYarnExtFactor:  firstFloat32(info.KeyValues["llama.rope.scaling.ext_factor"], 0),
		ropeYarnAttnFactor: firstFloat32(info.KeyValues["llama.rope.scaling.attn_factor"], 1.0),
		ffnUseSilu:         arch == "llama",
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
	if embInfo.Type == gguf.GGMLTypeF16 && useF16TokenEmbd {
		if b.tokenEmbdF16, err = loader.readTensorF16Raw("token_embd.weight"); err != nil {
			return nil, false, err
		}
	} else if b.tokenEmbd, err = loader.readTensorAsF32("token_embd.weight"); err != nil {
		return nil, false, err
	}
	if b.outputNorm, err = loader.readTensorAsF32("output_norm.weight"); err != nil {
		return nil, false, err
	}
	if len(b.outputNorm) != hidden {
		return nil, false, fmt.Errorf("output_norm.weight len=%d want=%d", len(b.outputNorm), hidden)
	}
	if useTokEmbOut {
		b.outputWeight = b.tokenEmbd
		b.outputWeightF16 = b.tokenEmbdF16
		b.outputRows = hidden
		b.outputCols = b.vocabDim
		b.outputTransposed = true
		if len(b.outputWeightF16) > 0 {
			b.outputWeightType = gguf.GGMLTypeF16
		} else {
			b.outputWeightType = gguf.GGMLTypeF32
		}
	} else if outInfo, ok := info.TensorByName("output.weight"); ok {
		if b.outputWeight, b.outputRows, b.outputCols, b.outputTransposed, err = loadLinearTensor(info, loader, "output.weight", hidden); err != nil {
			return nil, false, err
		}
		b.outputWeightType = outInfo.Type
		if outInfo.Type == gguf.GGMLTypeI2_S {
			packed, scale, _, err := loader.readTensorI2SPacked("output.weight")
			if err != nil {
				return nil, false, err
			}
			b.outputWeightPacked = packed
			b.outputWeightScale = scale
		}
	} else {
		// Some models tie output weights to token embeddings and omit output.weight.
		b.outputWeight = b.tokenEmbd
		b.outputWeightF16 = b.tokenEmbdF16
		b.outputRows = hidden
		b.outputCols = b.vocabDim
		b.outputTransposed = true
		if len(b.outputWeightF16) > 0 {
			b.outputWeightType = gguf.GGMLTypeF16
		} else {
			b.outputWeightType = gguf.GGMLTypeF32
		}
	}

	for idx := 0; ; idx++ {
		prefix := fmt.Sprintf("blk.%d.", idx)
		if _, ok := info.TensorByName(prefix + "attn_q.weight"); !ok {
			if idx == 0 {
				return nil, false, fmt.Errorf("missing tensor: %sattn_q.weight", prefix)
			}
			break
		}
		layer, err := loadLlamaLayer(info, loader, idx, prefix, hidden, b.attnHeads, b.kvHeads)
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

func loadLlamaLayer(info gguf.ModelInfo, loader *modelTensorLoader, layerIdx int, prefix string, hidden, attnHeads, kvHeads int) (llamaLayer, error) {
	var l llamaLayer
	var err error

	if l.attnNorm, err = loader.readTensorAsF32(prefix + "attn_norm.weight"); err != nil {
		return llamaLayer{}, err
	}
	if len(l.attnNorm) != hidden {
		return llamaLayer{}, fmt.Errorf("%sattn_norm.weight len=%d want=%d", prefix, len(l.attnNorm), hidden)
	}
	if _, ok := info.TensorByName(prefix + "attn_sub_norm.weight"); ok {
		if l.attnSubNorm, err = loader.readTensorAsF32(prefix + "attn_sub_norm.weight"); err != nil {
			return llamaLayer{}, err
		}
		if len(l.attnSubNorm) != hidden {
			return llamaLayer{}, fmt.Errorf("%sattn_sub_norm.weight len=%d want=%d", prefix, len(l.attnSubNorm), hidden)
		}
	}
	if l.ffnNorm, err = loader.readTensorAsF32(prefix + "ffn_norm.weight"); err != nil {
		return llamaLayer{}, err
	}
	if len(l.ffnNorm) != hidden {
		return llamaLayer{}, fmt.Errorf("%sffn_norm.weight len=%d want=%d", prefix, len(l.ffnNorm), hidden)
	}
	if l.attnQ, err = loadLinearWeight(info, loader, prefix+"attn_q.weight", hidden); err != nil {
		return llamaLayer{}, err
	}
	if l.attnK, err = loadLinearWeight(info, loader, prefix+"attn_k.weight", hidden); err != nil {
		return llamaLayer{}, err
	}
	if l.attnV, err = loadLinearWeight(info, loader, prefix+"attn_v.weight", hidden); err != nil {
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
	if l.attnOut, err = loadLinearWeight(info, loader, prefix+"attn_output.weight", qDim); err != nil {
		return llamaLayer{}, err
	}
	if linearOutputLen(l.attnOut) != hidden {
		return llamaLayer{}, fmt.Errorf("%sattn_output.weight output dim=%d want=%d", prefix, linearOutputLen(l.attnOut), hidden)
	}
	if l.ffnGate, err = loadLinearWeight(info, loader, prefix+"ffn_gate.weight", hidden); err != nil {
		return llamaLayer{}, err
	}
	if l.ffnUp, err = loadLinearWeight(info, loader, prefix+"ffn_up.weight", hidden); err != nil {
		return llamaLayer{}, err
	}
	if _, ok := info.TensorByName(prefix + "ffn_sub_norm.weight"); ok {
		if l.ffnSubNorm, err = loader.readTensorAsF32(prefix + "ffn_sub_norm.weight"); err != nil {
			return llamaLayer{}, err
		}
		if len(l.ffnSubNorm) != linearOutputLen(l.ffnGate) {
			return llamaLayer{}, fmt.Errorf("%sffn_sub_norm.weight len=%d want=%d", prefix, len(l.ffnSubNorm), linearOutputLen(l.ffnGate))
		}
	}
	if linearOutputLen(l.ffnGate) != linearOutputLen(l.ffnUp) {
		return llamaLayer{}, fmt.Errorf("%sincompatible ffn gate/up dims", prefix)
	}
	if l.ffnDown, err = loadLinearWeight(info, loader, prefix+"ffn_down.weight", linearOutputLen(l.ffnGate)); err != nil {
		return llamaLayer{}, err
	}
	if linearOutputLen(l.ffnDown) != hidden {
		return llamaLayer{}, fmt.Errorf("%sffn_down.weight output dim=%d want=%d", prefix, linearOutputLen(l.ffnDown), hidden)
	}
	if debugFFNLoad && layerIdx == 0 {
		if l.debugFFNGateF32, err = loader.readTensorAsF32(prefix + "ffn_gate.weight"); err != nil {
			return llamaLayer{}, err
		}
		if l.debugFFNUpF32, err = loader.readTensorAsF32(prefix + "ffn_up.weight"); err != nil {
			return llamaLayer{}, err
		}
		if l.debugFFNDownF32, err = loader.readTensorAsF32(prefix + "ffn_down.weight"); err != nil {
			return llamaLayer{}, err
		}
		fmt.Fprintf(os.Stderr, "debug ffn load: layer=%d gate=%d up=%d down=%d\n", layerIdx, len(l.debugFFNGateF32), len(l.debugFFNUpF32), len(l.debugFFNDownF32))
		fmt.Fprintf(os.Stderr, "debug ffn meta: gate rows=%d cols=%d transposed=%v qtype=%d\n", l.ffnGate.rows, l.ffnGate.cols, l.ffnGate.transposed, l.ffnGate.qtype)
		fmt.Fprintf(os.Stderr, "debug ffn meta: up rows=%d cols=%d transposed=%v qtype=%d\n", l.ffnUp.rows, l.ffnUp.cols, l.ffnUp.transposed, l.ffnUp.qtype)
		fmt.Fprintf(os.Stderr, "debug ffn meta: down rows=%d cols=%d transposed=%v qtype=%d\n", l.ffnDown.rows, l.ffnDown.cols, l.ffnDown.transposed, l.ffnDown.qtype)
		debugVecStats("debug attn_norm.weight", l.attnNorm)
		if len(l.attnSubNorm) > 0 {
			debugVecStats("debug attn_sub_norm.weight", l.attnSubNorm)
		}
		debugVecStats("debug ffn_norm.weight", l.ffnNorm)
		if len(l.ffnSubNorm) > 0 {
			debugVecStats("debug ffn_sub_norm.weight", l.ffnSubNorm)
		}
	}
	return l, nil
}

func runForwardTensorBlock(block *tensorBlock, seed int64, promptTokens []int32, out []int32, topk *topKWriter, forceTokens []int32, cfg samplingConfig) {
	switch block.mode {
	case tensorBlockModeProjection:
		runForwardProjectionBlock(block, seed, promptTokens, out, topk, cfg)
	case tensorBlockModeEmbeddingOutput:
		runForwardEmbeddingOutputBlock(block, seed, promptTokens, out, topk, cfg)
	case tensorBlockModeLlamaStack:
		runForwardLlamaStack(block, seed, promptTokens, out, topk, forceTokens, cfg)
	default:
		runForwardStub(uint32(block.vocabDim), seed, promptTokens, out, topk, cfg)
	}
}

func runForwardProjectionBlock(block *tensorBlock, seed int64, promptTokens []int32, out []int32, topk *topKWriter, cfg samplingConfig) {
	state := make([]float32, block.hiddenDim)
	tokenVec := make([]float32, block.hiddenDim)
	nextState := make([]float32, block.hiddenDim)
	logits := make([]float32, block.vocabDim)
	sampler := newSampler(seed)
	probs := make([]float32, block.vocabDim)
	idx := make([]int, block.vocabDim)
	for i := range idx {
		idx[i] = i
	}
	var topkEntries []TopKEntry
	var topkProbs []float32
	if cfg.topK > 0 {
		k := cfg.topK
		if k > block.vocabDim {
			k = block.vocabDim
		}
		topkEntries = make([]TopKEntry, k)
		topkProbs = make([]float32, k)
	}

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
			topk.append(i, logits)
		}

		next := sampleLogitsWithScratch(logits, cfg, sampler, probs, idx, topkEntries, topkProbs)
		if next < 0 {
			out[i] = 0
			continue
		}
		out[i] = int32(next)

		fillTokenVector(tokenVec, out[i])
		kernels.AddScaled(state, tokenVec, 0.05)
	}
}

func runForwardEmbeddingOutputBlock(block *tensorBlock, seed int64, promptTokens []int32, out []int32, topk *topKWriter, cfg samplingConfig) {
	state := make([]float32, block.hiddenDim)
	logits := make([]float32, block.vocabDim)
	scratch := make([]float32, block.hiddenDim)
	sampler := newSampler(seed)
	probs := make([]float32, block.vocabDim)
	idx := make([]int, block.vocabDim)
	for i := range idx {
		idx[i] = i
	}
	var topkEntries []TopKEntry
	var topkProbs []float32
	if cfg.topK > 0 {
		k := cfg.topK
		if k > block.vocabDim {
			k = block.vocabDim
		}
		topkEntries = make([]TopKEntry, k)
		topkProbs = make([]float32, k)
	}

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
			dataF16:    block.outputWeightF16,
			rows:       block.outputRows,
			cols:       block.outputCols,
			transposed: block.outputTransposed,
			qtype:      block.outputWeightType,
			i2sPacked:  block.outputWeightPacked,
			i2sScale:   block.outputWeightScale,
		}, state)
		if topk != nil {
			topk.append(i, logits)
		}

		next := sampleLogitsWithScratch(logits, cfg, sampler, probs, idx, topkEntries, topkProbs)
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
	if block.tokenEmbdType == gguf.GGMLTypeF16 && len(block.tokenEmbdF16) >= block.tokenEmbdRows*block.tokenEmbdCols {
		for r := 0; r < block.hiddenDim; r++ {
			dst[r] = kernels.Float16ToFloat32(block.tokenEmbdF16[r+block.tokenEmbdRows*idx])
		}
		return true
	}
	if debugEmbedRowMajor {
		for r := 0; r < block.hiddenDim; r++ {
			dst[r] = block.tokenEmbd[idx+block.tokenEmbdCols*r]
		}
		return true
	}
	for r := 0; r < block.hiddenDim; r++ {
		dst[r] = block.tokenEmbd[r+block.tokenEmbdRows*idx]
	}
	return true
}

func runForwardLlamaStack(block *tensorBlock, seed int64, promptTokens []int32, out []int32, topk *topKWriter, forceTokens []int32, cfg samplingConfig) {
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

	scratch := getLlamaRunScratch(block, maxSeq)
	defer putLlamaRunScratch(scratch)
	x := scratch.x
	n1 := scratch.n1
	n2 := scratch.n2
	logits := scratch.logits
	layerStates := scratch.layerState

	currentToken := seedToken(seed, block.vocabDim)
	startPos := 0
	if len(promptTokens) > 0 {
		currentToken = promptTokens[len(promptTokens)-1]
		for pos := 0; pos < len(promptTokens)-1; pos++ {
			runLlamaStackStep(block, layerStates, promptTokens[pos], pos, x, n1, n2, logits, false)
		}
		startPos = len(promptTokens) - 1
	}

	sampler := newSampler(seed)
	probs := make([]float32, block.vocabDim)
	idx := make([]int, block.vocabDim)
	for i := range idx {
		idx[i] = i
	}
	var topkEntries []TopKEntry
	var topkProbs []float32
	var stepProfile *llamaStepProfile
	if profileStep {
		stepProfile = &llamaStepProfile{}
	}
	if cfg.topK > 0 {
		k := cfg.topK
		if k > block.vocabDim {
			k = block.vocabDim
		}
		topkEntries = make([]TopKEntry, k)
		topkProbs = make([]float32, k)
	}
	for i := range out {
		stepStart := time.Time{}
		if stepProfile != nil {
			stepStart = time.Now()
		}
		stepPos := startPos + i
		fastGreedy := fastGreedyArgmax && cfg.temp <= 0 && topk == nil && i >= len(forceTokens) && !debugStep0
		var next int
		if fastGreedy {
			next = runLlamaStackStepProfile(block, layerStates, currentToken, stepPos, x, n1, n2, logits, false, stepProfile)
		} else {
			runLlamaStackStepProfile(block, layerStates, currentToken, stepPos, x, n1, n2, logits, true, stepProfile)
			if topk != nil {
				if stepProfile != nil {
					t := time.Now()
					topk.append(i, logits)
					stepProfile.topkCapture += time.Since(t)
				} else {
					topk.append(i, logits)
				}
			}
			if stepProfile != nil {
				t := time.Now()
				next = sampleLogitsWithScratch(logits, cfg, sampler, probs, idx, topkEntries, topkProbs)
				stepProfile.sample += time.Since(t)
			} else {
				next = sampleLogitsWithScratch(logits, cfg, sampler, probs, idx, topkEntries, topkProbs)
			}
		}
		if i < len(forceTokens) {
			next = int(forceTokens[i])
		}
		if next < 0 {
			out[i] = 0
			currentToken = 0
			continue
		}
		out[i] = int32(next)
		currentToken = out[i]
		if stepProfile != nil {
			stepProfile.addStep(time.Since(stepStart))
		}
	}
	stepProfile.log(block)
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

type llamaRunScratch struct {
	x          []float32
	n1         []float32
	n2         []float32
	logits     []float32
	layerState []llamaLayerState
}

type llamaStepProfile struct {
	steps       int
	stepTotal   time.Duration
	embed       time.Duration
	attn        time.Duration
	ffn         time.Duration
	ffnNorm     time.Duration
	ffnGateUp   time.Duration
	ffnAct      time.Duration
	ffnSubNorm  time.Duration
	ffnDown     time.Duration
	output      time.Duration
	sample      time.Duration
	topkCapture time.Duration
}

func (p *llamaStepProfile) addStep(total time.Duration) {
	p.steps++
	p.stepTotal += total
}

func (p *llamaStepProfile) log(block *tensorBlock) {
	if p == nil || p.steps == 0 {
		return
	}
	total := p.stepTotal
	if total <= 0 {
		return
	}
	pct := func(d time.Duration) float64 {
		return float64(d) * 100 / float64(total)
	}
	fmt.Fprintf(
		os.Stderr,
		"step_profile steps=%d layers=%d total=%s avg_step=%s embed=%s(%.1f%%) attn=%s(%.1f%%) ffn=%s(%.1f%%) ffn_norm=%s ffn_gate_up=%s ffn_act=%s ffn_subnorm=%s ffn_down=%s output=%s(%.1f%%) sample=%s(%.1f%%) topk=%s(%.1f%%)\n",
		p.steps,
		len(block.layers),
		total,
		total/time.Duration(p.steps),
		p.embed, pct(p.embed),
		p.attn, pct(p.attn),
		p.ffn, pct(p.ffn),
		p.ffnNorm,
		p.ffnGateUp,
		p.ffnAct,
		p.ffnSubNorm,
		p.ffnDown,
		p.output, pct(p.output),
		p.sample, pct(p.sample),
		p.topkCapture, pct(p.topkCapture),
	)
}

var llamaRunScratchPool sync.Pool

func resizeF32(buf []float32, n int) []float32 {
	if cap(buf) < n {
		return make([]float32, n)
	}
	return buf[:n]
}

func ensureLlamaLayerState(st *llamaLayerState, layer llamaLayer, hiddenDim, maxSeq, heads int) {
	kdim := linearOutputLen(layer.attnK)
	vdim := linearOutputLen(layer.attnV)
	qdim := linearOutputLen(layer.attnQ)
	ffnDim := linearOutputLen(layer.ffnGate)
	if heads <= 0 {
		heads = 1
	}
	st.q = resizeF32(st.q, qdim)
	st.k = resizeF32(st.k, kdim)
	st.v = resizeF32(st.v, vdim)
	st.attnAcc = resizeF32(st.attnAcc, qdim)
	st.attnOut = resizeF32(st.attnOut, hiddenDim)
	st.gate = resizeF32(st.gate, ffnDim)
	st.up = resizeF32(st.up, linearOutputLen(layer.ffnUp))
	st.ffnAct = resizeF32(st.ffnAct, ffnDim)
	st.ffnDown = resizeF32(st.ffnDown, hiddenDim)
	st.scores = resizeF32(st.scores, maxSeq*heads)
	st.keys = resizeF32(st.keys, maxSeq*kdim)
	st.values = resizeF32(st.values, maxSeq*vdim)
}

func getLlamaRunScratch(block *tensorBlock, maxSeq int) *llamaRunScratch {
	s, _ := llamaRunScratchPool.Get().(*llamaRunScratch)
	if s == nil {
		s = &llamaRunScratch{}
	}
	s.x = resizeF32(s.x, block.hiddenDim)
	s.n1 = resizeF32(s.n1, block.hiddenDim)
	s.n2 = resizeF32(s.n2, block.hiddenDim)
	s.logits = resizeF32(s.logits, block.vocabDim)
	if cap(s.layerState) < len(block.layers) {
		s.layerState = make([]llamaLayerState, len(block.layers))
	} else {
		s.layerState = s.layerState[:len(block.layers)]
	}
	for i := range block.layers {
		ensureLlamaLayerState(&s.layerState[i], block.layers[i], block.hiddenDim, maxSeq, block.attnHeads)
	}
	return s
}

func putLlamaRunScratch(s *llamaRunScratch) {
	if s == nil {
		return
	}
	llamaRunScratchPool.Put(s)
}

func makeLlamaLayerState(layer llamaLayer, hiddenDim, maxSeq, heads int) llamaLayerState {
	var st llamaLayerState
	ensureLlamaLayerState(&st, layer, hiddenDim, maxSeq, heads)
	return st
}

func runLlamaStackStep(block *tensorBlock, layerStates []llamaLayerState, token int32, pos int, x, n1, n2, logits []float32, computeLogits bool) int {
	return runLlamaStackStepProfile(block, layerStates, token, pos, x, n1, n2, logits, computeLogits, nil)
}

func runLlamaStackStepProfile(block *tensorBlock, layerStates []llamaLayerState, token int32, pos int, x, n1, n2, logits []float32, computeLogits bool, prof *llamaStepProfile) int {
	embedStart := time.Time{}
	if prof != nil {
		embedStart = time.Now()
	}
	embedded := embedToken(x, block, token)
	if !embedded {
		fillTokenVector(x, token)
	}
	if prof != nil {
		prof.embed += time.Since(embedStart)
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
		return -1
	}

	if disableLayers {
		if shouldDebug(pos) {
			fmt.Fprintln(os.Stderr, "debug layers: disabled")
		}
		rmsNormInto(n1, x, block.outputNorm, block.rmsEps)
		w := linearWeight{
			data:       block.outputWeight,
			dataF16:    block.outputWeightF16,
			rows:       block.outputRows,
			cols:       block.outputCols,
			transposed: block.outputTransposed,
			qtype:      block.outputWeightType,
			i2sPacked:  block.outputWeightPacked,
			i2sScale:   block.outputWeightScale,
		}
		if computeLogits {
			linearApplyIntoWeight(logits, w, n1)
		} else {
			return linearArgmaxWeight(w, n1)
		}
		if shouldDebug(pos) {
			debugVecStats("output_norm", n1)
			debugVecStats("logits", logits)
			debugStep0Printed = true
		}
		return -1
	}

	for i := range block.layers {
		layer := block.layers[i]
		st := &layerStates[i]

		attnStart := time.Time{}
		if prof != nil {
			attnStart = time.Now()
		}
		if !disableAttn {
			rmsNormInto(n1, x, layer.attnNorm, block.rmsEps)
			if debugStages && shouldDebug(pos) && i == 0 {
				debugStage("stage.attn_norm", n1, block, stageNormBuf, false)
			}
			if shouldDebug(pos) && i == 0 {
				debugVecStats("inp_embd", x)
				debugVecStats("attn_norm-0", n1)
			}
			if debugValues && shouldDebug(pos) && i == 0 {
				debugVecValues("inp_embd", x, debugValuesN)
				debugVecValues("attn_norm", n1, debugValuesN)
			}
			linearApplyQKV(st.q, st.k, st.v, layer.attnQ, layer.attnK, layer.attnV, n1)
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
				debugVecStats("Qcur-0", st.q)
				debugVecStats("Kcur-0", st.k)
				debugVecStats("Vcur-0", st.v)
			}
			if debugValues && shouldDebug(pos) && i == 0 {
				debugVecValues("Qcur", st.q, debugValuesN)
				debugVecValues("Kcur", st.k, debugValuesN)
				debugVecValues("Vcur", st.v, debugValuesN)
			}
			applyRoPEInPlace(st.q, pos, block.attnHeads, block.ropeFreqBase, block.ropeScale, block.ropeScalingType, block.ropeDim, block.ropeNeox, block.ropeYarnBetaFast, block.ropeYarnBetaSlow, block.ropeYarnExtFactor, block.ropeYarnAttnFactor)
			applyRoPEInPlace(st.k, pos, block.kvHeads, block.ropeFreqBase, block.ropeScale, block.ropeScalingType, block.ropeDim, block.ropeNeox, block.ropeYarnBetaFast, block.ropeYarnBetaSlow, block.ropeYarnExtFactor, block.ropeYarnAttnFactor)
			if debugAttnMeta && shouldDebug(pos) && i == 0 {
				debugVecSlice("q.post", st.q, 8)
				debugVecSlice("k.post", st.k, 8)
			}
			if shouldDebug(pos) && i == 0 {
				debugVecStats("q-0", st.q)
			}

			storeCacheVector(st.keys, pos, st.k)
			if debugParityStrict || debugStrictAttnRef {
				// Match ggml accumulation order in parity-strict mode.
				storeCacheVectorV(st.values, pos, st.v, block.kvHeads)
				causalAttentionMultiHeadIntoReference(st.attnAcc, st.q, st.keys, st.values, pos+1, block.attnHeads, block.kvHeads, len(st.k), len(st.v))
			} else if debugKVRowMajor {
				storeCacheVectorVRowMajor(st.values, pos, st.v, block.kvHeads)
				causalAttentionMultiHeadIntoRowMajor(st.attnAcc, st.scores, st.q, st.keys, st.values, pos+1, block.attnHeads, block.kvHeads, len(st.k), len(st.v), pos)
			} else {
				storeCacheVectorV(st.values, pos, st.v, block.kvHeads)
				causalAttentionMultiHeadInto(st.attnAcc, st.scores, st.q, st.keys, st.values, pos+1, block.attnHeads, block.kvHeads, len(st.k), len(st.v), pos)
			}

			applySubNormOrIdentity(n2, st.attnAcc, layer.attnSubNorm, block.rmsEps)
			if debugStages && shouldDebug(pos) && i == 0 {
				debugStage("stage.attn_sub_norm", n2, block, stageNormBuf, false)
			}
			if shouldDebug(pos) && i == 0 {
				debugVecStats("attn_sub_norm-0", n2)
			}
			if debugValues && shouldDebug(pos) && i == 0 {
				debugVecValues("kqv", st.attnAcc, debugValuesN)
				debugVecValues("attn_sub_norm", n2, debugValuesN)
			}
			linearApplyIntoWeight(st.attnOut, layer.attnOut, n2)
			kernels.AddScaled(x, st.attnOut, 1.0)
			if debugStages && shouldDebug(pos) && i == 0 {
				debugStage("stage.post_attn", x, block, stageNormBuf, false)
			}
			if debugValues && shouldDebug(pos) && i == 0 {
				debugVecValues("attn_o_out", st.attnOut, debugValuesN)
				debugVecValues("ffn_inp", x, debugValuesN)
			}
			if shouldDebug(pos) && i == 0 {
				debugVecStats("kqv-0", st.attnAcc)
				debugVecStats("kqv_out-0", st.attnAcc)
				debugVecStats("attn_o_out-0", st.attnOut)
				debugVecStats("x.post_attn", x)
			}
		} else if shouldDebug(pos) && i == 0 {
			fmt.Fprintln(os.Stderr, "debug attn: disabled")
		}
		if prof != nil {
			prof.attn += time.Since(attnStart)
		}

		ffnStart := time.Time{}
		if prof != nil {
			ffnStart = time.Now()
		}
		if !disableFFN {
			if debugStrictFFNRef {
				ffnNormStart := time.Time{}
				if prof != nil {
					ffnNormStart = time.Now()
				}
				rmsNormInto(n2, x, layer.ffnNorm, block.rmsEps)
				if prof != nil {
					prof.ffnNorm += time.Since(ffnNormStart)
				}
				gateUpStart := time.Time{}
				if prof != nil {
					gateUpStart = time.Now()
				}
				linearApplyIntoWeight(st.gate, layer.ffnGate, n2)
				linearApplyIntoWeight(st.up, layer.ffnUp, n2)
				if prof != nil {
					prof.ffnGateUp += time.Since(gateUpStart)
				}
				actStart := time.Time{}
				if prof != nil {
					actStart = time.Now()
				}
				ffnActivateReference(st.ffnAct, st.gate, st.up, block.ffnUseSilu)
				if prof != nil {
					prof.ffnAct += time.Since(actStart)
				}
				subNormStart := time.Time{}
				if prof != nil {
					subNormStart = time.Now()
				}
				applySubNormOrIdentity(st.up, st.ffnAct, layer.ffnSubNorm, block.rmsEps)
				if prof != nil {
					prof.ffnSubNorm += time.Since(subNormStart)
				}
				downStart := time.Time{}
				if prof != nil {
					downStart = time.Now()
				}
				linearApplyIntoWeight(st.ffnDown, layer.ffnDown, st.up)
				if prof != nil {
					prof.ffnDown += time.Since(downStart)
				}
				kernels.AddScaled(x, st.ffnDown, 1.0)
				if debugStages && shouldDebug(pos) && i == 0 {
					debugStage("stage.ffn_sub_norm", st.up, block, stageNormBuf, false)
					debugStage("stage.post_ffn", x, block, stageNormBuf, false)
				}
				if shouldDebug(pos) && i == 0 {
					debugVecStats("ffn_act", st.ffnAct)
					debugVecStats("ffn_down", st.ffnDown)
					debugVecStats("x.post_ffn", x)
				}
				if prof != nil {
					prof.ffn += time.Since(ffnStart)
				}
				continue
			}
			var gateRef []float32
			var upRef []float32
			var actRef []float32
			var haveFfnRef bool

			ffnNormStart := time.Time{}
			if prof != nil {
				ffnNormStart = time.Now()
			}
			rmsNormInto(n2, x, layer.ffnNorm, block.rmsEps)
			if prof != nil {
				prof.ffnNorm += time.Since(ffnNormStart)
			}
			if debugStages && shouldDebug(pos) && i == 0 {
				debugStage("stage.ffn_norm", n2, block, stageNormBuf, false)
			}
			if debugValues && shouldDebug(pos) && i == 0 {
				debugVecValues("ffn_norm", n2, debugValuesN)
			}
			gateUpStart := time.Time{}
			if prof != nil {
				gateUpStart = time.Now()
			}
			if debugFFNRefF32 && shouldDebug(pos) && i == 0 {
				if len(layer.debugFFNGateF32) == 0 || len(layer.debugFFNUpF32) == 0 {
					fmt.Fprintln(os.Stderr, "debug ffn_ref_f32: missing f32 weights (set BITNET_DEBUG_FFN_LOAD=1)")
				} else {
					wGate := linearWeight{
						data:       layer.debugFFNGateF32,
						rows:       layer.ffnGate.rows,
						cols:       layer.ffnGate.cols,
						transposed: layer.ffnGate.transposed,
						qtype:      gguf.GGMLTypeF32,
					}
					wUp := linearWeight{
						data:       layer.debugFFNUpF32,
						rows:       layer.ffnUp.rows,
						cols:       layer.ffnUp.cols,
						transposed: layer.ffnUp.transposed,
						qtype:      gguf.GGMLTypeF32,
					}
					linearApplyIntoWeight(st.gate, wGate, n2)
					linearApplyIntoWeight(st.up, wUp, n2)
				}
			} else if debugFFNTranspose {
				linearApplyIntoWeightTransposed(st.gate, layer.ffnGate, n2, !layer.ffnGate.transposed)
				linearApplyIntoWeightTransposed(st.up, layer.ffnUp, n2, !layer.ffnUp.transposed)
			} else {
				if ffnShareI2SQuant &&
					layer.ffnGate.qtype == gguf.GGMLTypeI2_S && len(layer.ffnGate.i2sPacked) > 0 &&
					layer.ffnUp.qtype == gguf.GGMLTypeI2_S && len(layer.ffnUp.i2sPacked) > 0 &&
					!debugI2SFloat {
					scratch := i8ScratchPool.Get().([]int8)
					if cap(scratch) < len(n2) {
						scratch = make([]int8, len(n2))
					} else {
						scratch = scratch[:len(n2)]
					}
					actScale, actSum := kernels.QuantizeRowI8S(scratch, n2)
					if debugI2SDisableActSum {
						actSum = 0
					}
					if debugI2SInvertActScale && actScale != 0 {
						actScale = 1 / actScale
					}
					linearApplyIntoWeightI2SQuantized(st.gate, layer.ffnGate, scratch, actScale, actSum)
					linearApplyIntoWeightI2SQuantized(st.up, layer.ffnUp, scratch, actScale, actSum)
					i8ScratchPool.Put(scratch[:0])
				} else if ffnParGateUp {
					var wg sync.WaitGroup
					wg.Add(1)
					go func() {
						defer wg.Done()
						linearApplyIntoWeight(st.gate, layer.ffnGate, n2)
					}()
					linearApplyIntoWeight(st.up, layer.ffnUp, n2)
					wg.Wait()
				} else {
					linearApplyIntoWeight(st.gate, layer.ffnGate, n2)
					linearApplyIntoWeight(st.up, layer.ffnUp, n2)
				}
			}
			if prof != nil {
				prof.ffnGateUp += time.Since(gateUpStart)
			}
			if debugFFNLoad && shouldDebug(pos) && i == 0 {
				debugFfnCompare("ffn_gate", st.gate, layer.ffnGate, layer.debugFFNGateF32, n2)
				debugFfnCompare("ffn_up", st.up, layer.ffnUp, layer.debugFFNUpF32, n2)
			}
			if debugFFNRef && shouldDebug(pos) && i == 0 {
				if len(layer.debugFFNGateF32) == 0 || len(layer.debugFFNUpF32) == 0 || len(layer.debugFFNDownF32) == 0 {
					fmt.Fprintln(os.Stderr, "debug ffn_ref: missing f32 weights (set BITNET_DEBUG_FFN_LOAD=1)")
				} else {
					gateRef = make([]float32, len(st.gate))
					upRef = make([]float32, len(st.up))
					wGate := linearWeight{
						data:       layer.debugFFNGateF32,
						rows:       layer.ffnGate.rows,
						cols:       layer.ffnGate.cols,
						transposed: layer.ffnGate.transposed,
						qtype:      gguf.GGMLTypeF32,
					}
					wUp := linearWeight{
						data:       layer.debugFFNUpF32,
						rows:       layer.ffnUp.rows,
						cols:       layer.ffnUp.cols,
						transposed: layer.ffnUp.transposed,
						qtype:      gguf.GGMLTypeF32,
					}
					linearApplyIntoWeight(gateRef, wGate, n2)
					linearApplyIntoWeight(upRef, wUp, n2)
					debugVecDiff("ffn_gate.ref.diff", st.gate, gateRef)
					debugVecDiff("ffn_up.ref.diff", st.up, upRef)
					haveFfnRef = true
				}
			}
			if debugValues && shouldDebug(pos) && i == 0 {
				debugVecValues("ffn_gate", st.gate, debugValuesN)
				debugVecValues("ffn_up", st.up, debugValuesN)
			}
			if shouldDebug(pos) && i == 0 {
				debugVecStats("ffn_norm", n2)
				debugVecStats("ffn_gate", st.gate)
				debugVecStats("ffn_up", st.up)
			}
			actStart := time.Time{}
			if prof != nil {
				actStart = time.Now()
			}
			ffnActivateInto(st.ffnAct, st.gate, st.up, block.ffnUseSilu)
			if prof != nil {
				prof.ffnAct += time.Since(actStart)
			}
			if debugFfnActRef && shouldDebug(pos) && i == 0 {
				refAct := make([]float32, len(st.ffnAct))
				ffnActivateReference(refAct, st.gate, st.up, block.ffnUseSilu)
				debugVecDiff("ffn_act.kernel_ref.diff", st.ffnAct, refAct)
			}
			if haveFfnRef {
				actRef = make([]float32, len(st.ffnAct))
				ffnActivateInto(actRef, gateRef, upRef, block.ffnUseSilu)
				debugVecDiff("ffn_act.ref.diff", st.ffnAct, actRef)
			}
			if debugValues && shouldDebug(pos) && i == 0 {
				debugVecValues("ffn_act", st.ffnAct, debugValuesN)
				debugVecValues("ffn_out", st.ffnAct, debugValuesN)
			}
			subNormStart := time.Time{}
			if prof != nil {
				subNormStart = time.Now()
			}
			applySubNormOrIdentity(st.up, st.ffnAct, layer.ffnSubNorm, block.rmsEps)
			if prof != nil {
				prof.ffnSubNorm += time.Since(subNormStart)
			}
			if haveFfnRef {
				upNormRef := make([]float32, len(st.up))
				applySubNormOrIdentity(upNormRef, actRef, layer.ffnSubNorm, block.rmsEps)
				debugVecDiff("ffn_sub_norm.ref.diff", st.up, upNormRef)
			}
			if debugStages && shouldDebug(pos) && i == 0 {
				debugStage("stage.ffn_sub_norm", st.up, block, stageNormBuf, false)
			}
			if debugValues && shouldDebug(pos) && i == 0 {
				debugVecValues("ffn_sub_norm", st.up, debugValuesN)
			}
			downStart := time.Time{}
			if prof != nil {
				downStart = time.Now()
			}
			if debugFFNTranspose {
				linearApplyIntoWeightTransposed(st.ffnDown, layer.ffnDown, st.up, !layer.ffnDown.transposed)
			} else {
				linearApplyIntoWeight(st.ffnDown, layer.ffnDown, st.up)
			}
			if prof != nil {
				prof.ffnDown += time.Since(downStart)
			}
			if haveFfnRef {
				downRef := make([]float32, len(st.ffnDown))
				wDown := linearWeight{
					data:       layer.debugFFNDownF32,
					rows:       layer.ffnDown.rows,
					cols:       layer.ffnDown.cols,
					transposed: layer.ffnDown.transposed,
					qtype:      gguf.GGMLTypeF32,
				}
				linearApplyIntoWeight(downRef, wDown, st.up)
				debugVecDiff("ffn_down.ref.diff", st.ffnDown, downRef)
			}
			if debugFFNLoad && shouldDebug(pos) && i == 0 {
				debugFfnCompare("ffn_down", st.ffnDown, layer.ffnDown, layer.debugFFNDownF32, st.up)
			}
			if debugValues && shouldDebug(pos) && i == 0 {
				debugVecValues("ffn_down", st.ffnDown, debugValuesN)
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
		if prof != nil {
			prof.ffn += time.Since(ffnStart)
		}
	}

	outputStart := time.Time{}
	if prof != nil {
		outputStart = time.Now()
	}
	rmsNormInto(n1, x, block.outputNorm, block.rmsEps)
	if debugStages && shouldDebug(pos) {
		debugStage("stage.output_norm", n1, block, stageNormBuf, true)
	}
	if debugValues && shouldDebug(pos) {
		debugVecValues("result_norm", n1, debugValuesN)
	}
	w := linearWeight{
		data:       block.outputWeight,
		dataF16:    block.outputWeightF16,
		rows:       block.outputRows,
		cols:       block.outputCols,
		transposed: block.outputTransposed,
		qtype:      block.outputWeightType,
		i2sPacked:  block.outputWeightPacked,
		i2sScale:   block.outputWeightScale,
	}
	if computeLogits {
		linearApplyIntoWeight(logits, w, n1)
		if prof != nil {
			prof.output += time.Since(outputStart)
		}
		if shouldDebug(pos) {
			debugVecStats("output_norm", n1)
			debugVecStats("logits", logits)
			debugStep0Printed = true
		}
		return -1
	}
	if prof != nil {
		prof.output += time.Since(outputStart)
	}
	return linearArgmaxWeight(w, n1)
}

func causalAttentionMultiHeadIntoGeneric(dst, scores, q, keys, values []float32, steps, qHeads, kvHeads, kStepDim, vStepDim int, pos int) {
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
	maxSeq := 0
	if vStepDim > 0 {
		maxSeq = len(values) / vStepDim
	}
	if maxSeq <= 0 {
		return
	}

	for h := 0; h < qHeads; h++ {
		qBase := h * headDim
		qh := q[qBase : qBase+headDim]
		kvHead := h * kvHeads / qHeads
		kBase := kvHead * headDim
		scale := float32(1.0 / math.Sqrt(float64(headDim)))
		maxScore := float32(-math.MaxFloat32)
		for i := 0; i < steps; i++ {
			kb := i*kStepDim + kBase
			var sum float32
			if debugStrictKQ {
				sum = dotF32GGML(qh, keys[kb:kb+headDim])
			} else if debugFastKQ {
				sum = dotF32FastN(keys, kb, qh, 0, headDim)
			} else if debugAttnF64 {
				var sum64 float64
				for j := 0; j < headDim; j++ {
					sum64 += float64(qh[j]) * float64(keys[kb+j])
				}
				sum = float32(sum64)
			} else {
				for j := 0; j < headDim; j++ {
					sum += qh[j] * keys[kb+j]
				}
			}
			s := sum * scale
			scores[h*steps+i] = s
			if s > maxScore {
				maxScore = s
			}
		}

		sum := softmaxInPlace(scores[h*steps:h*steps+steps], maxScore)
		if sum == 0 {
			continue
		}
		inv := 1 / sum
		if debugStrictAttention {
			for i := 0; i < steps; i++ {
				idx := h*steps + i
				scores[idx] *= inv
			}
		}
		if debugValues && h == 0 && shouldDebug(pos) && !debugSoftmaxPrinted {
			limit := steps
			if limit > debugValuesN {
				limit = debugValuesN
			}
			if limit > 0 {
				fmt.Fprint(os.Stderr, "debug_values kq_soft_max_ext values=")
				for i := 0; i < limit; i++ {
					if i > 0 {
						fmt.Fprint(os.Stderr, ",")
					}
					val := scores[i]
					if !debugStrictAttention {
						val *= inv
					}
					fmt.Fprintf(os.Stderr, "%.9g", val)
				}
				fmt.Fprintln(os.Stderr)
				debugSoftmaxPrinted = true
			}
		}
		if debugStrictAttention {
			vHeadBase := kvHead * headDim * maxSeq
			weights := scores[h*steps : h*steps+steps]
			for j := 0; j < headDim; j++ {
				rowBase := vHeadBase + j*maxSeq
				dst[qBase+j] += dotF32GGML(weights, values[rowBase:rowBase+steps])
			}
			continue
		}
		vHeadBase := kvHead * headDim * maxSeq
		if debugFastV && !debugAttnF64 {
			weights := scores[h*steps : h*steps+steps]
			for i := 0; i < steps; i++ {
				weights[i] *= inv
			}
			for j := 0; j < headDim; j++ {
				rowBase := vHeadBase + j*maxSeq
				dst[qBase+j] += dotF32FastN(values, rowBase, weights, 0, steps)
			}
			continue
		}
		if debugAttnF64 {
			for j := 0; j < headDim; j++ {
				var sum64 float64
				rowBase := vHeadBase + j*maxSeq
				for i := 0; i < steps; i++ {
					w := scores[h*steps+i] * inv
					sum64 += float64(values[rowBase+i]) * float64(w)
				}
				dst[qBase+j] += float32(sum64)
			}
			continue
		}
		for i := 0; i < steps; i++ {
			w := scores[h*steps+i] * inv
			for j := 0; j < headDim; j++ {
				vb := vHeadBase + j*maxSeq + i
				dst[qBase+j] += values[vb] * w
			}
		}
	}
}

func causalAttentionMultiHeadIntoReference(dst, q, keys, values []float32, steps, qHeads, kvHeads, kStepDim, vStepDim int) {
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
	if kStepDim/kvHeads != headDim || vStepDim/kvHeads != headDim {
		return
	}
	maxSeq := 0
	if vStepDim > 0 {
		maxSeq = len(values) / vStepDim
	}
	if maxSeq <= 0 {
		return
	}

	scores := make([]float32, steps)
	for h := 0; h < qHeads; h++ {
		qBase := h * headDim
		qh := q[qBase : qBase+headDim]
		kvHead := h * kvHeads / qHeads
		kBase := kvHead * headDim
		scale := float32(1.0 / math.Sqrt(float64(headDim)))
		maxScore := float32(-math.MaxFloat32)
		for i := 0; i < steps; i++ {
			kb := i*kStepDim + kBase
			sum := dotF32GGML(qh, keys[kb:kb+headDim])
			s := sum * scale
			scores[i] = s
			if s > maxScore {
				maxScore = s
			}
		}

		var sum float32
		for i := 0; i < steps; i++ {
			diff := scores[i] - maxScore
			var w float32
			if debugStrictExpf || debugFastExpf {
				w = expf32(diff)
			} else {
				w = float32(math.Exp(float64(diff)))
			}
			scores[i] = w
			sum += w
		}
		if sum == 0 {
			continue
		}
		inv := 1 / sum
		for i := 0; i < steps; i++ {
			scores[i] *= inv
		}

		vHeadBase := kvHead * headDim * maxSeq
		weights := scores[:steps]
		for j := 0; j < headDim; j++ {
			rowBase := vHeadBase + j*maxSeq
			dst[qBase+j] += dotF32GGML(weights, values[rowBase:rowBase+steps])
		}
	}
}

func dotF32GGML(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	const step = 32
	const epr = 8
	const arr = step / epr
	np := n & ^(step - 1)
	var sum0, sum1, sum2, sum3 float32
	for i := 0; i < np; i += step {
		for j := 0; j < arr; j++ {
			base := i + j*epr
			var s float32
			for k := 0; k < epr; k++ {
				s += a[base+k] * b[base+k]
			}
			switch j {
			case 0:
				sum0 += s
			case 1:
				sum1 += s
			case 2:
				sum2 += s
			default:
				sum3 += s
			}
		}
	}
	sum := sum0 + sum1 + sum2 + sum3
	for i := np; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

func mulRelu2Reference(dst, gate, up []float32) {
	n := len(dst)
	if len(gate) < n {
		n = len(gate)
	}
	if len(up) < n {
		n = len(up)
	}
	for i := 0; i < n; i++ {
		g := gate[i]
		if g < 0 {
			g = 0
		}
		dst[i] = g * g * up[i]
	}
	for i := n; i < len(dst); i++ {
		dst[i] = 0
	}
}

func mulSiluReference(dst, gate, up []float32) {
	n := len(dst)
	if len(gate) < n {
		n = len(gate)
	}
	if len(up) < n {
		n = len(up)
	}
	for i := 0; i < n; i++ {
		g := gate[i]
		dst[i] = (g / (1 + float32(math.Exp(float64(-g))))) * up[i]
	}
	for i := n; i < len(dst); i++ {
		dst[i] = 0
	}
}

func ffnActivateInto(dst, gate, up []float32, useSilu bool) {
	if useSilu {
		kernels.MulSiluInto(dst, gate, up)
		return
	}
	kernels.MulRelu2Into(dst, gate, up)
}

func ffnActivateReference(dst, gate, up []float32, useSilu bool) {
	if useSilu {
		mulSiluReference(dst, gate, up)
		return
	}
	mulRelu2Reference(dst, gate, up)
}

func expf32(x float32) float32 {
	// Cephes-style expf approximation in pure float32.
	const (
		expHi float32 = 88.3762626647949
		expLo float32 = -88.3762626647949
		log2e float32 = 1.44269504088896341
		ln2   float32 = 0.6931471805599453
		c0    float32 = 1.9875691500e-4
		c1    float32 = 1.3981999507e-3
		c2    float32 = 8.3334519073e-3
		c3    float32 = 4.1665795894e-2
		c4    float32 = 1.6666665459e-1
		c5    float32 = 5.0000001201e-1
	)
	if x > expHi {
		x = expHi
	} else if x < expLo {
		x = expLo
	}
	fx := x*log2e + 0.5
	n := int(fx)
	if float32(n) > fx {
		n--
	}
	x = x - float32(n)*ln2
	x2 := x * x
	px := c0
	px = px*x + c1
	px = px*x + c2
	px = px*x + c3
	px = px*x + c4
	px = px*x + c5
	px = px*x2 + x + 1

	// 2^n via bit manipulation.
	bits := uint32((n + 127) << 23)
	scale := math.Float32frombits(bits)
	return px * scale
}

func storeCacheVector(cache []float32, pos int, vec []float32) {
	storeCacheVectorImpl(cache, pos, vec)
}

func storeCacheVectorV(cache []float32, pos int, vec []float32, kvHeads int) {
	storeCacheVectorVImpl(cache, pos, vec, kvHeads)
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

func applyRoPEInPlace(v []float32, pos, heads int, base, scale float32, scalingType string, ropeDim int, ropeNeox bool, betaFast, betaSlow, extFactor, attnFactor float32) {
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
	thetaScale := math.Pow(float64(base), -2.0/float64(ropeDim))

	if scalingType != "yarn" {
		for h := 0; h < heads; h++ {
			offset := h * headDim
			theta := posf
			if ropeNeox {
				halfDim := ropeDim / 2
				for i := 0; i+1 < ropeDim; i += 2 {
					sinT, cosT := math.Sincos(theta)
					pair := i / 2
					x0 := v[offset+pair]
					x1 := v[offset+pair+halfDim]
					v[offset+pair] = x0*float32(cosT) - x1*float32(sinT)
					v[offset+pair+halfDim] = x0*float32(sinT) + x1*float32(cosT)
					theta *= thetaScale
				}
				continue
			}
			for i := 0; i+1 < ropeDim; i += 2 {
				sinT, cosT := math.Sincos(theta)
				x0 := v[offset+i]
				x1 := v[offset+i+1]
				v[offset+i] = x0*float32(cosT) - x1*float32(sinT)
				v[offset+i+1] = x0*float32(sinT) + x1*float32(cosT)
				theta *= thetaScale
			}
		}
		return
	}

	for h := 0; h < heads; h++ {
		offset := h * headDim
		theta := float32(posf)
		if ropeNeox {
			halfDim := ropeDim / 2
			for i := 0; i+1 < ropeDim; i += 2 {
				cosT, sinT := ropeYarnCosSin(theta, scale, betaFast, betaSlow, extFactor, attnFactor, i)
				pair := i / 2
				x0 := v[offset+pair]
				x1 := v[offset+pair+halfDim]
				v[offset+pair] = x0*cosT - x1*sinT
				v[offset+pair+halfDim] = x0*sinT + x1*cosT
				theta = float32(float64(theta) * thetaScale)
			}
			continue
		}
		for i := 0; i+1 < ropeDim; i += 2 {
			cosT, sinT := ropeYarnCosSin(theta, scale, betaFast, betaSlow, extFactor, attnFactor, i)
			x0 := v[offset+i]
			x1 := v[offset+i+1]
			v[offset+i] = x0*cosT - x1*sinT
			v[offset+i+1] = x0*sinT + x1*cosT
			theta = float32(float64(theta) * thetaScale)
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

func ropeYarnCosSin(thetaExtrap float32, scale, betaFast, betaSlow, extFactor, attnFactor float32, i0 int) (float32, float32) {
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
		theta = thetaInterp*(1-rampMix) + thetaExtrap*rampMix
		mscale *= 1.0 + 0.1*float32(math.Log(float64(1.0/freqScale)))
	}
	sinT, cosT := math.Sincos(float64(theta))
	return float32(cosT) * mscale, float32(sinT) * mscale
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
	kernels.RMSNormInto(dst, x, weight, eps)
}

func applySubNormOrIdentity(dst, x, weight []float32, eps float32) {
	if len(weight) == len(x) && len(x) > 0 {
		rmsNormInto(dst, x, weight, eps)
		return
	}
	copy(dst, x)
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
	if w.qtype == gguf.GGMLTypeF16 && len(w.dataF16) > 0 {
		for _, tok := range debugTokens {
			v := f16LogitForToken(w.dataF16, w.rows, w.cols, tok, x, w.transposed)
			alt := f16LogitForToken(w.dataF16, w.rows, w.cols, tok, x, !w.transposed)
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
		dataF16:    block.outputWeightF16,
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

func f16LogitForToken(mat []uint16, rows, cols, token int, x []float32, transposed bool) float32 {
	if transposed {
		if token < 0 || token >= cols || len(x) < rows {
			return 0
		}
		var sum float32
		for r := 0; r < rows; r++ {
			sum += kernels.Float16ToFloat32(mat[r+rows*token]) * x[r]
		}
		return sum
	}
	if token < 0 || token >= rows || len(x) < cols {
		return 0
	}
	var sum float32
	for c := 0; c < cols; c++ {
		sum += kernels.Float16ToFloat32(mat[token+rows*c]) * x[c]
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

func i2sPackedSetLocal(packed []byte, idx int, val byte) {
	if idx < 0 {
		return
	}
	const block = 128
	const blockBytes = 32
	bi := idx / block
	off := idx % block
	gp := off % 32
	group := off / 32
	p := bi*blockBytes + gp
	if p < 0 || p >= len(packed) {
		return
	}
	shift := uint(6 - 2*group)
	mask := byte(0x3 << shift)
	packed[p] = (packed[p] & ^mask) | ((val & 0x3) << shift)
}

func linearArgmaxWeight(w linearWeight, x []float32) int {
	if w.qtype == gguf.GGMLTypeF32 && len(w.data) > 0 {
		if w.transposed {
			return kernels.ArgmaxMatVecT(w.data, w.rows, w.cols, x)
		}
		if len(x) < w.cols || len(w.data) < w.rows*w.cols || w.rows <= 0 {
			return -1
		}
		bestID := 0
		bestVal := float32(-math.MaxFloat32)
		for r := 0; r < w.rows; r++ {
			var sum float64
			for c := 0; c < w.cols; c++ {
				sum += float64(w.data[r+w.rows*c]) * float64(x[c])
			}
			v := float32(sum)
			if v > bestVal {
				bestVal = v
				bestID = r
			}
		}
		return bestID
	}
	if w.qtype == gguf.GGMLTypeF16 && len(w.dataF16) > 0 {
		if w.transposed {
			return kernels.ArgmaxMatVecTF16(w.dataF16, w.rows, w.cols, x)
		}
		if len(x) < w.cols || len(w.dataF16) < w.rows*w.cols || w.rows <= 0 {
			return -1
		}
		bestID := 0
		bestVal := float32(-math.MaxFloat32)
		for r := 0; r < w.rows; r++ {
			var sum float64
			for c := 0; c < w.cols; c++ {
				sum += float64(kernels.Float16ToFloat32(w.dataF16[r+w.rows*c])) * float64(x[c])
			}
			v := float32(sum)
			if v > bestVal {
				bestVal = v
				bestID = r
			}
		}
		return bestID
	}
	return -1
}

func transposeI2SPacked(packed []byte, rows, cols int) []byte {
	if rows <= 0 || cols <= 0 {
		return nil
	}
	count := rows * cols
	out := make([]byte, (count+127)/128*32)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			src := r + rows*c
			dst := c + cols*r
			i2sPackedSetLocal(out, dst, i2sPackedAtLocal(packed, src))
		}
	}
	return out
}

func linearOutputLen(w linearWeight) int {
	if w.transposed {
		return w.cols
	}
	return w.rows
}

func linearApplyIntoWeight(dst []float32, w linearWeight, x []float32) {
	if w.qtype == gguf.GGMLTypeI2_S && len(w.i2sPacked) > 0 {
		if debugI2SFloat && !debugI2SForceQuant {
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
		linearApplyIntoWeightI2SQuantized(dst, w, scratch, actScale, actSum)
		i8ScratchPool.Put(scratch[:0])
		return
	}
	if w.qtype == gguf.GGMLTypeF16 && len(w.dataF16) > 0 {
		if w.transposed {
			kernels.MatVecTF16(dst, w.dataF16, w.rows, w.cols, x)
			return
		}
		if w.rows <= 0 || w.cols <= 0 || len(dst) < w.rows || len(x) < w.cols || len(w.dataF16) < w.rows*w.cols {
			return
		}
		for r := 0; r < w.rows; r++ {
			var sum float64
			for c := 0; c < w.cols; c++ {
				sum += float64(kernels.Float16ToFloat32(w.dataF16[r+w.rows*c])) * float64(x[c])
			}
			dst[r] = float32(sum)
		}
		return
	}
	if w.transposed {
		kernels.MatVecT(dst, w.data, w.rows, w.cols, x)
		return
	}
	kernels.MatVec(dst, w.data, w.rows, w.cols, x)
}

func linearApplyIntoWeightI2SQuantized(dst []float32, w linearWeight, scratch []int8, actScale float32, actSum int32) {
	if debugI2SRefOnce && !debugI2SRefOncePrinted {
		ref := make([]float32, len(dst))
		if w.transposed {
			kernels.MatVecTI2SI8SRef(ref, w.i2sPacked, w.rows, w.cols, scratch, w.i2sScale, actScale)
		} else {
			kernels.MatVecI2SI8SRef(ref, w.i2sPacked, w.rows, w.cols, scratch, w.i2sScale, actScale)
		}
		var maxAbs, maxRel float32
		for i := range ref {
			diff := float32(math.Abs(float64(dst[i] - ref[i])))
			if diff > maxAbs {
				maxAbs = diff
			}
			den := float32(math.Abs(float64(ref[i]))) + 1e-9
			rel := diff / den
			if rel > maxRel {
				maxRel = rel
			}
		}
		fmt.Fprintf(os.Stderr, "debug i2s_ref_once max_abs=%g max_rel=%g act_sum=%d act_scale=%g\n", maxAbs, maxRel, actSum, actScale)
		debugI2SRefOncePrinted = true
	}
	if debugI2SRefDot {
		if w.transposed {
			kernels.MatVecTI2SI8SRef(dst, w.i2sPacked, w.rows, w.cols, scratch, w.i2sScale, actScale)
		} else {
			kernels.MatVecI2SI8SRef(dst, w.i2sPacked, w.rows, w.cols, scratch, w.i2sScale, actScale)
		}
	} else {
		if w.transposed {
			if debugI2SMap3To1 {
				kernels.MatVecTI2SI8SMap(dst, w.i2sPacked, w.rows, w.cols, scratch, w.i2sScale, actScale, actSum)
			} else if debugI2SAltLayout {
				kernels.MatVecTI2SI8SAlt(dst, w.i2sPacked, w.rows, w.cols, scratch, w.i2sScale, actScale, actSum)
			} else if debugI2SScalar {
				kernels.MatVecTI2SI8SScalar(dst, w.i2sPacked, w.rows, w.cols, scratch, w.i2sScale, actScale, actSum)
			} else {
				kernels.MatVecTI2SI8S(dst, w.i2sPacked, w.rows, w.cols, scratch, w.i2sScale, actScale, actSum)
			}
		} else {
			if debugI2SMap3To1 {
				kernels.MatVecI2SI8SMap(dst, w.i2sPacked, w.rows, w.cols, scratch, w.i2sScale, actScale, actSum)
			} else if debugI2SAltLayout {
				kernels.MatVecI2SI8SAlt(dst, w.i2sPacked, w.rows, w.cols, scratch, w.i2sScale, actScale, actSum)
			} else if debugI2SScalar {
				kernels.MatVecI2SI8SScalar(dst, w.i2sPacked, w.rows, w.cols, scratch, w.i2sScale, actScale, actSum)
			} else {
				kernels.MatVecI2SI8S(dst, w.i2sPacked, w.rows, w.cols, scratch, w.i2sScale, actScale, actSum)
			}
		}
	}
	if debugI2SMatvecRef && !debugI2SMatvecPrinted {
		ref := make([]float32, len(dst))
		if w.transposed {
			kernels.MatVecTI2SI8SRef(ref, w.i2sPacked, w.rows, w.cols, scratch, w.i2sScale, actScale)
		} else {
			kernels.MatVecI2SI8SRef(ref, w.i2sPacked, w.rows, w.cols, scratch, w.i2sScale, actScale)
		}
		debugVecDiff("i2s_matvec.ref.diff", dst, ref)
		debugI2SMatvecPrinted = true
	}
}

func linearApplyIntoWeightTransposed(dst []float32, w linearWeight, x []float32, transposed bool) {
	w.transposed = transposed
	linearApplyIntoWeight(dst, w, x)
}

func linearApplyQKV(dstQ, dstK, dstV []float32, wQ, wK, wV linearWeight, x []float32) {
	if wQ.qtype == gguf.GGMLTypeF32 && wK.qtype == gguf.GGMLTypeF32 && wV.qtype == gguf.GGMLTypeF32 &&
		!wQ.transposed && !wK.transposed && !wV.transposed &&
		wQ.rows == wK.rows && wQ.rows == wV.rows &&
		wQ.cols == wK.cols && wQ.cols == wV.cols &&
		len(dstQ) >= wQ.rows && len(dstK) >= wQ.rows && len(dstV) >= wQ.rows &&
		len(x) >= wQ.cols &&
		len(wQ.data) >= wQ.rows*wQ.cols && len(wK.data) >= wQ.rows*wQ.cols && len(wV.data) >= wQ.rows*wQ.cols &&
		!debugParityStrict {
		size := wQ.rows * wQ.cols
		if debugQKVFusedMax > 0 && size <= debugQKVFusedMax {
			if debugFastQKVCol {
				matVec3F32Col(dstQ, dstK, dstV, wQ.data, wK.data, wV.data, wQ.rows, wQ.cols, x)
			} else {
				matVec3F32(dstQ, dstK, dstV, wQ.data, wK.data, wV.data, wQ.rows, wQ.cols, x)
			}
			return
		}
	}
	linearApplyIntoWeight(dstQ, wQ, x)
	linearApplyIntoWeight(dstK, wK, x)
	linearApplyIntoWeight(dstV, wV, x)
}

func matVec3F32(dstA, dstB, dstC []float32, matA, matB, matC []float32, rows, cols int, vec []float32) {
	if rows <= 0 || cols <= 0 {
		return
	}
	if len(dstA) < rows || len(dstB) < rows || len(dstC) < rows || len(vec) < cols {
		return
	}
	if len(matA) < rows*cols || len(matB) < rows*cols || len(matC) < rows*cols {
		return
	}
	for r := 0; r < rows; r++ {
		var sumA, sumB, sumC float64
		for c := 0; c < cols; c++ {
			idx := r + rows*c
			v := float64(vec[c])
			sumA += float64(matA[idx]) * v
			sumB += float64(matB[idx]) * v
			sumC += float64(matC[idx]) * v
		}
		dstA[r] = float32(sumA)
		dstB[r] = float32(sumB)
		dstC[r] = float32(sumC)
	}
}

func matVec3F32Col(dstA, dstB, dstC []float32, matA, matB, matC []float32, rows, cols int, vec []float32) {
	if rows <= 0 || cols <= 0 {
		return
	}
	if len(dstA) < rows || len(dstB) < rows || len(dstC) < rows || len(vec) < cols {
		return
	}
	if len(matA) < rows*cols || len(matB) < rows*cols || len(matC) < rows*cols {
		return
	}
	for r := 0; r < rows; r++ {
		dstA[r] = 0
		dstB[r] = 0
		dstC[r] = 0
	}
	for c := 0; c < cols; c++ {
		scale := vec[c]
		base := rows * c
		r := 0
		for ; r+7 < rows; r += 8 {
			a0 := matA[base+r] * scale
			a1 := matA[base+r+1] * scale
			a2 := matA[base+r+2] * scale
			a3 := matA[base+r+3] * scale
			a4 := matA[base+r+4] * scale
			a5 := matA[base+r+5] * scale
			a6 := matA[base+r+6] * scale
			a7 := matA[base+r+7] * scale

			b0 := matB[base+r] * scale
			b1 := matB[base+r+1] * scale
			b2 := matB[base+r+2] * scale
			b3 := matB[base+r+3] * scale
			b4 := matB[base+r+4] * scale
			b5 := matB[base+r+5] * scale
			b6 := matB[base+r+6] * scale
			b7 := matB[base+r+7] * scale

			c0 := matC[base+r] * scale
			c1 := matC[base+r+1] * scale
			c2 := matC[base+r+2] * scale
			c3 := matC[base+r+3] * scale
			c4 := matC[base+r+4] * scale
			c5 := matC[base+r+5] * scale
			c6 := matC[base+r+6] * scale
			c7 := matC[base+r+7] * scale

			dstA[r] += a0
			dstA[r+1] += a1
			dstA[r+2] += a2
			dstA[r+3] += a3
			dstA[r+4] += a4
			dstA[r+5] += a5
			dstA[r+6] += a6
			dstA[r+7] += a7

			dstB[r] += b0
			dstB[r+1] += b1
			dstB[r+2] += b2
			dstB[r+3] += b3
			dstB[r+4] += b4
			dstB[r+5] += b5
			dstB[r+6] += b6
			dstB[r+7] += b7

			dstC[r] += c0
			dstC[r+1] += c1
			dstC[r+2] += c2
			dstC[r+3] += c3
			dstC[r+4] += c4
			dstC[r+5] += c5
			dstC[r+6] += c6
			dstC[r+7] += c7
		}
		for ; r < rows; r++ {
			dstA[r] += matA[base+r] * scale
			dstB[r] += matB[base+r] * scale
			dstC[r] += matC[base+r] * scale
		}
	}
}

func loadLinearWeight(info gguf.ModelInfo, loader *modelTensorLoader, name string, inDim int) (linearWeight, error) {
	data, rows, cols, transposed, err := loadLinearTensor(info, loader, name, inDim)
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
		packed, scale, _, err := loader.readTensorI2SPacked(name)
		if err != nil {
			return linearWeight{}, err
		}
		if transposed && debugI2SPretransposeMax > 0 && rows*cols <= debugI2SPretransposeMax {
			if repacked := transposeI2SPacked(packed, rows, cols); len(repacked) > 0 {
				packed = repacked
				transposed = false
			}
		}
		w.i2sPacked = packed
		w.i2sScale = scale
	}
	return w, nil
}

func loadLinearTensor(info gguf.ModelInfo, loader *modelTensorLoader, name string, inDim int) ([]float32, int, int, bool, error) {
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
	data, err := loader.readTensorAsF32(name)
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

func runForwardStub(vocabSize uint32, seed int64, promptTokens []int32, out []int32, topk *topKWriter, cfg samplingConfig) {
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

	sampler := newSampler(seed)
	probs := make([]float32, logitsCap)
	idx := make([]int, logitsCap)
	for i := range idx {
		idx[i] = i
	}
	var topkEntries []TopKEntry
	var topkProbs []float32
	if cfg.topK > 0 {
		k := cfg.topK
		if k > logitsCap {
			k = logitsCap
		}
		topkEntries = make([]TopKEntry, k)
		topkProbs = make([]float32, k)
	}
	for i := range out {
		for id := 0; id < logitsCap; id++ {
			fillTokenVector(tokenVec, int32(id))
			logits[id] = kernels.Dot(state, tokenVec)
		}
		if topk != nil {
			topk.append(i, logits)
		}

		next := sampleLogitsWithScratch(logits, cfg, sampler, probs, idx, topkEntries, topkProbs)
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
	entries := make([]TopKEntry, k)
	count := 0
	minIdx := 0
	minVal := float32(0)
	for id, logit := range logits {
		entry := TopKEntry{TokenID: int32(id), Logit: logit}
		if count < k {
			entries[count] = entry
			if count == 0 || logit < minVal {
				minVal = logit
				minIdx = count
			}
			count++
			if count == k {
				minIdx = 0
				minVal = entries[0].Logit
				for i := 1; i < k; i++ {
					if entries[i].Logit < minVal {
						minVal = entries[i].Logit
						minIdx = i
					}
				}
			}
			continue
		}
		if logit <= minVal {
			continue
		}
		entries[minIdx] = entry
		minIdx = 0
		minVal = entries[0].Logit
		for i := 1; i < k; i++ {
			if entries[i].Logit < minVal {
				minVal = entries[i].Logit
				minIdx = i
			}
		}
	}
	out := entries[:count]
	for i := 1; i < len(out); i++ {
		key := out[i]
		j := i - 1
		for j >= 0 && out[j].Logit < key.Logit {
			out[j+1] = out[j]
			j--
		}
		out[j+1] = key
	}
	return append(dst, TopKStep{Step: step, Entries: out})
}

func fillTopK(entries []TopKEntry, logits []float32, k int) int {
	if k <= 0 || len(logits) == 0 || len(entries) == 0 {
		return 0
	}
	if k > len(logits) {
		k = len(logits)
	}
	if k > len(entries) {
		k = len(entries)
	}
	count := 0
	minIdx := 0
	minVal := float32(0)
	for id, logit := range logits {
		entry := TopKEntry{TokenID: int32(id), Logit: logit}
		if count < k {
			entries[count] = entry
			if count == 0 || logit < minVal {
				minVal = logit
				minIdx = count
			}
			count++
			if count == k {
				minIdx = 0
				minVal = entries[0].Logit
				for i := 1; i < k; i++ {
					if entries[i].Logit < minVal {
						minVal = entries[i].Logit
						minIdx = i
					}
				}
			}
			continue
		}
		if logit <= minVal {
			continue
		}
		entries[minIdx] = entry
		minIdx = 0
		minVal = entries[0].Logit
		for i := 1; i < k; i++ {
			if entries[i].Logit < minVal {
				minVal = entries[i].Logit
				minIdx = i
			}
		}
	}
	out := entries[:count]
	for i := 1; i < len(out); i++ {
		key := out[i]
		j := i - 1
		for j >= 0 && out[j].Logit < key.Logit {
			out[j+1] = out[j]
			j--
		}
		out[j+1] = key
	}
	return count
}

func sampleLogitsWithScratch(logits []float32, cfg samplingConfig, rng *sampler, probs []float32, idx []int, topkEntries []TopKEntry, topkProbs []float32) int {
	if len(logits) == 0 {
		return -1
	}
	if cfg.temp <= 0 {
		return kernels.Argmax(logits)
	}
	if cfg.topK > 0 {
		k := cfg.topK
		if k > len(logits) {
			k = len(logits)
		}
		entries := topkEntries
		if len(entries) < k {
			entries = make([]TopKEntry, k)
		}
		n := fillTopK(entries[:k], logits, k)
		if n == 0 {
			return kernels.Argmax(logits)
		}
		probsTopK := topkProbs
		if len(probsTopK) < n {
			probsTopK = make([]float32, n)
		}
		return sampleFromTopK(entries[:n], cfg.temp, cfg.topP, rng, probsTopK[:n])
	}
	if cfg.topP < 1 {
		return sampleFromTopP(logits, cfg.temp, cfg.topP, rng, probs, idx)
	}
	return sampleFromFull(logits, cfg.temp, rng, probs)
}

func sampleFromTopK(entries []TopKEntry, temp float32, topP float32, rng *sampler, probs []float32) int {
	if len(entries) == 0 {
		return -1
	}
	maxLogit := entries[0].Logit / temp
	for i := 1; i < len(entries); i++ {
		val := entries[i].Logit / temp
		if val > maxLogit {
			maxLogit = val
		}
	}
	if len(probs) < len(entries) {
		probs = make([]float32, len(entries))
	}
	probs = probs[:len(entries)]
	var sum float32
	for i := range entries {
		val := entries[i].Logit/temp - maxLogit
		p := expForSampling(val)
		probs[i] = p
		sum += p
	}
	if sum == 0 {
		return int(entries[0].TokenID)
	}
	inv := 1 / sum
	for i := range probs {
		probs[i] *= inv
	}
	limit := len(entries)
	if topP < 1 {
		var cum float32
		for i := 0; i < len(entries); i++ {
			cum += probs[i]
			if cum >= topP {
				limit = i + 1
				break
			}
		}
		if limit < len(entries) {
			var subSum float32
			for i := 0; i < limit; i++ {
				subSum += probs[i]
			}
			if subSum > 0 {
				invSub := 1 / subSum
				for i := 0; i < limit; i++ {
					probs[i] *= invSub
				}
			}
		}
	}
	r := rng.nextFloat()
	var cum float32
	for i := 0; i < limit; i++ {
		cum += probs[i]
		if r <= cum {
			return int(entries[i].TokenID)
		}
	}
	return int(entries[limit-1].TokenID)
}

func sampleFromFull(logits []float32, temp float32, rng *sampler, probs []float32) int {
	if len(logits) == 0 {
		return -1
	}
	maxLogit := logits[0] / temp
	for i := 1; i < len(logits); i++ {
		val := logits[i] / temp
		if val > maxLogit {
			maxLogit = val
		}
	}
	var sum float32
	for i := range logits {
		val := logits[i]/temp - maxLogit
		p := expForSampling(val)
		probs[i] = p
		sum += p
	}
	if sum == 0 {
		return kernels.Argmax(logits)
	}
	inv := 1 / sum
	for i := range probs {
		probs[i] *= inv
	}
	r := rng.nextFloat()
	var cum float32
	for i := range probs {
		cum += probs[i]
		if r <= cum {
			return i
		}
	}
	return len(probs) - 1
}

func sampleFromTopP(logits []float32, temp float32, topP float32, rng *sampler, probs []float32, idx []int) int {
	if len(logits) == 0 {
		return -1
	}
	if topPPrefilterK > 0 && topPPrefilterK < len(logits) {
		if id, ok := sampleFromTopPPrefilter(logits, temp, topP, rng, probs, idx, topPPrefilterK); ok {
			return id
		}
	}
	if topPHeapCap > 0 && topPHeapCap < len(logits) {
		if id, ok := sampleFromTopPHeap(logits, temp, topP, rng, topPHeapCap); ok {
			return id
		}
	}
	return sampleFromTopPSort(logits, temp, topP, rng, probs, idx)
}

func sampleFromTopPPrefilter(logits []float32, temp float32, topP float32, rng *sampler, probs []float32, idx []int, k int) (int, bool) {
	n := len(logits)
	if n == 0 || len(idx) < n || k <= 0 || k >= n {
		return -1, false
	}
	for i := 0; i < n; i++ {
		idx[i] = i
	}
	maxLogit := logits[0] / temp
	for i := 1; i < n; i++ {
		v := logits[i] / temp
		if v > maxLogit {
			maxLogit = v
		}
	}
	selectTopKIndices(idx[:n], logits, k)
	sortIndicesByLogitsDesc(idx[:k], logits)

	var total float32
	for i := 0; i < n; i++ {
		total += expForSampling(logits[i]/temp - maxLogit)
	}
	target := topP * total

	var prefixSum float32
	limit := k
	for i := 0; i < k; i++ {
		id := idx[i]
		p := expForSampling(logits[id]/temp - maxLogit)
		probs[id] = p
		prefixSum += p
		if prefixSum >= target {
			limit = i + 1
			break
		}
	}
	if prefixSum < target {
		return -1, false
	}
	r := rng.nextFloat() * prefixSum
	var run float32
	for i := 0; i < limit; i++ {
		id := idx[i]
		run += probs[id]
		if r <= run {
			return id, true
		}
	}
	return idx[limit-1], true
}

type topPEntry struct {
	id int
	p  float32
}

type topPMinHeap []topPEntry

func topPHeapUp(h topPMinHeap, i int) {
	for i > 0 {
		p := (i - 1) / 2
		if h[p].p <= h[i].p {
			break
		}
		h[p], h[i] = h[i], h[p]
		i = p
	}
}

func topPHeapDown(h topPMinHeap, i int) {
	n := len(h)
	for {
		l := 2*i + 1
		if l >= n {
			return
		}
		s := l
		r := l + 1
		if r < n && h[r].p < h[l].p {
			s = r
		}
		if h[i].p <= h[s].p {
			return
		}
		h[i], h[s] = h[s], h[i]
		i = s
	}
}

func sampleFromTopPHeap(logits []float32, temp float32, topP float32, rng *sampler, capN int) (int, bool) {
	if len(logits) == 0 || capN <= 0 {
		return -1, false
	}
	maxLogit := logits[0] / temp
	for i := 1; i < len(logits); i++ {
		v := logits[i] / temp
		if v > maxLogit {
			maxLogit = v
		}
	}
	h := make(topPMinHeap, 0, capN)
	var total, topSum float32
	for id := range logits {
		p := expForSampling(logits[id]/temp - maxLogit)
		total += p
		if len(h) < capN {
			h = append(h, topPEntry{id: id, p: p})
			topPHeapUp(h, len(h)-1)
			topSum += p
			continue
		}
		if p <= h[0].p {
			continue
		}
		topSum += p - h[0].p
		h[0] = topPEntry{id: id, p: p}
		topPHeapDown(h, 0)
	}
	target := topP * total
	if topSum < target {
		return -1, false
	}
	sortTopPEntriesDesc(h)
	var cum float32
	limit := len(h)
	for i := range h {
		cum += h[i].p
		if cum >= target {
			limit = i + 1
			break
		}
	}
	if cum == 0 {
		return kernels.Argmax(logits), true
	}
	r := rng.nextFloat() * cum
	var run float32
	for i := 0; i < limit; i++ {
		run += h[i].p
		if r <= run {
			return h[i].id, true
		}
	}
	return h[limit-1].id, true
}

func sampleFromTopPSort(logits []float32, temp float32, topP float32, rng *sampler, probs []float32, idx []int) int {
	if len(logits) == 0 {
		return -1
	}
	if len(idx) < len(logits) {
		return sampleFromFull(logits, temp, rng, probs)
	}
	n := len(logits)
	for i := 0; i < n; i++ {
		idx[i] = i
	}
	k := topPSortPrefix
	if k <= 0 {
		k = n
	}
	if k > n {
		k = n
	}
	limit := n
	var cum float32
	maxLogit := logits[0] / temp
	for i := 1; i < n; i++ {
		v := logits[i] / temp
		if v > maxLogit {
			maxLogit = v
		}
	}
	for {
		selectTopKIndices(idx[:n], logits, k)
		sortIndicesByLogitsDesc(idx[:k], logits)
		cum = 0
		limit = k
		for i := 0; i < k; i++ {
			id := idx[i]
			val := logits[id]/temp - maxLogit
			p := expForSampling(val)
			probs[id] = p
			cum += p
			if cum >= topP {
				limit = i + 1
				break
			}
		}
		if cum >= topP || k == n {
			break
		}
		k *= 2
		if k > n {
			k = n
		}
	}
	if cum == 0 {
		return kernels.Argmax(logits)
	}
	inv := 1 / cum
	for i := 0; i < limit; i++ {
		id := idx[i]
		probs[id] *= inv
	}
	r := rng.nextFloat()
	var run float32
	for i := 0; i < limit; i++ {
		id := idx[i]
		run += probs[id]
		if r <= run {
			return id
		}
	}
	return idx[limit-1]
}

func selectTopKIndices(idx []int, logits []float32, k int) {
	n := len(idx)
	if k <= 0 || k >= n {
		return
	}
	left, right := 0, n-1
	target := k - 1
	for left < right {
		pivot := logits[idx[(left+right)/2]]
		i, j := left, right
		for i <= j {
			for logits[idx[i]] > pivot {
				i++
			}
			for logits[idx[j]] < pivot {
				j--
			}
			if i <= j {
				idx[i], idx[j] = idx[j], idx[i]
				i++
				j--
			}
		}
		if target <= j {
			right = j
		} else if target >= i {
			left = i
		} else {
			return
		}
	}
}

func sortIndicesByLogitsDesc(idx []int, logits []float32) {
	slices.SortFunc(idx, func(a, b int) int {
		la := logits[a]
		lb := logits[b]
		if la > lb {
			return -1
		}
		if la < lb {
			return 1
		}
		return 0
	})
}

func sortTopPEntriesDesc(h topPMinHeap) {
	slices.SortFunc(h, func(a, b topPEntry) int {
		if a.p > b.p {
			return -1
		}
		if a.p < b.p {
			return 1
		}
		return 0
	})
}

func expForSampling(x float32) float32 {
	if debugStrictExpf || debugFastExpf {
		return expf32(x)
	}
	return float32(math.Exp(float64(x)))
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
