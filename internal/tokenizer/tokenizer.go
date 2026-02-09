package tokenizer

import (
	"container/heap"
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"

	"bitnet-go/internal/gguf"
)

type Tokenizer struct {
	addBOS           bool
	bosTokenID       int32
	unkTokenID       int32
	model            string
	preType          string
	tokens           []string
	hasSPMPrefix     bool
	hasBPEMerges     bool
	vocab            map[string]int32
	scores           []float32
	bpeRanks         map[string]int
	bpeRanksPair     map[bpePair]int
	byteEncode       [256]string
	byteDecode       map[string]byte
	trie             *trieNode
	byteTok          [256]int32
	bpeBuf           []string
	bpeWork          []string
	bpeWork2         []string
	bpeRankBuf       []int
	bpeByteBuf       []byte
	bpeRuneBuf       []rune
	bpeByteSym       [256]string
	bpeKeyBuf        []byte
	bpeChunkCache    *bpeChunkCache
	bpeChunkCacheCap int
	bpeMergeCache    map[bpePair]string
	bpeMergeCacheCap int
	spmChunkCache    *bpeChunkCache
	spmChunkCacheCap int
	spmSymbolPool    []spmSymbol
	spmHeapPool      spmBigramHeap
	spmMergePool     map[string][2]int
	spmIndexStack    []int
}

type bpePair struct {
	a string
	b string
}

func NewFromModelInfo(info gguf.ModelInfo) (*Tokenizer, error) {
	rawTokens, ok := info.KeyValues["tokenizer.ggml.tokens"]
	if !ok {
		return nil, fmt.Errorf("missing tokenizer.ggml.tokens")
	}
	tokens, ok := rawTokens.([]string)
	if !ok || len(tokens) == 0 {
		return nil, fmt.Errorf("tokenizer.ggml.tokens has invalid format")
	}

	scores, ok := info.KeyValues["tokenizer.ggml.scores"].([]float32)
	if !ok || len(scores) != len(tokens) {
		scores = make([]float32, len(tokens))
	}

	model := firstString(info.KeyValues["tokenizer.ggml.model"])
	defaultAddBOS := model == "llama"

	t := &Tokenizer{
		addBOS:           defaultAddBOS,
		bosTokenID:       int32(firstUint32(info.KeyValues["tokenizer.ggml.bos_token_id"])),
		unkTokenID:       int32(firstUint32(info.KeyValues["tokenizer.ggml.unknown_token_id"])),
		model:            model,
		preType:          firstString(info.KeyValues["tokenizer.ggml.pre"]),
		tokens:           tokens,
		vocab:            make(map[string]int32, len(tokens)),
		scores:           scores,
		bpeRanks:         make(map[string]int),
		bpeRanksPair:     make(map[bpePair]int),
		trie:             newTrieNode(),
		bpeChunkCacheCap: 256,
		bpeMergeCacheCap: 4096,
		spmChunkCacheCap: 256,
	}
	if v, ok := info.KeyValues["bitnet.tokenizer.bpe_cache_size"].(uint32); ok {
		t.bpeChunkCacheCap = int(v)
	}
	if v, ok := info.KeyValues["bitnet.tokenizer.spm_cache_size"].(uint32); ok {
		t.spmChunkCacheCap = int(v)
	}
	if v, ok := info.KeyValues["bitnet.tokenizer.bpe_merge_cache_size"].(uint32); ok {
		t.bpeMergeCacheCap = int(v)
	}
	for i := 0; i < 256; i++ {
		t.bpeByteSym[i] = string(byte(i))
	}
	for i := range t.byteTok {
		t.byteTok[i] = t.unkTokenID
	}
	if v, ok := info.KeyValues["tokenizer.ggml.add_bos_token"].(bool); ok {
		t.addBOS = v
	}

	for i, piece := range tokens {
		id := int32(i)
		t.vocab[piece] = id
		t.trie.insert(piece, id)
		if b, ok := parseByteToken(piece); ok {
			t.byteTok[b] = id
		}
	}
	t.byteEncode = buildByteEncoder()
	t.byteDecode = buildByteDecoder(t.byteEncode[:])

	if merges, ok := info.KeyValues["tokenizer.ggml.merges"].([]string); ok {
		t.hasBPEMerges = len(merges) > 0
		t.bpeRanksPair = make(map[bpePair]int, len(merges))
		for i, m := range merges {
			parts := strings.SplitN(m, " ", 2)
			if len(parts) != 2 {
				continue
			}
			t.bpeRanks[parts[0]+"\x00"+parts[1]] = i
			t.bpeRanksPair[bpePair{a: parts[0], b: parts[1]}] = i
		}
	}
	if t.model == "" && len(t.bpeRanks) > 0 {
		// Some GGUFs omit tokenizer.ggml.model but include BPE merges.
		t.model = "gpt2"
		t.addBOS = false
	}
	for _, tok := range tokens {
		if strings.HasPrefix(tok, "▁") {
			t.hasSPMPrefix = true
			break
		}
	}
	return t, nil
}

func (t *Tokenizer) Tokenize(prompt string) []int32 {
	if t.trie == nil {
		return nil
	}

	out := make([]int32, 0, len(prompt))
	if t.addBOS {
		out = append(out, t.bosTokenID)
	}

	if t.model == "llama" {
		normalized := normalizeSPM(prompt)
		if t.spmChunkCache == nil {
			t.spmChunkCache = newBPEChunkCache(t.spmChunkCacheCap)
		}
		if cached := t.spmChunkCache.get(normalized); cached != nil {
			out = append(out, cached...)
			return out
		}
		encoded := t.tokenizeSPM(normalized)
		t.spmChunkCache.add(normalized, encoded)
		out = append(out, encoded...)
		return out
	}
	if t.model == "gpt2" && len(t.bpeRanks) > 0 {
		out = append(out, t.tokenizeBPE(prompt)...)
		return out
	}

	text := prompt
	if !strings.HasPrefix(text, " ") {
		text = " " + text
	}
	out = append(out, t.tokenizeGreedy(text)...)
	return out
}

func (t *Tokenizer) Decode(tokens []int32) string {
	if len(tokens) == 0 || len(t.tokens) == 0 {
		return ""
	}
	if t.hasBPEMerges || t.model == "gpt2" {
		var out []byte
		for _, id := range tokens {
			if id < 0 || int(id) >= len(t.tokens) {
				continue
			}
			piece := t.tokens[id]
			for _, r := range piece {
				rs := string(r)
				if b, ok := t.byteDecode[rs]; ok {
					out = append(out, b)
				} else {
					out = append(out, rs...)
				}
			}
		}
		return string(out)
	}
	if t.model == "llama" || t.hasSPMPrefix {
		var b strings.Builder
		for _, id := range tokens {
			if id < 0 || int(id) >= len(t.tokens) {
				continue
			}
			piece := t.tokens[id]
			if piece == "▁" {
				b.WriteByte(' ')
				continue
			}
			if strings.HasPrefix(piece, "▁") {
				b.WriteByte(' ')
				b.WriteString(piece[3:])
				continue
			}
			b.WriteString(piece)
		}
		return b.String()
	}
	var b strings.Builder
	for _, id := range tokens {
		if id < 0 || int(id) >= len(t.tokens) {
			continue
		}
		b.WriteString(t.tokens[id])
	}
	return b.String()
}

func (t *Tokenizer) tokenizeBPE(prompt string) []int32 {
	if prompt == "" {
		return nil
	}
	if t.bpeChunkCache == nil {
		t.bpeChunkCache = newBPEChunkCache(t.bpeChunkCacheCap)
	}
	chunks := t.splitBPEPieces(prompt)
	if len(chunks) == 0 {
		chunks = []string{prompt}
	}
	out := make([]int32, 0, len(prompt))
	for _, chunk := range chunks {
		encoded := t.bpeChunkCache.get(chunk)
		if encoded == nil {
			encoded = t.encodeBPEWord(t.bpeByteMap(chunk))
			t.bpeChunkCache.add(chunk, encoded)
		}
		out = append(out, encoded...)
	}
	return out
}

func (t *Tokenizer) splitBPEPieces(text string) []string {
	switch strings.ToLower(t.preType) {
	case "llama3", "dbrx", "smaug":
		return splitLlama3(text)
	default:
		return splitGPT2(text)
	}
}

func splitGPT2(s string) []string {
	return splitByRules(s, false)
}

func splitLlama3(s string) []string {
	return splitByRules(s, true)
}

func splitByRules(s string, llama3 bool) []string {
	if isASCII(s) {
		return splitByRulesASCII(s, llama3)
	}
	rs := []rune(s)
	out := make([]string, 0, len(rs))
	for i := 0; i < len(rs); {
		// contractions
		if rs[i] == '\'' && i+1 < len(rs) {
			next := rs[i+1]
			if llama3 {
				next = unicode.ToLower(next)
			}
			if next == 's' || next == 't' || next == 'm' || next == 'd' {
				out = append(out, string(rs[i:i+2]))
				i += 2
				continue
			}
			if i+2 < len(rs) {
				n2 := rs[i+2]
				if llama3 {
					n2 = unicode.ToLower(n2)
				}
				if (next == 'r' && n2 == 'e') || (next == 'v' && n2 == 'e') || (next == 'l' && n2 == 'l') {
					out = append(out, string(rs[i:i+3]))
					i += 3
					continue
				}
			}
		}

		leadSpace := i < len(rs) && rs[i] == ' '
		j := i
		if leadSpace {
			j++
		}

		// letters
		if j < len(rs) && unicode.IsLetter(rs[j]) {
			k := j
			for k < len(rs) && unicode.IsLetter(rs[k]) {
				k++
			}
			out = append(out, string(rs[i:k]))
			i = k
			continue
		}
		// numbers
		if j < len(rs) && unicode.IsNumber(rs[j]) {
			k := j
			if llama3 {
				for k < len(rs) && unicode.IsNumber(rs[k]) {
					step := k + 3
					if step > len(rs) {
						step = len(rs)
					}
					out = append(out, string(rs[k:step]))
					k = step
				}
				i = k
				continue
			}
			for k < len(rs) && unicode.IsNumber(rs[k]) {
				k++
			}
			out = append(out, string(rs[i:k]))
			i = k
			continue
		}
		// punctuation/symbol block with optional leading space
		if j < len(rs) && !unicode.IsSpace(rs[j]) && !unicode.IsLetter(rs[j]) && !unicode.IsNumber(rs[j]) {
			k := j
			for k < len(rs) && !unicode.IsSpace(rs[k]) && !unicode.IsLetter(rs[k]) && !unicode.IsNumber(rs[k]) {
				k++
			}
			if llama3 {
				for k < len(rs) && (rs[k] == '\r' || rs[k] == '\n') {
					k++
				}
			}
			out = append(out, string(rs[i:k]))
			i = k
			continue
		}
		// whitespace block
		if unicode.IsSpace(rs[i]) {
			k := i
			for k < len(rs) && unicode.IsSpace(rs[k]) {
				k++
			}
			if llama3 {
				// \s*[\r\n]+
				nl := -1
				for p := i; p < k; p++ {
					if rs[p] == '\r' || rs[p] == '\n' {
						nl = p
					}
				}
				if nl >= 0 {
					out = append(out, string(rs[i:nl+1]))
					i = nl + 1
					continue
				}
			}
			// \s+(?!\S)
			if k-i > 1 && k < len(rs) {
				out = append(out, string(rs[i:k-1]))
				i = k - 1
				continue
			}
			out = append(out, string(rs[i:k]))
			i = k
			continue
		}

		out = append(out, string(rs[i:i+1]))
		i++
	}
	return out
}

func splitByRulesASCII(s string, llama3 bool) []string {
	out := make([]string, 0, len(s))
	for i := 0; i < len(s); {
		if s[i] == '\'' && i+1 < len(s) {
			next := s[i+1]
			if llama3 && next >= 'A' && next <= 'Z' {
				next = next - 'A' + 'a'
			}
			if next == 's' || next == 't' || next == 'm' || next == 'd' {
				out = append(out, s[i:i+2])
				i += 2
				continue
			}
			if i+2 < len(s) {
				n2 := s[i+2]
				if llama3 && n2 >= 'A' && n2 <= 'Z' {
					n2 = n2 - 'A' + 'a'
				}
				if (next == 'r' && n2 == 'e') || (next == 'v' && n2 == 'e') || (next == 'l' && n2 == 'l') {
					out = append(out, s[i:i+3])
					i += 3
					continue
				}
			}
		}

		leadSpace := s[i] == ' '
		j := i
		if leadSpace {
			j++
		}

		if j < len(s) && isASCIILetter(s[j]) {
			k := j
			for k < len(s) && isASCIILetter(s[k]) {
				k++
			}
			out = append(out, s[i:k])
			i = k
			continue
		}
		if j < len(s) && isASCIIDigit(s[j]) {
			k := j
			if llama3 {
				for k < len(s) && isASCIIDigit(s[k]) {
					step := k + 3
					if step > len(s) {
						step = len(s)
					}
					out = append(out, s[k:step])
					k = step
				}
				i = k
				continue
			}
			for k < len(s) && isASCIIDigit(s[k]) {
				k++
			}
			out = append(out, s[i:k])
			i = k
			continue
		}
		if j < len(s) && !isASCIISpace(s[j]) && !isASCIILetter(s[j]) && !isASCIIDigit(s[j]) {
			k := j
			for k < len(s) && !isASCIISpace(s[k]) && !isASCIILetter(s[k]) && !isASCIIDigit(s[k]) {
				k++
			}
			if llama3 {
				for k < len(s) && (s[k] == '\r' || s[k] == '\n') {
					k++
				}
			}
			out = append(out, s[i:k])
			i = k
			continue
		}
		if isASCIISpace(s[i]) {
			k := i
			for k < len(s) && isASCIISpace(s[k]) {
				k++
			}
			if llama3 {
				nl := -1
				for p := i; p < k; p++ {
					if s[p] == '\r' || s[p] == '\n' {
						nl = p
					}
				}
				if nl >= 0 {
					out = append(out, s[i:nl+1])
					i = nl + 1
					continue
				}
			}
			if k-i > 1 && k < len(s) {
				out = append(out, s[i:k-1])
				i = k - 1
				continue
			}
			out = append(out, s[i:k])
			i = k
			continue
		}

		out = append(out, s[i:i+1])
		i++
	}
	return out
}

func isASCII(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] >= utf8.RuneSelf {
			return false
		}
	}
	return true
}

const (
	asciiClassLetter = 1 << iota
	asciiClassDigit
	asciiClassSpace
)

var asciiClassTable = func() [256]uint8 {
	var tbl [256]uint8
	for c := byte('a'); c <= 'z'; c++ {
		tbl[c] |= asciiClassLetter
	}
	for c := byte('A'); c <= 'Z'; c++ {
		tbl[c] |= asciiClassLetter
	}
	for c := byte('0'); c <= '9'; c++ {
		tbl[c] |= asciiClassDigit
	}
	tbl[' '] |= asciiClassSpace
	tbl['\t'] |= asciiClassSpace
	tbl['\n'] |= asciiClassSpace
	tbl['\r'] |= asciiClassSpace
	tbl['\v'] |= asciiClassSpace
	tbl['\f'] |= asciiClassSpace
	return tbl
}()

func isASCIILetter(b byte) bool {
	return asciiClassTable[b]&asciiClassLetter != 0
}

func isASCIIDigit(b byte) bool {
	return asciiClassTable[b]&asciiClassDigit != 0
}

func isASCIISpace(b byte) bool {
	return asciiClassTable[b]&asciiClassSpace != 0
}

func (t *Tokenizer) encodeBPEWord(word string) []int32 {
	if word == "" {
		return nil
	}
	syms := t.bpeBuf[:0]
	if isASCII(word) {
		for i := 0; i < len(word); i++ {
			syms = append(syms, t.bpeByteSym[word[i]])
		}
	} else {
		runes := t.bpeRuneBuf[:0]
		for _, r := range word {
			runes = append(runes, r)
		}
		t.bpeRuneBuf = runes[:0]
		for i := range runes {
			syms = append(syms, string(runes[i]))
		}
	}
	ranks := t.bpeRankBuf[:0]
	if cap(ranks) < len(syms)-1 {
		ranks = make([]int, len(syms)-1)
	} else {
		ranks = ranks[:len(syms)-1]
	}
	rankFor := func(a, b string) int {
		if len(t.bpeRanksPair) > 0 {
			if r, ok := t.bpeRanksPair[bpePair{a: a, b: b}]; ok {
				return r
			}
			return -1
		}
		key := t.pairKey(a, b)
		if r, ok := t.bpeRanks[key]; ok {
			return r
		}
		return -1
	}
	for i := 0; i < len(syms)-1; i++ {
		ranks[i] = rankFor(syms[i], syms[i+1])
	}
	for {
		if len(syms) < 2 {
			break
		}
		bestRank := int(^uint(0) >> 1)
		bestIdx := -1
		for i := 0; i < len(ranks); i++ {
			r := ranks[i]
			if r >= 0 && r < bestRank {
				bestRank = r
				bestIdx = i
			}
		}
		if bestIdx < 0 {
			break
		}
		merged := t.mergePair(syms[bestIdx], syms[bestIdx+1])
		syms[bestIdx] = merged
		copy(syms[bestIdx+1:], syms[bestIdx+2:])
		syms = syms[:len(syms)-1]
		copy(ranks[bestIdx:], ranks[bestIdx+1:])
		ranks = ranks[:len(syms)-1]
		if bestIdx-1 >= 0 {
			ranks[bestIdx-1] = rankFor(syms[bestIdx-1], syms[bestIdx])
		}
		if bestIdx < len(ranks) {
			ranks[bestIdx] = rankFor(syms[bestIdx], syms[bestIdx+1])
		}
	}
	t.bpeRankBuf = ranks[:0]

	out := make([]int32, 0, len(syms))
	for _, s := range syms {
		if id, ok := t.vocab[s]; ok {
			out = append(out, id)
			continue
		}
		for _, r := range s {
			if id, ok := t.vocab[string(r)]; ok {
				out = append(out, id)
			} else {
				out = append(out, t.unkTokenID)
			}
		}
	}
	t.bpeBuf = syms[:0]
	return out
}

func (t *Tokenizer) pairKey(left, right string) string {
	keyBuf := t.bpeKeyBuf
	keyBuf = keyBuf[:0]
	keyBuf = append(keyBuf, left...)
	keyBuf = append(keyBuf, 0)
	keyBuf = append(keyBuf, right...)
	key := string(keyBuf)
	t.bpeKeyBuf = keyBuf[:0]
	return key
}

func (t *Tokenizer) mergePair(left, right string) string {
	if t.bpeMergeCacheCap <= 0 {
		return left + right
	}
	if t.bpeMergeCache == nil {
		t.bpeMergeCache = make(map[bpePair]string, t.bpeMergeCacheCap)
	} else if len(t.bpeMergeCache) >= t.bpeMergeCacheCap {
		for k := range t.bpeMergeCache {
			delete(t.bpeMergeCache, k)
		}
	}
	key := bpePair{a: left, b: right}
	if v, ok := t.bpeMergeCache[key]; ok {
		return v
	}
	merged := left + right
	t.bpeMergeCache[key] = merged
	return merged
}

func (t *Tokenizer) bpeByteMap(s string) string {
	n := len(s)
	buf := t.bpeByteBuf
	if cap(buf) < n*2 {
		buf = make([]byte, 0, n*2)
	} else {
		buf = buf[:0]
	}
	for i := 0; i < n; i++ {
		buf = append(buf, t.byteEncode[s[i]]...)
	}
	t.bpeByteBuf = buf[:0]
	return string(buf)
}

func (t *Tokenizer) tokenizeGreedy(text string) []int32 {
	out := make([]int32, 0, len(text))
	for i := 0; i < len(text); {
		bestLen := 0
		bestID := t.unkTokenID
		for _, m := range t.trie.match(text, i) {
			if m.length > bestLen {
				bestLen = m.length
				bestID = m.id
			}
		}
		if bestLen > 0 {
			out = append(out, bestID)
			i += bestLen
			continue
		}
		out = append(out, t.byteTok[text[i]])
		i++
	}
	return out
}

func (t *Tokenizer) tokenizeSPM(text string) []int32 {
	syms := t.spmSymbolPool[:0]
	if cap(syms) < len(text) {
		syms = make([]spmSymbol, 0, len(text))
	}
	for i := 0; i < len(text); {
		_, size := utf8.DecodeRuneInString(text[i:])
		if size <= 0 {
			size = 1
		}
		syms = append(syms, spmSymbol{start: i, n: size, prev: len(syms) - 1, next: len(syms) + 1})
		i += size
	}
	if len(syms) == 0 {
		return nil
	}
	syms[len(syms)-1].next = -1

	q := t.spmHeapPool[:0]
	t.spmHeapPool = q[:0]
	heap.Init(&q)
	revMerge := t.spmMergePool
	if revMerge == nil {
		revMerge = make(map[string][2]int, len(syms))
	} else {
		for k := range revMerge {
			delete(revMerge, k)
		}
	}

	tryAddBigram := func(left, right int) {
		if left < 0 || right < 0 || left >= len(syms) || right >= len(syms) {
			return
		}
		if syms[left].n == 0 || syms[right].n == 0 {
			return
		}
		piece := text[syms[left].start : syms[right].start+syms[right].n]
		id, ok := t.vocab[piece]
		if !ok || int(id) >= len(t.scores) {
			return
		}
		heap.Push(&q, spmBigram{left: left, right: right, score: t.scores[id], size: len(piece)})
		revMerge[piece] = [2]int{left, right}
	}

	for i := 1; i < len(syms); i++ {
		tryAddBigram(i-1, i)
	}

	for q.Len() > 0 {
		bg := heap.Pop(&q).(spmBigram)
		left := &syms[bg.left]
		right := &syms[bg.right]
		if left.n == 0 || right.n == 0 || left.n+right.n != bg.size {
			continue
		}
		left.n += right.n
		right.n = 0
		left.next = right.next
		if right.next >= 0 {
			syms[right.next].prev = bg.left
		}
		tryAddBigram(left.prev, bg.left)
		tryAddBigram(bg.left, left.next)
	}

	out := make([]int32, 0, len(text))
	intStack := t.spmIndexStack[:0]
	if cap(intStack) < len(text) {
		intStack = make([]int, 0, len(text))
	}
	for i := 0; i != -1; i = syms[i].next {
		if syms[i].n == 0 {
			continue
		}
		intStack = append(intStack, i)
		for len(intStack) > 0 {
			idx := intStack[len(intStack)-1]
			intStack = intStack[:len(intStack)-1]
			if idx < 0 || idx >= len(syms) {
				continue
			}
			s := syms[idx]
			if s.n == 0 {
				continue
			}
			piece := text[s.start : s.start+s.n]
			if id, ok := t.vocab[piece]; ok {
				out = append(out, id)
				continue
			}
			if pair, ok := revMerge[piece]; ok {
				// push right then left to preserve order
				intStack = append(intStack, pair[1])
				intStack = append(intStack, pair[0])
				continue
			}
			for i := s.start; i < s.start+s.n; i++ {
				out = append(out, t.byteTok[text[i]])
			}
		}
	}
	t.spmSymbolPool = syms[:0]
	t.spmHeapPool = q[:0]
	t.spmMergePool = revMerge
	t.spmIndexStack = intStack[:0]
	return out
}

func normalizeSPM(prompt string) string {
	return "▁" + strings.ReplaceAll(prompt, " ", "▁")
}

func buildByteEncoder() [256]string {
	var enc [256]string
	bs := make([]int, 0, 256)
	for i := int('!'); i <= int('~'); i++ {
		bs = append(bs, i)
	}
	for i := 0xA1; i <= 0xAC; i++ {
		bs = append(bs, i)
	}
	for i := 0xAE; i <= 0xFF; i++ {
		bs = append(bs, i)
	}
	seen := make(map[int]struct{}, len(bs))
	for _, v := range bs {
		seen[v] = struct{}{}
	}
	cs := append([]int(nil), bs...)
	n := 0
	for b := 0; b < 256; b++ {
		if _, ok := seen[b]; ok {
			continue
		}
		bs = append(bs, b)
		cs = append(cs, 256+n)
		n++
	}
	for i := range bs {
		enc[bs[i]] = string(rune(cs[i]))
	}
	return enc
}

func buildByteDecoder(encoder []string) map[string]byte {
	dec := make(map[string]byte, 256)
	for b := 0; b < 256; b++ {
		dec[encoder[b]] = byte(b)
	}
	return dec
}

func firstUint32(v any) uint32 {
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
	return 0
}

func firstString(v any) string {
	s, _ := v.(string)
	return s
}

type trieNode struct {
	children map[byte]*trieNode
	hasID    bool
	id       int32
}

func newTrieNode() *trieNode {
	return &trieNode{children: make(map[byte]*trieNode)}
}

func (n *trieNode) insert(piece string, id int32) {
	cur := n
	for i := 0; i < len(piece); i++ {
		b := piece[i]
		child, ok := cur.children[b]
		if !ok {
			child = newTrieNode()
			cur.children[b] = child
		}
		cur = child
	}
	cur.hasID = true
	cur.id = id
}

type match struct {
	length int
	id     int32
}

func (n *trieNode) match(text string, start int) []match {
	cur := n
	out := make([]match, 0, 4)
	for i := start; i < len(text); i++ {
		child, ok := cur.children[text[i]]
		if !ok {
			break
		}
		cur = child
		if cur.hasID {
			out = append(out, match{length: i - start + 1, id: cur.id})
		}
	}
	return out
}

type spmSymbol struct {
	start int
	n     int
	prev  int
	next  int
}

type spmBigram struct {
	left  int
	right int
	score float32
	size  int
}

type spmBigramHeap []spmBigram

func (h spmBigramHeap) Len() int { return len(h) }
func (h spmBigramHeap) Less(i, j int) bool {
	if h[i].score == h[j].score {
		return h[i].left < h[j].left
	}
	return h[i].score > h[j].score
}
func (h spmBigramHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *spmBigramHeap) Push(x any)   { *h = append(*h, x.(spmBigram)) }
func (h *spmBigramHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

func parseByteToken(piece string) (byte, bool) {
	if len(piece) != 6 || piece[0] != '<' || piece[1] != '0' || piece[2] != 'x' || piece[5] != '>' {
		return 0, false
	}
	hex := piece[3:5]
	var v byte
	for i := 0; i < 2; i++ {
		v <<= 4
		c := hex[i]
		switch {
		case c >= '0' && c <= '9':
			v |= c - '0'
		case c >= 'a' && c <= 'f':
			v |= c - 'a' + 10
		case c >= 'A' && c <= 'F':
			v |= c - 'A' + 10
		default:
			return 0, false
		}
	}
	return v, true
}
