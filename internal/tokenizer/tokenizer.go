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
	vocab            map[string]int32
	scores           []float32
	bpeRanks         map[string]int
	byteEncode       [256]string
	trie             *trieNode
	byteTok          [256]int32
	bpeBuf           []string
	bpeWork          []string
	bpeWork2         []string
	bpeByteBuf       []byte
	bpeRuneBuf       []rune
	bpeByteSym       [256]string
	bpeKeyBuf        []byte
	bpeChunkCache    *bpeChunkCache
	bpeChunkCacheCap int
	spmChunkCache    *bpeChunkCache
	spmChunkCacheCap int
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
		vocab:            make(map[string]int32, len(tokens)),
		scores:           scores,
		bpeRanks:         make(map[string]int),
		trie:             newTrieNode(),
		bpeChunkCacheCap: 256,
		spmChunkCacheCap: 256,
	}
	if v, ok := info.KeyValues["bitnet.tokenizer.bpe_cache_size"].(uint32); ok {
		t.bpeChunkCacheCap = int(v)
	}
	if v, ok := info.KeyValues["bitnet.tokenizer.spm_cache_size"].(uint32); ok {
		t.spmChunkCacheCap = int(v)
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

	if merges, ok := info.KeyValues["tokenizer.ggml.merges"].([]string); ok {
		for i, m := range merges {
			parts := strings.SplitN(m, " ", 2)
			if len(parts) != 2 {
				continue
			}
			t.bpeRanks[parts[0]+"\x00"+parts[1]] = i
		}
	}
	if t.model == "" && len(t.bpeRanks) > 0 {
		// Some GGUFs omit tokenizer.ggml.model but include BPE merges.
		t.model = "gpt2"
		t.addBOS = false
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

func isASCIILetter(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')
}

func isASCIIDigit(b byte) bool {
	return b >= '0' && b <= '9'
}

func isASCIISpace(b byte) bool {
	switch b {
	case ' ', '\t', '\n', '\r', '\v', '\f':
		return true
	default:
		return false
	}
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
	keyBuf := t.bpeKeyBuf
	for {
		if len(syms) < 2 {
			break
		}
		bestRank := int(^uint(0) >> 1)
		bestIdx := -1
		for i := 0; i < len(syms)-1; i++ {
			key := t.pairKey(syms[i], syms[i+1])
			rank, ok := t.bpeRanks[key]
			if !ok {
				continue
			}
			if rank < bestRank {
				bestRank = rank
				bestIdx = i
			}
		}
		if bestIdx < 0 {
			break
		}
		merged := t.mergePair(syms[bestIdx], syms[bestIdx+1])
		next := t.bpeWork[:0]
		next = append(next, syms[:bestIdx]...)
		next = append(next, merged)
		next = append(next, syms[bestIdx+2:]...)
		syms, next = next, syms
		t.bpeWork = next[:0]
	}
	t.bpeKeyBuf = keyBuf[:0]

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
	return left + right
}

func (t *Tokenizer) bpeByteMap(s string) string {
	raw := []byte(s)
	n := len(raw)
	buf := t.bpeByteBuf
	if cap(buf) < n*2 {
		buf = make([]byte, 0, n*2)
	} else {
		buf = buf[:0]
	}
	for _, x := range raw {
		buf = append(buf, t.byteEncode[x]...)
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
	syms := make([]spmSymbol, 0, len(text))
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

	q := &spmBigramHeap{}
	heap.Init(q)
	revMerge := make(map[string][2]int)

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
		heap.Push(q, spmBigram{left: left, right: right, score: t.scores[id], size: len(piece)})
		revMerge[piece] = [2]int{left, right}
	}

	for i := 1; i < len(syms); i++ {
		tryAddBigram(i-1, i)
	}

	for q.Len() > 0 {
		bg := heap.Pop(q).(spmBigram)
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
	var resegment func(idx int)
	resegment = func(idx int) {
		s := syms[idx]
		piece := text[s.start : s.start+s.n]
		if id, ok := t.vocab[piece]; ok {
			out = append(out, id)
			return
		}
		if pair, ok := revMerge[piece]; ok {
			resegment(pair[0])
			resegment(pair[1])
			return
		}
		for i := s.start; i < s.start+s.n; i++ {
			out = append(out, t.byteTok[text[i]])
		}
	}

	for i := 0; i != -1; i = syms[i].next {
		if syms[i].n == 0 {
			continue
		}
		resegment(i)
	}
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

type match struct {
	length int
	id     int32
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
