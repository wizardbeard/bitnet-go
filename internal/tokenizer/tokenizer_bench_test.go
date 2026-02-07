package tokenizer

import (
	"testing"

	"bitnet-go/internal/gguf"
)

func BenchmarkSplitGPT2(b *testing.B) {
	text := "Hello, world! 12345 It's tokenizer time.\nNew line."
	for i := 0; i < b.N; i++ {
		_ = splitGPT2(text)
	}
}

func BenchmarkSplitLlama3(b *testing.B) {
	text := "Hello, world! 1234567890 It's tokenizer time.\nNew line."
	for i := 0; i < b.N; i++ {
		_ = splitLlama3(text)
	}
}

func BenchmarkTokenizeBPE(b *testing.B) {
	info := gguf.ModelInfo{
		KeyValues: map[string]any{
			"tokenizer.ggml.model":            "gpt2",
			"tokenizer.ggml.tokens":           []string{"<unk>", "Ġ", "h", "e", "l", "o", "Ġh", "Ġhe", "Ġhel", "Ġhell", "Ġhello"},
			"tokenizer.ggml.merges":           []string{"Ġ h", "Ġh e", "Ġhe l", "Ġhel l", "Ġhell o"},
			"tokenizer.ggml.bos_token_id":     uint32(1),
			"tokenizer.ggml.unknown_token_id": uint32(0),
		},
	}
	tok, err := NewFromModelInfo(info)
	if err != nil {
		b.Fatalf("NewFromModelInfo: %v", err)
	}
	text := " hello hello hello"
	for i := 0; i < b.N; i++ {
		_ = tok.Tokenize(text)
	}
}

func BenchmarkTokenizeBPECold(b *testing.B) {
	info := gguf.ModelInfo{
		KeyValues: map[string]any{
			"tokenizer.ggml.model":            "gpt2",
			"tokenizer.ggml.tokens":           []string{"<unk>", "Ġ", "h", "e", "l", "o", "Ġh", "Ġhe", "Ġhel", "Ġhell", "Ġhello"},
			"tokenizer.ggml.merges":           []string{"Ġ h", "Ġh e", "Ġhe l", "Ġhel l", "Ġhell o"},
			"tokenizer.ggml.bos_token_id":     uint32(1),
			"tokenizer.ggml.unknown_token_id": uint32(0),
		},
	}
	text := " hello hello hello"
	for i := 0; i < b.N; i++ {
		tok, err := NewFromModelInfo(info)
		if err != nil {
			b.Fatalf("NewFromModelInfo: %v", err)
		}
		_ = tok.Tokenize(text)
	}
}

func BenchmarkTokenizeSPM(b *testing.B) {
	info := gguf.ModelInfo{
		KeyValues: map[string]any{
			"tokenizer.ggml.model":            "llama",
			"tokenizer.ggml.tokens":           []string{"<unk>", "<s>", "▁", "h", "e", "l", "o", "▁h", "▁he", "▁hel", "▁hell", "▁hello"},
			"tokenizer.ggml.scores":           []float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			"tokenizer.ggml.bos_token_id":     uint32(1),
			"tokenizer.ggml.unknown_token_id": uint32(0),
		},
	}
	tok, err := NewFromModelInfo(info)
	if err != nil {
		b.Fatalf("NewFromModelInfo: %v", err)
	}
	text := "Hello hello hello"
	for i := 0; i < b.N; i++ {
		_ = tok.Tokenize(text)
	}
}

func BenchmarkTokenizeSPMCold(b *testing.B) {
	info := gguf.ModelInfo{
		KeyValues: map[string]any{
			"tokenizer.ggml.model":            "llama",
			"tokenizer.ggml.tokens":           []string{"<unk>", "<s>", "▁", "h", "e", "l", "o", "▁h", "▁he", "▁hel", "▁hell", "▁hello"},
			"tokenizer.ggml.scores":           []float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			"tokenizer.ggml.bos_token_id":     uint32(1),
			"tokenizer.ggml.unknown_token_id": uint32(0),
		},
	}
	text := "Hello hello hello"
	for i := 0; i < b.N; i++ {
		tok, err := NewFromModelInfo(info)
		if err != nil {
			b.Fatalf("NewFromModelInfo: %v", err)
		}
		_ = tok.Tokenize(text)
	}
}
