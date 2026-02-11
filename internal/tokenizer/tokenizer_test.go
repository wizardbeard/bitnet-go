package tokenizer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"bitnet-go/internal/gguf"
)

func TestTokenizerGreedy(t *testing.T) {
	info := gguf.ModelInfo{
		KeyValues: map[string]any{
			"tokenizer.ggml.tokens":           []string{"<unk>", "<s>", " ", " Hello", " Bit", "Net", "H", "ello"},
			"tokenizer.ggml.bos_token_id":     uint32(1),
			"tokenizer.ggml.unknown_token_id": uint32(0),
			"tokenizer.ggml.add_bos_token":    true,
		},
	}

	tok, err := NewFromModelInfo(info)
	if err != nil {
		t.Fatalf("NewFromModelInfo() error = %v", err)
	}

	got := tok.Tokenize("Hello BitNet")
	want := []int32{1, 3, 4, 5}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d (%v)", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestTokenizerNoBOS(t *testing.T) {
	info := gguf.ModelInfo{
		KeyValues: map[string]any{
			"tokenizer.ggml.tokens":           []string{"<unk>", "<s>", " ", " hi"},
			"tokenizer.ggml.bos_token_id":     uint32(1),
			"tokenizer.ggml.unknown_token_id": uint32(0),
			"tokenizer.ggml.add_bos_token":    false,
		},
	}

	tok, err := NewFromModelInfo(info)
	if err != nil {
		t.Fatalf("NewFromModelInfo() error = %v", err)
	}
	got := tok.Tokenize("hi")
	if len(got) != 1 || got[0] != 3 {
		t.Fatalf("got %v, want [3]", got)
	}
}

func TestTokenizerByteFallback(t *testing.T) {
	info := gguf.ModelInfo{
		KeyValues: map[string]any{
			"tokenizer.ggml.tokens":           []string{"<unk>", "<s>", " ", "<0x41>"},
			"tokenizer.ggml.bos_token_id":     uint32(1),
			"tokenizer.ggml.unknown_token_id": uint32(0),
			"tokenizer.ggml.add_bos_token":    false,
		},
	}
	tok, err := NewFromModelInfo(info)
	if err != nil {
		t.Fatalf("NewFromModelInfo() error = %v", err)
	}
	got := tok.Tokenize("A")
	if len(got) != 2 || got[0] != 2 || got[1] != 3 {
		t.Fatalf("got %v, want [2 3]", got)
	}
}

func TestTokenizerFixturePrompt(t *testing.T) {
	root := filepath.Join("..", "..", "testdata")
	info, err := gguf.ReadModelInfo(filepath.Join(root, "stories15M-q8_0.gguf"))
	if err != nil {
		t.Fatalf("ReadModelInfo() error = %v", err)
	}
	tok, err := NewFromModelInfo(info)
	if err != nil {
		t.Fatalf("NewFromModelInfo() error = %v", err)
	}

	got := tok.Tokenize("Hello BitNet")
	want := []int32{1, 15043, 18531, 6779}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d (%v)", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token[%d] = %d, want %d", i, got[i], want[i])
		}
	}

	promptBytes, err := os.ReadFile(filepath.Join(root, "prompt.txt"))
	if err != nil {
		t.Fatalf("ReadFile(prompt.txt) error = %v", err)
	}
	expectedPromptTokensBytes, err := os.ReadFile(filepath.Join(root, "expected.prompt_tokens.json"))
	if err != nil {
		t.Fatalf("ReadFile(expected.prompt_tokens.json) error = %v; run scripts/run_ref.sh", err)
	}
	var wantPrompt []int32
	if err := json.Unmarshal(expectedPromptTokensBytes, &wantPrompt); err != nil {
		t.Fatalf("Unmarshal(expected.prompt_tokens.json) error = %v", err)
	}
	gotPrompt := tok.Tokenize(strings.TrimRight(string(promptBytes), "\r\n"))
	if len(gotPrompt) != len(wantPrompt) {
		t.Fatalf("prompt token len = %d, want %d (got=%v want=%v)", len(gotPrompt), len(wantPrompt), gotPrompt, wantPrompt)
	}
	for i := range wantPrompt {
		if gotPrompt[i] != wantPrompt[i] {
			t.Fatalf("prompt token[%d] = %d, want %d", i, gotPrompt[i], wantPrompt[i])
		}
	}
}

func TestTokenizerBPEMerges(t *testing.T) {
	info := gguf.ModelInfo{
		KeyValues: map[string]any{
			"tokenizer.ggml.model":            "gpt2",
			"tokenizer.ggml.tokens":           []string{"<unk>", "Ġ", "h", "e", "l", "o", "Ġh", "Ġhe", "Ġhel", "Ġhell", "Ġhello"},
			"tokenizer.ggml.merges":           []string{"Ġ h", "Ġh e", "Ġhe l", "Ġhel l", "Ġhell o"},
			"tokenizer.ggml.bos_token_id":     uint32(1),
			"tokenizer.ggml.unknown_token_id": uint32(0),
			"tokenizer.ggml.add_bos_token":    false,
		},
	}
	tok, err := NewFromModelInfo(info)
	if err != nil {
		t.Fatalf("NewFromModelInfo() error = %v", err)
	}
	got := tok.Tokenize(" hello")
	if len(got) != 1 || got[0] != 10 {
		t.Fatalf("got %v, want [10]", got)
	}
}

func TestTokenizerBPELlama3NumberChunking(t *testing.T) {
	info := gguf.ModelInfo{
		KeyValues: map[string]any{
			"tokenizer.ggml.model":            "gpt2",
			"tokenizer.ggml.pre":              "llama3",
			"tokenizer.ggml.tokens":           []string{"<unk>", "123", "4"},
			"tokenizer.ggml.merges":           []string{},
			"tokenizer.ggml.bos_token_id":     uint32(1),
			"tokenizer.ggml.unknown_token_id": uint32(0),
			"tokenizer.ggml.add_bos_token":    false,
		},
	}
	tok, err := NewFromModelInfo(info)
	if err != nil {
		t.Fatalf("NewFromModelInfo() error = %v", err)
	}
	got := tok.Tokenize("1234")
	if len(got) < 2 {
		t.Fatalf("got %v, want at least 2 tokens", got)
	}
	if got[len(got)-2] != 1 || got[len(got)-1] != 2 {
		t.Fatalf("got %v, want suffix [1 2]", got)
	}
}

func TestTokenizerGPT2FixturePrompt(t *testing.T) {
	assertFixturePromptTokens(
		t,
		"ggml-vocab-gpt-2.gguf",
		"gpt2.prompt.txt",
		"expected.gpt2_prompt_tokens.json",
		"run scripts/run_ref_tokenizer_variants.sh",
	)
}

func TestTokenizerFalconFixturePrompt(t *testing.T) {
	assertFixturePromptTokens(
		t,
		"ggml-vocab-falcon.gguf",
		"falcon.prompt.txt",
		"expected.falcon_prompt_tokens.json",
		"run scripts/run_ref_tokenizer_variants.sh",
	)
}

func TestTokenizerQwen2FixturePrompt(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping qwen2 fixture in short mode")
	}
	assertFixturePromptTokens(
		t,
		"ggml-vocab-qwen2.gguf",
		"qwen2.prompt.txt",
		"expected.qwen2_prompt_tokens.json",
		"run scripts/run_ref_tokenizer_variants.sh",
	)
}

func TestTokenizerYarnFixturePrompt(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping yarn fixture in short mode")
	}
	assertFixturePromptTokensFromModelFixture(
		t,
		"model_fixture_yarn.txt",
		"yarn.prompt.txt",
		"expected.yarn.prompt_tokens.json",
		"run scripts/run_ref_tokenizer_variants.sh",
	)
}

func TestTokenizerI2SFixturePrompt(t *testing.T) {
	assertFixturePromptTokensFromModelFixtureIfPresent(
		t,
		"model_fixture_i2s.txt",
		"prompt.txt",
		"expected.i2s.prompt_tokens.json",
		"run scripts/run_ref_i2s.sh",
	)
}

func TestTokenizerI2S2BFixturePrompt(t *testing.T) {
	assertFixturePromptTokensFromModelFixtureIfPresent(
		t,
		"model_fixture_i2s_2b.txt",
		"prompt.txt",
		"expected.i2s_2b.prompt_tokens.json",
		"run scripts/run_ref_i2s_2b.sh",
	)
}

func assertFixturePromptTokens(t *testing.T, modelFile, promptFile, expectedFile, hint string) {
	t.Helper()
	root := filepath.Join("..", "..", "testdata")
	info, err := gguf.ReadModelInfo(filepath.Join(root, modelFile))
	if err != nil {
		t.Fatalf("ReadModelInfo(%s) error = %v; %s", modelFile, err, hint)
	}
	tok, err := NewFromModelInfo(info)
	if err != nil {
		t.Fatalf("NewFromModelInfo(%s) error = %v", modelFile, err)
	}

	promptBytes, err := os.ReadFile(filepath.Join(root, promptFile))
	if err != nil {
		t.Fatalf("ReadFile(%s) error = %v; %s", promptFile, err, hint)
	}
	expectedBytes, err := os.ReadFile(filepath.Join(root, expectedFile))
	if err != nil {
		t.Fatalf("ReadFile(%s) error = %v; %s", expectedFile, err, hint)
	}

	var want []int32
	if err := json.Unmarshal(expectedBytes, &want); err != nil {
		t.Fatalf("Unmarshal(%s) error = %v", expectedFile, err)
	}
	got := tok.Tokenize(strings.TrimRight(string(promptBytes), "\r\n"))
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d (got=%v want=%v)", len(got), len(want), got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func assertFixturePromptTokensFromModelFixture(t *testing.T, modelFixtureFile, promptFile, expectedFile, hint string) {
	t.Helper()
	root := filepath.Join("..", "..", "testdata")
	modelFixture, err := os.ReadFile(filepath.Join(root, modelFixtureFile))
	if err != nil {
		t.Fatalf("ReadFile(%s) error = %v; %s", modelFixtureFile, err, hint)
	}
	modelFile := strings.TrimSpace(string(modelFixture))
	if modelFile == "" {
		t.Fatalf("%s is empty; %s", modelFixtureFile, hint)
	}
	assertFixturePromptTokens(t, modelFile, promptFile, expectedFile, hint)
}

func assertFixturePromptTokensFromModelFixtureIfPresent(t *testing.T, modelFixtureFile, promptFile, expectedFile, hint string) {
	t.Helper()
	root := filepath.Join("..", "..", "testdata")
	modelFixture, err := os.ReadFile(filepath.Join(root, modelFixtureFile))
	if err != nil {
		t.Fatalf("ReadFile(%s) error = %v; %s", modelFixtureFile, err, hint)
	}
	modelFile := strings.TrimSpace(string(modelFixture))
	if modelFile == "" {
		t.Fatalf("%s is empty; %s", modelFixtureFile, hint)
	}
	modelPath := filepath.Join(root, modelFile)
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("skipping fixture prompt parity; model missing: %s", modelPath)
	}
	assertFixturePromptTokens(t, modelFile, promptFile, expectedFile, hint)
}
