package bitnet

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

func TestParitySmoke(t *testing.T) {
	modelPath := filepath.Join("..", "..", "testdata", "minimal.gguf")
	session, err := LoadModel(context.Background(), modelPath)
	if err != nil {
		t.Fatalf("LoadModel() error = %v", err)
	}

	result, err := session.Generate(context.Background(), GenerateRequest{
		Prompt:    "hello",
		Seed:      1,
		MaxTokens: 4,
	})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}

	if len(result.TokenIDs) != 4 {
		t.Fatalf("len(TokenIDs) = %d, want 4", len(result.TokenIDs))
	}
}

func TestParityVectorsPresent(t *testing.T) {
	root := filepath.Join("..", "..", "testdata")

	tokenBytes, err := os.ReadFile(filepath.Join(root, "expected.tokens.json"))
	if err != nil {
		t.Fatalf("read expected.tokens.json: %v", err)
	}

	var tokens []int32
	if err := json.Unmarshal(tokenBytes, &tokens); err != nil {
		t.Fatalf("decode expected.tokens.json: %v", err)
	}
	if len(tokens) == 0 {
		t.Fatalf("expected.tokens.json is empty; run scripts/run_ref.sh to freeze vectors")
	}

	topkBytes, err := os.ReadFile(filepath.Join(root, "expected.topk_logits.json"))
	if err != nil {
		t.Fatalf("read expected.topk_logits.json: %v", err)
	}
	var topk []topKStep
	if err := json.Unmarshal(topkBytes, &topk); err != nil {
		t.Fatalf("decode expected.topk_logits.json: %v", err)
	}
	if len(topk) != len(tokens) {
		t.Fatalf("topk length mismatch: got=%d want=%d", len(topk), len(tokens))
	}
	for i, step := range topk {
		if step.Step != i {
			t.Fatalf("topk step mismatch at index %d: got=%d want=%d", i, step.Step, i)
		}
		if len(step.Entries) == 0 {
			t.Fatalf("topk entries empty at step %d", i)
		}
		if step.Entries[0].TokenID != tokens[i] {
			t.Fatalf("topk best token mismatch at step %d: got=%d want=%d", i, step.Entries[0].TokenID, tokens[i])
		}
		prev := float32(math.MaxFloat32)
		for j, entry := range step.Entries {
			if !isFinite(float64(entry.Logit)) {
				t.Fatalf("non-finite logit at step=%d rank=%d", i, j)
			}
			if j > 0 && entry.Logit > prev {
				t.Fatalf("topk logits not sorted desc at step=%d rank=%d", i, j)
			}
			prev = entry.Logit
		}
	}

	timingBytes, err := os.ReadFile(filepath.Join(root, "expected.timings.json"))
	if err != nil {
		t.Fatalf("read expected.timings.json: %v", err)
	}
	var timings []timingStep
	if err := json.Unmarshal(timingBytes, &timings); err != nil {
		t.Fatalf("decode expected.timings.json: %v", err)
	}
	if len(timings) != len(tokens) {
		t.Fatalf("timings length mismatch: got=%d want=%d", len(timings), len(tokens))
	}
	for i, step := range timings {
		if step.Step != i {
			t.Fatalf("timing step mismatch at index %d: got=%d want=%d", i, step.Step, i)
		}
		if step.Ms < 0 || !isFinite(step.Ms) {
			t.Fatalf("invalid timing at step=%d: %f", i, step.Ms)
		}
	}
}

func TestParityAgainstFrozenVectors(t *testing.T) {
	if os.Getenv("BITNET_ENFORCE_PARITY") != "1" {
		t.Skip("set BITNET_ENFORCE_PARITY=1 to enforce strict parity while runtime is under active porting")
	}

	root := filepath.Join("..", "..", "testdata")

	tokenBytes, err := os.ReadFile(filepath.Join(root, "expected.tokens.json"))
	if err != nil {
		t.Fatalf("read expected.tokens.json: %v", err)
	}
	var want []int32
	if err := json.Unmarshal(tokenBytes, &want); err != nil {
		t.Fatalf("decode expected.tokens.json: %v", err)
	}

	promptBytes, err := os.ReadFile(filepath.Join(root, "prompt.txt"))
	if err != nil {
		t.Fatalf("read prompt.txt: %v", err)
	}
	promptBytes = bytesTrimSpace(promptBytes)

	modelFixture, err := os.ReadFile(filepath.Join(root, "model_fixture.txt"))
	if err != nil {
		t.Fatalf("read model_fixture.txt: %v", err)
	}
	modelPath := filepath.Join(root, string(bytesTrimSpace(modelFixture)))

	session, err := LoadModel(context.Background(), modelPath)
	if err != nil {
		t.Fatalf("LoadModel() error = %v", err)
	}

	got, err := session.Generate(context.Background(), GenerateRequest{
		Prompt:    string(promptBytes),
		Seed:      1,
		MaxTokens: len(want),
	})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if len(got.TokenIDs) != len(want) {
		t.Fatalf("token length mismatch: got=%d want=%d", len(got.TokenIDs), len(want))
	}
	for i := range want {
		if got.TokenIDs[i] != want[i] {
			t.Fatalf("token mismatch at step %d: got=%d want=%d", i, got.TokenIDs[i], want[i])
		}
	}

	topkBytes, err := os.ReadFile(filepath.Join(root, "expected.topk_logits.json"))
	if err != nil {
		t.Fatalf("read expected.topk_logits.json: %v", err)
	}
	var wantTopK []topKStep
	if err := json.Unmarshal(topkBytes, &wantTopK); err != nil {
		t.Fatalf("decode expected.topk_logits.json: %v", err)
	}
	if len(got.TopK) != len(wantTopK) {
		t.Fatalf("topk step mismatch: got=%d want=%d", len(got.TopK), len(wantTopK))
	}
	atol := envFloat32("BITNET_PARITY_LOGIT_ATOL", 1e-3)
	rtol := envFloat32("BITNET_PARITY_LOGIT_RTOL", 1e-3)
	strictK := envInt("BITNET_PARITY_TOPK_STRICT", 1)
	for i := range wantTopK {
		if got.TopK[i].Step != wantTopK[i].Step {
			t.Fatalf("topk step id mismatch at index %d: got=%d want=%d", i, got.TopK[i].Step, wantTopK[i].Step)
		}
		if len(got.TopK[i].Entries) != len(wantTopK[i].Entries) {
			t.Fatalf("topk entry count mismatch at step %d: got=%d want=%d", i, len(got.TopK[i].Entries), len(wantTopK[i].Entries))
		}
		if strictK < 1 {
			strictK = 1
		}
		if strictK > len(wantTopK[i].Entries) {
			strictK = len(wantTopK[i].Entries)
		}
		for j := 0; j < strictK; j++ {
			g := got.TopK[i].Entries[j]
			w := wantTopK[i].Entries[j]
			if g.TokenID != w.TokenID {
				t.Fatalf("topk token mismatch step=%d rank=%d: got=%d want=%d", i, j, g.TokenID, w.TokenID)
			}
			if !closeLogit(g.Logit, w.Logit, atol, rtol) {
				t.Fatalf("topk logit mismatch step=%d rank=%d: got=%f want=%f atol=%f rtol=%f", i, j, g.Logit, w.Logit, atol, rtol)
			}
		}
	}
}

func TestParityAgainstYarnVectors(t *testing.T) {
	if os.Getenv("BITNET_ENFORCE_YARN") != "1" {
		t.Skip("set BITNET_ENFORCE_YARN=1 to enforce YaRN parity vectors")
	}

	root := filepath.Join("..", "..", "testdata")

	tokenBytes, err := os.ReadFile(filepath.Join(root, "expected.yarn.tokens.json"))
	if err != nil {
		t.Fatalf("read expected.yarn.tokens.json: %v", err)
	}
	var want []int32
	if err := json.Unmarshal(tokenBytes, &want); err != nil {
		t.Fatalf("decode expected.yarn.tokens.json: %v", err)
	}
	if len(want) == 0 {
		t.Fatalf("expected.yarn.tokens.json is empty; run scripts/run_ref_yarn.sh to freeze vectors")
	}

	promptBytes, err := os.ReadFile(filepath.Join(root, "yarn.prompt.txt"))
	if err != nil {
		t.Fatalf("read yarn.prompt.txt: %v", err)
	}
	promptBytes = bytesTrimSpace(promptBytes)

	modelFixture, err := os.ReadFile(filepath.Join(root, "model_fixture_yarn.txt"))
	if err != nil {
		t.Fatalf("read model_fixture_yarn.txt: %v", err)
	}
	modelPath := filepath.Join(root, string(bytesTrimSpace(modelFixture)))

	session, err := LoadModel(context.Background(), modelPath)
	if err != nil {
		t.Fatalf("LoadModel() error = %v", err)
	}

	got, err := session.Generate(context.Background(), GenerateRequest{
		Prompt:    string(promptBytes),
		Seed:      1,
		MaxTokens: len(want),
	})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if len(got.TokenIDs) != len(want) {
		t.Fatalf("token length mismatch: got=%d want=%d", len(got.TokenIDs), len(want))
	}
	for i := range want {
		if got.TokenIDs[i] != want[i] {
			t.Fatalf("token mismatch at step %d: got=%d want=%d", i, got.TokenIDs[i], want[i])
		}
	}

	topkBytes, err := os.ReadFile(filepath.Join(root, "expected.yarn.topk_logits.json"))
	if err != nil {
		t.Fatalf("read expected.yarn.topk_logits.json: %v", err)
	}
	var wantTopK []topKStep
	if err := json.Unmarshal(topkBytes, &wantTopK); err != nil {
		t.Fatalf("decode expected.yarn.topk_logits.json: %v", err)
	}
	if len(got.TopK) != len(wantTopK) {
		t.Fatalf("topk step mismatch: got=%d want=%d", len(got.TopK), len(wantTopK))
	}
	atol := envFloat32("BITNET_PARITY_LOGIT_ATOL", 1e-3)
	rtol := envFloat32("BITNET_PARITY_LOGIT_RTOL", 3e-2)
	strictK := envInt("BITNET_PARITY_TOPK_STRICT", 1)
	for i := range wantTopK {
		if got.TopK[i].Step != wantTopK[i].Step {
			t.Fatalf("topk step id mismatch at index %d: got=%d want=%d", i, got.TopK[i].Step, wantTopK[i].Step)
		}
		if len(got.TopK[i].Entries) != len(wantTopK[i].Entries) {
			t.Fatalf("topk entry count mismatch at step %d: got=%d want=%d", i, len(got.TopK[i].Entries), len(wantTopK[i].Entries))
		}
		if strictK < 1 {
			strictK = 1
		}
		if strictK > len(wantTopK[i].Entries) {
			strictK = len(wantTopK[i].Entries)
		}
		for j := 0; j < strictK; j++ {
			g := got.TopK[i].Entries[j]
			w := wantTopK[i].Entries[j]
			if g.TokenID != w.TokenID {
				t.Fatalf("topk token mismatch step=%d rank=%d: got=%d want=%d", i, j, g.TokenID, w.TokenID)
			}
			if !closeLogit(g.Logit, w.Logit, atol, rtol) {
				t.Fatalf("topk logit mismatch step=%d rank=%d: got=%f want=%f atol=%f rtol=%f", i, j, g.Logit, w.Logit, atol, rtol)
			}
		}
	}
}

func TestParityAgainstI2SVectors(t *testing.T) {
	if os.Getenv("BITNET_ENFORCE_I2S") != "1" {
		t.Skip("set BITNET_ENFORCE_I2S=1 to enforce i2_s parity vectors")
	}

	root := filepath.Join("..", "..", "testdata")

	tokenBytes, err := os.ReadFile(filepath.Join(root, "expected.i2s.tokens.json"))
	if err != nil {
		t.Fatalf("read expected.i2s.tokens.json: %v; run scripts/run_ref_i2s.sh", err)
	}
	var want []int32
	if err := json.Unmarshal(tokenBytes, &want); err != nil {
		t.Fatalf("decode expected.i2s.tokens.json: %v", err)
	}
	if len(want) == 0 {
		t.Fatalf("expected.i2s.tokens.json is empty; run scripts/run_ref_i2s.sh to freeze vectors")
	}

	promptBytes, err := os.ReadFile(filepath.Join(root, "prompt.txt"))
	if err != nil {
		t.Fatalf("read prompt.txt: %v", err)
	}
	promptBytes = bytesTrimSpace(promptBytes)

	modelFixture, err := os.ReadFile(filepath.Join(root, "model_fixture_i2s.txt"))
	if err != nil {
		t.Fatalf("read model_fixture_i2s.txt: %v", err)
	}
	modelPath := filepath.Join(root, string(bytesTrimSpace(modelFixture)))

	session, err := LoadModel(context.Background(), modelPath)
	if err != nil {
		t.Fatalf("LoadModel() error = %v", err)
	}

	got, err := session.Generate(context.Background(), GenerateRequest{
		Prompt:    string(promptBytes),
		Seed:      1,
		MaxTokens: len(want),
	})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if len(got.TokenIDs) != len(want) {
		t.Fatalf("token length mismatch: got=%d want=%d", len(got.TokenIDs), len(want))
	}
	for i := range want {
		if got.TokenIDs[i] != want[i] {
			t.Fatalf("token mismatch at step %d: got=%d want=%d", i, got.TokenIDs[i], want[i])
		}
	}

	topkBytes, err := os.ReadFile(filepath.Join(root, "expected.i2s.topk_logits.json"))
	if err != nil {
		t.Fatalf("read expected.i2s.topk_logits.json: %v", err)
	}
	var wantTopK []topKStep
	if err := json.Unmarshal(topkBytes, &wantTopK); err != nil {
		t.Fatalf("decode expected.i2s.topk_logits.json: %v", err)
	}
	if len(got.TopK) != len(wantTopK) {
		t.Fatalf("topk step mismatch: got=%d want=%d", len(got.TopK), len(wantTopK))
	}
	atol := envFloat32("BITNET_PARITY_LOGIT_ATOL", 6e-2)
	rtol := envFloat32("BITNET_PARITY_LOGIT_RTOL", 6e-2)
	strictK := envInt("BITNET_PARITY_TOPK_STRICT", 1)
	for i := range wantTopK {
		if got.TopK[i].Step != wantTopK[i].Step {
			t.Fatalf("topk step id mismatch at index %d: got=%d want=%d", i, got.TopK[i].Step, wantTopK[i].Step)
		}
		if len(got.TopK[i].Entries) != len(wantTopK[i].Entries) {
			t.Fatalf("topk entry count mismatch at step %d: got=%d want=%d", i, len(got.TopK[i].Entries), len(wantTopK[i].Entries))
		}
		if strictK < 1 {
			strictK = 1
		}
		if strictK > len(wantTopK[i].Entries) {
			strictK = len(wantTopK[i].Entries)
		}
		for j := 0; j < strictK; j++ {
			g := got.TopK[i].Entries[j]
			w := wantTopK[i].Entries[j]
			if g.TokenID != w.TokenID {
				t.Fatalf("topk token mismatch step=%d rank=%d: got=%d want=%d", i, j, g.TokenID, w.TokenID)
			}
			if !closeLogit(g.Logit, w.Logit, atol, rtol) {
				t.Fatalf("topk logit mismatch step=%d rank=%d: got=%f want=%f atol=%f rtol=%f", i, j, g.Logit, w.Logit, atol, rtol)
			}
		}
	}
}

func TestParityAgainstI2S2BVectors(t *testing.T) {
	if os.Getenv("BITNET_ENFORCE_I2S_2B") != "1" {
		t.Skip("set BITNET_ENFORCE_I2S_2B=1 to enforce i2_s 2B parity vectors")
	}

	root := filepath.Join("..", "..", "testdata")

	tokenBytes, err := os.ReadFile(filepath.Join(root, "expected.i2s_2b.tokens.json"))
	if err != nil {
		t.Fatalf("read expected.i2s_2b.tokens.json: %v; run scripts/run_ref_i2s_2b.sh", err)
	}
	var want []int32
	if err := json.Unmarshal(tokenBytes, &want); err != nil {
		t.Fatalf("decode expected.i2s_2b.tokens.json: %v", err)
	}
	if len(want) == 0 {
		t.Fatalf("expected.i2s_2b.tokens.json is empty; run scripts/run_ref_i2s_2b.sh to freeze vectors")
	}

	promptBytes, err := os.ReadFile(filepath.Join(root, "prompt.txt"))
	if err != nil {
		t.Fatalf("read prompt.txt: %v", err)
	}
	promptBytes = bytesTrimSpace(promptBytes)

	modelFixture, err := os.ReadFile(filepath.Join(root, "model_fixture_i2s_2b.txt"))
	if err != nil {
		t.Fatalf("read model_fixture_i2s_2b.txt: %v", err)
	}
	modelPath := filepath.Join(root, string(bytesTrimSpace(modelFixture)))

	session, err := LoadModel(context.Background(), modelPath)
	if err != nil {
		t.Fatalf("LoadModel() error = %v", err)
	}

	got, err := session.Generate(context.Background(), GenerateRequest{
		Prompt:    string(promptBytes),
		Seed:      1,
		MaxTokens: len(want),
	})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if len(got.TokenIDs) != len(want) {
		t.Fatalf("token length mismatch: got=%d want=%d", len(got.TokenIDs), len(want))
	}
	for i := range want {
		if got.TokenIDs[i] != want[i] {
			t.Fatalf("token mismatch at step %d: got=%d want=%d", i, got.TokenIDs[i], want[i])
		}
	}

	topkBytes, err := os.ReadFile(filepath.Join(root, "expected.i2s_2b.topk_logits.json"))
	if err != nil {
		t.Fatalf("read expected.i2s_2b.topk_logits.json: %v", err)
	}
	var wantTopK []topKStep
	if err := json.Unmarshal(topkBytes, &wantTopK); err != nil {
		t.Fatalf("decode expected.i2s_2b.topk_logits.json: %v", err)
	}
	if len(got.TopK) != len(wantTopK) {
		t.Fatalf("topk step mismatch: got=%d want=%d", len(got.TopK), len(wantTopK))
	}
	atol := envFloat32("BITNET_PARITY_LOGIT_ATOL", 6e-2)
	rtol := envFloat32("BITNET_PARITY_LOGIT_RTOL", 6e-2)
	strictK := envInt("BITNET_PARITY_TOPK_STRICT", 1)
	for i := range wantTopK {
		if got.TopK[i].Step != wantTopK[i].Step {
			t.Fatalf("topk step id mismatch at index %d: got=%d want=%d", i, got.TopK[i].Step, wantTopK[i].Step)
		}
		if len(got.TopK[i].Entries) != len(wantTopK[i].Entries) {
			t.Fatalf("topk entry count mismatch at step %d: got=%d want=%d", i, len(got.TopK[i].Entries), len(wantTopK[i].Entries))
		}
		if strictK < 1 {
			strictK = 1
		}
		if strictK > len(wantTopK[i].Entries) {
			strictK = len(wantTopK[i].Entries)
		}
		for j := 0; j < strictK; j++ {
			g := got.TopK[i].Entries[j]
			w := wantTopK[i].Entries[j]
			if g.TokenID != w.TokenID {
				t.Fatalf("topk token mismatch step=%d rank=%d: got=%d want=%d", i, j, g.TokenID, w.TokenID)
			}
			if !closeLogit(g.Logit, w.Logit, atol, rtol) {
				t.Fatalf("topk logit mismatch step=%d rank=%d: got=%f want=%f atol=%f rtol=%f", i, j, g.Logit, w.Logit, atol, rtol)
			}
		}
	}
}

func TestParityAgainstI2S2BSmoke(t *testing.T) {
	if os.Getenv("BITNET_ENFORCE_I2S_2B_SMOKE") != "1" {
		t.Skip("set BITNET_ENFORCE_I2S_2B_SMOKE=1 to run i2_s 2B smoke parity")
	}

	root := filepath.Join("..", "..", "testdata")

	tokenBytes, err := os.ReadFile(filepath.Join(root, "expected.i2s_2b.tokens.json"))
	if err != nil {
		t.Fatalf("read expected.i2s_2b.tokens.json: %v; run scripts/run_ref_i2s_2b.sh", err)
	}
	var want []int32
	if err := json.Unmarshal(tokenBytes, &want); err != nil {
		t.Fatalf("decode expected.i2s_2b.tokens.json: %v", err)
	}
	if len(want) == 0 {
		t.Fatalf("expected.i2s_2b.tokens.json is empty; run scripts/run_ref_i2s_2b.sh to freeze vectors")
	}

	promptBytes, err := os.ReadFile(filepath.Join(root, "prompt.txt"))
	if err != nil {
		t.Fatalf("read prompt.txt: %v", err)
	}
	promptBytes = bytesTrimSpace(promptBytes)

	modelFixture, err := os.ReadFile(filepath.Join(root, "model_fixture_i2s_2b.txt"))
	if err != nil {
		t.Fatalf("read model_fixture_i2s_2b.txt: %v", err)
	}
	modelPath := filepath.Join(root, string(bytesTrimSpace(modelFixture)))

	session, err := LoadModel(context.Background(), modelPath)
	if err != nil {
		t.Fatalf("LoadModel() error = %v", err)
	}

	got, err := session.Generate(context.Background(), GenerateRequest{
		Prompt:    string(promptBytes),
		Seed:      1,
		MaxTokens: 1,
	})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if len(got.TokenIDs) != 1 {
		t.Fatalf("token length mismatch: got=%d want=1", len(got.TokenIDs))
	}
	if got.TokenIDs[0] != want[0] {
		t.Fatalf("token mismatch at step 0: got=%d want=%d", got.TokenIDs[0], want[0])
	}
}

func bytesTrimSpace(b []byte) []byte {
	start := 0
	for start < len(b) && (b[start] == ' ' || b[start] == '\t' || b[start] == '\n' || b[start] == '\r') {
		start++
	}
	end := len(b)
	for end > start && (b[end-1] == ' ' || b[end-1] == '\t' || b[end-1] == '\n' || b[end-1] == '\r') {
		end--
	}
	return b[start:end]
}

type topKEntry struct {
	TokenID int32   `json:"token_id"`
	Logit   float32 `json:"logit"`
}

type topKStep struct {
	Step    int         `json:"step"`
	Entries []topKEntry `json:"entries"`
}

type timingStep struct {
	Step int     `json:"step"`
	Ms   float64 `json:"ms"`
}

func isFinite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}

func closeLogit(got, want, atol, rtol float32) bool {
	diff := float32(math.Abs(float64(got - want)))
	if diff <= atol {
		return true
	}
	scale := float32(math.Abs(float64(want)))
	return diff <= atol+rtol*scale
}

func envFloat32(key string, fallback float32) float32 {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	f, err := strconv.ParseFloat(v, 32)
	if err != nil {
		return fallback
	}
	return float32(f)
}

func envInt(key string, fallback int) int {
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
