package bitnet

import (
	"context"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestIQModelSmoke(t *testing.T) {
	if os.Getenv("BITNET_ENFORCE_IQ") != "1" {
		t.Skip("set BITNET_ENFORCE_IQ=1 to run IQ model smoke test")
	}

	root := filepath.Join("..", "..", "testdata")
	fixtureBytes, err := os.ReadFile(filepath.Join(root, "model_fixture_iq.txt"))
	if err != nil {
		t.Skip("missing model_fixture_iq.txt; run scripts/fetch_testdata_gguf.sh with BITNET_FETCH_IQ=1")
	}
	modelPath := filepath.Join(root, strings.TrimSpace(string(fixtureBytes)))
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("IQ fixture not present: %s", modelPath)
	}

	maxTokens := 1
	if v := os.Getenv("BITNET_IQ_MAX_TOKENS"); v != "" {
		if parsed, err := strconv.Atoi(v); err == nil && parsed >= 0 {
			maxTokens = parsed
		}
	}
	if maxTokens > 1 {
		maxTokens = 1
	}

	session, err := LoadModel(context.Background(), modelPath)
	if err != nil {
		t.Fatalf("LoadModel() error = %v", err)
	}
	got, err := session.Generate(context.Background(), GenerateRequest{
		Prompt:    "Hello",
		Seed:      1,
		MaxTokens: maxTokens,
	})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if len(got.TokenIDs) != maxTokens {
		t.Fatalf("token length mismatch: got=%d want=%d", len(got.TokenIDs), maxTokens)
	}
}
