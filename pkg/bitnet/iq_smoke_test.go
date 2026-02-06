package bitnet

import (
	"context"
	"os"
	"path/filepath"
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

	session, err := LoadModel(context.Background(), modelPath)
	if err != nil {
		t.Fatalf("LoadModel() error = %v", err)
	}
	got, err := session.Generate(context.Background(), GenerateRequest{
		Prompt:    "Hello",
		Seed:      1,
		MaxTokens: 1,
	})
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if len(got.TokenIDs) != 1 {
		t.Fatalf("token length mismatch: got=%d want=1", len(got.TokenIDs))
	}
}
