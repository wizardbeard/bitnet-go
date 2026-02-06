package gguf

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

type iqHashSpec struct {
	Tensor string `json:"tensor"`
	Count  int    `json:"count"`
	SHA256 string `json:"sha256"`
}

func TestIQFixtureHash(t *testing.T) {
	if os.Getenv("BITNET_ENFORCE_IQ") != "1" {
		t.Skip("set BITNET_ENFORCE_IQ=1 to enforce IQ fixture hash")
	}

	root := filepath.Join("..", "..", "testdata")
	modelFixture, err := os.ReadFile(filepath.Join(root, "model_fixture_iq.txt"))
	if err != nil {
		t.Skip("missing model_fixture_iq.txt; run scripts/fetch_testdata_gguf.sh with BITNET_FETCH_IQ=1")
	}
	modelPath := filepath.Join(root, strings.TrimSpace(string(modelFixture)))
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("IQ fixture not present: %s", modelPath)
	}

	hashBytes, err := os.ReadFile(filepath.Join(root, "expected.iq_hash.json"))
	if err != nil {
		t.Skip("missing expected.iq_hash.json; run scripts/gen_iq_fixture_hash.sh")
	}
	var spec iqHashSpec
	if err := json.Unmarshal(hashBytes, &spec); err != nil {
		t.Fatalf("decode expected.iq_hash.json: %v", err)
	}
	if spec.Tensor == "" || spec.Count <= 0 || spec.SHA256 == "" {
		t.Fatalf("invalid expected.iq_hash.json contents")
	}

	if v := os.Getenv("BITNET_IQ_COUNT"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			spec.Count = n
		}
	}

	info, err := ReadModelInfo(modelPath)
	if err != nil {
		t.Fatalf("ReadModelInfo() error = %v", err)
	}
	data, err := ReadTensorAsF32(modelPath, info, spec.Tensor)
	if err != nil {
		t.Fatalf("ReadTensorAsF32(%q) error = %v", spec.Tensor, err)
	}
	if spec.Count > len(data) {
		spec.Count = len(data)
	}

	h := sha256.New()
	var buf [4]byte
	for i := 0; i < spec.Count; i++ {
		binary.LittleEndian.PutUint32(buf[:], math.Float32bits(data[i]))
		_, _ = h.Write(buf[:])
	}
	got := fmt.Sprintf("%x", h.Sum(nil))
	if got != spec.SHA256 {
		t.Fatalf("IQ hash mismatch: got=%s want=%s (tensor=%s count=%d)", got, spec.SHA256, spec.Tensor, spec.Count)
	}
}
