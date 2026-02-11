package gguf

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

func TestMaintainedFixtureTensorTypesSupported(t *testing.T) {
	root := filepath.Join("..", "..", "testdata")
	models := maintainedFixtureModels(t, root)
	if len(models) == 0 {
		t.Skip("no maintained fixture models found")
	}

	for _, modelPath := range models {
		modelPath := modelPath
		t.Run(filepath.Base(modelPath), func(t *testing.T) {
			info, err := ReadModelInfo(modelPath)
			if err != nil {
				t.Skipf("skipping unreadable fixture %s: %v", modelPath, err)
			}
			unsupported := map[uint32]int{}
			for _, ti := range info.Tensors {
				if !IsTensorTypeSupportedAsF32(ti.Type) {
					unsupported[ti.Type]++
				}
			}
			if len(unsupported) == 0 {
				return
			}
			keys := make([]uint32, 0, len(unsupported))
			for typ := range unsupported {
				keys = append(keys, typ)
			}
			sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
			items := make([]string, 0, len(keys))
			for _, typ := range keys {
				items = append(items, fmt.Sprintf("%d(%s) x%d", typ, TensorTypeString(typ), unsupported[typ]))
			}
			t.Fatalf("fixture %s has unsupported tensor types for ReadTensorAsF32: %s", modelPath, strings.Join(items, ", "))
		})
	}
}

func maintainedFixtureModels(t *testing.T, root string) []string {
	t.Helper()
	candidates := []string{
		"stories15M-q8_0.gguf",
		"YarnGPT2b.f16.gguf",
		"ggml-model-i2_s.gguf",
		"bitnet_b1_58-large.IQ4_XS.gguf",
		"bitnet_b1_58-xl.IQ3_M.gguf",
		"bitnet_b1_58-xl.IQ4_XS.gguf",
		"smollm2-135m-instruct-iq4_xs-imat.gguf",
	}
	fixturePointers := []string{
		"model_fixture.txt",
		"model_fixture_yarn.txt",
		"model_fixture_i2s.txt",
		"model_fixture_i2s_2b.txt",
		"model_fixture_iq.txt",
	}

	seen := map[string]struct{}{}
	out := make([]string, 0, len(candidates)+len(fixturePointers))
	add := func(path string) {
		if path == "" {
			return
		}
		if _, ok := seen[path]; ok {
			return
		}
		if _, err := os.Stat(path); err != nil {
			return
		}
		seen[path] = struct{}{}
		out = append(out, path)
	}

	for _, name := range candidates {
		add(filepath.Join(root, name))
	}
	for _, ptr := range fixturePointers {
		data, err := os.ReadFile(filepath.Join(root, ptr))
		if err != nil {
			continue
		}
		modelFile := strings.TrimSpace(string(data))
		if modelFile == "" {
			continue
		}
		add(filepath.Join(root, modelFile))
	}
	sort.Strings(out)
	return out
}
