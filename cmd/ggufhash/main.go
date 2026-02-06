package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"

	"bitnet-go/internal/gguf"
)

type hashSpec struct {
	Tensor string `json:"tensor"`
	Count  int    `json:"count"`
	SHA256 string `json:"sha256"`
}

func main() {
	modelPath := flag.String("model", "", "path to GGUF model")
	tensorName := flag.String("tensor", "", "tensor name (optional)")
	count := flag.Int("count", 4096, "number of elements to hash")
	outPath := flag.String("out", "", "output JSON path (optional)")
	flag.Parse()

	if *modelPath == "" {
		log.Fatal("missing --model")
	}

	info, err := gguf.ReadModelInfo(*modelPath)
	if err != nil {
		log.Fatalf("read gguf model info: %v", err)
	}

	name := *tensorName
	if name == "" {
		for i := range info.Tensors {
			t := info.Tensors[i]
			if isIQType(t.Type) {
				name = t.Name
				break
			}
		}
		if name == "" {
			log.Fatal("no IQ tensor found; use --tensor to specify")
		}
	}

	data, err := gguf.ReadTensorAsF32(*modelPath, info, name)
	if err != nil {
		log.Fatalf("read tensor %q: %v", name, err)
	}
	if *count <= 0 || *count > len(data) {
		*count = len(data)
	}

	h := sha256.New()
	var buf [4]byte
	for i := 0; i < *count; i++ {
		binary.LittleEndian.PutUint32(buf[:], math.Float32bits(data[i]))
		_, _ = h.Write(buf[:])
	}
	sum := fmt.Sprintf("%x", h.Sum(nil))

	spec := hashSpec{
		Tensor: name,
		Count:  *count,
		SHA256: sum,
	}
	out, err := json.MarshalIndent(spec, "", "  ")
	if err != nil {
		log.Fatalf("marshal hash spec: %v", err)
	}
	out = append(out, '\n')

	if *outPath == "" {
		os.Stdout.Write(out)
		return
	}
	if err := os.WriteFile(*outPath, out, 0o644); err != nil {
		log.Fatalf("write %s: %v", *outPath, err)
	}
}

func isIQType(t uint32) bool {
	switch t {
	case gguf.GGMLTypeIQ2_XXS,
		gguf.GGMLTypeIQ2_XS,
		gguf.GGMLTypeIQ3_XXS,
		gguf.GGMLTypeIQ1_S,
		gguf.GGMLTypeIQ4_NL,
		gguf.GGMLTypeIQ3_S,
		gguf.GGMLTypeIQ2_S,
		gguf.GGMLTypeIQ4_XS,
		gguf.GGMLTypeIQ1_M:
		return true
	default:
		return false
	}
}
