package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"

	"bitnet-go/pkg/bitnet"
)

type topKEntry struct {
	TokenID int32   `json:"token_id"`
	Logit   float32 `json:"logit"`
}

type topKStep struct {
	Step    int         `json:"step"`
	Entries []topKEntry `json:"entries"`
}

func main() {
	var (
		modelPath   = flag.String("model", "", "Path to GGUF model")
		promptFile  = flag.String("prompt-file", "", "Path to prompt file")
		expectedTop = flag.String("expected-topk", "", "Path to expected top-k logits JSON")
		step        = flag.Int("step", 0, "Step index to compare")
		trimPrompt  = flag.Bool("trim", true, "Trim prompt whitespace")
		seed        = flag.Int64("seed", 1, "Seed for generation")
	)
	flag.Parse()

	if *modelPath == "" || *promptFile == "" || *expectedTop == "" {
		fmt.Fprintln(os.Stderr, "missing required --model, --prompt-file, or --expected-topk")
		flag.Usage()
		os.Exit(2)
	}

	promptBytes, err := os.ReadFile(*promptFile)
	if err != nil {
		log.Fatalf("read prompt: %v", err)
	}
	if *trimPrompt {
		promptBytes = bytesTrimSpace(promptBytes)
	}

	expBytes, err := os.ReadFile(*expectedTop)
	if err != nil {
		log.Fatalf("read expected topk: %v", err)
	}
	var expected []topKStep
	if err := json.Unmarshal(expBytes, &expected); err != nil {
		log.Fatalf("decode expected topk: %v", err)
	}
	if *step < 0 || *step >= len(expected) {
		log.Fatalf("step %d out of range (expected len=%d)", *step, len(expected))
	}

	session, err := bitnet.LoadModel(context.Background(), *modelPath)
	if err != nil {
		log.Fatalf("load model: %v", err)
	}

	got, err := session.Generate(context.Background(), bitnet.GenerateRequest{
		Prompt:    string(promptBytes),
		Seed:      *seed,
		MaxTokens: *step + 1,
	})
	if err != nil {
		log.Fatalf("generate: %v", err)
	}
	if *step >= len(got.TopK) {
		log.Fatalf("got topk steps=%d, need step=%d", len(got.TopK), *step)
	}

	fmt.Printf("step=%d\n", *step)
	fmt.Println("expected:")
	for i, e := range expected[*step].Entries {
		fmt.Printf("  %d: token=%d logit=%g\n", i, e.TokenID, e.Logit)
	}
	fmt.Println("got:")
	for i, e := range got.TopK[*step].Entries {
		fmt.Printf("  %d: token=%d logit=%g\n", i, e.TokenID, e.Logit)
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
