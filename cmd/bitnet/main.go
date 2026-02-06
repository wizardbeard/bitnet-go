package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"

	"bitnet-go/pkg/bitnet"
)

func main() {
	var (
		modelPath = flag.String("model", "", "Path to model file (GGUF for now)")
		prompt    = flag.String("prompt", "", "Prompt text")
		seed      = flag.Int64("seed", 1, "Deterministic seed")
		maxTokens = flag.Int("max-tokens", 32, "Maximum tokens to generate")
	)
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "missing required --model")
		flag.Usage()
		os.Exit(2)
	}

	session, err := bitnet.LoadModel(context.Background(), *modelPath)
	if err != nil {
		log.Fatalf("load model: %v", err)
	}

	result, err := session.Generate(context.Background(), bitnet.GenerateRequest{
		Prompt:    *prompt,
		Seed:      *seed,
		MaxTokens: *maxTokens,
	})
	if err != nil {
		log.Fatalf("generate: %v", err)
	}

	info := session.ModelInfo()
	fmt.Printf(
		"model=%s arch=%s gguf_version=%d tensors=%d kv=%d ctx=%d vocab=%d tokens=%d output=%q\n",
		info.Path,
		info.Architecture,
		info.GGUFVersion,
		info.Tensors,
		info.KVCount,
		info.ContextLength,
		info.VocabSize,
		len(result.TokenIDs),
		result.Text,
	)
}
