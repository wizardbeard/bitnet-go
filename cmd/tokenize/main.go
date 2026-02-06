package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"

	"bitnet-go/internal/gguf"
	"bitnet-go/internal/tokenizer"
)

func main() {
	var (
		modelPath  = flag.String("model", "", "Path to GGUF model")
		prompt     = flag.String("prompt", "", "Prompt text (overrides --prompt-file)")
		promptFile = flag.String("prompt-file", "", "Path to prompt file")
	)
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "missing required --model")
		flag.Usage()
		os.Exit(2)
	}

	text := *prompt
	if text == "" {
		if *promptFile == "" {
			fmt.Fprintln(os.Stderr, "missing --prompt or --prompt-file")
			flag.Usage()
			os.Exit(2)
		}
		b, err := os.ReadFile(*promptFile)
		if err != nil {
			log.Fatalf("read prompt file: %v", err)
		}
		text = string(b)
	}

	info, err := gguf.ReadModelInfo(*modelPath)
	if err != nil {
		log.Fatalf("read gguf model info: %v", err)
	}
	tok, err := tokenizer.NewFromModelInfo(info)
	if err != nil {
		log.Fatalf("init tokenizer: %v", err)
	}
	ids := tok.Tokenize(text)
	enc := json.NewEncoder(os.Stdout)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(ids); err != nil {
		log.Fatalf("encode tokens: %v", err)
	}
}
