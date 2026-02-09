package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"bitnet-go/pkg/bitnet"
)

func main() {
	var (
		modelPath = flag.String("model", "", "Path to model file (GGUF for now)")
		prompt    = flag.String("prompt", "", "Prompt text")
		systemMsg = flag.String("system", "", "System message (Llama chat template)")
		userMsg   = flag.String("user", "", "User message (Llama chat template)")
		assistant = flag.String("assistant", "", "Prior assistant message (Llama chat template)")
		chatFile  = flag.String("chat-history", "", "Path to chat history file (role:content per line)")
		useChat   = flag.Bool("chat-template", false, "Use Llama chat template for system/user/assistant")
		seed      = flag.Int64("seed", 1, "Deterministic seed")
		maxTokens = flag.Int("max-tokens", 32, "Maximum tokens to generate")
		temp      = flag.Float64("temp", 0, "Sampling temperature (0 = greedy)")
		topP      = flag.Float64("top-p", 1, "Top-p nucleus sampling")
		topK      = flag.Int("top-k", 0, "Top-k sampling (0 = disabled)")
	)
	var history chatHistory
	flag.Var(&history, "chat", "Chat history item (role:content). Repeatable. Roles: system,user,assistant")
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

	finalPrompt := *prompt
	combinedHistory := history.items
	if *chatFile != "" {
		fileHistory, err := loadChatHistoryFile(*chatFile)
		if err != nil {
			log.Fatalf("load chat history: %v", err)
		}
		combinedHistory = append(fileHistory, combinedHistory...)
	}
	if *useChat || *systemMsg != "" || *userMsg != "" || *assistant != "" || len(combinedHistory) > 0 {
		finalPrompt = formatLlamaChatWithHistory(*systemMsg, *userMsg, *assistant, combinedHistory)
	}

	result, err := session.Generate(context.Background(), bitnet.GenerateRequest{
		Prompt:    finalPrompt,
		Seed:      *seed,
		MaxTokens: *maxTokens,
		Temp:      float32(*temp),
		TopP:      float32(*topP),
		TopK:      *topK,
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

type chatEntry struct {
	role    string
	content string
}

type chatHistory struct {
	items []chatEntry
}

func (h *chatHistory) String() string {
	return fmt.Sprintf("%d items", len(h.items))
}

func (h *chatHistory) Set(value string) error {
	parts := strings.SplitN(value, ":", 2)
	if len(parts) != 2 {
		return fmt.Errorf("chat item must be role:content, got %q", value)
	}
	role := strings.ToLower(strings.TrimSpace(parts[0]))
	content := strings.TrimSpace(parts[1])
	if content == "" {
		return fmt.Errorf("chat item content is empty")
	}
	switch role {
	case "system", "user", "assistant":
	default:
		return fmt.Errorf("invalid chat role %q", role)
	}
	h.items = append(h.items, chatEntry{role: role, content: content})
	return nil
}

func formatLlamaChat(systemMsg, userMsg, assistantMsg string) string {
	var b strings.Builder
	b.WriteString("<|begin_of_text|>")
	if systemMsg != "" {
		b.WriteString("<|system|>\n")
		b.WriteString(systemMsg)
		b.WriteString("\n<|end_of_text|>")
	}
	if userMsg != "" {
		b.WriteString("<|user|>\n")
		b.WriteString(userMsg)
		b.WriteString("\n<|end_of_text|>")
	}
	if assistantMsg != "" {
		b.WriteString("<|assistant|>\n")
		b.WriteString(assistantMsg)
		b.WriteString("\n<|end_of_text|>")
	}
	b.WriteString("<|assistant|>\n")
	return b.String()
}

func formatLlamaChatWithHistory(systemMsg, userMsg, assistantMsg string, history []chatEntry) string {
	var b strings.Builder
	b.WriteString("<|begin_of_text|>")
	for _, entry := range history {
		b.WriteString("<|")
		b.WriteString(entry.role)
		b.WriteString("|>\n")
		b.WriteString(entry.content)
		b.WriteString("\n<|end_of_text|>")
	}
	if systemMsg != "" {
		b.WriteString("<|system|>\n")
		b.WriteString(systemMsg)
		b.WriteString("\n<|end_of_text|>")
	}
	if userMsg != "" {
		b.WriteString("<|user|>\n")
		b.WriteString(userMsg)
		b.WriteString("\n<|end_of_text|>")
	}
	if assistantMsg != "" {
		b.WriteString("<|assistant|>\n")
		b.WriteString(assistantMsg)
		b.WriteString("\n<|end_of_text|>")
	}
	b.WriteString("<|assistant|>\n")
	return b.String()
}

func loadChatHistoryFile(path string) ([]chatEntry, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var history chatHistory
	scanner := bufio.NewScanner(file)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if err := history.Set(line); err != nil {
			return nil, fmt.Errorf("line %d: %w", lineNo, err)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return history.items, nil
}
