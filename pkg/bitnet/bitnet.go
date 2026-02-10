package bitnet

import (
	"context"
	"fmt"

	"bitnet-go/internal/runtime"
)

type GenerateRequest struct {
	Prompt             string
	Seed               int64
	MaxTokens          int
	Temp               float32
	TopP               float32
	TopK               int
	DisableTopKCapture bool
}

type GenerateResult struct {
	TokenIDs []int32
	Text     string
	TopK     []TopKStep
}

type TopKEntry struct {
	TokenID int32
	Logit   float32
}

type TopKStep struct {
	Step    int
	Entries []TopKEntry
}

type ModelInfo struct {
	Path          string
	GGUFVersion   uint32
	Tensors       uint64
	KVCount       uint64
	Architecture  string
	ContextLength uint32
	VocabSize     uint32
}

type Session struct {
	rt *runtime.Runtime
}

func LoadModel(ctx context.Context, modelPath string) (*Session, error) {
	rt, err := runtime.New(ctx, modelPath)
	if err != nil {
		return nil, err
	}
	return &Session{rt: rt}, nil
}

func (s *Session) ModelInfo() ModelInfo {
	meta := s.rt.Metadata()
	return ModelInfo{
		Path:          meta.Path,
		GGUFVersion:   meta.Version,
		Tensors:       meta.TensorCount,
		KVCount:       meta.KVCount,
		Architecture:  meta.Architecture,
		ContextLength: meta.ContextLength,
		VocabSize:     meta.VocabSize,
	}
}

func (s *Session) Generate(ctx context.Context, req GenerateRequest) (GenerateResult, error) {
	if req.MaxTokens < 0 {
		return GenerateResult{}, fmt.Errorf("max tokens must be >= 0")
	}
	raw, err := s.rt.Generate(ctx, runtime.GenerateRequest{
		Prompt:             req.Prompt,
		Seed:               req.Seed,
		MaxTokens:          req.MaxTokens,
		Temp:               req.Temp,
		TopP:               req.TopP,
		TopK:               req.TopK,
		DisableTopKCapture: req.DisableTopKCapture,
	})
	if err != nil {
		return GenerateResult{}, err
	}
	topk := make([]TopKStep, 0, len(raw.TopK))
	for _, step := range raw.TopK {
		entries := make([]TopKEntry, 0, len(step.Entries))
		for _, entry := range step.Entries {
			entries = append(entries, TopKEntry{
				TokenID: entry.TokenID,
				Logit:   entry.Logit,
			})
		}
		topk = append(topk, TopKStep{
			Step:    step.Step,
			Entries: entries,
		})
	}
	return GenerateResult{
		TokenIDs: raw.TokenIDs,
		Text:     raw.Text,
		TopK:     topk,
	}, nil
}
