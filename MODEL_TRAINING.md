# Model Training Plan (1.5B)

This repo is inference-first. This document defines the separate training track required to produce 1.5B-class models that can be exported to GGUF and consumed by the Go runtime.

## Decisions (Chosen Defaults)

- Tokenizer: SentencePiece (SPM).
  - Reason: aligns with Llama-style tooling, stable SPM segmentation, and existing GGUF metadata conventions.
- Context length: 4K.
  - Reason: balances training cost with usability; extensible later via RoPE scaling.
- Precision: bfloat16 (bf16).
  - Reason: better numeric stability than fp16 and widely supported on modern GPUs.
- Training budget: start with a scaled run (100M–300M) to validate the full pipeline; scale to 1.5B once parity and export/import are proven.

## Model Spec (Target 1.5B)

These are draft targets intended to be frozen before training starts:
- Architecture: Llama-style decoder-only transformer.
- Layers: 24–28 (finalize based on hidden size).
- Hidden size: 2048–2560.
- Heads: 16–20 (head dim 128).
- KV heads: grouped (e.g., 4–8) to reduce KV cache size.
- FFN size: 4x hidden size (SwiGLU).
- RoPE: base 10000, dimension count = head dim (default), with optional scaling for 8K+ later.
- Vocab: SPM vocab size 32k.

Note: the exact layer/hidden/head counts must be finalized to hit the 1.5B parameter budget. Once chosen, they must be treated as a contract for GGUF export and inference.

## Training Stack

- Framework: PyTorch + FSDP (recommended).
  - Alternative: DeepSpeed ZeRO-3 if preferred.
- Mixed precision: bf16 with grad scaling if needed.
- Activation checkpointing: enabled by default.
- Optimizer: AdamW (beta1=0.9, beta2=0.95).
- LR schedule: cosine decay with warmup (e.g., 1–2% of total steps).
- Gradient clipping: 1.0.
- Determinism: fixed seeds and logged RNG state; deterministic dataloader order.

## Data Pipeline

- Use a streaming, sharded dataset format.
- Pre-tokenize with the frozen SPM tokenizer into binary shards.
- Record hash manifests for shard integrity and reproducibility.
- Maintain a held-out eval set for perplexity tracking.

## Export + Interop Plan

- Checkpoints saved in safetensors.
- Export step:
  - Convert to GGUF with full metadata:
    - `llama.attention.head_count`, `llama.attention.head_count_kv`
    - `llama.rope.freq_base`, `llama.rope.dimension_count`
    - tokenizer metadata keys and vocab blobs
  - Export both fp16/bf16 (for reference) and quantized (i2_s, IQ*) for inference.
- Validate load in Go runtime with fixed prompt + seed (parity harness).

## Validation Loop (Milestones)

1. Tiny model (e.g., 10–50M) trains end-to-end, exports to GGUF, loads in Go.
2. Medium model (100–300M) reaches stable loss and parity vs PyTorch for fixed prompts.
3. Full 1.5B training run once pipeline is proven and inference parity is tight.

## Immediate Next Actions

1. Freeze the model spec (layers/hidden/heads) to hit 1.5B exactly.
2. Define tokenizer training recipe and produce the SPM model + vocab.
3. Build the training repo skeleton (data loader, model, training loop, export).
4. Add a GGUF export script aligned with our Go runtime metadata expectations.

