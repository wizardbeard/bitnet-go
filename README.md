# bitnet-go

Initial Go scaffold for porting BitNet CPU inference.

## Commands

- `go test ./...`
 - `go test ./... -run TestParity -count=1`
- `go test ./... -bench . -benchmem`
- `BITNET_ENFORCE_YARN=1 go test ./... -run TestParityAgainstYarnVectors -count=1`
- `./scripts/fetch_testdata_gguf.sh`

Note: `go test ./...` can take ~3 minutes because tokenizer fixture tests are slow; plan CI timeouts accordingly.
- `go run ./cmd/bitnet --help`

## Benchmarks (Snapshot)

All results below were recorded on 2026-02-08 (i7-11800H, Linux, amd64).

| Benchmark | Result | Notes |
| --- | --- | --- |
| Attention `steps=64/h=8/d=64` | row‑major `41.0us`, generic `42.9us` | row‑major faster |
| Attention `steps=128/h=8/d=64` | row‑major `78.0us`, generic `86.2us` | row‑major faster |
| Attention `steps=256/h=16/d=64` | row‑major `325.1us`, generic `365.1us` | row‑major faster |
| i2_s MatVec `r=512/c=512` | dispatch `171,953ns`, generic `671,031ns` | dispatch faster |
| i2_s MatVecT `r=512/c=512` | dispatch `149,885ns`, generic `639,439ns` | dispatch faster |
| f32 MatVec `r=1024/c=1024` | dispatch `625,111ns`, generic `1,999,773ns` | dispatch faster |
| f32 MatVecT `r=1024/c=1024` | dispatch `778,114ns`, generic `974,746ns` | dispatch faster |
| RMSNorm `n=4096` | `3379ns` | optimized dispatch |
| Softmax `steps=256` | dispatch `1766ns`, generic `1784ns` | small gain |
| KQV accumulation `steps=256/d=64` | fast `6518ns`, fast_n `8427ns`, ggml `14075ns` | fast wins |
| Output projection (f32) | `87.9ms` | fast col‑accum path |
| Llama layer step `h=1024/ffn=4096/heads=16/steps=128` | `27.9ms` | end‑to‑end kernel mix |
| QKV matvec `r=1024/c=1024` | separate `1,969,612ns`, fused `1,920,684ns`, fused_col `2,387,984ns` | fused gated to separate |
| Tokenize BPE (hot) | `~198ns/op`, 3 allocs | GPT2 fixture |
| Tokenize SPM (hot) | `~117ns/op`, 3 allocs | Llama fixture |

## Reference harness (Phase 0)

## Testdata GGUFs

GGUF fixtures are no longer committed. Use `./scripts/fetch_testdata_gguf.sh` to fetch the required files
into `testdata/` locally, and CI will run the same script before tests.

Overrides:
- `BITNET_FORCE_FETCH=1` (redownload even if present)
- `BITNET_GPT2_VOCAB_URL`, `BITNET_FALCON_VOCAB_URL`, `BITNET_QWEN2_VOCAB_URL`
- `BITNET_FETCH_YARN=1`, `BITNET_YARN_MODEL_URL`, `BITNET_YARN_MODEL_FILE`
- `BITNET_FETCH_IQ=1`, `BITNET_IQ_MODEL_URL`, `BITNET_IQ_MODEL_FILE`
- `BITNET_FETCH_I2S=1`, `BITNET_I2S_MODEL_URL`, `BITNET_I2S_MODEL_FILE`, `BITNET_I2S_MODEL_SHA256`
- `BITNET_FETCH_I2S_2B=1`, `BITNET_I2S_2B_MODEL_URL`, `BITNET_I2S_2B_MODEL_FILE`

CI note:
- CI runs `./scripts/fetch_testdata_gguf.sh` with `BITNET_FETCH_I2S_2B=1` and
  `BITNET_I2S_2B_MODEL_URL` set to the BitNet 2B i2_s GGUF, so the 2B smoke parity
  test can run when enabled.
- CI also runs non-gating benchmark jobs (`bench-smoke`, `bench-kernels`, `bench-runtime`) to track perf regressions.

Optional IQ fixture hash:
- `BITNET_FETCH_IQ=1 ./scripts/fetch_testdata_gguf.sh`
- `./scripts/gen_iq_fixture_hash.sh` (writes `testdata/expected.iq_hash.json`)
- `BITNET_ENFORCE_IQ=1 go test ./internal/gguf -run TestIQFixtureHash -count=1`

`./scripts/fetch_ref_model.sh`:
- Downloads a small GGUF fixture into `testdata/` (default: `stories15M-q8_0.gguf`)
- Updates `testdata/model_fixture.txt`
- Supports overrides:
  - `BITNET_REF_MODEL_URL`
  - `BITNET_REF_MODEL_FILE`
  - `BITNET_REF_MODEL_SHA256`

`./scripts/build_ref.sh`:
- Locates upstream BitNet C++ source (`BITNET_REF_SRC` override)
- Builds with CMake under `.ref/build`
- Copies detected inference binary to `.ref/bin/ref-infer`
- If `include/bitnet-lut-kernels.h` is missing, either:
  - run upstream setup first (`python <ref-src>/setup_env.py -md <model_dir> -q i2_s`), or
  - set `BITNET_REF_AUTOGEN_KERNEL_HEADER=1` to copy a preset kernel header

`./scripts/run_ref.sh`:
- Runs the reference binary and captures `.ref/reference.trace`
- Generates:
  - `testdata/expected.tokens.json`
  - `testdata/expected.topk_logits.json`
  - `testdata/expected.timings.json`
  - `testdata/expected.prompt_tokens.json`

`./scripts/run_ref_tokenizer.sh`:
- Builds a vocab-only tokenizer tracer (`.ref/bin/ref-tokenize`)
- Generates non-SPM tokenizer reference vector:
  - `testdata/expected.gpt2_prompt_tokens.json`
- Uses fixture files:
  - `testdata/ggml-vocab-gpt-2.gguf`
  - `testdata/gpt2.prompt.txt`

`./scripts/run_ref_tokenizer_variants.sh`:
- Fetches gpt2/falcon/qwen2 vocab-only fixtures
- Generates tokenizer reference vectors:
  - `testdata/expected.gpt2_prompt_tokens.json`
  - `testdata/expected.falcon_prompt_tokens.json`
  - `testdata/expected.qwen2_prompt_tokens.json`
- Default mode auto-builds and uses a dedicated tracer binary (`.ref/bin/ref-trace`) to emit exact per-step:
  - token IDs
  - top-k logits
  - decode timings
- You can still override with `BITNET_REF_RUN_CMD` to use a custom runner.

By default, `run_ref.sh` expects trace lines in this format:
- `TOKEN step=0 id=123`
- `TOPK step=0 entries=123:5.12,77:4.98`
- `TIME step=0 ms=0.42`

If upstream CLI output differs, provide a wrapper command via `BITNET_REF_RUN_CMD` that emits these lines.

## Current status

- Phase 1 started: CLI + thin core API scaffold.
- GGUF model reader now parses:
  two header counts, scalar metadata KV values, array counts, and tensor descriptors.
- Basic tokenizer plumbing added:
  - captures `tokenizer.ggml.tokens` from GGUF
  - adds SPM merge-queue tokenizer path for llama-style vocab (fixture prompt parity check included)
  - adds GPT2/BPE path with:
    - regex pretokenization
    - byte-to-unicode mapping
    - merge-rank application from `tokenizer.ggml.merges`
    - `tokenizer.ggml.pre` dispatch (currently includes GPT2 baseline + llama3-style splitter)
  - keeps a greedy fallback path for other scaffolding
- Phase 2 stepping-stone added:
  - `internal/kernels` naive ops (`Dot`, `AddScaled`, `Argmax`) with unit tests
  - runtime now uses a deterministic minimal forward loop with procedural weights
    to exercise prompt->state->logits->token flow before true tensor kernels land
- GGUF tensor layout stepping-stone added:
  - captures model alignment (`general.alignment`, default 32)
  - computes aligned tensor data start offset
  - provides naive tensor loaders by tensor name:
    - `f32`
    - `q8_0` (decoded to `float32`)
- Runtime tensor-backed stepping-stone added:
  - if model includes `bitnet_go.state_proj` and `bitnet_go.logits_proj` (`f32`),
    runtime executes a naive single-block forward path using loaded tensor weights
  - if model includes llama-style `token_embd.weight` + `output.weight`,
    runtime executes a naive embedding/output path with real model tensors
  - if model includes llama-style block-0 tensors and `output_norm.weight`,
    runtime executes a naive transformer-stack path:
    - RMSNorm (`attn_norm`, `ffn_norm`, `output_norm`)
    - attention projections (`attn_q/k/v/output`)
    - head-aware attention partitioning using `llama.attention.head_count`
    - grouped-query style KV head mapping using `llama.attention.head_count_kv`
    - RoPE on q/k using `llama.rope.freq_base` (fallback default applied)
    - basic RoPE scaling support via `llama.rope.scaling.type` + `llama.rope.scaling.factor`
    - supports `llama.rope.dimension_count` to limit RoPE rotation dims
    - SwiGLU-style MLP (`ffn_gate`, `ffn_up`, `ffn_down`)
    - sequence-aware causal attention with an in-memory K/V cache
    - supports multiple sequential `blk.N.*` layers (starting at `blk.0`)
  - keeps the prior procedural forward fallback when those tensors are absent
- Phase 0 scripts are functional and configurable.
- Parity testing:
  - `BITNET_ENFORCE_PARITY=1` enables strict token parity against frozen vectors.
  - Optional logits parity uses:
    - `BITNET_PARITY_LOGIT_ATOL` (default `1e-3`)
    - `BITNET_PARITY_LOGIT_RTOL` (default `1e-3`)
  - i2_s parity uses relaxed tolerances due to FFN activation amplification:
    - `BITNET_I2S_LOGIT_ATOL` (default `2e-1`)
    - `BITNET_I2S_LOGIT_RTOL` (default `1e-1`)
    - `BITNET_I2S_TOPK_STRICT` (default `3`)
    - `BITNET_I2S_RELAX_TOPK` (default `1`, compares top‑K as a set when enabled)
    - `BITNET_I2S_FORCE_LOGIT_ATOL` / `BITNET_I2S_FORCE_LOGIT_RTOL` (default `7e-1`)
  - CI will run i2_s parity only when the referenced model fixtures exist in `testdata/`.
  - Set `BITNET_DISABLE_TOPK=1` to skip top‑K capture (perf optimization outside parity runs).
  - `BITNET_FAST_COL_MATVEC_AUTO=1` (default) enables a faster column‑accumulation matvec for large f32 projections when not in parity‑strict mode.
    - Set `BITNET_FAST_COL_MATVEC=1` to force it on; set `BITNET_FAST_COL_MATVEC_AUTO=0` to disable auto behavior.
  - `BITNET_FAST_V_DOT=1` (default) uses a cache‑friendly value accumulation loop in attention when not in parity‑strict mode.
  - `BITNET_KV_ROWMAJOR=1` (default) stores the V cache in row‑major `[head][pos][dim]` layout for faster attention accumulation.
    - Set `BITNET_KV_ROWMAJOR=0` to use the legacy `[head][dim][pos]` layout.
  - `BITNET_FAST_QKV_COL=1` enables a column‑accumulation path for fused f32 Q/K/V projection (opt‑in).
  - `BITNET_QKV_FUSED_MAX` caps fused Q/K/V projection by `rows*cols` (default `65536`); larger sizes fall back to separate matvecs.
