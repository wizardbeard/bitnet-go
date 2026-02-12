# bitnet-go

Initial Go scaffold for porting BitNet CPU inference.
Training support (diverges from upstream) will live in this same repo so we can share types and GGUF export logic.

## Commands

- `go test ./...`
 - `go test ./... -run TestParity -count=1`
- `go test ./... -bench . -benchmem`
- `BITNET_ENFORCE_YARN=1 go test ./... -run TestParityAgainstYarnVectors -count=1`
- `./scripts/fetch_testdata_gguf.sh`
- `sh ./scripts/bench_infer.sh`
- `sh ./scripts/bench_i2s_kernels.sh`
- `sh ./scripts/bench_i2s_kernels_sweep.sh`
- `sh ./scripts/select_i2s_defaults.sh .bench/i2s-kernels-sweep-summary.tsv`
- `sh ./scripts/audit_cpu_parity.sh` (full parity matrix audit; set `BITNET_AUDIT_FETCH=1` to fetch fixtures first)
  - emits stage-by-stage PASS/FAIL and writes a markdown table to `GITHUB_STEP_SUMMARY` in CI
- Chat prompt template (Llama):
`go run ./cmd/bitnet --chat-template --system "You are helpful." --user "Hello"`
- Chat history (repeatable):
`go run ./cmd/bitnet --chat "system:You are helpful." --chat "user:Hello" --chat "assistant:Hi!" --chat "user:What is BitNet?" --chat-template`
- Chat history file:
`go run ./cmd/bitnet --chat-history testdata/chat_history.txt --chat-template`
Format: `role:content` per line. Blank lines and `#` comments are ignored.
- Auto procs (uses `NumCPU-2`, min 1):
`go run ./cmd/bitnet --model testdata/ggml-model-i2_s.gguf --prompt "Hello" --max-tokens 32`
- Sampling controls:
`go run ./cmd/bitnet --prompt "Hello" --temp 0.8 --top-p 0.9 --top-k 40`

Note: `go test ./...` can take ~3 minutes because tokenizer fixture tests are slow; plan CI timeouts accordingly.
- `go run ./cmd/bitnet --help`
- `go run ./cmd/bitnet-train --help`

## Training

Training is a new track hosted in this repo (not part of the upstream inference implementation).
See `MODEL_TRAINING.md` for decisions, model spec targets, and the export/interop plan.

## Benchmarks (Snapshot)

All results below were recorded on 2026-02-08 (i7-11800H, Linux, amd64).

Quick inference benchmark (wall time and tok/s):
`sh ./scripts/bench_infer.sh`
Repeated perf harness (runtime + e2e medians/p95):
`sh ./scripts/bench_perf_repeat.sh`
Thread-sweep wrapper for repeated perf harness:
`sh ./scripts/bench_perf_repeat_matrix.sh`
Drift report vs checked-in baseline:
`sh ./scripts/report_perf_repeat_drift.sh .bench/perf-repeat-summary.tsv testdata/perf-repeat-summary-baseline.tsv`
Overrides:
- `BITNET_BENCH_MODEL`
- `BITNET_BENCH_PROMPT` or `BITNET_BENCH_PROMPT_FILE`
- `BITNET_BENCH_SYSTEM`
- `BITNET_BENCH_TOKENS`
- `BITNET_BENCH_TEMP`
- `BITNET_BENCH_PROCS`
- `BITNET_BENCH_BATCH`
- `BITNET_BENCH_SWEEP=1` (run batch sweep 1/2/4)
- `BITNET_REPEAT_RUNS` (for `bench_perf_repeat.sh`, default `5`)
- `BITNET_REPEAT_THREADS` (for `bench_perf_repeat_matrix.sh`, default `"1 4 6 8"`)
- `BITNET_FORCE_AVX2=1` (force AVX2 i2_s i8_s matvec fast path on amd64+cgo; auto-detects when available)
- `BITNET_MATVEC_THREADS` (enable parallel i2_s i8_s matvec when AVX2 is unavailable; tune per host, `4-8` is a good starting range on 8-core CPUs)
- `BITNET_I2S_I8S_DISABLE_FAST=1` (disable AVX2 i2_s+i8_s fast paths to tune fallback dispatch behavior)
- `BITNET_I2S_I8S_PAR_ROWS_MIN` / `BITNET_I2S_I8S_PAR_COLS_MIN` (parallel fallback thresholds, defaults `512`)
- `BITNET_I2S_I8S_PAR_CHUNK_ROWS` / `BITNET_I2S_I8S_PAR_CHUNK_COLS` (parallel fallback chunk overrides; default auto)
- `BITNET_I2S_I8S_BLOCK_MIN_ROWS` (minimum rows for block-decode path in fallback kernels, default `256`)
- `BITNET_I2S_I8S_FAST_MIN_ELEMS` (minimum `rows*cols` to use AVX2 fast path when available, default `0`)
- `BITNET_I2S_I8S_FAST_PAR_COLS_MIN` (minimum output cols for transposed fast-range parallel split; default `512`)
- `BITNET_I2S_I8S_FAST_PAR_NT_COLS_MIN` (minimum input cols for non-transposed fast-range parallel split; default `0` / disabled)
  - sweep note (i7-11800H, fallback path): current defaults outperformed tested alternatives (`min_1024`, fixed chunk sizes, `block_min_rows=128`)
  - host note (i7-11800H): repeat-harness sweep favored `BITNET_MATVEC_THREADS=8` over `1/4/6` for end-to-end medians
- Arm64-specific overrides use the same suffix with `BITNET_ARM64_` prefix (example: `BITNET_ARM64_I2S_I8S_BLOCK_MIN_ROWS=256`).
- `BITNET_I2S_I8S_POOL` (set `0` to disable reusable fallback worker pool)
- `BITNET_I2S_I8S_POOL_WORKERS` (override fallback worker pool size; default `GOMAXPROCS`)
- `BITNET_TOPP_HEAP_CAP` (opt-in bounded-heap top-p sampler candidate cap; default `0` = use existing full-sort top-p path)
- `BITNET_TOPP_SORT_PREFIX` (initial prefix size for partial-selection top-p sort path; default `0` = full-sort, set `>0` to enable partial-selection)
- `BITNET_TOPP_PREFILTER_K` (opt-in top-p prefilter candidate cap before full-sort fallback; default `0` = disabled)

| Benchmark | Result | Notes |
| --- | --- | --- |
| Attention `steps=64/h=8/d=64` | row‑major `34.2us`, generic `41.3us` | row‑major faster |
| Attention `steps=128/h=8/d=64` | row‑major `66.8us`, generic `80.6us` | row‑major faster |
| Attention `steps=256/h=16/d=64` | row‑major `275.5us`, generic `326.1us` | row‑major faster |
| i2_s MatVec `r=512/c=512` | dispatch `88,960ns`, generic `660,277ns` | AVX2 path ~7.4x faster |
| i2_s MatVecT `r=512/c=512` | dispatch `149,885ns`, generic `639,439ns` | dispatch faster |
| f32 MatVec `r=1024/c=1024` | dispatch `553,173ns`, generic `2,147,608ns` | dispatch faster |
| f32 MatVecT `r=1024/c=1024` | dispatch `774,732ns`, generic `977,864ns` | dispatch faster |
| RMSNorm `n=4096` | `3379ns` | optimized dispatch |
| Softmax `steps=256` | dispatch `1506ns` (with `BITNET_FAST_EXPF=1`) | expf approximation |
| RoPE `h=8/d=64` | `~2986ns` | `math.Sincos` fast path |
| KQV accumulation `steps=256/d=64` | fast `6518ns`, fast_n `8427ns`, ggml `14075ns` | fast wins |
| Output projection (f32) | `87.9ms` | fast col‑accum path |
| Llama layer step `h=1024/ffn=4096/heads=16/steps=128` | `27.9ms` | end‑to‑end kernel mix |
| QKV matvec `r=256/c=256` | separate `119,386ns`, fused `139,802ns`, fused_col `106,445ns` | fused_col wins for small |
| Tokenize BPE (hot) | `~196ns/op`, 3 allocs | GPT2 fixture |
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
- CI runs a dedicated `cpu-parity-audit` job via `./scripts/audit_cpu_parity.sh` (with fixture fetch enabled) to continuously re-verify the full parity matrix through one command.
- The main `go` CI job now focuses on fmt/vet/unit tests and benchmark artifacts; parity-specific checks are centralized in `cpu-parity-audit` to avoid duplicated coverage.
- CI also runs non-gating benchmark jobs (`bench-smoke`, `bench-kernels`, `bench-runtime`) to track perf regressions.
- CI also runs a non-gating targeted i2_s+i8_s kernel benchmark (`bench-i2s-kernels`) and uploads the result artifact (`.bench/i2s-kernels.txt`).
- CI also runs a non-gating repeat-harness thread sweep (`bench-perf-repeat`) and uploads:
  - `.bench/perf-repeat-summary.tsv` (medians by thread)
  - `.bench/perf-repeat-best.env` (selected best thread on that runner)
  - `.bench/perf-repeat-drift.tsv` (delta vs `testdata/perf-repeat-summary-baseline.tsv`)
  - `.bench/perf-repeat-threads*.tsv` (raw per-run results)
  - summary schema is validated in CI via `scripts/validate_perf_repeat_summary.sh` before selecting best thread

Baseline refresh:
- after reviewing a new stable summary, update `testdata/perf-repeat-summary-baseline.tsv` from `.bench/perf-repeat-summary.tsv` in the same commit as the related perf tuning change.
- CI also runs a non-gating arm64 i2_s benchmark+sweep job and uploads arm64 artifacts (`bench-i2s-kernels-arm64`, `bench-i2s-sweep-arm64`).
  - arm64 job also uploads machine-readable sweep summary (`bench-i2s-sweep-summary-arm64`) and suggested env defaults (`i2s-defaults-arm64`).

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
    - quantized: `q8_0`, `q8_1`, `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q2_k`, `q3_k`, `q4_k`, `q5_k`, `q6_k`, `q8_k`, `tq1_0`, `tq2_0`, `i2_s`, and IQ variants
    - numeric: `f16`, `bf16`, `f64`, `i8`, `i16`, `i32`, `i64`
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
  - Runtime defaults (used when env overrides are not set) for base logits parity:
    - `BITNET_PARITY_LOGIT_ATOL` (default `1e-3`)
    - `BITNET_PARITY_LOGIT_RTOL` (default `1e-3`)
    - `BITNET_PARITY_TOPK_STRICT` (default `1`; compare top-1 exactly unless overridden)
  - CI-pinned parity policy (authoritative for merge gating):
    - base: `BITNET_PARITY_LOGIT_ATOL=1e-1`, `BITNET_PARITY_LOGIT_RTOL=1e-1`, `BITNET_PARITY_TOPK_STRICT=1`
      - rationale: current base fixture logits show stable token/top-1 parity with larger absolute/relative drift than `1e-3`; CI pins the observed stable threshold.
    - YaRN: `BITNET_PARITY_LOGIT_ATOL=1e-3`, `BITNET_PARITY_LOGIT_RTOL=3e-2`, `BITNET_PARITY_TOPK_STRICT=1`
    - i2_s / i2_s 2B (teacher-forced CI): `BITNET_I2S_FORCE_LOGIT_ATOL=7e-2`, `BITNET_I2S_FORCE_LOGIT_RTOL=7e-2`, `BITNET_I2S_TOPK_STRICT=3`, `BITNET_PARITY_FORCE_RELAX_TOPK=1`
  - i2_s parity runtime defaults (non teacher-forced) remain relaxed due to FFN activation amplification:
    - `BITNET_I2S_LOGIT_ATOL` (default `2e-1`)
    - `BITNET_I2S_LOGIT_RTOL` (default `1e-1`)
    - `BITNET_I2S_TOPK_STRICT` (default `3`)
    - `BITNET_I2S_RELAX_TOPK` (default `1`, compares top‑K as a set when enabled)
    - `BITNET_I2S_FORCE_LOGIT_ATOL` / `BITNET_I2S_FORCE_LOGIT_RTOL` (default `7e-2` when `BITNET_PARITY_FORCE=1`)
    - `BITNET_PARITY_FORCE_RELAX_TOPK` (default `1` when `BITNET_PARITY_FORCE=1`)
  - CI will run i2_s parity only when the referenced model fixtures exist in `testdata/`.
  - Seed determinism fixture checks can be run in stochastic sampling mode via:
    - `BITNET_SEED_DETERMINISM_TEMP` (CI pins `0.8`)
    - `BITNET_SEED_DETERMINISM_TOP_P` (CI pins `0.9`)
    - `BITNET_SEED_DETERMINISM_TOP_K` (CI pins `40`)
  - Set `BITNET_DISABLE_TOPK=1` to skip top‑K capture (perf optimization outside parity runs).
  - `BITNET_FAST_COL_MATVEC_AUTO=1` (default) enables a faster column‑accumulation matvec for large f32 projections when not in parity‑strict mode.
    - Set `BITNET_FAST_COL_MATVEC=1` to force it on; set `BITNET_FAST_COL_MATVEC_AUTO=0` to disable auto behavior.
  - `BITNET_FAST_V_DOT=1` (default) uses a cache‑friendly value accumulation loop in attention when not in parity‑strict mode.
  - `BITNET_KV_ROWMAJOR=1` (default) stores the V cache in row‑major `[head][pos][dim]` layout for faster attention accumulation.
    - Set `BITNET_KV_ROWMAJOR=0` to use the legacy `[head][dim][pos]` layout.
  - `BITNET_FAST_QKV_COL=1` enables a column‑accumulation path for fused f32 Q/K/V projection (opt‑in).
  - `BITNET_QKV_FUSED_MAX` caps fused Q/K/V projection by `rows*cols` (default `65536`); larger sizes fall back to separate matvecs.
  - `BITNET_STRICT_ATTENTION_REF=1` routes attention through the ggml-order reference accumulation (debug/analysis).
  - `BITNET_STRICT_FFN_REF=1` routes FFN through the reference activation path (debug/analysis).
  - `BITNET_I2S_REF_DOT=1` uses the map‑to‑{-1,0,1} reference dot for i2_s (ignores actSum; debug/analysis).
  - `BITNET_I2S_REF_ONCE=1` runs a one‑off i2_s ref‑dot comparison and prints max abs/rel deltas.
  - `BITNET_I2S_MAP3_TO1=1` maps i2_s q=3 to 1 before actSum (debug/analysis).
  - `BITNET_I2S_ALT_LAYOUT=1` treats packed i2_s weights as row‑major for debug/layout comparison.
  - `BITNET_I2S_SCALAR=1` forces scalar i2_s dot (no block decode) for drift analysis.
  - `BITNET_FFN_SHARE_I2S_QUANT` controls shared i2_s FFN input quantization for `ffn_gate` + `ffn_up`.
    - Default is enabled (`1` behavior). Set `BITNET_FFN_SHARE_I2S_QUANT=0` to disable for A/B checks.
  - `BITNET_FFN_SHARE_I2S_DOWN` controls shared i2_s FFN down-projection quantization scratch reuse.
    - Default is enabled (`1` behavior). Set `BITNET_FFN_SHARE_I2S_DOWN=0` to disable for A/B checks.
  - `BITNET_REF_I2S_DOT=1` (ref tracer) emits a ggml i2_s dot for `ffn_norm-0` against `BITNET_REF_I2S_DOT_TENSOR` and `BITNET_REF_I2S_DOT_ROW`.

## CPU Parity Status Matrix

Legend:
- `Yes` = covered and enforced in current CI flow.
- `Cond` = covered when fixture/model is present on host/runner.
- `No` = not currently enforced.

| Fixture family | Tokenizer prompt vectors | Token/logit parity vectors | Smoke parity | Seed determinism | GGUF type compatibility |
| --- | --- | --- | --- | --- | --- |
| Base (`model_fixture.txt`) | Yes (`expected.prompt_tokens.json`) | Yes (`BITNET_ENFORCE_PARITY=1`, `atol/rtol=1e-1`, top-1 strict in CI) | N/A | Yes | Cond |
| YaRN (`model_fixture_yarn.txt`) | Yes (`expected.yarn.prompt_tokens.json`) | Yes (`BITNET_ENFORCE_YARN=1`, strict+tolerance-pinned in CI) | N/A | Yes | Cond |
| i2_s (`model_fixture_i2s.txt`) | Cond (`expected.i2s.prompt_tokens.json`) | Yes (teacher-forced strict in CI) | Yes | Yes | Cond |
| i2_s 2B (`model_fixture_i2s_2b.txt`) | Cond (`expected.i2s_2b.prompt_tokens.json`) | Yes (teacher-forced strict in CI) | Yes | Yes | Cond |
| Tokenizer vocab-only (gpt2/falcon/qwen2) | Yes (CI enforces GPT2/Falcon/Qwen2 prompt vectors) | N/A | N/A | N/A | N/A |

Notes:
- `TestMaintainedFixtureTensorTypesSupported` enforces GGUF tensor-type decode support for all maintained fixtures that are present locally.
- CI currently fetches YaRN and i2_s 2B fixtures by default and enforces base + YaRN + i2_s + i2_s 2B parity checks.
