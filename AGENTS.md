# AGENTS.md — BitNet → Go (porting guide for coding agents)

This repository is **microsoft/BitNet** (“bitnet.cpp”), an inference framework for 1-bit / 1.58-bit LLMs, with optimized CPU and GPU kernels. :contentReference[oaicite:0]{index=0}  
Primary code areas in this repo include: `src/`, `include/`, `gpu/`, `preset_kernels/`, `utils/`, and `3rdparty/`. :contentReference[oaicite:1]{index=1}

## Mission

Port BitNet’s inference stack to **Golang** with:
- Feature parity for **CPU inference first**
- Deterministic correctness parity vs the existing C++ implementation
- Competitive performance (via Go assembly and/or cgo only where strictly necessary)
- A clean Go module layout and stable public API

## Non-goals (initially)

- Full GPU parity (treat as Phase 3+)
- Reproducing CMake build UX in Go
- Rewriting every experimental utility script on day 1 (Python in root can remain until the Go CLI fully replaces it)

## Working definition of “done”

A Go CLI can:
1. Load a supported model format used by this repo (commonly GGUF flows in practice)
2. Run token generation with the same prompt+seed producing matching logits/token IDs vs the reference binary (within defined tolerances)
3. Match throughput benchmarks on at least one x86_64 and one ARM64 host within an acceptable gap for the chosen implementation strategy

## Repo map (what to read first)

Start here:
- `README.md` (build/run expectations, supported models, overall framing) :contentReference[oaicite:2]{index=2}
- `src/` and `include/` (CPU inference core) :contentReference[oaicite:3]{index=3}

Defer until later:
- `gpu/` (GPU kernels and integration) :contentReference[oaicite:4]{index=4}
- `preset_kernels/` (pre-tuned kernels and tuning artifacts) :contentReference[oaicite:5]{index=5}
- `3rdparty/` (vendored dependencies; treat as “read-only” during early phases) :contentReference[oaicite:6]{index=6}

## Porting strategy (phased)

### Phase 0 — Ground truth harness (mandatory)
Goal: create an objective oracle for correctness.

Deliverables:
- A repeatable “reference run” script that:
  - Builds the C++ binary
  - Runs inference on a tiny model + tiny prompt
  - Emits:
    - token IDs per step
    - logits (top-K) per step
    - timings per step
- A frozen test vector set under `testdata/`:
  - prompt.txt
  - expected.tokens.json
  - expected.topk_logits.json
  - model fixture descriptor (path or download hint)

Rules:
- Always compare Go outputs to the reference harness on every change set.
- Never “fix” a mismatch by weakening tests until you have a root cause.

### Phase 1 — Go CLI + thin core API (scaffolding)
Goal: establish the shape of the Go program without porting kernels yet.

Recommended Go layout (suggested):
- `go.mod` at repo root
- `/cmd/bitnet/` — CLI entrypoint (flags, logging, IO)
- `/internal/` — implementation details (not public)
- `/pkg/bitnet/` — stable API (model load, session, generate)
- `/internal/gguf/` — model format reader (or whatever format this repo uses in practice)
- `/internal/math/` — low-bit packing, LUT ops, quant helpers
- `/internal/kernels/` — CPU kernel implementations (Phase 2)

Outputs:
- `bitnet` CLI that can parse args and load model metadata
- Basic prompt tokenization plumbing (align with reference implementation behavior)

### Phase 2 — CPU inference parity (core port)
Goal: reproduce end-to-end CPU inference.

Approach:
- Port model loading + tensor layout rules first
- Port attention/MLP blocks in a correctness-first mode
- Port 1-bit / 1.58-bit specific kernels last, with clear interfaces

Kernel guidance:
- Implement a “naive” Go kernel first (pure Go, clear correctness)
- Add optimized variants behind build tags:
  - `//go:build amd64` and `//go:build arm64`
- Keep an escape hatch for cgo for ultra-hot loops only if needed.
  - If cgo is used, isolate it in `/internal/cgo/` and wrap with stable Go interfaces.

### Phase 3 — Performance and CPU feature work
Goal: recover most of the speedups.

Tactics:
- Add architecture-specific fast paths:
  - amd64: AVX2/AVX-512 via Go assembly or cgo intrinsics
  - arm64: NEON via Go assembly
- Add kernel tiling parameters that mirror the reference system’s tunable blocks
- Add microbenchmarks (`go test -bench`) per kernel and per block

### Phase 4 — GPU parity (optional / later)
Goal: mirror `gpu/` behavior.
- Prefer a separate Go package boundary so CPU work stays clean
- Consider leaving GPU as “external backend” via FFI initially

## Correctness policy

- Default tolerance: exact match on token IDs for deterministic settings.
- Logits tolerance: define:
  - top-K set equality OR
  - max absolute delta threshold
  - max relative delta threshold
- Any tolerance change requires a written justification in the PR description.

## Coding conventions (Go)

- No “clever” generics in hot loops.
- Avoid allocations in step loops:
  - preallocate buffers
  - reuse scratch arenas
- Keep data layouts explicit:
  - document endian, packing, bitwidth, and stride assumptions
- Separate “model format” from “runtime tensor layout”.
- Keep all unsafe usage localized and commented with invariants.

## Build / run commands (Go side)

Agents should add (or update) these targets:
- `go test ./...`
- `go test ./... -run TestParity -count=1`
- `go test ./... -bench . -benchmem`
- `go run ./cmd/bitnet --help`

If a reference build is needed in CI:
- Provide a `./scripts/build_ref.sh` and `./scripts/run_ref.sh`
- Keep scripts POSIX-sh compatible.

## CI expectations

Add minimal CI steps:
- Go fmt/vet/test
- Parity test against the frozen test vector
- Benchmark smoke (non-gating) on at least one runner class

## Issue triage tags for the port

Use labels (or prefixes) consistently:
- `port/cpu-core`
- `port/model-io`
- `port/kernels-naive`
- `port/kernels-opt`
- `port/tokenizer`
- `perf/amd64`
- `perf/arm64`
- `backend/gpu`

## Agent operating rules

1. Start every task by identifying the smallest reference behavior you can lock down.
2. Prefer small PRs:
   - one subsystem per PR
3. Never refactor while porting unless it unblocks parity or performance.
4. Document all assumptions about tensor shapes, packing, and quantization.
5. Keep a living `PORTING_NOTES.md` with:
   - mismatches discovered
   - resolved root causes
   - model format details confirmed from source

## Suggested first tasks (ordered)

1. Add `testdata/` fixtures + reference runner script (Phase 0).
2. Create `cmd/bitnet` skeleton + basic logging/flags (Phase 1).
3. Implement model header/metadata reader (Phase 1).
4. Implement a minimal forward pass with stub kernels (Phase 2).
5. Replace stubs with naive correct kernels (Phase 2).
6. Add parity tests and make them green (Phase 2).
7. Add optimized kernel variants behind build tags (Phase 3).

---
Context: BitNet repository structure and scope (folders, purpose) are defined in the upstream repo metadata and README. :contentReference[oaicite:7]{index=7}
