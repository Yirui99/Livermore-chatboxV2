# Changelog

All notable changes to this project. Versions follow [SemVer](https://semver.org/).
Every number below comes from a script in `benchmarks/`; raw outputs are committed next to it.

## [0.1.0] — unreleased

Course project restructured into an installable library + apps. RAG and inference
logic were moved, not rewritten.

### Added — operations (stage 5)
- `~/.livermore/config.yaml`: notes/index dirs, embed model, top_k, backend, device, models and revisions,
  max_tokens, temperature, top_p, timeout, host/port. Precedence: built-in default < config file < CLI flag.
  Unknown keys, wrong types and out-of-range values give one-line errors with a suggestion.
  `livermore config show | init | path`. There are no chunk parameters because there is no chunking:
  each JSONL record is one note.
- **Default temperature 0.2** (was 0.7 in the CLI/server and 0.8 in the UI). The app, server, CLI and (stage 4)
  eval all read the same config, so daily use and evaluation run with identical settings.
- Model cache `~/.livermore/models/<owner>--<name>/`, pinned to the revisions the shipped index and benchmarks
  used (`livermore.models.PINNED`), with `manifest.json` (size + sha256 for LFS files, git sha1 otherwise).
  On first use a model is copied from the local Hugging Face cache if it holds that revision (verified offline:
  blob names are hashes), otherwise downloaded from the Hub with progress and checked against the Hub's hashes.
  Loading does a size check; `livermore doctor --verify` re-hashes. `livermore models list | fetch [repo|all] [--force]`.
- Device fallback: `device: mps` (or `cuda`) when unavailable → CPU with a warning; MLX without Metal → CPU with a
  warning. Traces record `device` and `device_requested`.
- Human-readable errors (no traceback) for: missing/unreadable/inconsistent index, missing/empty/malformed notes,
  missing/truncated/corrupted model files, gated or unreachable models, missing scratch checkpoint, bad config.
- `livermore doctor [--verify]`: Python, platform, packages, MLX Metal, torch MPS, config, model files, scratch
  checkpoint, notes, index (and its embed model/dim vs config), trace dir. Exit 1 on any problem.
- `tests/test_ops.py` (config, model cache, fallback, errors).

### Fixed — stage 5
- MLX CPU fallback really runs on CPU. mlx_lm binds its generation stream to the default device at import time,
  and `import mlx_lm.generate` returns the *function* (the package shadows the module). The first version of the
  fallback only relabelled the device: it measured the same 363 tok/s as the GPU. Now it measures 5 tok/s, the same
  as a clean mlx_lm-on-CPU control with identical output. Covered by a regression test.

Verified: loading from `~/.livermore/models` is behaviour-identical (equivalence PASS; MLX greedy output identical to
loading by repo id).

### Added — tracing and usage log (stage 3)
- `livermore.trace`: every `ask()` is a span tree `ask → embed_query / retrieve / build_prompt / generate`,
  following Next AI's `trace.js`. Attributes include backend, model, device (backend and embedder), dtype,
  prompt/completion tokens, TTFT, tokens/s, hit ids and scores, the query and the answer.
  Written to `~/.livermore/traces/YYYY-MM-DD/trace_<id>.jsonl` (UTC day), plus one summary row per trace in
  `~/.livermore/traces/index.jsonl`. `LIVERMORE_HOME` moves it.
- Failures are traced too, each as a `status=error` span with an `error_kind`: `backend_error` (exception in
  generation, re-raised), `generation_timeout` (`timeout_s`, default 120 s; generation is actually stopped and the
  partial text kept), `empty_retrieval` (no notes found; the answer still proceeds). Exceptions in the other
  stages are recorded as `<stage>_error`.
- 👍/👎 per answer: buttons in Streamlit, `POST /v1/feedback` on the server, `livermore feedback <id> up|down`.
  Written as a `feedback` span into the trace and as a row in `index.jsonl`; the last click wins.
- `livermore stats --since 7d`: query count, failure rate, active days, latency p50/p95/mean (nearest-rank),
  TTFT, retrieval time, tokens/s, error kinds, feedback; grouped `--by backend day entry device`.
  `--list [--rated down]`, `--trace <id>`, `--json`. Traffic with `entry=eval|test` is excluded unless `--include-eval`.
- `livermore app` (opens the Streamlit app), `--timeout` on `ask`/`serve`, `tests/test_trace.py`.

### Changed — stage 3
- `import livermore` imports its light submodules eagerly. Importing them lazily let
  `from livermore.ask import X` rebind `livermore.ask` to the module, breaking `livermore.ask(...)`.
  `import livermore` takes 15 ms and still loads no torch/faiss/mlx.
- Streamlit keeps retrieved notes and the trace id with each message, so notes and 👍/👎 stay visible in history.

Benchmarks after stage 3 (`benchmarks/after_stage3/`): all within noise of the baseline, and equivalence still
PASS. Query embedding P50 measured 4.13 ms against 3.92 ms at baseline. The benchmark script doesn't use package
code, so to check this I ran the **unmodified pre-refactor checkout** 5 times at the same time:
P50 4.14–5.39 ms, vs. 4.08–4.22 ms for this tree (plus one 6.22 ms outlier). The difference is machine state,
not the refactor.

### Added
- `livermore` package (`pip install -e .`, `livermore --version`)
  - `livermore.index`: `build(notes_dir) -> Index`, `load(path) -> Index`
  - `livermore.retrieve`: `Index.search(q, k) -> list[Hit]`
  - `livermore.generate`: `Backend` protocol + `get_backend()`; implementations in
    `livermore/backends/`: `torch` (HF), `mlx` (mlx-lm 4-bit), `scratch` (from-scratch Transformer)
  - `livermore.ask`: `ask(q, index, backend) -> Answer{text, hits, trace_id, usage, timings_ms}`
  - `livermore.serve`: OpenAI-compatible HTTP server (`POST /v1/chat/completions` incl. `stream`,
    `GET /v1/models`, `GET /health`); stdlib only, no new dependencies
  - CLI: `livermore build | ask | serve`
- MLX is now a generation backend in the app and server, and the default (`backend: mlx`).
  Before this it only existed inside `benchmark_mlx_4bit.py`; the app used HF torch fp32 on CPU.
  `--backend torch` restores the old path.
- `benchmarks/equivalence.py`: checks the package against outputs captured from the
  pre-refactor code (retrieval ids/scores, prompt bytes, greedy generations). The existing
  benchmark scripts don't import app code, so they cannot catch a behaviour change by themselves.
- `benchmarks/run_all.sh`, `benchmarks/compare.py`, raw outputs in `benchmarks/baseline/`
  (pre-refactor, commit 693a450) and `benchmarks/after_stage2/`.

### Changed
- The from-scratch Transformer moved from `transformer/*.py` to `livermore/scratch/`, with imports
  made package-relative. Data, corpus, paper and checkpoints remain under `transformer/`.
  Teaching scripts are run as modules (see `transformer/README.md`).
- The Streamlit app moved to `apps/streamlit_app.py`. It only calls `livermore.*`; the CSS is unchanged.
- The backtest moved to `examples/backtest/` as its own Streamlit app. It is no longer part of the package.
- Benchmarks moved to `benchmarks/`. Only import paths changed (`from livermore.scratch import ops`).
- `.gitignore` fixed: the old `*.h5*.pt` pattern matched nothing.

### Removed
- `rag_qa.py`, `build_dataset.py` → `livermore ask` / `livermore build` (same index format,
  same prompt, same decoding)
- `streamlit_app.py`, `pages/` → `apps/streamlit_app.py`, `examples/backtest/app.py`

### Fixed
- Importing faiss before torch made CPU `SentenceTransformer.encode` segfault on macOS
  (duplicate libomp). The package now always loads torch first. Before the refactor this never
  showed up only because `rag_qa.py` happened to import torch first.
- Checkpoints that pickle `config.ModelConfig` (`seq2seq_best.pt`, `seq2seq_final.pt`) can still be
  loaded after the move (`livermore.scratch.load_checkpoint`).
- Temperature 0 in the UI now means greedy decoding. Before, it was passed to `do_sample=True`
  and raised an error.

### Verified (Apple M4 Max, 36 GB, torch 2.9.1, mlx-lm 0.31.3)
Equivalence vs. pre-refactor code: retrieval 15/15 queries with identical top-10 ids, max |Δscore| 0;
prompts 15/15 byte-identical; torch greedy 3/3 identical; scratch greedy 2/2 identical
(plus `seq2seq_best.pt` checked against the old code). Rebuilding the index from `data/`
reproduces `kb_data/` (same 1,367 docs, max |Δvector| 2.3e-7).

Benchmarks before → after (single runs unless noted; the scripts themselves are unchanged):

| metric | before | after |
|---|---|---|
| query embedding P50 | 3.92 ms | 4.08 ms |
| retrieval (embed + FAISS) mean / P95 | 4.20 / 5.73 ms | 4.28 / 5.86 ms |
| MLX 4-bit generation | 304.4 t/s | 322.0 t/s |
| HF MPS fp16, median of 5 | 64.45 t/s | 62.05 t/s |
| HF CPU fp32, median of 5 | 31.08 t/s | 31.52 t/s |
| custom ops vs PyTorch (`test_ops_correctness.py`) | 15/15 pass | 15/15 pass |

### Known issues (carried over, not changed here)
- `bench_matmul.py` reports ❌ for 3 of 6 sizes: `allclose(atol=1e-5)` is too strict for fp32
  accumulation at K≥256. Its "memory" column uses tracemalloc, which does not see torch
  allocations, so it always prints 0.00 MB.
- The torch backend tokenizes a chat-templated prompt that already begins with `<|begin_of_text|>`,
  so the model sees two BOS tokens. Kept as-is so answers stay identical; to be revisited with eval (stage 4).
- The scratch model produces mostly ungrammatical text (e.g. "A ruleistent ad rest…"). This is the
  model's quality, unchanged by the move.
