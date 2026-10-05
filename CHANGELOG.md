# Changelog

All notable changes to this project. Versions follow [SemVer](https://semver.org/).
Every number below comes from a script in `benchmarks/`; raw outputs are committed next to it.

## [0.1.0] — unreleased

Course project restructured into an installable library + apps. RAG and inference
logic were moved, not rewritten.

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
- MLX is now a selectable generation backend in the app and server. Before this it only
  existed inside `benchmark_mlx_4bit.py`.
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
