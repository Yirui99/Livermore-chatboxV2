# Livermore-chatbox (Performance & Strategy Optimization Branch)

> **Disclaimer**: Originally developed as a university course project; this repository contains my implementation and subsequent optimization work.
>
> 1. **Hardware Acceleration & Quantization**: Ported the PyTorch inference pipeline to Apple Silicon (MPS fp16) yielding a 2.7x speedup, and implemented MLX native 4-bit quantization achieving 325 tokens/sec (10.3x speedup over the original CPU fp32 baseline) with a 93% memory reduction.
> 2. **RAG Latency Profiling**: Built a comprehensive benchmarking suite for the FAISS + SentenceTransformer RAG pipeline, identifying and resolving JIT cold-start bottlenecks.
> 3. **Operator Performance Analysis**: Conducted numerical correctness and latency benchmarks comparing custom-built Transformer operators (using broadcast+sum) against native PyTorch BLAS implementations, quantifying the performance-vs-education tradeoff (native being ~80x faster).
> 4. **Quantitative Strategy Enhancement**: Augmented the baseline breakout strategy with multi-indicator confirmation (RSI, MACD, Bollinger Bands, Volume MA) and ATR-based dynamic trailing stops, reducing max drawdown by 26% on average across tech stocks.

## Usage (v0.1.0)

```bash
pip install -e ".[app]"
livermore ask "When should I cut my losses?"            # --backend torch | mlx | scratch
livermore serve --backend mlx                            # OpenAI-compatible, http://127.0.0.1:8000/v1
livermore app                                            # Streamlit UI, with 👍/👎 per answer
livermore stats --since 7d                               # usage from ~/.livermore/traces
```

The full README is coming in a later stage. See `CHANGELOG.md` for what changed.

## Benchmarks & Optimization Reports
All benchmark scripts were authored as part of this optimization effort. They live in `benchmarks/`;
`bash benchmarks/run_all.sh <outdir>` runs them all. The backtest is in `examples/backtest/`.