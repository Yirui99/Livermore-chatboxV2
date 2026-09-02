# Livermore-chatbox (Performance & Strategy Optimization Branch)

> **Disclaimer**: This repository is a personal fork/branch based on a university course team project (original contributors: Yirui UI, Yuki, Mengge Luo). The original repository implemented a custom sequence-to-sequence Transformer and a base retrieval-augmented generation (RAG) system for a trading chatbot. 
> 
> **My personal contributions in this branch focus exclusively on performance optimization, quantization, and quantitative strategy enhancement:**
> 1. **Hardware Acceleration & Quantization**: Ported the PyTorch inference pipeline to Apple Silicon (MPS fp16) yielding a 2.7x speedup, and implemented MLX native 4-bit quantization achieving 325 tokens/sec (10.3x speedup over the original CPU fp32 baseline) with a 93% memory reduction.
> 2. **RAG Latency Profiling**: Built a comprehensive benchmarking suite for the FAISS + SentenceTransformer RAG pipeline, identifying and resolving JIT cold-start bottlenecks.
> 3. **Operator Performance Analysis**: Conducted numerical correctness and latency benchmarks comparing custom-built Transformer operators (using broadcast+sum) against native PyTorch BLAS implementations, quantifying the performance-vs-education tradeoff (native being ~80x faster).
> 4. **Quantitative Strategy Enhancement**: Augmented the baseline breakout strategy with multi-indicator confirmation (RSI, MACD, Bollinger Bands, Volume MA) and ATR-based dynamic trailing stops, reducing max drawdown by 26% on average across tech stocks.

## Benchmarks & Optimization Reports
All benchmark scripts and backtesting tools were authored as part of this optimization effort. Check the scripts prefixed with `bench_` or `benchmark_` for reproducible tests.