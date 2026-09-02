"""
MLX 4-bit Quantized LLM Benchmark
Compares:
  - OLD: HuggingFace transformers on CPU (fp32)
  - MID: HuggingFace transformers on MPS (fp16)
  - NEW: MLX native 4-bit quantized on Apple Silicon

Uses mlx-community/Llama-3.2-1B-Instruct-4bit (pre-quantized).
"""
import time
import sys
import os

def benchmark_hf(device_name, dtype_name, model_name, prompt, max_new_tokens=100):
    """Benchmark HuggingFace transformers generation."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
    }
    dtype = dtype_map[dtype_name]

    print(f"\n--- HuggingFace: {device_name.upper()} + {dtype_name} ---")

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device_name)
    load_time = time.perf_counter() - t0
    print(f"  Model load: {load_time:.2f} s")

    # Memory estimate
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"  Model size in memory: {param_bytes / 1024 / 1024:.1f} MB")

    inputs = tokenizer(prompt, return_tensors="pt").to(device_name)

    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=2, do_sample=False)
    if device_name == "mps":
        torch.mps.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    if device_name == "mps":
        torch.mps.synchronize()
    gen_time = time.perf_counter() - t0

    gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    n_tokens = len(gen_tokens)
    tps = n_tokens / gen_time if gen_time > 0 else 0

    print(f"  Generated: {n_tokens} tokens in {gen_time:.2f} s")
    print(f"  Speed:     {tps:.2f} tokens/sec")

    # Cleanup
    del model, tokenizer
    import gc; gc.collect()
    if device_name == "mps":
        torch.mps.empty_cache()

    return tps, gen_time, load_time, param_bytes


def benchmark_mlx(model_name_mlx, prompt, max_new_tokens=100):
    """Benchmark MLX 4-bit quantized generation."""
    from mlx_lm import load, generate
    import mlx.core as mx

    print(f"\n--- MLX 4-bit Quantized ---")
    print(f"  Model: {model_name_mlx}")

    t0 = time.perf_counter()
    model, tokenizer = load(model_name_mlx)
    load_time = time.perf_counter() - t0
    print(f"  Model load: {load_time:.2f} s")

    # Warmup
    _ = generate(model, tokenizer, prompt=prompt, max_tokens=2, verbose=False)
    mx.eval(mx.zeros(1))  # sync

    # Benchmark
    t0 = time.perf_counter()
    response = generate(model, tokenizer, prompt=prompt, max_tokens=max_new_tokens, verbose=False)
    mx.eval(mx.zeros(1))  # sync
    gen_time = time.perf_counter() - t0

    # Count tokens in response
    n_tokens = len(tokenizer.encode(response))
    tps = n_tokens / gen_time if gen_time > 0 else 0

    print(f"  Generated: {n_tokens} tokens in {gen_time:.2f} s")
    print(f"  Speed:     {tps:.2f} tokens/sec")
    print(f"  Response:  {response[:200]}...")

    return tps, gen_time, load_time


def main():
    print("=" * 70)
    print("MLX 4-BIT QUANTIZATION BENCHMARK")
    print("=" * 70)

    hf_model = "meta-llama/Llama-3.2-1B-Instruct"
    mlx_model = "mlx-community/Llama-3.2-1B-Instruct-4bit"

    prompt = "You are a trading coach in the style of Jesse Livermore. Explain the importance of cutting losses quickly and letting profits run."
    max_tokens = 100

    # 1. CPU fp32 (old baseline)
    tps_cpu, time_cpu, load_cpu, mem_cpu = benchmark_hf("cpu", "float32", hf_model, prompt, max_tokens)

    # 2. MPS fp16 (previous optimization)
    tps_mps, time_mps, load_mps, mem_mps = benchmark_hf("mps", "float16", hf_model, prompt, max_tokens)

    # 3. MLX 4-bit (new)
    tps_mlx, time_mlx, load_mlx = benchmark_mlx(mlx_model, prompt, max_tokens)

    # Results
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"{'Method':<25} | {'Load (s)':<10} | {'Gen (s)':<10} | {'Speed (t/s)':<12} | {'Mem (MB)':<10} | {'vs CPU'}")
    print("-" * 90)
    print(f"{'CPU fp32 (old)':<25} | {load_cpu:<10.2f} | {time_cpu:<10.2f} | {tps_cpu:<12.2f} | {mem_cpu/1024/1024:<10.1f} | 1.0x")
    print(f"{'MPS fp16 (prev)':<25} | {load_mps:<10.2f} | {time_mps:<10.2f} | {tps_mps:<12.2f} | {mem_mps/1024/1024:<10.1f} | {tps_mps/tps_cpu:.2f}x")
    print(f"{'MLX 4-bit (new)':<25} | {load_mlx:<10.2f} | {time_mlx:<10.2f} | {tps_mlx:<12.2f} | {'~350':<10} | {tps_mlx/tps_cpu:.2f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
