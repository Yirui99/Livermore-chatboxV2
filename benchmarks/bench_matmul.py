"""
Benchmark: ops.matmul (broadcast+sum) vs torch.matmul (BLAS)
Measures: latency (ms) and peak memory for varying matrix sizes.
"""
import torch
import time
import sys
import os
from livermore.scratch import ops

def measure_matmul(fn, A, B, warmup=5, repeat=50, label=""):
    """Run fn(A, B), return avg_ms and peak_memory_bytes."""
    device = A.device

    # Warmup
    for _ in range(warmup):
        _ = fn(A, B)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    # Reset peak memory
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "mps":
        # MPS doesn't have reset_peak_memory_stats; we just note current
        pass

    # Timed runs
    times = []
    for _ in range(repeat):
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

        t0 = time.perf_counter()
        out = fn(A, B)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    avg_ms = sum(times) / len(times)
    p95_ms = sorted(times)[int(0.95 * len(times))]

    # Peak memory (only reliable on CUDA)
    peak_mem = 0
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated()

    return avg_ms, p95_ms, peak_mem


def run_benchmark():
    device = torch.device("cpu")  # Force CPU for fair comparison
    print("=" * 80)
    print("BENCHMARK: ops.matmul (broadcast+sum) vs torch.matmul (BLAS)")
    print(f"Device: {device}")
    print("=" * 80)

    # Test configurations: (B, T, K, D) — simulating attention matmul
    configs = [
        # (batch, seq_len, inner_dim) — A: (B,T,K), B: (B,K,D)
        ("Small  (1, 32, 64)",      1,  32,  64,  64),
        ("Medium (1, 64, 128)",     1,  64, 128, 128),
        ("Medium (1, 128, 256)",    1, 128, 256, 256),
        ("Large  (1, 256, 256)",    1, 256, 256, 256),
        ("Attn   (1, 4, 128, 128)", 4, 128, 128, 128),  # simulates multi-head
        ("Large  (1, 4, 256, 256)", 4, 256, 256, 256),
    ]

    print(f"\n{'Config':<30} | {'ops.matmul (ms)':<18} | {'torch.matmul (ms)':<20} | {'Speedup':<10} | {'Correct'}")
    print("-" * 110)

    for name, *dims in configs:
        if len(dims) == 4:
            B, T, K, D = dims
            A = torch.randn(B, T, K, device=device)
            B_mat = torch.randn(B, K, D, device=device)
        else:
            # Should not happen with our configs
            continue

        # ops.matmul
        avg_custom, p95_custom, _ = measure_matmul(ops.matmul, A, B_mat, warmup=3, repeat=20, label="ops")

        # torch.matmul
        avg_native, p95_native, _ = measure_matmul(torch.matmul, A, B_mat, warmup=3, repeat=20, label="torch")

        # Correctness
        out_custom = ops.matmul(A, B_mat)
        out_native = torch.matmul(A, B_mat)
        correct = torch.allclose(out_custom, out_native, atol=1e-5, rtol=1e-5)

        speedup = avg_custom / avg_native if avg_native > 0 else float('inf')
        print(f"{name:<30} | {avg_custom:>12.3f} ms    | {avg_native:>14.3f} ms    | {speedup:>7.1f}x    | {'✅' if correct else '❌'}")

    # =====================================================================
    # Memory comparison for large matrix
    # =====================================================================
    print("\n" + "=" * 80)
    print("MEMORY COMPARISON (large matrix, CPU)")
    print("=" * 80)

    import tracemalloc

    for size_label, B, T, K, D in [("256x256", 1, 256, 256, 256), ("512x512", 1, 512, 512, 512)]:
        A = torch.randn(B, T, K)
        B_mat = torch.randn(B, K, D)

        # ops.matmul memory
        tracemalloc.start()
        _ = ops.matmul(A, B_mat)
        _, peak_custom = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # torch.matmul memory
        tracemalloc.start()
        _ = torch.matmul(A, B_mat)
        _, peak_native = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        ratio = peak_custom / peak_native if peak_native > 0 else float('inf')
        print(f"  {size_label}: ops.matmul peak = {peak_custom / 1024 / 1024:.2f} MB | torch.matmul peak = {peak_native / 1024 / 1024:.2f} MB | ratio = {ratio:.1f}x")


if __name__ == "__main__":
    run_benchmark()
