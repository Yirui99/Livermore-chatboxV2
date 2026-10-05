"""Side-by-side of the headline numbers in two run_all.sh output dirs.
    python benchmarks/compare.py baseline after_stage2
"""
import os
import re
import sys

PATTERNS = [
    ("rag_latency", r"Embed:\s+mean=([\d.]+) ms\s+P50=([\d.]+)", ["embed mean ms", "embed P50 ms"]),
    ("rag_latency", r"Retrieval: mean=([\d.]+) ms\s+P50=([\d.]+) ms\s+P95=([\d.]+)", ["retrieval mean ms", "retrieval P50 ms", "retrieval P95 ms"]),
    ("rag_latency", r"Search:\s+mean=([\d.]+)", ["faiss search mean ms"]),
    ("rag_latency", r"Index:\s+(\d+) docs, dim=(\d+)", ["index docs", "index dim"]),
    ("mlx_4bit", r"CPU fp32 \(old\)\s+\|[^|]+\|[^|]+\|\s+([\d.]+)", ["mlx-bench CPU fp32 t/s"]),
    ("mlx_4bit", r"MPS fp16 \(prev\)\s+\|[^|]+\|[^|]+\|\s+([\d.]+)", ["mlx-bench MPS fp16 t/s"]),
    ("mlx_4bit", r"MLX 4-bit \(new\)\s+\|[^|]+\|[^|]+\|\s+([\d.]+)", ["MLX 4-bit t/s"]),
    ("mps_median", r"Median: ([\d.]+) t/s.*\n(?:.*\n)*?.*Median: ([\d.]+) t/s", ["CPU fp32 median t/s", "MPS fp16 median t/s"]),
    ("rag_generation", r"Old CPU Time: [\d.]+s \| Speed: ([\d.]+).*\nNew MPS Time: [\d.]+s \| Speed: ([\d.]+)", ["rag-bench CPU t/s", "rag-bench MPS t/s"]),
    ("ops_correctness", r"(✅ PASS)", None),
    ("matmul", r"(❌)", None),
    ("strategy", r"Strategy Return: ([-\d.]+)%", ["AAPL strategy %"]),
]


def extract(d):
    out = {}
    for f, pat, names in PATTERNS:
        p = os.path.join("benchmarks", d, f + ".txt")
        txt = open(p).read() if os.path.exists(p) else ""
        if names is None:
            out[f"{f}: count {pat[1:-1]}"] = str(len(re.findall(pat, txt)))
            continue
        m = re.search(pat, txt)
        for i, n in enumerate(names):
            out[n] = m.group(i + 1) if m else "n/a"
    return out


a, b = sys.argv[1], sys.argv[2]
A, B = extract(a), extract(b)
print(f"{'metric':<34}{a:>14}{b:>14}{'Δ%':>9}")
for k in A:
    try:
        d = f"{(float(B[k]) - float(A[k])) / float(A[k]) * 100:+.1f}"
    except (ValueError, ZeroDivisionError):
        d = ""
    print(f"{k:<34}{A[k]:>14}{B[k]:>14}{d:>9}")
