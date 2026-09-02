"""
RAG Latency Benchmark (v2 — fixed warmup + correct percentile reporting)
  - Embedding latency (SentenceTransformer encode)
  - FAISS search latency
  - Index statistics

Bug fixes from v1:
  - Added 10-query warmup to eliminate JIT/cold-start outliers
  - Mean and percentiles now computed from the SAME cleaned dataset
"""
import time
import pickle
import faiss
import numpy as np
import sys
import os
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))


def load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def percentile(sorted_list, pct):
    """Compute percentile from a pre-sorted list."""
    idx = int(pct / 100.0 * len(sorted_list))
    idx = min(idx, len(sorted_list) - 1)
    return sorted_list[idx]


def benchmark_latency():
    config = load_yaml_config("config_rag.yaml")

    print("=" * 70)
    print("RAG LATENCY BENCHMARK v2")
    print("=" * 70)

    # ---- 1. Index Statistics ----
    print("\n--- Index Statistics ---")
    docs_path = config['build_dataset']['docs_path']
    index_path = config['build_dataset']['index_path']

    with open(docs_path, "rb") as f:
        docs = pickle.load(f)
    index = faiss.read_index(index_path)

    n_docs = len(docs)
    n_vectors = index.ntotal
    dim = index.d
    index_size_mb = os.path.getsize(index_path) / 1024 / 1024
    docs_size_mb = os.path.getsize(docs_path) / 1024 / 1024
    avg_doc_len = sum(len(d) for d in docs) / n_docs if n_docs > 0 else 0

    print(f"  Documents:          {n_docs}")
    print(f"  Vectors in index:   {n_vectors}")
    print(f"  Embedding dim:      {dim}")
    print(f"  Index file size:    {index_size_mb:.2f} MB")
    print(f"  Docs file size:     {docs_size_mb:.2f} MB")
    print(f"  Avg doc length:     {avg_doc_len:.0f} chars")

    # ---- 2. Load embedding model ----
    from sentence_transformers import SentenceTransformer
    embed_model_name = config['build_dataset']['embed_model']
    emb_model = SentenceTransformer(embed_model_name)

    test_queries = [
        "What is the most important rule in trading?",
        "How to manage risk in stock trading?",
        "When should I cut my losses?",
        "How to identify a trend reversal?",
        "What is position sizing?",
        "How do you handle a losing streak?",
        "What makes a great trader?",
        "When to add to a winning position?",
        "How to control emotions in trading?",
        "What is the role of patience in trading?",
    ]

    # ---- WARMUP: 10 queries to eliminate JIT/cold-start ----
    print("\n--- Warmup (10 queries, discarded) ---")
    for q in test_queries[:10]:
        emb = emb_model.encode([q], convert_to_numpy=True, normalize_embeddings=False)
        faiss.normalize_L2(emb)
        _ = index.search(emb, 3)
    print("  Done.")

    # ---- 3. Embedding Latency (N=100, post-warmup) ----
    queries_100 = (test_queries * 10)[:100]
    top_k = config['rag_qa']['top_k']

    print(f"\n--- Embedding Latency (N=100 queries, post-warmup) ---")
    embed_times = []
    for q in queries_100:
        t0 = time.perf_counter()
        emb = emb_model.encode([q], convert_to_numpy=True, normalize_embeddings=False)
        faiss.normalize_L2(emb)
        t1 = time.perf_counter()
        embed_times.append((t1 - t0) * 1000)

    embed_sorted = sorted(embed_times)
    avg_embed = sum(embed_times) / len(embed_times)
    p50_embed = percentile(embed_sorted, 50)
    p95_embed = percentile(embed_sorted, 95)
    p99_embed = percentile(embed_sorted, 99)

    print(f"  Model:   {embed_model_name}")
    print(f"  Mean:    {avg_embed:.2f} ms")
    print(f"  P50:     {p50_embed:.2f} ms")
    print(f"  P95:     {p95_embed:.2f} ms")
    print(f"  P99:     {p99_embed:.2f} ms")
    print(f"  Min:     {embed_sorted[0]:.2f} ms")
    print(f"  Max:     {embed_sorted[-1]:.2f} ms")

    # Sanity check
    assert p95_embed >= avg_embed * 0.5, f"BUG: P95 ({p95_embed:.2f}) << Mean ({avg_embed:.2f}), warmup may be insufficient"

    # ---- 4. FAISS Search Latency (N=100) ----
    print(f"\n--- FAISS Search Latency (N=100 queries, top_k={top_k}) ---")
    search_times = []
    for q in queries_100:
        emb = emb_model.encode([q], convert_to_numpy=True, normalize_embeddings=False)
        faiss.normalize_L2(emb)
        t0 = time.perf_counter()
        D, I = index.search(emb, top_k)
        t1 = time.perf_counter()
        search_times.append((t1 - t0) * 1000)

    search_sorted = sorted(search_times)
    avg_search = sum(search_times) / len(search_times)
    p50_search = percentile(search_sorted, 50)
    p95_search = percentile(search_sorted, 95)
    p99_search = percentile(search_sorted, 99)

    print(f"  Mean:    {avg_search:.4f} ms")
    print(f"  P50:     {p50_search:.4f} ms")
    print(f"  P95:     {p95_search:.4f} ms")
    print(f"  P99:     {p99_search:.4f} ms")

    # ---- 5. Combined retrieval ----
    print("\n--- Combined Retrieval Latency (embed + search, N=100) ---")
    combined_times = [e + s for e, s in zip(embed_times, search_times)]
    combined_sorted = sorted(combined_times)
    avg_combined = sum(combined_times) / len(combined_times)
    p50_combined = percentile(combined_sorted, 50)
    p95_combined = percentile(combined_sorted, 95)

    print(f"  Mean:    {avg_combined:.2f} ms")
    print(f"  P50:     {p50_combined:.2f} ms")
    print(f"  P95:     {p95_combined:.2f} ms")

    # ---- 6. Retrieval Quality Sample ----
    print("\n--- Retrieval Quality Sample ---")
    sample_query = "What is the most important rule in trading?"
    emb = emb_model.encode([sample_query], convert_to_numpy=True, normalize_embeddings=False)
    faiss.normalize_L2(emb)
    D, I = index.search(emb, top_k)

    print(f"  Query: \"{sample_query}\"")
    for i, (idx, score) in enumerate(zip(I[0], D[0])):
        doc_preview = docs[idx][:150].replace('\n', ' ')
        print(f"  Top-{i+1} (score={score:.4f}): {doc_preview}...")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Index:     {n_docs} docs, dim={dim}")
    print(f"  Embed:     mean={avg_embed:.2f} ms  P50={p50_embed:.2f} ms  P95={p95_embed:.2f} ms")
    print(f"  Search:    mean={avg_search:.4f} ms  P50={p50_search:.4f} ms  P95={p95_search:.4f} ms")
    print(f"  Retrieval: mean={avg_combined:.2f} ms  P50={p50_combined:.2f} ms  P95={p95_combined:.2f} ms")
    print(f"  Bottleneck: {'Embedding' if avg_embed > avg_search else 'Search'} ({max(avg_embed, avg_search):.2f} ms)")
    print("=" * 70)


if __name__ == "__main__":
    benchmark_latency()
