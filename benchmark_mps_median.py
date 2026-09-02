"""
MPS fp16 Speed — 3-run median benchmark

Resolves the inconsistency between 38.14 t/s (benchmark_rag.py)
and 60.80 t/s (benchmark_mlx_4bit.py) by running 3 identical trials
and reporting min/median/max.
"""
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
PROMPT = "You are a trading coach in the style of Jesse Livermore. Explain the importance of cutting losses quickly and letting profits run."
MAX_NEW_TOKENS = 100
NUM_RUNS = 5


def run_single_trial(device_name, dtype, trial_num):
    """Single trial: load model, warmup, generate, measure."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype).to(device_name)

    inputs = tokenizer(PROMPT, return_tensors="pt").to(device_name)

    # Warmup (2 rounds to ensure GPU kernels are compiled)
    with torch.no_grad():
        for _ in range(3):
            _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    if device_name == "mps":
        torch.mps.synchronize()

    # Timed generation
    if device_name == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    if device_name == "mps":
        torch.mps.synchronize()
    gen_time = time.perf_counter() - t0

    gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    n_tokens = len(gen_tokens)
    tps = n_tokens / gen_time if gen_time > 0 else 0

    # Cleanup
    del model, tokenizer
    gc.collect()
    if device_name == "mps":
        torch.mps.empty_cache()

    return tps, gen_time, n_tokens


def main():
    print("=" * 70)
    print("MPS FP16 SPEED — 3-RUN MEDIAN BENCHMARK")
    print(f"Model: {MODEL_NAME}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"Runs per config: {NUM_RUNS}")
    print("=" * 70)

    configs = [
        ("CPU fp32", "cpu", torch.float32),
        ("MPS fp16", "mps", torch.float16),
    ]

    for label, device_name, dtype in configs:
        print(f"\n--- {label} ({NUM_RUNS} trials) ---")

        results = []
        for i in range(NUM_RUNS):
            tps, gen_time, n_tokens = run_single_trial(device_name, dtype, i + 1)
            print(f"  Trial {i+1}: {tps:.2f} t/s ({n_tokens} tokens in {gen_time:.2f}s)")
            results.append(tps)

        results.sort()
        median_tps = results[NUM_RUNS // 2]
        print(f"  ---")
        print(f"  Min:    {results[0]:.2f} t/s")
        print(f"  Median: {median_tps:.2f} t/s  ← USE THIS")
        print(f"  Max:    {results[-1]:.2f} t/s")

    print("\n" + "=" * 70)
    print("DONE — Use the MEDIAN values for reporting.")
    print("=" * 70)


if __name__ == "__main__":
    main()
