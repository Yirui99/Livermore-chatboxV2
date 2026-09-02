import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml
import sys

def load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def benchmark_generation(device_name, dtype, model_name, prompt):
    print(f"\n--- Benchmarking on {device_name.upper()} ({dtype}) ---")
    
    # Load model and tokenizer
    start_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    ).to(device_name)
    load_time = time.time() - start_load
    print(f"Model load time: {load_time:.2f} s")

    # Prepare inputs
    inputs = tokenizer(prompt, return_tensors="pt").to(device_name)

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=2, do_sample=False)

    # Benchmark Generation
    print("Generating...")
    start_gen = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False, # deterministic for fair comparison
        )
    gen_time = time.time() - start_gen
    
    gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    num_tokens = len(gen_tokens)
    
    tokens_per_sec = num_tokens / gen_time if gen_time > 0 else 0
    print(f"Generation time: {gen_time:.2f} s")
    print(f"Tokens generated: {num_tokens}")
    print(f"Speed: {tokens_per_sec:.2f} tokens/sec")
    
    # Clear memory
    del model
    del tokenizer
    if device_name == "mps":
        torch.mps.empty_cache()
        
    return tokens_per_sec, gen_time

if __name__ == "__main__":
    config = load_yaml_config("config_rag.yaml")
    model_name = config["model_name"]
    
    prompt = "You are a trading coach. Summarize the key principles of Jesse Livermore's trading strategy in one paragraph."
    
    print(f"Model: {model_name}")
    
    # OLD behavior on Mac (CPU, float32)
    old_speed, old_time = benchmark_generation("cpu", torch.float32, model_name, prompt)
    
    # NEW behavior on Mac (MPS, float16)
    if torch.backends.mps.is_available():
        new_speed, new_time = benchmark_generation("mps", torch.float16, model_name, prompt)
        
        speedup = new_speed / old_speed if old_speed > 0 else 0
        print(f"\n====================================")
        print(f"QUANTITATIVE RESULTS:")
        print(f"Old CPU Time: {old_time:.2f}s | Speed: {old_speed:.2f} tokens/s")
        print(f"New MPS Time: {new_time:.2f}s | Speed: {new_speed:.2f} tokens/s")
        print(f"Performance Multiplier: {speedup:.2f}x faster generation")
        print(f"====================================")
    else:
        print("\nMPS is not available on this machine. Cannot benchmark MPS.")
