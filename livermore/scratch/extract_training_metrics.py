"""
Training Metrics Extraction:
  - Read checkpoint files for stored loss values
  - Report SFT dataset size, model config, parameter counts
"""
import torch
import os
import sys
import json
import inspect
import dataclasses
from torch.serialization import add_safe_globals

from . import config
from . import load_checkpoint

def allow_config_dataclasses():
    add_safe_globals([obj for _, obj in inspect.getmembers(config)
                      if inspect.isclass(obj) and dataclasses.is_dataclass(obj)])

def count_parameters(state_dict):
    """Count total parameters from a state dict."""
    total = 0
    for k, v in state_dict.items():
        total += v.numel()
    return total

def main():
    allow_config_dataclasses()

    print("=" * 70)
    print("TRAINING METRICS EXTRACTION")
    print("=" * 70)

    # ---- 1. Dataset stats ----
    print("\n--- Dataset Statistics ---")
    sft_path = "data/sft_dataset.jsonl" if os.path.exists("data/sft_dataset.jsonl") else "../data/sft_dataset.jsonl"
    if os.path.exists(sft_path):
        with open(sft_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        n_pairs = len(lines)
        # Sample a few
        samples = []
        for l in lines[:3]:
            try:
                obj = json.loads(l)
                samples.append({k: str(v)[:80] for k, v in obj.items()})
            except:
                pass
        print(f"  SFT dataset path: {sft_path}")
        print(f"  Total QA pairs:   {n_pairs}")
        if samples:
            print(f"  Sample keys:      {list(samples[0].keys())}")
            for i, s in enumerate(samples[:2]):
                print(f"  Sample {i+1}:        {s}")
    else:
        print(f"  SFT dataset not found at {sft_path}")

    # ---- 2. Checkpoint analysis ----
    ckpt_dir = "checkpoints"
    ckpt_files = [
        ("Pre-train best",  os.path.join(ckpt_dir, "seq2seq_best.pt")),
        ("Pre-train final", os.path.join(ckpt_dir, "seq2seq_final.pt")),
        ("SFT best",        os.path.join(ckpt_dir, "sft_best.pt")),
        ("SFT final",       os.path.join(ckpt_dir, "sft.pt")),
    ]

    print("\n--- Checkpoint Analysis ---")
    for label, path in ckpt_files:
        if not os.path.exists(path):
            print(f"  {label:<18}: {path} — NOT FOUND")
            continue

        ckpt = load_checkpoint(path, map_location="cpu", weights_only=False)
        file_size_mb = os.path.getsize(path) / 1024 / 1024

        # Extract config
        raw_cfg = ckpt.get("cfg", None)
        if isinstance(raw_cfg, dict):
            cfg_info = raw_cfg
        elif raw_cfg is not None:
            cfg_info = dataclasses.asdict(raw_cfg) if dataclasses.is_dataclass(raw_cfg) else str(raw_cfg)
        else:
            cfg_info = "N/A"

        # Count parameters
        model_params = count_parameters(ckpt["model"]) if "model" in ckpt else 0
        tok_emb_params = count_parameters(ckpt["tok_emb"]) if "tok_emb" in ckpt else 0
        pos_emb_params = count_parameters(ckpt["pos_emb"]) if "pos_emb" in ckpt else 0
        total_params = model_params + tok_emb_params + pos_emb_params

        # Check for loss
        stored_loss = ckpt.get("loss", ckpt.get("val_loss", ckpt.get("best_val_loss", "N/A")))
        epoch = ckpt.get("epoch", "N/A")

        print(f"\n  {label} ({path}):")
        print(f"    File size:       {file_size_mb:.1f} MB")
        print(f"    Keys:            {list(ckpt.keys())}")
        print(f"    Model params:    {model_params:,} ({model_params/1e6:.2f}M)")
        print(f"    Tok emb params:  {tok_emb_params:,}")
        print(f"    Pos emb params:  {pos_emb_params:,}")
        print(f"    Total params:    {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"    Stored loss:     {stored_loss}")
        print(f"    Epoch:           {epoch}")

    # ---- 3. Model architecture summary ----
    print("\n--- Model Architecture (from config) ---")
    try:
        path = os.path.join(ckpt_dir, "sft_best.pt")
        if not os.path.exists(path):
            path = os.path.join(ckpt_dir, "seq2seq_best.pt")
        ckpt = load_checkpoint(path, map_location="cpu", weights_only=False)
        raw_cfg = ckpt.get("cfg")
        if isinstance(raw_cfg, dict):
            # Rebuild nested dataclass configs from dicts
            build_kwargs = {}
            for k, v in raw_cfg.items():
                if k == "self_attn" and isinstance(v, dict):
                    build_kwargs[k] = config.SelfAttnConfig(**v)
                elif k == "cross_attn" and isinstance(v, dict):
                    build_kwargs[k] = config.CrossAttnConfig(**v)
                elif k == "ffn" and isinstance(v, dict):
                    build_kwargs[k] = config.FFNConfig(**v)
                elif k == "addnorm" and isinstance(v, dict):
                    build_kwargs[k] = config.AddNormConfig(**v)
                else:
                    build_kwargs[k] = v
            cfg = config.ModelConfig(**build_kwargs)
        elif raw_cfg is not None:
            cfg = raw_cfg
        else:
            cfg = config.ModelConfig()

        print(f"    vocab_size:  {cfg.vocab_size}")
        print(f"    d_model:     {cfg.d_model}")
        print(f"    n_layer:     {cfg.n_layer}")
        print(f"    n_ctx:       {cfg.n_ctx}")
        print(f"    n_head:      {cfg.self_attn.n_head}")
        print(f"    d_ff:        {cfg.ffn.d_ff}")
        print(f"    dropout:     {cfg.dropout}")
    except Exception as e:
        print(f"    Error loading config: {e}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
