"""Defaults. Stage 5 turns this into ~/.livermore/config.yaml + CLI overrides.

Values mirror config_rag.yaml as of v0.1.0 so the refactor does not change behaviour.
"""
import os
from dataclasses import dataclass, field

# Repo root when installed with `pip install -e .`; data paths default to it until stage 5.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _root(*p):
    return os.path.join(PROJECT_ROOT, *p)


@dataclass
class Settings:
    notes_dir: str = field(default_factory=lambda: _root("data"))
    index_dir: str = field(default_factory=lambda: _root("kb_data"))
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 3

    backend: str = "mlx"
    device: str = "auto"
    hf_model: str = "meta-llama/Llama-3.2-1B-Instruct"
    mlx_model: str = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    scratch_ckpt: str = field(default_factory=lambda: _root("transformer", "checkpoints", "sft_best.pt"))
    scratch_tokenizer: str = field(
        default_factory=lambda: _root("transformer", "tokenizer", "corpus", "processed", "tokenizer.json"))

    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    timeout_s: float = 120.0  # generation wall-clock limit; exceeded -> generation_timeout span

    host: str = "127.0.0.1"
    port: int = 8000

    def backend_kwargs(self, name: str | None = None) -> dict:
        name = name or self.backend
        if name == "torch":
            return {"model": self.hf_model, "device": self.device}
        if name == "mlx":
            return {"model": self.mlx_model}
        if name == "scratch":
            return {"ckpt": self.scratch_ckpt, "tokenizer": self.scratch_tokenizer, "device": self.device}
        raise ValueError(f"unknown backend {name!r}")
