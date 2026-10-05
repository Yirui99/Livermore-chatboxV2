"""Generation backends.

A backend turns a chat (list of {"role", "content"}) into a prompt string and
streams text for it. Three implementations, one file each under backends/:

  torch    HF transformers (Llama-3.2-1B-Instruct, fp32 on CPU / fp16 on CUDA|MPS)
  mlx      mlx-lm, 4-bit (mlx-community/Llama-3.2-1B-Instruct-4bit)
  scratch  the from-scratch 13.26M seq2seq Transformer (livermore.scratch)
"""
from __future__ import annotations

import importlib
from typing import Iterator, Protocol, runtime_checkable


@runtime_checkable
class Backend(Protocol):
    name: str
    model: str
    device: str
    # Token counts of the most recent generate() call: {"prompt_tokens", "completion_tokens"}.
    last_usage: dict

    def apply_chat_template(self, messages: list[dict]) -> str: ...

    def generate(self, prompt: str, max_tokens: int, **sampling) -> Iterator[str]: ...


_REGISTRY = {
    "torch": "livermore.backends.torch_hf:TorchBackend",
    "mlx": "livermore.backends.mlx:MLXBackend",
    "scratch": "livermore.backends.scratch:ScratchBackend",
}

BACKENDS = tuple(_REGISTRY)


def get_backend(name: str, **kwargs) -> Backend:
    if name not in _REGISTRY:
        raise ValueError(f"unknown backend {name!r}; choose from {', '.join(_REGISTRY)}")
    mod, cls = _REGISTRY[name].split(":")
    return getattr(importlib.import_module(mod), cls)(**kwargs)
