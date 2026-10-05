"""The from-scratch seq2seq Transformer (livermore.scratch) as a backend.

The model is a 13.26M-parameter encoder-decoder with a 512-token context, trained
on Livermore Q->A pairs. As in the original Streamlit page, it receives only the
user's question: retrieved notes are not fed to it.
"""
from __future__ import annotations

import os
from typing import Iterator

from .._device import resolve_torch_device


class ScratchBackend:
    name = "scratch"

    def __init__(self, ckpt: str | None = None, tokenizer: str | None = None, device: str = "auto"):
        import torch
        from ..config import Settings
        from ..scratch.generate import build_modules

        ckpt = ckpt or Settings().scratch_ckpt
        tokenizer = tokenizer or Settings().scratch_tokenizer

        for what, p in (("checkpoint", ckpt), ("tokenizer", tokenizer)):
            if not os.path.exists(p):
                raise FileNotFoundError(f"scratch {what} not found: {p}")
        self.model = os.path.basename(ckpt)
        self.ckpt = ckpt
        self.device = resolve_torch_device(device)
        self.dtype = "float32"
        self.last_usage: dict = {}
        self._dev = torch.device(self.device)
        (self.tok, self.cfg, self.tok_emb, self.pos_emb,
         self.lm, kind, self.pad_id) = build_modules(ckpt, tokenizer, self._dev)
        if kind != "seq2seq":
            raise ValueError(f"expected a seq2seq checkpoint, got {kind}: {ckpt}")

    def apply_chat_template(self, messages: list[dict]) -> str:
        return next(m["content"] for m in reversed(messages) if m["role"] == "user")

    def generate(self, prompt: str, max_tokens: int, temperature: float = 0.8, top_k: int = 0,
                 min_new_tokens: int = 12, no_repeat_ngram_size: int = 3, repetition_penalty: float = 1.1,
                 **_) -> Iterator[str]:
        from ..scratch.generate import generate_seq2seq
        text = generate_seq2seq(prompt, self.tok, self.cfg, self.tok_emb, self.pos_emb, self.lm, self.pad_id,
                                max_new_tokens=max_tokens, temperature=temperature, top_k=top_k,
                                min_new_tokens=min_new_tokens, no_repeat_ngram_size=no_repeat_ngram_size,
                                repetition_penalty=repetition_penalty, device=self._dev)
        self.last_usage = {"prompt_tokens": len(self.tok.encode(prompt).ids),
                           "completion_tokens": len(self.tok.encode(text, add_special_tokens=False).ids)}
        yield text
