"""HF transformers backend. Model loading and decoding are unchanged from rag_qa.RAGQA."""
from __future__ import annotations

import threading
from typing import Iterator

from .._device import resolve_torch_device


class TorchBackend:
    name = "torch"

    def __init__(self, model: str = "meta-llama/Llama-3.2-1B-Instruct", device: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = model
        self.device = resolve_torch_device(device)
        self.dtype = "float16" if self.device in ("cuda", "mps") else "float32"
        self.last_usage: dict = {}
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.lm = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float16 if self.device in ["cuda", "mps"] else torch.float32,
        ).to(self.device)

    def apply_chat_template(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(self, prompt: str, max_tokens: int, temperature: float = 0.7, top_p: float = 0.9,
                 **_) -> Iterator[str]:
        import torch
        from transformers import TextIteratorStreamer

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        n_prompt = inputs["input_ids"].shape[1]
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        if temperature and temperature > 0:
            sampling = dict(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            sampling = dict(do_sample=False)
        result: dict = {}

        def run():
            try:
                with torch.no_grad():
                    out = self.lm.generate(**inputs, max_new_tokens=max_tokens, streamer=streamer, **sampling)
                result["n"] = out.shape[1] - n_prompt
            except BaseException as e:  # surface in the consumer thread
                result["error"] = e
                streamer.end()

        t = threading.Thread(target=run, daemon=True)
        t.start()
        yield from (t for t in streamer if t)
        t.join()
        if "error" in result:
            raise result["error"]
        self.last_usage = {"prompt_tokens": n_prompt, "completion_tokens": result["n"]}
