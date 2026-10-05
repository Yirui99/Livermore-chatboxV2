"""mlx-lm 4-bit backend.

All MLX work runs on one dedicated thread: MLX streams are per-thread, and both
Streamlit and the HTTP server call us from arbitrary worker threads.
"""
from __future__ import annotations

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

_DONE = object()


class MLXBackend:
    name = "mlx"

    def __init__(self, model: str = "mlx-community/Llama-3.2-1B-Instruct-4bit", **_):
        self.model = model
        self.device = "metal"
        self.dtype = "4bit"
        self.last_usage: dict = {}
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
        self._pool.submit(self._load).result()

    def _load(self):
        import mlx.core as mx
        from mlx_lm import load
        self.lm, self.tokenizer = load(self.model)
        self.device = str(mx.default_device())

    def apply_chat_template(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(self, prompt: str, max_tokens: int, temperature: float = 0.7, top_p: float = 0.9,
                 **_) -> Iterator[str]:
        q: queue.Queue = queue.Queue()
        stop = threading.Event()

        def run():
            from mlx_lm import stream_generate
            from mlx_lm.sample_utils import make_sampler
            try:
                sampler = make_sampler(temp=temperature or 0.0, top_p=top_p if temperature else 0.0)
                last = None
                for r in stream_generate(self.lm, self.tokenizer, prompt, max_tokens=max_tokens, sampler=sampler):
                    last = r
                    q.put(r.text)
                    if stop.is_set():
                        break
                if last is not None:
                    self.last_usage = {"prompt_tokens": last.prompt_tokens,
                                       "completion_tokens": last.generation_tokens}
                q.put(_DONE)
            except BaseException as e:
                q.put(e)

        self._pool.submit(run)
        try:
            while True:
                item = q.get()
                if item is _DONE:
                    return
                if isinstance(item, BaseException):
                    raise item
                if item:
                    yield item
        finally:
            stop.set()
