"""mlx-lm 4-bit backend.

All MLX work runs on one dedicated thread: MLX streams are per-thread, and both
Streamlit and the HTTP server call us from arbitrary worker threads.
"""
from __future__ import annotations

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

from .. import models
from .._device import warn
from ..errors import BackendUnavailable, ModelFileError

_DONE = object()


class MLXBackend:
    name = "mlx"

    def __init__(self, model: str = "mlx-community/Llama-3.2-1B-Instruct-4bit", revision: str | None = None, **_):
        try:
            import mlx.core  # noqa: F401
            import mlx_lm  # noqa: F401
        except ImportError as e:
            raise BackendUnavailable(f"the mlx backend needs Apple Silicon with mlx-lm installed ({e})",
                                     "use `--backend torch`, or set `backend: torch` in ~/.livermore/config.yaml")
        self.model = model
        self.revision = models.revision_for(model, revision)
        self.device_requested = "gpu"
        self.device = "gpu"
        self.dtype = "4bit"
        self.last_usage: dict = {}
        self._path = models.ensure(model, revision)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
        self._pool.submit(self._load).result()

    def _load(self):
        import sys

        import mlx.core as mx
        from mlx_lm import load
        # Not `import mlx_lm.generate`: mlx_lm/__init__ shadows the module with the function.
        mlx_generate = sys.modules["mlx_lm.generate"]
        if not mx.metal.is_available():
            warn("MLX: Metal GPU not available, running on CPU. 4-bit MLX on CPU is very slow "
                 "(5 tok/s measured on an M4 Max vs ~360 on its GPU); `backend: torch` is the better CPU choice")
            mx.set_default_device(mx.cpu)
            # mlx_lm fixes its generation stream to the default device *at import time*;
            # without this the "CPU" fallback would keep running on the GPU stream.
            mlx_generate.generation_stream = mx.new_thread_local_stream(mx.cpu)
        self.device = "gpu" if mx.default_device() == mx.gpu else "cpu"
        try:
            self.lm, self.tokenizer = load(self._path)
        except Exception as e:
            raise ModelFileError(f"could not load {self.model} from {self._path}: {type(e).__name__}: {str(e)[:200]}",
                                 f"run `livermore doctor --verify`, or `livermore models fetch {self.model} --force`")

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
