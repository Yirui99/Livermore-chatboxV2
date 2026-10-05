"""ask(q, index, backend) -> Answer: retrieve, build the prompt, generate."""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Callable

from .retrieve import Hit, search_embedding
from .trace import Trace

# Verbatim from the pre-refactor rag_qa.RAGQA.build_prompt (trading-notes corpus default).
SYSTEM_TEMPLATE = (
    "You are a trading psychology and risk management coach in the style of Jesse Livermore.\n"
    "You will answer the user’s question using the retrieved Q&A notes below as your primary reference.\n"
    "Use the notes to guide and support your answer.\n"
    "You may summarize, rephrase, or generalize the ideas from the notes when answering.\n"
    "Do NOT hallucinate information that does not appear in the notes.\n"
    "Only say 'I'm unsure' if the notes contain no concepts that are relevant to the question.\n\n"
    "================= PLAYBOOK NOTES ================\n"
    "{context}\n"
    "=================================================\n")


class GenerationTimeout(TimeoutError):
    def __init__(self, timeout_s: float, partial: str):
        super().__init__(f"generation exceeded {timeout_s:g}s")
        self.timeout_s = timeout_s
        self.partial = partial


@dataclass
class Answer:
    text: str
    hits: list[Hit]
    trace_id: str
    usage: dict = field(default_factory=dict)
    timings_ms: dict = field(default_factory=dict)


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[Note {i+1}]\n{h.text}" for i, h in enumerate(hits))


def build_messages(query: str, hits: list[Hit], system_template: str = SYSTEM_TEMPLATE) -> list[dict]:
    return [
        {"role": "system", "content": system_template.replace("{context}", format_context(hits))},
        {"role": "user", "content": query},
    ]


def ask(query: str, index, backend, k: int = 3, max_tokens: int = 512, temperature: float = 0.7,
        top_p: float = 0.9, on_token: Callable[[str], None] | None = None, trace_id: str | None = None,
        entry: str = "api", timeout_s: float | None = None, trace: bool = True, **sampling) -> Answer:
    """Retrieve k notes, build the prompt, generate. Every call is traced (see livermore.trace)
    unless trace=False. `entry` says where the call came from (cli / serve / streamlit / eval)."""
    from . import __version__

    tr = Trace("ask", trace_id=trace_id, attributes={
        "entry": entry, "version": __version__, "query": query[:2000],
        "backend": backend.name, "model": backend.model, "device": backend.device,
        "device_requested": getattr(backend, "device_requested", None),
        "revision": getattr(backend, "revision", None),
        "dtype": getattr(backend, "dtype", None), "embed_model": index.embed_model,
        "k": k, "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p,
    })
    tr.root["attributes"].update(tr.attributes)
    stage, error, kind = "embed_query", None, None
    hits, parts = [], []
    try:
        cold = index._embedder is None
        with tr.span("embed_query", model=index.embed_model, cold_start=cold) as s:
            q = index.embed(query)
            s["attributes"].update(dim=int(q.shape[1]), device=str(index.embedder.device))

        stage = "retrieve"
        with tr.span("retrieve", k=k, n_notes=len(index)) as s:
            hits = search_embedding(index, q, k)
            s["attributes"].update(n_hits=len(hits), hit_ids=[h.id for h in hits],
                                   scores=[round(h.score, 4) for h in hits],
                                   top_score=round(hits[0].score, 4) if hits else None)
            if not hits:
                Trace.fail(s, "retrieval returned no notes", "empty_retrieval")

        stage = "build_prompt"
        with tr.span("build_prompt", n_notes=len(hits)) as s:
            prompt = backend.apply_chat_template(build_messages(query, hits))
            s["attributes"]["prompt_chars"] = len(prompt)

        stage = "generate"
        s = tr.start("generate", backend=backend.name, model=backend.model, device=backend.device,
                     device_requested=getattr(backend, "device_requested", None),
                     dtype=getattr(backend, "dtype", None), max_tokens=max_tokens, timeout_s=timeout_s)
        t0, ttft = time.perf_counter(), None
        gen = backend.generate(prompt, max_tokens, temperature=temperature, top_p=top_p, **sampling)
        try:
            for chunk in gen:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                parts.append(chunk)
                if on_token:
                    on_token(chunk)
                if timeout_s and time.perf_counter() - t0 > timeout_s:
                    raise GenerationTimeout(timeout_s, "".join(parts))
        except GenerationTimeout as e:
            kind = "generation_timeout"
            s["attributes"]["partial_chars"] = len(e.partial)
            Trace.fail(s, e, kind)
            tr.end(s)
            raise
        except Exception as e:
            kind = "backend_error"
            Trace.fail(s, e, kind)
            tr.end(s)
            raise
        finally:
            gen.close()
        dt = time.perf_counter() - t0
        usage = dict(backend.last_usage)
        n = usage.get("completion_tokens", 0)
        s["attributes"].update(
            prompt_tokens=usage.get("prompt_tokens", 0), completion_tokens=n,
            ttft_ms=round(ttft * 1000, 1) if ttft is not None else None,
            tokens_per_s=round(n / dt, 1) if dt > 0 else None,
            decode_tokens_per_s=round((n - 1) / (dt - ttft), 1) if ttft is not None and n > 1 and dt > ttft else None,
            finish_reason="length" if n >= max_tokens else "stop")
        tr.end(s)
    except Exception as e:
        error, kind = e, kind or f"{stage}_error"
        raise
    finally:
        text = "".join(parts).strip()
        tr.root["attributes"].update(answer=text[:4000], n_hits=len(hits))
        if trace:
            try:
                tr.finish(error, kind)
            except OSError as werr:  # never let logging break answering
                print(f"livermore: could not write trace: {werr}", file=sys.stderr)

    spans = {sp["name"]: sp for sp in tr.spans}
    return Answer(
        text=text,
        hits=hits,
        trace_id=tr.trace_id,
        usage=usage,
        timings_ms={n: spans[n]["end_ms"] - spans[n]["start_ms"]
                    for n in ("embed_query", "retrieve", "build_prompt", "generate") if n in spans},
    )
