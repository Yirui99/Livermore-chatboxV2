"""ask(q, index, backend) -> Answer: retrieve, build the prompt, generate."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

from .retrieve import Hit

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
        **sampling) -> Answer:
    trace_id = trace_id or uuid.uuid4().hex[:16]
    t0 = time.perf_counter()
    hits = index.search(query, k=k)
    t1 = time.perf_counter()
    prompt = backend.apply_chat_template(build_messages(query, hits))
    t2 = time.perf_counter()
    parts = []
    for chunk in backend.generate(prompt, max_tokens, temperature=temperature, top_p=top_p, **sampling):
        parts.append(chunk)
        if on_token:
            on_token(chunk)
    t3 = time.perf_counter()
    return Answer(
        text="".join(parts).strip(),
        hits=hits,
        trace_id=trace_id,
        usage=dict(backend.last_usage),
        timings_ms={"retrieve": (t1 - t0) * 1e3, "build_prompt": (t2 - t1) * 1e3, "generate": (t3 - t2) * 1e3},
    )
