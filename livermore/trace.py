"""Span-tree tracing, ported from Next AI (server/src/trace.js).

Every ask() is one trace: a root span with children embed_query / retrieve /
build_prompt / generate. On finish the spans go to
    $LIVERMORE_HOME/traces/YYYY-MM-DD/trace_<id>.jsonl   (one span per line, day in UTC)
and one summary row is appended to
    $LIVERMORE_HOME/traces/index.jsonl
`livermore stats` reads only index.jsonl; `--trace <id>` reads the span file.
Feedback (👍/👎) is appended as a span to the trace file and as a row to index.jsonl.
"""
from __future__ import annotations

import json
import os
import secrets
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone

_WRITE_LOCK = threading.Lock()
_ID_RE_CHARS = set("0123456789abcdef")


from .config import home  # noqa: E402


def trace_root() -> str:
    return os.path.join(home(), "traces")


def new_id() -> str:
    return secrets.token_hex(8)


def _err(e) -> str:
    msg = e if isinstance(e, str) else f"{type(e).__name__}: {e}"
    return msg[:500]


class Trace:
    def __init__(self, name: str, attributes: dict | None = None, trace_id: str | None = None):
        self.trace_id = trace_id or new_id()
        self.name = name
        self.started_at = datetime.now(timezone.utc)
        self._t0 = time.perf_counter()
        self._seq = 0
        self.spans: list[dict] = []
        self.attributes = dict(attributes or {})  # trace-level, merged into the index row
        self.root = self._open(name, None)

    def _ms(self) -> int:
        return round((time.perf_counter() - self._t0) * 1000)

    def _open(self, name, parent) -> dict:
        self._seq += 1
        return {"trace_id": self.trace_id, "span_id": secrets.token_hex(4), "parent_span_id": parent,
                "name": name, "start_ms": self._ms(), "end_ms": None, "attributes": {},
                "status": "ok", "error": None, "_seq": self._seq}

    def _close(self, span):
        span["end_ms"] = self._ms()
        self.spans.append(span)

    @contextmanager
    def span(self, name: str, **attributes):
        s = self.start(name, **attributes)
        try:
            yield s
        except BaseException as e:
            self.end(s, e)
            raise
        else:
            self.end(s, s["error"] if s["status"] == "error" else None)

    def start(self, name: str, parent: str | None = None, **attributes) -> dict:
        s = self._open(name, parent or self.root["span_id"])
        s["attributes"].update(attributes)
        return s

    def end(self, span: dict, error=None):
        if error is not None:
            span["status"] = "error"
            span["error"] = _err(error)
        self._close(span)

    @staticmethod
    def fail(span: dict, error, kind: str):
        """Mark a span as failed without raising (e.g. empty retrieval)."""
        span["status"] = "error"
        span["error"] = _err(error)
        span["attributes"]["error_kind"] = kind

    def finish(self, error=None, error_kind: str | None = None) -> str:
        if error is not None:
            self.root["status"] = "error"
            self.root["error"] = _err(error)
            if error_kind:
                self.root["attributes"]["error_kind"] = error_kind
        self._close(self.root)
        self.root["attributes"]["duration_ms"] = self.root["end_ms"]
        spans = [{k: v for k, v in s.items() if k != "_seq"} for s in sorted(self.spans, key=lambda s: s["_seq"])]
        by_name = {s["name"]: s for s in spans}
        gen = by_name.get("generate", {}).get("attributes", {})
        error_kinds = sorted({s["attributes"]["error_kind"] for s in spans if s["attributes"].get("error_kind")})
        row = {
            "trace_id": self.trace_id, "name": self.name, "time": self.started_at.isoformat(timespec="milliseconds"),
            "status": self.root["status"], "total_latency_ms": self.root["end_ms"],
            "error_kinds": error_kinds,
            "prompt_tokens": gen.get("prompt_tokens", 0), "completion_tokens": gen.get("completion_tokens", 0),
            "ttft_ms": gen.get("ttft_ms"), "tokens_per_s": gen.get("tokens_per_s"),
            **{f"{n}_ms": by_name[n]["end_ms"] - by_name[n]["start_ms"]
               for n in ("embed_query", "retrieve", "build_prompt", "generate") if n in by_name},
            **self.attributes,
        }
        day = self.started_at.strftime("%Y-%m-%d")
        d = os.path.join(trace_root(), day)
        with _WRITE_LOCK:
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"trace_{self.trace_id}.jsonl"), "w") as f:
                f.writelines(json.dumps(s, ensure_ascii=False) + "\n" for s in spans)
            with open(os.path.join(trace_root(), "index.jsonl"), "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return self.trace_id


# ---------------- reading ----------------

def _valid_id(tid: str) -> bool:
    return 0 < len(tid) <= 16 and set(tid) <= _ID_RE_CHARS


def find_trace_file(trace_id: str) -> str | None:
    """Exact id or unique prefix. Ids are validated so a request can't walk the filesystem."""
    root = trace_root()
    if not _valid_id(trace_id) or not os.path.isdir(root):
        return None
    matches = []
    for day in sorted(os.listdir(root)):
        dp = os.path.join(root, day)
        if os.path.isdir(dp):
            matches += [os.path.join(dp, f) for f in os.listdir(dp) if f.startswith(f"trace_{trace_id}")]
    return matches[0] if len(matches) == 1 else None


def read_trace(trace_id: str) -> list[dict] | None:
    p = find_trace_file(trace_id)
    if not p:
        return None
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def read_index() -> list[dict]:
    p = os.path.join(trace_root(), "index.jsonl")
    if not os.path.exists(p):
        return []
    rows = []
    with open(p) as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a torn last line from a crash must not break stats
    return rows


# ---------------- feedback ----------------

def record_feedback(trace_id: str, rating: int, comment: str | None = None, entry: str | None = None) -> bool:
    """rating: +1 (👍) or -1 (👎). Appends a `feedback` span to the trace and a row to index.jsonl."""
    if rating not in (1, -1):
        raise ValueError("rating must be +1 or -1")
    p = find_trace_file(trace_id)
    if not p:
        return False
    with open(p) as f:
        root = next(json.loads(l) for l in f if l.strip() and json.loads(l)["parent_span_id"] is None)
    now = datetime.now(timezone.utc)
    span = {"trace_id": root["trace_id"], "span_id": secrets.token_hex(4), "parent_span_id": root["span_id"],
            "name": "feedback", "start_ms": None, "end_ms": None,
            "attributes": {"rating": rating, "comment": comment, "time": now.isoformat(timespec="milliseconds")},
            "status": "ok", "error": None}
    row = {"trace_id": root["trace_id"], "name": "feedback", "time": now.isoformat(timespec="milliseconds"),
           "rating": rating, "comment": comment, "entry": entry}
    with _WRITE_LOCK:
        with open(p, "a") as f:
            f.write(json.dumps(span, ensure_ascii=False) + "\n")
        with open(os.path.join(trace_root(), "index.jsonl"), "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return True
