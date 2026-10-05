"""OpenAI-compatible HTTP server (stdlib only).

  POST /v1/chat/completions   RAG over the last user message; stream=true supported (SSE)
  GET  /v1/models
  GET  /health                index status + current backend

The response carries the standard OpenAI fields plus a `livermore` object with
the trace id and the retrieved notes. Client system messages are ignored: the
system prompt is the one that carries the retrieved notes.
"""
from __future__ import annotations

import json
import threading
import uuid
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__
from .ask import GenerationTimeout, ask
from .trace import record_feedback, trace_root


def _last_user_text(messages) -> str | None:
    for m in reversed(messages or []):
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, list):  # content parts
            c = "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
        if isinstance(c, str) and c.strip():
            return c
    return None


def _hits_json(hits):
    return [{"id": h.id, "score": round(h.score, 6), "text": h.text} for h in hits]


class App:
    def __init__(self, index, backend, defaults):
        self.index = index
        self.backend = backend
        self.defaults = defaults  # livermore.config.Settings
        self.lock = threading.Lock()  # one model, one generation at a time
        self.started = time.time()

    def health(self) -> dict:
        b = self.backend
        return {
            "status": "ok",
            "version": __version__,
            "uptime_s": round(time.time() - self.started, 1),
            "index": self.index.info(),
            "backend": {"name": b.name, "model": b.model, "revision": getattr(b, "revision", None), "device": b.device,
                        "device_requested": getattr(b, "device_requested", None), "dtype": getattr(b, "dtype", None)},
            "traces": trace_root(),
        }

    def params(self, body: dict) -> dict:
        d = self.defaults
        return {
            "k": int(body.get("top_k_notes", d.top_k)),
            "max_tokens": int(body.get("max_tokens") or body.get("max_completion_tokens") or d.max_tokens),
            "temperature": float(body.get("temperature", d.temperature)),
            "top_p": float(body.get("top_p", d.top_p)),
            "timeout_s": d.timeout_s,
            "entry": "serve",
        }


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = f"livermore/{__version__}"

        def log_message(self, fmt, *args):  # quieter than the default
            print(f"[serve] {self.address_string()} {fmt % args}", flush=True)

        def _json(self, code: int, obj: dict):
            data = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, code: int, msg: str, typ: str = "invalid_request_error"):
            self._json(code, {"error": {"message": msg, "type": typ}})

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/")
            if path == "/health":
                return self._json(200, app.health())
            if path == "/v1/models":
                return self._json(200, {"object": "list", "data": [
                    {"id": app.backend.model, "object": "model", "created": int(app.started), "owned_by": "livermore"}]})
            self._error(404, f"no route {self.path}")

        def do_POST(self):
            route = self.path.split("?")[0].rstrip("/")
            if route not in ("/v1/chat/completions", "/v1/feedback"):
                return self._error(404, f"no route {self.path}")
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, json.JSONDecodeError):
                return self._error(400, "request body is not valid JSON")
            if route == "/v1/feedback":
                return self._feedback(body)
            query = _last_user_text(body.get("messages"))
            if not query:
                return self._error(400, "messages must contain a non-empty user message")
            try:
                p = app.params(body)
            except (TypeError, ValueError) as e:
                return self._error(400, f"bad parameter: {e}")
            created = int(time.time())
            model = app.backend.model
            if body.get("stream"):
                return self._stream(query, p, created, model)
            try:
                with app.lock:
                    ans = ask(query, app.index, app.backend, **p)
            except GenerationTimeout as e:
                return self._error(504, str(e), "timeout")
            except Exception as e:
                return self._error(500, f"{type(e).__name__}: {e}", "server_error")
            u = ans.usage
            self._json(200, {
                "id": f"chatcmpl-{ans.trace_id}",
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": ans.text},
                             "finish_reason": _finish(u, p)}],
                "usage": {"prompt_tokens": u.get("prompt_tokens", 0),
                          "completion_tokens": u.get("completion_tokens", 0),
                          "total_tokens": u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)},
                "livermore": {"trace_id": ans.trace_id, "hits": _hits_json(ans.hits), "timings_ms": ans.timings_ms},
            })

        def _feedback(self, body):
            rating = {"up": 1, "down": -1, 1: 1, -1: -1}.get(body.get("rating"))
            if not isinstance(body.get("trace_id"), str) or not body["trace_id"] or rating is None:
                return self._error(400, 'expected {"trace_id": "...", "rating": "up" | "down"}')
            if not record_feedback(body["trace_id"], rating, body.get("comment"), entry="serve"):
                return self._error(404, f"trace not found: {body['trace_id']}")
            self._json(200, {"ok": True})

        def _stream(self, query, p, created, model):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            trace_id = uuid.uuid4().hex[:16]
            cid = f"chatcmpl-{trace_id}"

            def send(obj):
                self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()

            def chunk(delta, finish=None, extra=None):
                obj = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                if extra:
                    obj.update(extra)
                send(obj)

            first = True

            def on_token(t):
                nonlocal first
                if first:
                    chunk({"role": "assistant", "content": ""})
                    first = False
                chunk({"content": t})

            try:
                with app.lock:
                    ans = ask(query, app.index, app.backend, on_token=on_token, trace_id=trace_id, **p)
                if first:
                    chunk({"role": "assistant", "content": ""})
                chunk({}, _finish(ans.usage, p),
                      {"livermore": {"trace_id": ans.trace_id, "hits": _hits_json(ans.hits)}})
            except BrokenPipeError:
                return
            except Exception as e:
                send({"error": {"message": f"{type(e).__name__}: {e}", "type": "server_error"}})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    return Handler


def _finish(usage: dict, p: dict) -> str:
    return "length" if usage.get("completion_tokens", 0) >= p["max_tokens"] else "stop"


def serve(index, backend, settings, host: str | None = None, port: int | None = None):
    index.embed("warmup")  # load the embedder now, not on the first request
    app = App(index, backend, settings)
    host = host or settings.host
    port = port or settings.port
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    print(f"livermore {__version__} serving on http://{host}:{port}  "
          f"(backend={backend.name} model={backend.model} device={backend.device}, index={len(index)} notes)",
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
