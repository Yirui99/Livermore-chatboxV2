"""`livermore` command line."""
from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from .config import Settings
from .generate import BACKENDS


def _quiet_hf():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _add_backend_args(p, s: Settings):
    p.add_argument("--backend", choices=BACKENDS, default=s.backend)
    p.add_argument("--device", default=s.device, help="auto | cpu | mps | cuda (torch/scratch backends)")
    p.add_argument("--index", default=s.index_dir, help="index directory")
    p.add_argument("-k", "--top-k", type=int, default=s.top_k)
    p.add_argument("--timeout", type=float, default=s.timeout_s, help="generation time limit, seconds")


def _load(args, s: Settings):
    from .generate import get_backend
    from .index import load
    s.device = args.device
    s.timeout_s = args.timeout
    index = load(args.index)
    backend = get_backend(args.backend, **s.backend_kwargs(args.backend))
    return index, backend


def cmd_build(args, s: Settings):
    from .index import build
    idx = build(args.notes, out_dir=args.out)
    print(f"built index: {len(idx)} notes, dim={idx.dim} -> {args.out}")


def cmd_ask(args, s: Settings):
    from .ask import ask
    index, backend = _load(args, s)
    ans = ask(args.question, index, backend, k=args.top_k, max_tokens=args.max_tokens,
              temperature=args.temperature, top_p=s.top_p, entry=args.entry, timeout_s=s.timeout_s,
              on_token=lambda t: print(t, end="", flush=True))
    print("\n")
    for i, h in enumerate(ans.hits, 1):
        print(f"[{i}] note {h.id}  score={h.score:.4f}  {h.text[:100].replace(chr(10), ' ')}")
    print(f"\ntrace={ans.trace_id} (rate it: livermore feedback {ans.trace_id} up|down) usage={ans.usage} "
          + " ".join(f"{k}={v:.0f}ms" for k, v in ans.timings_ms.items()))


def cmd_serve(args, s: Settings):
    from .serve import serve
    s.top_k = args.top_k
    index, backend = _load(args, s)
    serve(index, backend, s, host=args.host, port=args.port)


def cmd_stats(args, s: Settings):
    from . import stats
    if args.trace:
        sys.exit(stats.show_trace(args.trace))
    if args.list:
        return stats.list_queries(args.since, args.rated, args.include_eval)
    if args.json:
        return stats.report_json(args.since, args.include_eval)
    stats.report(args.since, tuple(args.by) if args.by else ("backend", "day"), args.include_eval)


def cmd_feedback(args, s: Settings):
    from .trace import record_feedback
    ok = record_feedback(args.trace_id, 1 if args.rating == "up" else -1, args.comment, entry="cli")
    print("recorded" if ok else f"trace not found: {args.trace_id}")
    sys.exit(0 if ok else 1)


def cmd_app(args, s: Settings):
    from .config import PROJECT_ROOT
    script = os.path.join(PROJECT_ROOT, "apps", "streamlit_app.py")
    if not os.path.exists(script):
        sys.exit(f"Streamlit app not found at {script} (it ships with the source checkout, not the wheel)")
    os.execvp(sys.executable, [sys.executable, "-m", "streamlit", "run", script, "--server.port", str(args.port)])


def main(argv=None):
    _quiet_hf()
    s = Settings()
    ap = argparse.ArgumentParser(prog="livermore", description="Local notes Q&A engine for Apple Silicon")
    ap.add_argument("--version", action="version", version=f"livermore {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build", help="build the vector index from a notes directory")
    p.add_argument("--notes", default=s.notes_dir)
    p.add_argument("--out", default=s.index_dir)
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("ask", help="ask one question")
    p.add_argument("question")
    _add_backend_args(p, s)
    p.add_argument("--max-tokens", type=int, default=s.max_tokens)
    p.add_argument("--temperature", type=float, default=s.temperature)
    p.add_argument("--entry", default="cli", help=argparse.SUPPRESS)
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("serve", help="OpenAI-compatible HTTP server")
    _add_backend_args(p, s)
    p.add_argument("--host", default=s.host)
    p.add_argument("--port", type=int, default=s.port)
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("stats", help="usage report from traces")
    p.add_argument("--since", default="7d", help="7d | 24h | 4w | YYYY-MM-DD | all")
    p.add_argument("--by", nargs="*", choices=["backend", "day", "entry", "device"])
    p.add_argument("--list", action="store_true", help="list every query with its rating")
    p.add_argument("--rated", choices=["up", "down", "none"], help="with --list")
    p.add_argument("--trace", help="show one trace (id or unique prefix)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--include-eval", action="store_true", help="also count entry=eval/test traffic")
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("feedback", help="rate an answer: livermore feedback <trace_id> up|down")
    p.add_argument("trace_id")
    p.add_argument("rating", choices=["up", "down"])
    p.add_argument("--comment")
    p.set_defaults(fn=cmd_feedback)

    p = sub.add_parser("app", help="open the Streamlit app")
    p.add_argument("--port", type=int, default=8501)
    p.set_defaults(fn=cmd_app)

    args = ap.parse_args(argv)
    try:
        args.fn(args, s)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
