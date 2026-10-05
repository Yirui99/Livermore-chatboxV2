"""`livermore` command line. Settings: built-in default < ~/.livermore/config.yaml < flags."""
from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from .errors import LivermoreError
from .generate import BACKENDS


def _quiet_hf():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _settings(args):
    from .config import Settings
    s = Settings.load(args.config)
    s.override(**{k: getattr(args, k, None) for k in
                  ("backend", "device", "top_k", "timeout_s", "max_tokens", "temperature", "top_p",
                   "host", "port", "notes_dir", "index_dir")})
    return s


def _add_gen_args(p):
    p.add_argument("--backend", choices=BACKENDS)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], help="torch/scratch backends")
    p.add_argument("--index", dest="index_dir", help="index directory")
    p.add_argument("-k", "--top-k", dest="top_k", type=int, help="notes retrieved per question")
    p.add_argument("--max-tokens", dest="max_tokens", type=int)
    p.add_argument("--temperature", type=float)
    p.add_argument("--top-p", dest="top_p", type=float)
    p.add_argument("--timeout", dest="timeout_s", type=float, help="generation time limit, seconds")


def _load(s):
    from .generate import get_backend
    from .index import load
    index = load(s.index_dir, device=s.device)
    index.embed_revision = index.embed_revision or s.embed_revision
    backend = get_backend(s.backend, **s.backend_kwargs())
    return index, backend


def cmd_build(args):
    from .index import build
    s = _settings(args)
    idx = build(s.notes_dir, out_dir=s.index_dir, embed_model=s.embed_model, embed_revision=s.embed_revision)
    print(f"built index: {len(idx)} notes, dim={idx.dim} -> {s.index_dir}")


def cmd_ask(args):
    from .ask import ask
    s = _settings(args)
    index, backend = _load(s)
    ans = ask(args.question, index, backend, k=s.top_k, max_tokens=s.max_tokens, temperature=s.temperature,
              top_p=s.top_p, entry=args.entry, timeout_s=s.timeout_s,
              on_token=lambda t: print(t, end="", flush=True))
    print("\n")
    for i, h in enumerate(ans.hits, 1):
        print(f"[{i}] note {h.id}  score={h.score:.4f}  {h.text[:100].replace(chr(10), ' ')}")
    print(f"\ntrace={ans.trace_id} (rate it: livermore feedback {ans.trace_id} up|down) usage={ans.usage} "
          + " ".join(f"{k}={v:.0f}ms" for k, v in ans.timings_ms.items()))


def cmd_serve(args):
    from .serve import serve
    s = _settings(args)
    index, backend = _load(s)
    serve(index, backend, s)


def cmd_stats(args):
    from . import stats
    if args.trace:
        sys.exit(stats.show_trace(args.trace))
    if args.list:
        return stats.list_queries(args.since, args.rated, args.include_eval)
    if args.json:
        return stats.report_json(args.since, args.include_eval)
    stats.report(args.since, tuple(args.by) if args.by else ("backend", "day"), args.include_eval)


def cmd_feedback(args):
    from .trace import record_feedback
    ok = record_feedback(args.trace_id, 1 if args.rating == "up" else -1, args.comment, entry="cli")
    print("recorded" if ok else f"trace not found: {args.trace_id}")
    sys.exit(0 if ok else 1)


def cmd_app(args):
    from .config import PROJECT_ROOT
    script = os.path.join(PROJECT_ROOT, "apps", "streamlit_app.py")
    if not os.path.exists(script):
        raise LivermoreError(f"the Streamlit app is not at {script}",
                             "it ships with the source checkout; install with `pip install -e .` from a clone")
    if args.config:
        os.environ["LIVERMORE_CONFIG"] = os.path.abspath(os.path.expanduser(args.config))
    os.execvp(sys.executable, [sys.executable, "-m", "streamlit", "run", script, "--server.port", str(args.port)])


def cmd_doctor(args):
    from .doctor import run
    sys.exit(run(args.config, verify=args.verify))


def cmd_models(args):
    from . import models
    s = _settings(args)
    repos = {s.embed_model: s.embed_revision, s.hf_model: s.hf_revision, s.mlx_model: s.mlx_revision}
    if args.action == "list":
        for repo, rev in repos.items():
            problems = models.check(repo, rev)
            print(f"  {'✓' if not problems else '✗'} {repo} @ {models.revision_for(repo, rev)[:10]}  "
                  + ("ok" if not problems else problems[0]))
        print(f"  cache: {os.path.dirname(models.local_dir('x/y'))}")
        return
    if args.repo is None:  # default: what the configured backend needs
        targets = {s.embed_model: s.embed_revision}
        if s.backend == "torch":
            targets[s.hf_model] = s.hf_revision
        if s.backend == "mlx":
            targets[s.mlx_model] = s.mlx_revision
    elif args.repo == "all":
        targets = repos
    else:
        targets = {args.repo: repos.get(args.repo)}
    for repo, rev in targets.items():
        path = models.fetch(repo, rev, force=args.force)
        print(f"  ✓ {repo} -> {path}")


def cmd_config(args):
    from .config import Settings, default_config_path, render_template
    path = os.path.expanduser(args.config or default_config_path())
    if args.action == "path":
        print(path)
    elif args.action == "init":
        if os.path.exists(path) and not args.force:
            raise LivermoreError(f"{path} already exists", "use --force to overwrite it")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(render_template())
        print(f"wrote {path}")
    else:
        s = Settings.load(args.config)
        print(f"# {path}{'' if os.path.exists(path) else ' (not present; built-in defaults)'}")
        for k, v, src in s.describe():
            print(f"{k:<18} {str(v):<62} # {'config' if src == path else src}")


def main(argv=None):
    _quiet_hf()
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="config file (default: ~/.livermore/config.yaml or $LIVERMORE_CONFIG)")

    ap = argparse.ArgumentParser(prog="livermore", description="Local notes Q&A engine for Apple Silicon")
    ap.add_argument("--version", action="version", version=f"livermore {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def cmd(name, fn, help):
        p = sub.add_parser(name, help=help, parents=[common])
        p.set_defaults(fn=fn)
        return p

    p = cmd("ask", cmd_ask, "ask one question")
    p.add_argument("question")
    _add_gen_args(p)
    p.add_argument("--entry", default="cli", help=argparse.SUPPRESS)

    p = cmd("serve", cmd_serve, "OpenAI-compatible HTTP server")
    _add_gen_args(p)
    p.add_argument("--host")
    p.add_argument("--port", type=int)

    p = cmd("app", cmd_app, "open the Streamlit app")
    p.add_argument("--port", type=int, default=8501)

    p = cmd("build", cmd_build, "build the vector index from a notes directory")
    p.add_argument("--notes", dest="notes_dir")
    p.add_argument("--out", dest="index_dir")

    p = cmd("stats", cmd_stats, "usage report from traces")
    p.add_argument("--since", default="7d", help="7d | 24h | 4w | YYYY-MM-DD | all")
    p.add_argument("--by", nargs="*", choices=["backend", "day", "entry", "device"])
    p.add_argument("--list", action="store_true", help="list every query with its rating")
    p.add_argument("--rated", choices=["up", "down", "none"], help="with --list")
    p.add_argument("--trace", help="show one trace (id or unique prefix)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--include-eval", action="store_true", help="also count entry=eval/test traffic")

    p = cmd("feedback", cmd_feedback, "rate an answer: livermore feedback <trace_id> up|down")
    p.add_argument("trace_id")
    p.add_argument("rating", choices=["up", "down"])
    p.add_argument("--comment")

    p = cmd("doctor", cmd_doctor, "check install, config, models, index, traces")
    p.add_argument("--verify", action="store_true", help="also re-hash every model file")

    p = cmd("models", cmd_models, "list or download model files")
    p.add_argument("action", choices=["list", "fetch"])
    p.add_argument("repo", nargs="?", help="repo id, or 'all' (default: what the configured backend needs)")
    p.add_argument("--force", action="store_true", help="delete and re-download")
    p.add_argument("--backend", choices=BACKENDS)

    p = cmd("config", cmd_config, "show | init | path")
    p.add_argument("action", nargs="?", default="show", choices=["show", "init", "path"])
    p.add_argument("--force", action="store_true")

    args = ap.parse_args(argv)
    try:
        args.fn(args)
    except LivermoreError as e:
        print(f"livermore: error: {e.message}", file=sys.stderr)
        if e.hint:
            print(f"  hint: {e.hint}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
