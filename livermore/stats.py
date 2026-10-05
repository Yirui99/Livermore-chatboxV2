"""`livermore stats`: usage report over $LIVERMORE_HOME/traces/index.jsonl.

    livermore stats --since 7d                   queries, latency p50/p95, failure rate, by backend, by day
    livermore stats --since 30d --by entry
    livermore stats --list [--rated down]        every query with its 👍/👎 (for picking eval questions)
    livermore stats --trace <id-or-prefix>       one span tree
    livermore stats --json                       the summary as JSON (README table source)

Only real usage counts: rows with entry in EXCLUDED_ENTRIES (eval, test) are left out
unless --include-eval. Percentiles are nearest-rank, as in Next AI. Days are local time.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from .trace import read_index, read_trace

EXCLUDED_ENTRIES = {"eval", "test"}


def parse_since(s: str | None) -> datetime | None:
    if not s or s == "all":
        return None
    m = re.fullmatch(r"(\d+)([hdw])", s)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        return datetime.now(timezone.utc) - timedelta(**{{"h": "hours", "d": "days", "w": "weeks"}[unit]: n})
    return datetime.fromisoformat(s).astimezone(timezone.utc)  # YYYY-MM-DD (local midnight)


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    return xs[max(0, math.ceil(len(xs) * p) - 1)] if xs else None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _t(r) -> datetime:
    return datetime.fromisoformat(r["time"])


def _local_day(r) -> str:
    return _t(r).astimezone().strftime("%Y-%m-%d")


def load(since: str | None = "7d", include_eval: bool = False):
    cutoff = parse_since(since)
    rows = read_index()
    asks = [r for r in rows if r.get("name") == "ask"
            and (cutoff is None or _t(r) >= cutoff)
            and (include_eval or r.get("entry") not in EXCLUDED_ENTRIES)]
    ratings = {}
    for r in rows:  # last click wins
        if r.get("name") == "feedback":
            ratings[r["trace_id"]] = r["rating"]
    for r in asks:
        r["rating"] = ratings.get(r["trace_id"])
    return asks, cutoff


def summarize(asks: list[dict]) -> dict:
    ok = [r for r in asks if r["status"] == "ok"]
    lat = [r["total_latency_ms"] for r in ok]
    days = Counter(_local_day(r) for r in asks)
    kinds = Counter(k for r in asks for k in r.get("error_kinds", []))
    up = sum(1 for r in asks if r.get("rating") == 1)
    down = sum(1 for r in asks if r.get("rating") == -1)
    return {
        "queries": len(asks), "ok": len(ok), "failed": len(asks) - len(ok),
        "failure_rate": (len(asks) - len(ok)) / len(asks) if asks else None,
        "active_days": len(days), "per_active_day_mean": _mean(list(days.values())),
        "per_active_day_max": max(days.values()) if days else None,
        "latency_ms": {"n": len(lat), "p50": pct(lat, .5), "p95": pct(lat, .95), "mean": _mean(lat)},
        "ttft_ms": {"p50": pct([r.get("ttft_ms") for r in ok], .5), "p95": pct([r.get("ttft_ms") for r in ok], .95)},
        "retrieval_ms": {"p50": pct([(r.get("embed_query_ms") or 0) + (r.get("retrieve_ms") or 0) for r in ok], .5),
                         "p95": pct([(r.get("embed_query_ms") or 0) + (r.get("retrieve_ms") or 0) for r in ok], .95)},
        "tokens_per_s_p50": pct([r.get("tokens_per_s") for r in ok], .5),
        "error_kinds": dict(kinds),
        "feedback": {"up": up, "down": down, "rated": up + down},
    }


def _group(asks, key):
    g = defaultdict(list)
    for r in asks:
        g[key(r)].append(r)
    return g


def _fmt(x, d=0):
    return "n/a" if x is None else f"{x:,.{d}f}"


def _table(head, rows):
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h)) for i, h in enumerate(head)]
    for r in [head, *rows]:
        print("  " + "  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))


def _group_rows(asks, key):
    out = []
    for k, rs in sorted(_group(asks, key).items()):
        s = summarize(rs)
        out.append([k, s["queries"], s["failed"], _fmt(s["failure_rate"] * 100, 1) + "%",
                    _fmt(s["latency_ms"]["p50"]), _fmt(s["latency_ms"]["p95"]),
                    _fmt(s["ttft_ms"]["p50"]), _fmt(s["tokens_per_s_p50"], 1),
                    s["feedback"]["up"], s["feedback"]["down"]])
    return out


GROUP_HEAD = ["n", "fail", "fail%", "p50_ms", "p95_ms", "ttft_p50", "tok/s_p50", "👍", "👎"]
GROUPS = {
    "backend": lambda r: f"{r.get('backend')} ({r.get('model', '?').split('/')[-1]})",
    "day": _local_day,
    "entry": lambda r: r.get("entry", "?"),
    "device": lambda r: str(r.get("device")),
}


def report(since="7d", by=("backend", "day"), include_eval=False):
    asks, cutoff = load(since, include_eval)
    s = summarize(asks)
    window = "all time" if cutoff is None else f"since {cutoff.astimezone():%Y-%m-%d %H:%M} ({since})"
    print(f"Livermore usage, {window}{' incl. eval' if include_eval else ''}")
    if not asks:
        print("  no queries recorded")
        return
    print(f"  queries      {s['queries']}  (ok {s['ok']}, failed {s['failed']} = {_fmt(s['failure_rate'] * 100, 1)}%)"
          f"   active days {s['active_days']}, per active day mean {_fmt(s['per_active_day_mean'], 1)} max {s['per_active_day_max']}")
    L = s["latency_ms"]
    print(f"  latency      p50 {_fmt(L['p50'])} ms  p95 {_fmt(L['p95'])} ms  mean {_fmt(L['mean'])} ms  (successful, n={L['n']})")
    print(f"    ttft       p50 {_fmt(s['ttft_ms']['p50'])} ms  p95 {_fmt(s['ttft_ms']['p95'])} ms")
    print(f"    retrieval  p50 {_fmt(s['retrieval_ms']['p50'])} ms  p95 {_fmt(s['retrieval_ms']['p95'])} ms  (embed + search)")
    print(f"    generation p50 {_fmt(s['tokens_per_s_p50'], 1)} tokens/s (end to end, incl. prefill)")
    ek = s["error_kinds"]
    print("  errors       " + (", ".join(f"{k} {v}" for k, v in sorted(ek.items())) if ek else "none"))
    fb = s["feedback"]
    print(f"  feedback     👍 {fb['up']}  👎 {fb['down']}  ({fb['rated']}/{s['queries']} rated"
          + (f"; 👎 share of rated {fb['down'] / fb['rated'] * 100:.0f}%)" if fb["rated"] else ")"))
    for g in by:
        print(f"\nBy {g}")
        _table([g, *GROUP_HEAD], _group_rows(asks, GROUPS[g]))


def report_json(since="7d", include_eval=False):
    asks, cutoff = load(since, include_eval)
    out = {"since": since, "cutoff": cutoff.isoformat() if cutoff else None, **summarize(asks),
           "by_backend": {k: summarize(v) for k, v in _group(asks, GROUPS["backend"]).items()}}
    print(json.dumps(out, indent=2, ensure_ascii=False))


def list_queries(since="7d", rated=None, include_eval=False):
    asks, _ = load(since, include_eval)
    want = {"up": 1, "down": -1}.get(rated)
    sym = {1: "👍", -1: "👎", None: "  "}
    for r in asks:
        if rated == "none" and r.get("rating") is not None:
            continue
        if want is not None and r.get("rating") != want:
            continue
        st = "ok " if r["status"] == "ok" else "ERR"
        print(f"{_t(r).astimezone():%m-%d %H:%M}  {r['trace_id']}  {sym[r.get('rating')]}  {st}  "
              f"{r.get('backend', '?'):<7} {_fmt(r.get('total_latency_ms')):>6} ms  {r.get('query', '')[:90]}")


def show_trace(tid: str):
    spans = read_trace(tid)
    if not spans:
        print(f"trace not found (or ambiguous prefix): {tid}")
        return 1
    kids = defaultdict(list)
    for s in spans:
        kids[s["parent_span_id"]].append(s)

    def show(s, depth):
        dur = "" if s["start_ms"] is None else f" [{s['start_ms']}-{s['end_ms']}ms, {s['end_ms'] - s['start_ms']}ms]"
        attrs = " ".join(f"{k}={json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, str)) else v}"
                         for k, v in s["attributes"].items() if k != "answer")
        print(f"{'  ' * depth}{s['name']}{dur} {s['status']}{' ERROR: ' + s['error'] if s['error'] else ''}")
        if attrs:
            print(f"{'  ' * (depth + 1)}{attrs}")
        if s["attributes"].get("answer"):
            print(f"{'  ' * (depth + 1)}answer={s['attributes']['answer'][:300]!r}")
        for c in kids[s["span_id"]]:
            show(c, depth + 1)

    print(f"trace {spans[0]['trace_id']}")
    for root in kids[None]:
        show(root, 0)
    return 0
