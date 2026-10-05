"""
Behavioural equivalence check for the library refactor.

  --capture   run the pre-refactor code (rag_qa.RAGQA, transformer/generate.py)
              and write benchmarks/baseline/equivalence.json. Only works on a
              pre-refactor checkout (captured at commit 693a450).
  --check     run the same inputs through livermore.* and diff against the fixture

The benchmark scripts do not import the app code, so they cannot detect a
behaviour change introduced by moving RAG/inference into the package. This can.

Usage (from repo root):
  python benchmarks/equivalence.py --capture   # once, on the old code
  python benchmarks/equivalence.py --check
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ROOT, "benchmarks", "baseline", "equivalence.json")

QUERIES = [
    "What is the most important rule in trading?",
    "How to manage risk in stock trading?",
    "When should I cut my losses?",
    "How to identify a trend reversal?",
    "What is position sizing?",
    "How do you handle a losing streak?",
    "What makes a great trader?",
    "When to add to a winning position?",
    "How to control emotions in trading?",
    "What is the role of patience in trading?",
    "Why did Livermore go bankrupt?",
    "What did Livermore think about tips from other people?",
    "How should I react when the market moves against me?",
    "Is it better to average down on a losing trade?",
    "What is a pivotal point?",
]
GEN_QUERIES = QUERIES[:3]
SEARCH_K = 10
GEN_TOKENS = 64
SCRATCH_CKPT = os.path.join(ROOT, "transformer", "checkpoints", "sft_best.pt")
SCRATCH_TOKENIZER = os.path.join(ROOT, "transformer", "tokenizer", "corpus", "processed", "tokenizer.json")
SCRATCH_QUERIES = ["who are you?", "How should I handle losses?"]


def _greedy_hf(model, tokenizer, prompt, device):
    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=GEN_TOKENS, do_sample=False)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def capture():
    import torch
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    from rag_qa import RAGQA, load_yaml_config

    rag = RAGQA(load_yaml_config("config_rag.yaml"))
    doc_id = {d: i for i, d in enumerate(rag.docs)}

    fx = {"device": rag.device, "search_k": SEARCH_K, "gen_tokens": GEN_TOKENS,
          "retrieval": {}, "prompt": {}, "greedy": {}, "scratch": {}}
    for q in QUERIES:
        docs, scores = rag.retrieve(q, top_k=SEARCH_K)
        fx["retrieval"][q] = {"ids": [doc_id[d] for d in docs], "scores": scores}
        fx["prompt"][q] = rag.build_prompt(q, docs[:3])
    for q in GEN_QUERIES:
        fx["greedy"][q] = _greedy_hf(rag.model, rag.tokenizer, fx["prompt"][q], rag.device)

    sys.path.insert(0, os.path.join(ROOT, "transformer"))
    from generate import build_modules, generate_seq2seq
    dev = torch.device("cpu")
    tok, cfg, tok_emb, pos_emb, model, kind, pad_id = build_modules(SCRATCH_CKPT, SCRATCH_TOKENIZER, dev)
    for q in SCRATCH_QUERIES:
        fx["scratch"][q] = generate_seq2seq(q, tok, cfg, tok_emb, pos_emb, model, pad_id,
                                            max_new_tokens=GEN_TOKENS, temperature=0.0, top_k=0,
                                            min_new_tokens=0, no_repeat_ngram_size=3,
                                            repetition_penalty=1.1, device=dev)

    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    with open(FIXTURE, "w") as f:
        json.dump(fx, f, indent=2, ensure_ascii=False)
    print(f"wrote {FIXTURE}")


def check():
    from livermore import index as lindex
    from livermore.ask import build_messages
    from livermore.generate import get_backend

    with open(FIXTURE) as f:
        fx = json.load(f)
    failures = []

    idx = lindex.load(os.path.join(ROOT, "kb_data"))
    max_dscore = 0.0
    for q, ref in fx["retrieval"].items():
        hits = idx.search(q, k=fx["search_k"])
        ids = [h.id for h in hits]
        if ids != ref["ids"]:
            failures.append(f"retrieval ids differ for {q!r}: {ids} vs {ref['ids']}")
        max_dscore = max(max_dscore, max(abs(h.score - s) for h, s in zip(hits, ref["scores"])))
    print(f"retrieval: {len(fx['retrieval'])} queries, ids identical={not failures}, max |Δscore|={max_dscore:.2e}")
    if max_dscore > 1e-5:
        failures.append(f"retrieval scores drift {max_dscore:.2e}")

    backend = get_backend("torch", device=fx["device"])
    n_prompt_ok = 0
    for q, ref in fx["prompt"].items():
        hits = idx.search(q, k=3)
        p = backend.apply_chat_template(build_messages(q, hits))
        if p == ref:
            n_prompt_ok += 1
        else:
            failures.append(f"prompt differs for {q!r}")
    print(f"prompt: {n_prompt_ok}/{len(fx['prompt'])} byte-identical")

    n_gen_ok = 0
    for q, ref in fx["greedy"].items():
        out = "".join(backend.generate(fx["prompt"][q], fx["gen_tokens"], temperature=0.0)).strip()
        if out == ref:
            n_gen_ok += 1
        else:
            failures.append(f"greedy output differs for {q!r}:\n  new: {out!r}\n  old: {ref!r}")
    print(f"torch greedy: {n_gen_ok}/{len(fx['greedy'])} identical")

    scratch = get_backend("scratch", device="cpu", ckpt=SCRATCH_CKPT)
    n_s_ok = 0
    for q, ref in fx["scratch"].items():
        out = "".join(scratch.generate(q, fx["gen_tokens"], temperature=0.0, top_k=0, min_new_tokens=0,
                                       no_repeat_ngram_size=3, repetition_penalty=1.1))
        if out == ref:
            n_s_ok += 1
        else:
            failures.append(f"scratch output differs for {q!r}:\n  new: {out!r}\n  old: {ref!r}")
    print(f"scratch greedy: {n_s_ok}/{len(fx['scratch'])} identical")

    if failures:
        print("\nFAIL")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print("\nPASS: refactored code is behaviour-identical to the captured baseline")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--capture", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args()
    capture() if a.capture else check()
