# inspect_special_tokens.py
import argparse, json, os, inspect, dataclasses
import torch
from torch.serialization import add_safe_globals
from tokenizers import Tokenizer
import config

def allow_config_dataclasses():
    add_safe_globals([obj for _, obj in inspect.getmembers(config)
                      if inspect.isclass(obj) and dataclasses.is_dataclass(obj)])

def load_ckpt_cfg(path):
    allow_config_dataclasses()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    raw_cfg = ckpt.get("cfg")
    if isinstance(raw_cfg, dict):
        try:
            cfg_obj = config.ModelConfig(**raw_cfg)
        except Exception:
            cfg_obj = raw_cfg
    else:
        cfg_obj = raw_cfg
    return ckpt, cfg_obj

def load_tok_ids(tok_path):
    tok = Tokenizer.from_file(tok_path)
    def _id(s, default):
        x = tok.token_to_id(s)
        return default if x is None else x
    return tok, {
        "PAD": _id("<pad>", 0),
        "BOS": _id("<s>", 1),
        "EOS": _id("</s>", 2),
        "UNK": _id("<unk>", 3),
        "vocab_size": tok.get_vocab_size(),
    }

def maybe_peek_prefix(jsonl_path, keys=("prompt","src","question","input"), topn=10):
    info = {"file": jsonl_path, "exists": os.path.exists(jsonl_path), "prefixed_qa_count": 0, "total": 0, "samples": []}
    if not info["exists"]:
        return info
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            for k in keys:
                if k in o and isinstance(o[k], str):
                    s = o[k].strip()
                    info["total"] += 1
                    if s.lower().startswith("qa:"):
                        info["prefixed_qa_count"] += 1
                    if len(info["samples"]) < topn:
                        info["samples"].append({k: s[:120]})
                    break
    return info

def get_cfg_special_ids(cfg_obj):
    # 尝试从 ckpt.cfg 中读取特殊符号（若未存储则返回 None）
    fields = {}
    for name in ("pad_id","bos_id","eos_id","unk_id","vocab_size"):
        v = None
        if hasattr(cfg_obj, name):
            v = getattr(cfg_obj, name)
        elif isinstance(cfg_obj, dict) and name in cfg_obj:
            v = cfg_obj[name]
        fields[name] = v
    return fields

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--sft_jsonl", default="tokenizer/corpus/raw/sft_dataset.jsonl")
    args = ap.parse_args()

    ckpt, cfg_obj = load_ckpt_cfg(args.ckpt)
    tok, tok_ids = load_tok_ids(args.tokenizer)
    cfg_ids = get_cfg_special_ids(cfg_obj)

    print("=== Check: vocab sizes ===")
    print(f"ckpt cfg.vocab_size = {cfg_ids['vocab_size']}")
    print(f"tokenizer_vocab_size = {tok_ids['vocab_size']}")
    ok_vocab = (cfg_ids["vocab_size"] == tok_ids["vocab_size"])
    print(f"Vocab match: {ok_vocab}")

    print("\n=== Token IDs (tokenizer) ===")
    print({k: tok_ids[k] for k in ("PAD","BOS","EOS","UNK")})

    print("\n=== Token IDs (from ckpt cfg, if any) ===")
    print({k: cfg_ids[k] for k in ("pad_id","bos_id","eos_id","unk_id")})

    if all(cfg_ids[k] is not None for k in ("pad_id","bos_id","eos_id","unk_id")):
        print("\nSpecial IDs match check:")
        for (name_tok, name_cfg) in [("PAD","pad_id"),("BOS","bos_id"),("EOS","eos_id"),("UNK","unk_id")]:
            same = (tok_ids[name_tok] == cfg_ids[name_cfg])
            print(f"  {name_tok}: tokenizer={tok_ids[name_tok]} vs ckpt={cfg_ids[name_cfg]}  -> match={same}")
    else:
        print("\n[Note] ckpt.cfg 未存储特殊符号 ID（常见情况）。训练时通常直接用 tokenizer 的 ID。")

    print("\n=== SFT prefix quick peek ===")
    info = maybe_peek_prefix(args.sft_jsonl)
    if not info["exists"]:
        print(f"file not found: {info['file']}")
    else:
        print(f"scanned: {info['total']} lines; startswith 'qa:' = {info['prefixed_qa_count']}")
        print("samples:", info["samples"])

if __name__ == "__main__":
    main()
