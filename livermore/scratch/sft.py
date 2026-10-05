#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, time, math, argparse, random
from dataclasses import asdict
from typing import List, Tuple, Optional
import inspect, dataclasses
from torch.serialization import add_safe_globals


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.serialization import add_safe_globals
from tqdm import tqdm

# === 你的项目内模块 ===
import config
from models import TransformerSeq2Seq
from layers.token_embedding import TokenEmbedding
from layers.positional_encoding import SinusoidalPositionalEncoding
# -----------------------------
# Tokenizer & special tokens
# -----------------------------
try:
    from tokenizers import Tokenizer
except ImportError:
    raise RuntimeError("Please `pip install tokenizers` first.")

def load_tokenizer(path: str):
    tok = Tokenizer.from_file(path)
    def _id(s, default):
        x = tok.token_to_id(s)
        return default if x is None else x
    PAD = _id("<pad>", 0)
    BOS = _id("<s>", 1)
    EOS = _id("</s>", 2)
    UNK = _id("<unk>", 3)
    return tok, PAD, BOS, EOS, UNK

# -----------------------------
# Dataset
# -----------------------------
class QADatasetJSONL(Dataset):
    """读取 jsonl，每行至少包含一对问答。
       支持常见键名：('src','tgt') / ('question','answer') / ('input','output')
       可选在输入前加前缀（如 'qa: '），便于样式对齐。
    """
    def __init__(self, jsonl_path: str, tokenizer: "Tokenizer",
                 max_src_len: int = 256, max_tgt_len: int = 256,
                 src_key: Optional[str] = None, tgt_key: Optional[str] = None,
                 src_prefix: str = ""):
        self.path = jsonl_path
        self.tok = tokenizer
        self.max_src = max_src_len
        self.max_tgt = max_tgt_len
        self.src_key = src_key
        self.tgt_key = tgt_key
        self.src_prefix = src_prefix

        self.samples: List[Tuple[List[int], List[int]]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                # 解析键
                if self.src_key is None or self.tgt_key is None:
                    for a, b in [("prompt","response"), ("src","tgt"),
                                ("question","answer"), ("input","output")]:

                        if a in obj and b in obj:
                            sk, tk = a, b
                            break
                    else:
                        # 不符合预期的行跳过
                        continue
                else:
                    sk, tk = self.src_key, self.tgt_key
                    if sk not in obj or tk not in obj:
                        continue

                src_text = (self.src_prefix + str(obj[sk])).strip()
                tgt_text = str(obj[tk]).strip()
                if not src_text or not tgt_text:
                    continue

                src_ids = self.tok.encode(src_text).ids[:self.max_src]
                tgt_ids = self.tok.encode(tgt_text).ids[:(self.max_tgt - 2)]  # 预留 <s> 和 </s>
                self.samples.append((src_ids, tgt_ids))

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples parsed from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def _pad_1d(x: List[int], L: int, pad_id: int) -> torch.Tensor:
    if len(x) >= L:
        return torch.tensor(x[:L], dtype=torch.long)
    out = torch.full((L,), pad_id, dtype=torch.long)
    out[:len(x)] = torch.tensor(x, dtype=torch.long)
    return out


class Collator:
    """把 (src_ids, tgt_ids) 补齐到定长；目标侧加 <s> 开头、</s> 结尾。"""
    def __init__(self, pad_id: int, bos_id: int, eos_id: int, max_src: int, max_tgt: int):
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.max_src = max_src
        self.max_tgt = max_tgt

    def __call__(self, batch: List[Tuple[List[int], List[int]]]):
        src_list, tgt_list = zip(*batch)
        # tgt: [B, Lt] 形如 <s> y ... </s>
        tgt_with_tokens = [[self.bos_id] + t + [self.eos_id] for t in tgt_list]

        src_pad = torch.stack([_pad_1d(s, self.max_src, self.pad_id) for s in src_list], dim=0)
        tgt_pad = torch.stack([_pad_1d(t, self.max_tgt, self.pad_id) for t in tgt_with_tokens], dim=0)

        src_kpm = (src_pad != self.pad_id)   # True 表示有效 token
        tgt_kpm = (tgt_pad != self.pad_id)

        return src_pad, src_kpm, tgt_pad, tgt_kpm


def shift_tgt(tgt_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # 输入 tgt_ids 形如 <s> y1 y2 ... </s> PAD...
    # y_in = <s> y1 ... y_{T-1}
    # y_out =      y1 ... y_{T-1} </s>
    return tgt_ids[:, :-1], tgt_ids[:, 1:]


def cyc_lr(step: int, warmup_updates: int, base_lr: float) -> float:
    if step < warmup_updates:
        return base_lr * (step + 1) / max(1, warmup_updates)
    return base_lr

# -----------------------------
# Build + Load
# -----------------------------
def build_modules_from_ckpt(ckpt_path: str, device: torch.device):
    # 允许旧 ckpt 里序列化的 ModelConfig（PyTorch 2.6+ 安全变更）
    add_safe_globals([config.ModelConfig])

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False) 
    raw_cfg = ckpt.get("cfg", None)

    if isinstance(raw_cfg, dict):
        cfg = config.ModelConfig(**raw_cfg)
    else:
        cfg = raw_cfg if raw_cfg is not None else config.ModelConfig()

    tok_emb = TokenEmbedding(cfg.vocab_size, cfg.d_model)
    pos_emb = SinusoidalPositionalEncoding(cfg.n_ctx, cfg.d_model)
    model = TransformerSeq2Seq(cfg)

    # 加载权重（与你的保存键一致）
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    if "tok_emb" in ckpt:
        tok_emb.load_state_dict(ckpt["tok_emb"])
    if "pos_emb" in ckpt:
        pos_emb.load_state_dict(ckpt["pos_emb"])

    tok_emb.to(device)
    pos_emb.to(device)
    model.to(device)

    return cfg, tok_emb, pos_emb, model


# -----------------------------
# Train / Eval
# -----------------------------
def run_epoch(dl, model, tok_emb, pos_emb, loss_fn, opt, scaler, device,
              grad_accum: int, warmup_updates: int, base_lr: float,
              global_step: int, use_amp: bool, pad_id: int) -> Tuple[float, int, int]:
    model.train()
    run_loss = 0.0
    tokens_seen = 0
    total_iters = len(dl)

    pbar = tqdm(dl, total=total_iters, leave=False, desc="train")
    for it, (src_ids, src_kpm, tgt_ids, tgt_kpm) in enumerate(pbar, start=1):
        src_ids = src_ids.to(device, non_blocking=True)
        tgt_ids = tgt_ids.to(device, non_blocking=True)
        src_kpm  = src_kpm.to(device, non_blocking=True)
        tgt_kpm  = tgt_kpm.to(device, non_blocking=True)

        B, Ls = src_ids.size()
        y_in, y_out = shift_tgt(tgt_ids)
        _, Lt_in = y_in.size()

        src_emb = tok_emb(src_ids) + pos_emb(B, Ls, device)
        tgt_emb = tok_emb(y_in)   + pos_emb(B, Lt_in, device)

        lr = cyc_lr(global_step, warmup_updates, base_lr)
        if opt is not None:
            for pg in opt.param_groups:
                pg["lr"] = lr

        with torch.autocast(device_type=("cuda" if device.type=="cuda" else "cpu"),
                             dtype=(torch.bfloat16 if device.type=="cuda" else torch.float32),
                             enabled=use_amp):
            logits = model(src_emb, tgt_emb, src_kpm=src_kpm, tgt_kpm=tgt_kpm[:, :Lt_in])
            # 训练时避免采到 PAD 的梯度影响：这里 CrossEntropy 的 ignore_index=pad_id 已处理
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y_out.reshape(-1))
            loss = loss / grad_accum

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (it % grad_accum == 0) or (it == total_iters):
            if opt is not None:
                if scaler is not None and scaler.is_enabled():
                    scaler.step(opt); scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
            global_step += 1

        run_loss += loss.item() * grad_accum
        tokens_seen += (src_kpm.sum().item() + tgt_kpm.sum().item())
        try:
            pbar.set_postfix_str(f"loss={run_loss/it:.4f} lr={lr:.2e}")
        except Exception:
            pass

    return run_loss / max(1, total_iters), tokens_seen, global_step


@torch.no_grad()
def evaluate(dl, model, tok_emb, pos_emb, loss_fn, device, use_amp: bool) -> float:
    model.eval()
    val_loss = 0.0
    for src_ids, src_kpm, tgt_ids, tgt_kpm in tqdm(dl, total=len(dl), leave=False, desc="valid"):
        src_ids = src_ids.to(device, non_blocking=True)
        tgt_ids = tgt_ids.to(device, non_blocking=True)
        src_kpm  = src_kpm.to(device, non_blocking=True)
        tgt_kpm  = tgt_kpm.to(device, non_blocking=True)

        B, Ls = src_ids.size()
        y_in, y_out = shift_tgt(tgt_ids)
        _, Lt_in = y_in.size()

        src_emb = tok_emb(src_ids) + pos_emb(B, Ls, device)
        tgt_emb = tok_emb(y_in)   + pos_emb(B, Lt_in, device)

        with torch.autocast(device_type=("cuda" if device.type=="cuda" else "cpu"),
                             dtype=(torch.bfloat16 if device.type=="cuda" else torch.float32),
                             enabled=use_amp):
            logits = model(src_emb, tgt_emb, src_kpm=src_kpm, tgt_kpm=tgt_kpm[:, :Lt_in])
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y_out.reshape(-1))
        val_loss += loss.item()
    return val_loss / max(1, len(dl))


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tokenizer/corpus/raw/sft_dataset.jsonl")
    ap.add_argument("--tokenizer", default="tokenizer/corpus/processed/tokenizer.json")
    ap.add_argument("--init_ckpt", default="checkpoints/seq2seq_best.pt")
    ap.add_argument("--save_dir", default="checkpoints")
    ap.add_argument("--save_name", default="sft.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--base_lr", type=float, default=1e-4)
    ap.add_argument("--warmup_updates", type=int, default=200)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--max_src_len", type=int, default=128)
    ap.add_argument("--max_tgt_len", type=int, default=128)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--num_workers", type=int, default=0)  # Windows 友好
    ap.add_argument("--src_key", type=str, default="")     # 留空则自动识别
    ap.add_argument("--tgt_key", type=str, default="")
    ap.add_argument("--src_prefix", type=str, default="")  # 如需 'qa: ' 可在此指定
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    device = torch.device(args.device)

    # tokenizer & special ids
    tok, PAD_ID, BOS_ID, EOS_ID, UNK_ID = load_tokenizer(args.tokenizer)

    # dataset & split
    ds = QADatasetJSONL(args.dataset, tok, args.max_src_len, args.max_tgt_len,
                        src_key=(args.src_key or None),
                        tgt_key=(args.tgt_key or None),
                        src_prefix=args.src_prefix)
    n_total = len(ds)
    n_val = max(1, int(args.val_ratio * n_total))
    n_train = n_total - n_val
    ds_train, ds_val = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    collate = Collator(PAD_ID, BOS_ID, EOS_ID, args.max_src_len, args.max_tgt_len)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, drop_last=False,
                          collate_fn=collate)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True, drop_last=False,
                          collate_fn=collate)

    # modules from checkpoint
    cfg, tok_emb, pos_emb, model = build_modules_from_ckpt(args.init_ckpt, device)

    # optimizer / loss / scaler
    opt = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=0.01, betas=(0.9, 0.95))
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler('cuda', enabled=(args.amp and device.type=="cuda"))

    # train
    os.makedirs(args.save_dir, exist_ok=True)
    best_val = float("inf")
    epochs_no_improve = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_tokens, global_step = run_epoch(
            dl_train, model, tok_emb, pos_emb, loss_fn, opt, scaler, device,
            args.grad_accum, args.warmup_updates, args.base_lr,
            global_step, args.amp, PAD_ID
        )
        dt = time.time() - t0
        print(f"[epoch {epoch}] train_loss={tr_loss:.4f} tokens={int(tr_tokens)} time={dt:.1f}s", flush=True)

        # validate every epoch
        val_loss = evaluate(dl_val, model, tok_emb, pos_emb, loss_fn, device, args.amp)
        print(f"[epoch {epoch}] val_loss={val_loss:.4f} best={best_val:.4f} no_improve={epochs_no_improve}", flush=True)

        # save best
        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            best_path = os.path.join(args.save_dir, "sft_best.pt")
            torch.save({
                "cfg": asdict(cfg) if hasattr(cfg, "__dict__") else cfg,
                "model": model.state_dict(),
                "tok_emb": tok_emb.state_dict(),
                "pos_emb": pos_emb.state_dict(),
            }, best_path)
            print(f"[best] epoch {epoch} → saved {best_path} (val_loss={val_loss:.4f})", flush=True)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"[early stop] patience={args.patience} reached at epoch {epoch}", flush=True)
                break

    # final save
    final_path = os.path.join(args.save_dir, args.save_name)
    torch.save({
        "cfg": asdict(cfg) if hasattr(cfg, "__dict__") else cfg,
        "model": model.state_dict(),
        "tok_emb": tok_emb.state_dict(),
        "pos_emb": pos_emb.state_dict(),
    }, final_path)
    print(f"[done] saved {final_path}", flush=True)


if __name__ == "__main__":
    main()
