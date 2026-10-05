# train.py
from __future__ import annotations
import math, os, time, random
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from . import config
from .models import TransformerSeq2Seq
from tokenizers import Tokenizer
from .layers.token_embedding import TokenEmbedding
from .layers.positional_encoding import SinusoidalPositionalEncoding

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TOKENIZER_JSON = os.path.join("tokenizer", "corpus", "processed", "tokenizer.json")
tok = Tokenizer.from_file(TOKENIZER_JSON)
PAD_ID = tok.token_to_id("<pad>") if tok.token_to_id("<pad>") is not None else 0
BOS_ID = tok.token_to_id("<s>")   if tok.token_to_id("<s>")   is not None else 1
EOS_ID = tok.token_to_id("</s>")  if tok.token_to_id("</s>")  is not None else 2
UNK_ID = tok.token_to_id("<unk>") if tok.token_to_id("<unk>") is not None else 3

class LinePairDataset(Dataset):
    def __init__(self, text_path: str, tokenizer: Tokenizer):
        with open(text_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        pairs: List[Tuple[List[int], List[int]]] = []
        for i in range(len(lines) - 1):
            src = tokenizer.encode(lines[i]).ids
            tgt = tokenizer.encode(lines[i + 1]).ids
            pairs.append((src, tgt))
        self.data = pairs

    def __len__(self): return len(self.data)

    def __getitem__(self, idx: int):
        src_ids, tgt_ids = self.data[idx]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

def _pad_1d(x: torch.Tensor, L: int, pad_id: int) -> torch.Tensor:
    if x.size(0) >= L:
        return x[:L]
    out = x.new_full((L,), pad_id)
    out[: x.size(0)] = x
    return out

class Seq2SeqCollator:
    def __init__(self, pad_id: int, max_src_len: int, max_tgt_len: int):
        self.pad_id = pad_id
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def set_caps(self, max_src_len: int, max_tgt_len: int):
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __call__(self, batch):
        src_list, tgt_list = zip(*batch)
        Ls, Lt = self.max_src_len, self.max_tgt_len
        src_ids = torch.stack([_pad_1d(x, Ls, self.pad_id) for x in src_list], dim=0)
        tgt_ids = torch.stack([_pad_1d(x, Lt, self.pad_id) for x in tgt_list], dim=0)
        src_mask = (src_ids != self.pad_id).long()
        tgt_mask = (tgt_ids != self.pad_id).long()
        return src_ids, src_mask, tgt_ids, tgt_mask

def cyc_lr(it: int, warmup: int, base_lr: float) -> float:
    if it < warmup:
        return base_lr * (it + 1) / warmup
    return base_lr

def shift_tgt(tgt_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T = tgt_ids.size()
    y_in = tgt_ids[:, :-1].contiguous()
    y_out = tgt_ids[:, 1:].contiguous()
    return y_in, y_out

def build_modules() -> tuple[config.ModelConfig, TokenEmbedding, SinusoidalPositionalEncoding, TransformerSeq2Seq]:
    cfg = config.ModelConfig()
    tok_emb = TokenEmbedding(cfg.vocab_size, cfg.d_model, pad_id=PAD_ID).to(device)
    pos_emb = SinusoidalPositionalEncoding(cfg.n_ctx, cfg.d_model).to(device)
    model = TransformerSeq2Seq(cfg, return_probs=False).to(device)

    @torch.no_grad()
    def _locate_head_param(m: TransformerSeq2Seq) -> torch.nn.Parameter:
        V, D = m.vocab_size, m.d_model
        expect = V * D
        cand_dv = None
        for _, p in m.head.named_parameters(recurse=True):
            if p.ndim == 2 and p.numel() == expect:
                if tuple(p.shape) == (V, D):
                    return p
                if tuple(p.shape) == (D, V):
                    cand_dv = p
        if cand_dv is not None:
            return cand_dv
        raise RuntimeError("LMHead not found")

    with torch.no_grad():
        W_head = _locate_head_param(model)
        W_tok = tok_emb.weight
        if tuple(W_head.shape) == (cfg.vocab_size, cfg.d_model):
            W_head.copy_(W_tok)
        elif tuple(W_head.shape) == (cfg.d_model, cfg.vocab_size):
            W_head.copy_(W_tok.t())
        else:
            raise RuntimeError(f"unexpected LMHead shape: {tuple(W_head.shape)}")

    return cfg, tok_emb, pos_emb, model

def main():
    BATCH_SIZE = 4
    GRAD_ACCUM = 4
    BASE_LR = 3e-4
    WARMUP_UPDATES = 1000
    NUM_EPOCHS = 1000
    VAL_EVERY_EPOCHS = 1
    EARLY_PATIENCE = 15
    USE_AMP = True
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    cfg, tok_emb, pos_emb, model = build_modules()

    MAX_SRC_LEN = cfg.n_ctx
    MAX_TGT_LEN = cfg.n_ctx // 2

    ds_all = LinePairDataset(os.path.join("tokenizer", "corpus", "raw", "Reminiscences_of_a_Stock_Operator.txt"), tok)
    n_total = len(ds_all)
    n_val = max(1, int(0.05 * n_total))
    n_train = n_total - n_val
    ds_train, ds_val = torch.utils.data.random_split(ds_all, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))

    collate = Seq2SeqCollator(PAD_ID, MAX_SRC_LEN, MAX_TGT_LEN)
    NUM_WORKERS = 4
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=(device.type=="cuda"), collate_fn=collate)
    dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(device.type=="cuda"), collate_fn=collate)

    opt = torch.optim.AdamW(model.parameters(), lr=BASE_LR, betas=(0.9, 0.95), weight_decay=0.1)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    scaler = torch.amp.GradScaler('cuda', enabled=(USE_AMP and device.type=="cuda" and amp_dtype in (torch.float16, torch.bfloat16)))

    best_val = float("inf")
    epochs_no_improve = 0
    global_step = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        t0 = time.time()
        run_loss = 0.0
        tokens_seen = 0

        for it, (src_ids, src_mask, tgt_ids, tgt_mask) in enumerate(dl_train, start=1):
            src_ids = src_ids.to(device, non_blocking=True)
            tgt_ids = tgt_ids.to(device, non_blocking=True)
            src_mask = src_mask.to(device, non_blocking=True)
            tgt_mask = tgt_mask.to(device, non_blocking=True)

            B, Ls = src_ids.size()
            y_in, y_out = shift_tgt(tgt_ids)
            _, Lt_in = y_in.size()

            src_emb = tok_emb(src_ids) + pos_emb(B, Ls, device)
            tgt_emb = tok_emb(y_in)   + pos_emb(B, Lt_in, device)

            lr = cyc_lr(global_step, WARMUP_UPDATES, BASE_LR)
            for pg in opt.param_groups: pg["lr"] = lr

            with torch.autocast(device_type="cuda" if device.type=="cuda" else "cpu", dtype=amp_dtype, enabled=USE_AMP):
                logits = model(src_emb, tgt_emb, src_kpm=src_mask, tgt_kpm=tgt_mask[:, :Lt_in])
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), y_out.reshape(-1))
                loss = loss / GRAD_ACCUM

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if it % GRAD_ACCUM == 0:
                if scaler.is_enabled():
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1

            run_loss += loss.item() * GRAD_ACCUM
            tokens_seen += (src_mask.sum().item() + tgt_mask.sum().item())

        avg_train = run_loss / max(1, len(dl_train))
        dt = time.time() - t0
        print(f"[epoch {epoch}] train_loss={avg_train:.4f} tokens={int(tokens_seen)} time={dt:.1f}s", flush=True)

        if (epoch % VAL_EVERY_EPOCHS) == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for src_ids, src_mask, tgt_ids, tgt_mask in dl_val:
                    src_ids = src_ids.to(device, non_blocking=True)
                    tgt_ids = tgt_ids.to(device, non_blocking=True)
                    src_mask = src_mask.to(device, non_blocking=True)
                    tgt_mask = tgt_mask.to(device, non_blocking=True)

                    B, Ls = src_ids.size()
                    y_in, y_out = shift_tgt(tgt_ids)
                    _, Lt_in = y_in.size()

                    src_emb = tok_emb(src_ids) + pos_emb(B, Ls, device)
                    tgt_emb = tok_emb(y_in)   + pos_emb(B, Lt_in, device)

                    with torch.autocast(device_type="cuda" if device.type=="cuda" else "cpu", dtype=amp_dtype, enabled=USE_AMP):
                        logits = model(src_emb, tgt_emb, src_kpm=src_mask, tgt_kpm=tgt_mask[:, :Lt_in])
                        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y_out.reshape(-1))

                    val_loss += loss.item()

                val_loss /= max(1, len(dl_val))
                print(f"[epoch {epoch}] val_loss={val_loss:.4f} best={best_val:.4f} no_improve={epochs_no_improve}", flush=True)

                
                if val_loss + 1e-6 < best_val:
                    best_val = val_loss
                    epochs_no_improve = 0
                    os.makedirs("checkpoints", exist_ok=True)
                    torch.save({
                        "cfg": cfg,
                        "model": model.state_dict(),
                        "tok_emb": tok_emb.state_dict(),
                        "pos_emb": pos_emb.state_dict(),
                    }, os.path.join("checkpoints", "seq2seq_best.pt"))
                    print(f"[best] epoch {epoch} → saved checkpoints/seq2seq_best.pt (val_loss={val_loss:.4f})", flush=True)
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= EARLY_PATIENCE:
                        print(f"[early stop] patience={EARLY_PATIENCE} reached at epoch {epoch}", flush=True)
                        break

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "cfg": cfg,
        "model": model.state_dict(),
        "tok_emb": tok_emb.state_dict(),
        "pos_emb": pos_emb.state_dict(),
    }, os.path.join("checkpoints", "seq2seq_final.pt"))

if __name__ == "__main__":
    main()
