# From-scratch Transformer — data, checkpoints, teaching notes

The code lives in [`livermore/scratch/`](../livermore/scratch). This directory holds what it trains on
and what it produces:

| path | what |
|---|---|
| `1706.03762v7.pdf` | *Attention Is All You Need* (Vaswani et al.) |
| `tokenizer/corpus/raw/` | *Reminiscences of a Stock Operator* + SFT pairs |
| `tokenizer/corpus/processed/` | BPE tokenizer (`tokenizer.json`, vocab 11,502) and src/tgt pairs |
| `checkpoints/` | `seq2seq_*.pt` (pre-training), `sft*.pt` (fine-tuned); not in git |

Model: encoder-decoder, d_model 256, 4 layers, 4 heads, context 512, 13,261,824 parameters.
Every op (`softmax`, `layer_norm`, `gelu`, `matmul`) is hand-written in `livermore/scratch/ops.py`.
`benchmarks/test_ops_correctness.py` checks each op against PyTorch.

The scripts use paths relative to this directory, so run them from here as modules:

```bash
cd transformer
python -m livermore.scratch.train                      # pre-train seq2seq -> checkpoints/seq2seq_best.pt
python -m livermore.scratch.sft --dataset ../data/sft_dataset.jsonl
python -m livermore.scratch.generate --mode seq2seq --ckpt checkpoints/sft_best.pt --src "who are you?"
python -m livermore.scratch.extract_training_metrics
```

As a `livermore` backend: `livermore ask "..." --backend scratch`. It answers from the
question alone; retrieved notes are not fed to it.
