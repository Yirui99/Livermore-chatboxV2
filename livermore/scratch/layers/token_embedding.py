# layers/token_embedding.py
import torch
import torch.nn as nn
from .. import ops

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, pad_id: int = 0, std: float = 0.02):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        nn.init.normal_(self.weight, mean=0.0, std=std)
        self.pad_id = pad_id
        self.scale = d_model ** 0.5

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return ops.embedding(ids, self.weight, pad_id=self.pad_id, scale=self.scale)
