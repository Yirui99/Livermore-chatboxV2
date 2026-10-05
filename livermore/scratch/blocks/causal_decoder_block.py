# blocks/causal_decoder_block.py
import torch
import torch.nn as nn
from .. import config
from ..layers import MaskedMultiHeadAttention, FeedForward, AddNormPre

class CausalDecoderBlock(nn.Module):

    def __init__(self, cfg: config.ModelConfig):
        super().__init__()
        self.mha = MaskedMultiHeadAttention(cfg)  
        self.ffn = FeedForward(cfg)
        self.an1 = AddNormPre(cfg)
        self.an2 = AddNormPre(cfg)

    def forward(self, x: torch.Tensor, *, self_kpm: torch.Tensor | None = None) -> torch.Tensor:
        x = self.an1(x, lambda u: self.mha(u, key_padding_mask=self_kpm))
        x = self.an2(x, lambda u: self.ffn(u))
        return x
