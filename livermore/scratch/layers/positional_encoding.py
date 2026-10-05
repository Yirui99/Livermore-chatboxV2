# layers/positional_encoding_sinusoidal.py
import math, torch, torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int, base: float = 10000.0):
        super().__init__()
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        i = torch.arange(0, d_model, 2, dtype=torch.float32).unsqueeze(0)
        angle = pos / (base ** (i / d_model))
        pe = torch.empty(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = angle.sin()
        pe[:, 1::2] = angle.cos()
        if d_model % 2 == 1:
            pe[:, -1] = 0
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, B: int, T: int, device=None, dtype=None, offset: int = 0):
        x = self.pe[offset:offset+T]
        if device is not None or dtype is not None:
            x = x.to(device=device or x.device, dtype=dtype or x.dtype)
        return x.unsqueeze(0).expand(B, T, -1)
