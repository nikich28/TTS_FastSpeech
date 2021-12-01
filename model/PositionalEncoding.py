import torch.nn as nn
import torch
import math


class PositionalEncoding(nn.Module):
    '''
    classic Positional Encoding for transformers
    '''
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, (d_model * 2 + 1) // 2, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term[:d_model // 2 + d_model % 2])
        pe[:, 0, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
