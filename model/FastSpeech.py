import torch.nn as nn
from .PositionalEncoding import PositionalEncoding
from .FFTBlock import FFTBlock
from .LengthRegulator import LengthRegulator


class FastSpeechModel(nn.Module):
    def __init__(self, dict_size, emb_dim, n_blocks1, n_blocks2, n_heads, conv_size, kernel_size, duration_size,
                 attn_hidden, output_size, device, dropout, alpha):
        super().__init__()
        self.emb = nn.Embedding(dict_size, emb_dim)
        self.pe1 = PositionalEncoding(emb_dim)
        self.pe2 = PositionalEncoding(emb_dim)
        self.linear = nn.Linear(emb_dim, output_size)
        self.blocks1 = nn.ModuleList([
            FFTBlock(n_heads, attn_hidden, emb_dim, conv_size, kernel_size, dropout) for _ in range(n_blocks1)
        ])
        self.blocks2 = nn.ModuleList([
            FFTBlock(n_heads, attn_hidden, emb_dim, conv_size, kernel_size, dropout) for _ in range(n_blocks2)
        ])
        self.LR = LengthRegulator(emb_dim, duration_size, device, dropout, alpha)

    def forward(self, x):
        tkn = x.tokens
        emb = self.pe1(self.emb(tkn))
        for b in self.blocks1:
            emb = b(emb)
        tkn = emb

        tkn, predicted_duration = self.LR(tkn, target_duration=x.durations)
        emb = self.pe2(tkn)
        for b in self.blocks2:
            emb = b(emb)
        tkn = emb

        return (self.linear(tkn).transpose(1, 2), predicted_duration)
