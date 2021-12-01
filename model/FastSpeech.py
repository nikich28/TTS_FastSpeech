import torch.nn as nn
from PositionalEncoding import PositionalEncoding
from FFTBlock import FFTBlock
from LengthRegulator import LengthRegulator


class FastSpeechModel(nn.Module):
    def __init__(self, dict_size, emb_dim, n_blocks1, n_blocks2, n_heads, conv_size, kernel_size, duration_size,
                 attn_hidden, output_size, dropout, alpha):
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
        self.LR = LengthRegulator(emb_dim, duration_size, dropout, alpha)

    def forward(self, x):
        emb = self.pe1(self.emb(x))
        x = self.blocks1(emb)

        x, predicted_duration = self.LR(x, target_duration=x.durations)
        emb = self.pe2(x)
        x = self.blocks2(emb)

        return (self.linear(x), predicted_duration)
