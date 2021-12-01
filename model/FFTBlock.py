import torch.nn as nn
from .Attention import MultiHeadAttention


class FFTBlock(nn.Module):
    '''
    here we use conv1d instead of linear layers
    and we use pre-norm
    '''
    def __init__(self, n_heads, attn_size, input_size, hidden_size, kernel_size, dropout):
        super().__init__()

        self.attn = MultiHeadAttention(
            input_size=input_size,
            hidden_size=attn_size,
            n_heads=n_heads,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(input_size)
        self.act = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding='same',
        )
        self.conv2 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=input_size,
            kernel_size=kernel_size,
            padding='same',
        )

    def forward(self, x, mask=None):
        tmp = x.clone()
        tmp = self.norm1(tmp)
        tmp = self.attn(tmp, tmp, tmp, mask)
        x += self.dropout1(tmp)

        tmp = x.clone()
        tmp = self.norm2(tmp)
        tmp = self.dropout2(self.conv2(self.act(self.conv1(tmp.transpose(1, 2))))).transpose(1, 2)
        return x + tmp
