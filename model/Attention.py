import torch.nn as nn
import torch
import numpy as np


class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, hidden_size, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = input_size
        self.n_heads = n_heads
        self.hidden = hidden_size
        self.q = nn.Linear(input_size, self.n_heads * self.hidden)
        self.k = nn.Linear(input_size, self.n_heads * self.hidden)
        self.v = nn.Linear(input_size, self.n_heads * self.hidden)
        self.linear = nn.Linear(self.n_heads * self.hidden, input_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        n_batches = query.size(0)
        key = self.k(key).view(n_batches, -1, self.n_heads, self.hidden).transpose(1, 2)
        query = self.q(query).view(n_batches, -1, self.n_heads, self.hidden).transpose(1, 2)
        value = self.v(value).view(n_batches, -1, self.n_heads, self.hidden).transpose(1, 2)

        #scaled dot product attention
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.hidden)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        attn_probs = self.softmax(attn_scores)
        attn_probs = self.dropout(attn_probs)
        z = torch.matmul(attn_probs, value)
        z = z.transpose(1, 2).contiguous().view(n_batches, -1, self.n_heads * self.d_v)
        return self.linear(z)
