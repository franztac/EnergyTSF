import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class PositionalEmbedding(nn.Module):
    """Fixed Sin/Cos Embedding (Standard)"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)    # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    

    def forward(self, x):
        # x: [Batch, Seq_Len, d_model]
        return self.pe[:, :x.size(1)]




class LearnablePositionalEmbedding(nn.Module):
    """Directly adapted from the 2020 classic TSTransformer."""
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.empty(1, max_len, d_model)) 
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        # x: [Batch, Seq_Len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)




class TimeEmbedding(nn.Module):
    """Time-wise: [B, L, V] -> [B, L, H]"""
    def __init__(self, channels, d_model, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(channels, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, V] 
        return self.dropout(self.proj(x))   # [B, L, H]



class PatchEmbedding(nn.Module):
    pass



class VariateEmbedding(nn.Module):
    """Variate-wise (Flipped): [B, L, V] -> [B, V, D]"""
    def __init__(self, lookback, d_model, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(lookback, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, V]
        x = x.permute(0, 2, 1) # [B, V, L]
        return self.dropout(self.proj(x))  # [B, V, D]