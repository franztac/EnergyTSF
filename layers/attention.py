import torch
import torch.nn as nn
import torch.nn.functional as F



class TemporalAttention(nn.Module):
    def __init__(self, d_model, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.attention_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1, bias=False)
        )

    def forward(self, x):
        # x: [B, L, H]
        scores = self.attention_net(x)  # [B, L, 1]
        weights = F.softmax(scores / self.temperature, dim=1) # [B, L, 1]
        context = torch.sum(x * weights, dim=1)     # [B, H]
        return context