import torch
import torch.nn as nn
from utils.tools import count_parameters, print_diagnostics
from layers.embed import TimeEmbedding, VariateEmbedding
from layers.blocks import GRUWithAdapterTime, GRUWithAdapterVariate
from layers.attention import TemporalAttention



class GRUTimeModel(nn.Module):
    """
    Standard Time-wise GRU.
    Tokens: Time-steps [B, L, H]
    Pooling: Required to collapse L dimension.
    """
    def __init__(self, configs,):
        super().__init__()
        self.horizon = configs.horizon
        self.channels = configs.channels
        self.pooling_type = configs.pooling   # 'last' or 'attention'

        # 1. Embedding Choice
        self.embedding = TimeEmbedding(
            channels=configs.channels, 
            d_model=configs.d_model, 
            dropout=configs.dropout,
        )

        # 2. Sequential Processor
        self.encoder = GRUWithAdapterTime(
            d_model=configs.d_model,
            num_layers=configs.num_layers,
            dropout=configs.dropout,
            bidirectional=configs.bidirectional,
        )
        
        # 3. Pooling Mechanism
        if self.pooling_type == "attention":
            self.pooler = TemporalAttention(
                d_model=configs.d_model,
                temperature=configs.temperature,
            )

        # 4. Prediction Head
        self.output_layer = nn.Linear(
            configs.d_model, configs.horizon * configs.channels
        )


    def forward(self, x):
        # x: [B, L, V]
        x = self.embedding(x)  # [B, L, H]
        x = self.encoder(x)  # [B, L, H]

        if self.pooling_type == "last":
            x = x[:, -1, :]    # [B, H]
        else:
            x = self.pooler(x) # [B, H]

        x = self.output_layer(x)  # [B, T*V]
        return x.view(-1, self.horizon, self.channels)  # [B, T, V]
    



class GRUVariateModel(nn.Module):
    """
    Flipped Variate-wise GRU.
    Tokens: Channels [B, V, D]
    Pooling: None (Bypassed).
    """
    def __init__(self, configs):
        super().__init__()
        self.horizon = configs.horizon
        self.channels = configs.channels

        self.embedding = VariateEmbedding(
            lookback=configs.window,
            d_model=configs.d_model,
            dropout=configs.dropout
        )
        self.encoder = GRUWithAdapterVariate(
            d_model=configs.d_model,
            num_layers=configs.num_layers,
            dropout=configs.dropout,
            bidirectional=configs.bidirectional
        )
        # Projects D dimension directly to T (Horizon)
        self.output_layer = nn.Linear(
            configs.d_model, self.horizon
        )


    def forward(self, x):
        # x shape: [B, L, V]
        x = self.embedding(x)    # [B, V, D]
        x = self.encoder(x)      # [B, V, D]
        x = self.output_layer(x) # [B, V, T]
        return x.permute(0, 2, 1) # [B, T, V]



class SimpleGRUModel(nn.Module):
    def __init__(self, configs, verbose: bool = True):
        super().__init__()
        self.token_style = configs.token_style
        
        if self.token_style == "time":
            self.model = GRUTimeModel(configs=configs)
        else:
            self.model = GRUVariateModel(configs=configs)
            

        # --- 🤖 Diagnostics ---
        if verbose:
            params, size_mb = count_parameters(self)
            print_diagnostics(self, configs, params, size_mb)

    def forward(self, x):
        # x: [B, L, V]
        return self.model(x)    # [B, T, V]