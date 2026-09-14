import torch
import torch.nn as nn
from utils.tools import count_parameters, print_diagnostics
from layers.embed import TimeEmbedding, VariateEmbedding
from layers.attention import TemporalAttention
from layers.blocks import MambaEncoderTime, MambaEncoderVariate




class MambaTimeModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.horizon = configs.horizon
        self.channels = configs.channels
        self.pooling_type = configs.pooling

        self.embedding = TimeEmbedding(
            channels=configs.channels, 
            d_model=configs.d_model, 
            dropout=configs.dropout,
        )
        
        # 1. Instantiate the Prototype Layer
        self.encoder_layer = MambaEncoderTime(
            d_model=configs.d_model, 
            d_state=configs.d_state, 
            d_conv=configs.d_conv,
            expand=configs.expand, 
            dt_rank=configs.dt_rank, 
            dt_min=configs.dt_min, 
            dt_max=configs.dt_max,
            use_fast_path=getattr(configs, "use_fast_path", True), 
            bidirectional=configs.bidirectional,
            d_ff=getattr(configs, "d_ff", None), 
            dropout=configs.dropout, 
            activation=getattr(configs, "activation", "gelu"),
        )


        # 2. Stack the Prototype N times
        self.encoder = nn.Sequential(*[self.encoder_layer for _ in range(configs.num_layers)])

        if self.pooling_type == "attention":
            self.pooler = TemporalAttention(
                d_model=configs.d_model,
                temperature=configs.temperature,
            )

        self.head = nn.Linear(configs.d_model, configs.horizon * configs.channels)

    
    def forward(self, x):
        # x: [B, L, V] -> [B, L, H]
        x = self.embedding(x)
        x = self.encoder(x)
        
        # Collapse L dimension
        x = x[:, -1, :] if self.pooling_type == "last" else self.pooler(x) # [B, H]
        
        # Project to Horizon
        out = self.head(x) # [B, T*V]
        return out.view(-1, self.horizon, self.channels)   # [B, T, V]
    




class MambaVariateModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.horizon = configs.horizon
        self.channels = configs.channels

        self.embedding = VariateEmbedding(configs.window, configs.d_model, configs.dropout)
        
        # 1. Instantiate the Prototype Layer (Always Bidirectional)
        self.encoder_layer = MambaEncoderVariate(
            d_model=configs.d_model, 
            d_state=configs.d_state, 
            d_conv=configs.d_conv,
            expand=configs.expand, 
            dt_rank=configs.dt_rank, 
            dt_min=configs.dt_min, 
            dt_max=configs.dt_max,
            use_fast_path=getattr(configs, "use_fast_path", True),
            d_ff=getattr(configs, "d_ff", None), 
            dropout=configs.dropout, 
            activation=getattr(configs, "activation", "gelu"),
        )

        # 2. Stack the Prototype N times
        self.encoder = nn.Sequential(*[self.encoder_layer for _ in range(configs.num_layers)])

        # Linear head projects D (history info) directly to T (future horizon)
        self.head = nn.Linear(configs.d_model, configs.horizon)


    def forward(self, x):
        # x: [B, L, V] -> [B, V, D]
        x = self.embedding(x)
        x = self.encoder(x)
        
        # Linear Head per variable: [B, V, D] -> [B, V, T]
        x = self.head(x)
        
        # Final Reorder: [B, V, T] -> [B, T, V]
        return x.permute(0, 2, 1).contiguous()
    



class MambaModel(nn.Module):
    """
    Modular Wrapper for the 4 runs suite.
    """
    def __init__(self, configs, verbose: bool = True):
        super().__init__()
        self.token_style = configs.token_style
        
        if self.token_style == "time":
            self.model = MambaTimeModel(configs)
        else:
            self.model = MambaVariateModel(configs)

        # Corposus Diagnostics using updated print_diagnostics from utils/tools.py
        if verbose:
            params, size_mb = count_parameters(self)
            print_diagnostics(self, configs, params, size_mb)

    def forward(self, x):
        return self.model(x)   # [B, L, V] -> [B, T, V]