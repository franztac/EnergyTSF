import torch
import torch.nn as nn
from mamba_ssm import Mamba



class GRUWithAdapterTime(nn.Module):
    """
    Encoder for Time-wise tokenization.
    Input: [B, L, H] | Output: [B, L, H]
    """
    def __init__(
        self, 
        d_model, 
        num_layers, 
        dropout, 
        bidirectional=True,
    ):
        super().__init__()

        # Avoid PyTorch warning
        gru_internal_dropout = dropout if num_layers > 1 else 0.0
        
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_internal_dropout,
            bidirectional=bidirectional,
        )

        # Projects [B, L, H*2] back to [B, L, H] if bidirectional
        self.adapter = nn.Linear(d_model * 2, d_model) if bidirectional else nn.Identity()

        # Explicit dropout ensures 'dropout' hparam is used even if num_layers=1
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        # x: [B, L, H]
        out, _ = self.gru(x)   # out: [B, L, H*2] or [B, L, H]
        out = self.adapter(out) # Always [B, L, H]
        return self.dropout(out) # Apply regularization before returning
    


class GRUWithAdapterVariate(nn.Module):
    """
    Encoder for Variate-wise (Flipped) tokenization.
    Input: [B, V, D] | Output: [B, V, D]
    """
    def __init__(
        self, 
        d_model, 
        num_layers, 
        dropout, 
        bidirectional=True
    ):
        super().__init__()

        # Avoid PyTorch warning for single-layer GRU
        gru_internal_dropout = dropout if num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_internal_dropout,
            bidirectional=bidirectional
        )

        self.adapter = nn.Linear(
            d_model * 2, 
            d_model
        ) if bidirectional else nn.Identity()

        # Explicit dropout to match Transformer-style regularization behavior
        self.dropout = nn.Dropout(dropout)                        


    def forward(
        self, 
        x
    ):
        # x: [B, V, D]
        out, _ = self.gru(x) 
        # out: [B, V, D*2] or [B, V, D]
        out = self.adapter(out)
        return self.dropout(out)




# --- Sublayer 1: The Mixers (Mamba SSM) ---
class MambaSublayerTime(nn.Module):
    """
    Mixer for Time-wise tokenization. Scans Temporal axis L.
    Input/Output: [B, L, H]
    """
    def __init__(
        self, 
        d_model, 
        d_state=16,
        d_conv=4,
        expand=2, 
        dt_rank="auto", 
        dt_min=0.001, 
        dt_max=0.1, 
        use_fast_path=True, 
        bidirectional=False,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        
        mamba_params = dict(
            d_model=d_model, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand,
            dt_rank=dt_rank, 
            dt_min=dt_min, 
            dt_max=dt_max, 
            use_fast_path=use_fast_path,
        )
        self.mamba_fwd = Mamba(**mamba_params)
        self.mamba_bwd = Mamba(**mamba_params) if bidirectional else None
        self.norm = nn.LayerNorm(d_model)


    def forward(self, x):
        # x: [B, L, H] | returns [B, L, H]
        residual = x
        y_fwd = self.mamba_fwd(x)
        if self.bidirectional and self.mamba_bwd is not None:
            x_bwd = torch.flip(x, dims=[1])    # Flip along L dimension
            y_bwd = self.mamba_bwd(x_bwd)
            y_bwd = torch.flip(y_bwd, dims=[1])  # unflip to realign token order
            y_fwd = y_fwd + y_bwd
            
        return self.norm(residual + y_fwd) # Post-Norm




class MambaSublayerVariate(nn.Module):
    """
    Mixer for Variate-wise (Flipped) tokenization.
    Scans Variate axis V. Always Bidirectional.
    Input/Output: [B, V, D]
    """
    def __init__(
        self, 
        d_model, 
        d_state=16,
        d_conv=4,
        expand=2, 
        dt_rank="auto",
        dt_min=0.001, 
        dt_max=0.1, 
        use_fast_path=True,
    ):
        super().__init__()
        mamba_params = dict(
            d_model=d_model,
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand,
            dt_rank=dt_rank,
            dt_min=dt_min, 
            dt_max=dt_max, 
            use_fast_path=use_fast_path,
        )

        self.mamba_fwd = Mamba(**mamba_params)
        self.mamba_bwd = Mamba(**mamba_params)
        self.norm = nn.LayerNorm(d_model)


    def forward(self, x):
        # x: (B, V, D) | returns: (B, V, D)
        residual = x
        y_fwd = self.mamba_fwd(x)
        x_bwd = torch.flip(x, dims=[1])    # Flip along Variate dimension
        y_bwd = self.mamba_bwd(x_bwd)      
        y_bwd = torch.flip(y_bwd, dims=[1])   # Flip back to align variable tokens
        
        return self.norm(residual + y_fwd + y_bwd) # Post-Norm




# --- Sublayer 2: The Refiner (FFN) ---
class FFNSublayerBlock(nn.Module):
    """
    Feed-Forward: Residual -> FFN -> LayerNorm.
    Input/Output: [B, L, H] or [B, V, D]
    """
    def __init__(
        self, 
        d_model, 
        d_ff=None,
        dropout=0.1,
        activation="gelu",
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        act = nn.GELU() if activation == "gelu" else (nn.ReLU() if activation == "relu" else nn.SiLU())

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            act,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)


    def forward(self, x):
        # Residual connection + LayerNorm
        return self.norm(x + self.ffn(x))
    



# --- Specialized Full Encoder Layers ---
class MambaEncoderTime(nn.Module):
    def __init__(
        self,
        d_model,
        d_state,
        d_conv,
        expand, 
        dt_rank,
        dt_min,
        dt_max,
        use_fast_path,
        bidirectional,
        d_ff,
        dropout,
        activation,
    ):
        super().__init__()
        self.sublayer1 = MambaSublayerTime(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            use_fast_path=use_fast_path,
            bidirectional=bidirectional,
        )
        self.sublayer2 = FFNSublayerBlock(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
        )

    def forward(self, x):
        return self.sublayer2(self.sublayer1(x))
    



class MambaEncoderVariate(nn.Module):
    def __init__(
        self, 
        d_model, 
        d_state, 
        d_conv, 
        expand, 
        dt_rank, 
        dt_min, 
        dt_max, 
        use_fast_path, 
        d_ff,
        dropout, 
        activation,
    ):
        
        super().__init__()
        self.sublayer1 = MambaSublayerVariate(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            use_fast_path=use_fast_path,
        )
        self.sublayer2 = FFNSublayerBlock(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
        )

    
    def forward(self, x):
        return self.sublayer2(self.sublayer1(x))
        