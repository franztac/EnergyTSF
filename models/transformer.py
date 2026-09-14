import torch
import torch.nn as nn
from utils.tools import count_parameters, print_diagnostics
from layers.embed import (
    TimeEmbedding, 
    VariateEmbedding, 
    PositionalEmbedding,
    LearnablePositionalEmbedding,
)
from layers.attention import TemporalAttention



class TransformerTimeModel(nn.Module):
    """
    TIME tokenization:
        input BLV -> embed V->H -> BLH
        + positional encoding (sin or learnable)
        -> TransformerEncoder -> BLH
        -> pooling (last or attention) -> BH
        -> linear head -> B(T*V) -> reshape -> BTV
    """
    def __init__(self, configs):
        super().__init__()
        self.horizon = configs.horizon
        self.channels = configs.channels
        self.pooling_type = configs.pooling   # "last" | "attention"
        
        # ---- Embedding: pointwise V -> H ----
        self.value_embedding = TimeEmbedding(
            channels=configs.channels, 
            d_model=configs.d_model, 
            dropout=configs.dropout,
        )

        # ---- Positional encoding switch (time only) ----
        # Rule:
        # - "sin": module returns PE tensor [1,L,H] -> you add it
        # - "learnable": module returns transformed x (adds + dropout internally)
        pos_emb_type = getattr(configs, "pos_emb", "sin")
        self.pos_emb_type = pos_emb_type

        if pos_emb_type == "sin":
            self.pos_embedding = PositionalEmbedding(d_model=configs.d_model)
        elif pos_emb_type == "learnable":
            # reuse global dropout; no extra pos_dropout name (your preference)
            self.pos_embedding = LearnablePositionalEmbedding(
                d_model=configs.d_model,
                dropout=configs.dropout,
                max_len=max(1024, configs.window),
            )
        else:
            raise ValueError(f"Invalid pos_emb for time token: {pos_emb_type}")
        

        # ---- TransformerEncoderLayer params (config-driven) ----

        # Usually d_ff = 4*d_model in vanilla Transformer
        d_ff = configs.dim_feedforward if configs.dim_feedforward is not None else 4 * configs.d_model
        configs.dim_feedforward = d_ff
        
        nhead = configs.nhead
        if configs.d_model % nhead != 0:
            raise ValueError(f"d_model ({configs.d_model}) must be divisible by nhead ({nhead}).")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=configs.d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=configs.dropout,
            activation=configs.activation,         # "relu" | "gelu"
            layer_norm_eps=configs.layer_norm_eps,
            batch_first=configs.batch_first,
            norm_first=configs.norm_first,
        )


        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=configs.num_layers,
        )
        
        # ---- Pooling (time token always) ----
        if self.pooling_type == "attention":
            self.pooler = TemporalAttention(
                d_model=configs.d_model,
                temperature=configs.temperature,
            )
        elif self.pooling_type == "last":
            self.pooler = None
        else:
            raise ValueError(f"Invalid pooling type: {self.pooling_type}")
        
        # ---- Head ----
        self.output_layer = nn.Linear(
            configs.d_model, self.horizon * self.channels)


    def forward(self, x):
        # x: [B, L, V]
        x = self.value_embedding(x)      # [B, L, H]
        
        # apply PE
        if self.pos_emb_type == "sin":
            x = x + self.pos_embedding(x)    # add [1,L,H]
        else:
            x = self.pos_embedding(x)       # module applies add+dropout internally

        # encoder (no mask passed => full attention)
        x = self.encoder(x)              # [B, L, H]
        
        # pooling -> [B, H]
        if self.pooling_type == "last":
            x = x[:, -1, :]              
        else:
            x = self.pooler(x)           

        x = self.output_layer(x)         # [B, T * V]
        return x.view(-1, self.horizon, self.channels)  # [B, T, V]
    



class TransformerVariateModel(nn.Module):
    """
    VARIATE tokenization (inverted):
        input BLV -> flip + embed L->D -> BVD
        -> TransformerEncoder -> BVD
        -> linear head D->T -> BVT -> permute -> BTV

    No pooling. No positional encoding (your rule: variates unordered).
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
        
        # Usually d_ff = 4*d_model in vanilla Transformer
        d_ff = configs.dim_feedforward if configs.dim_feedforward is not None else 4 * configs.d_model
        configs.dim_feedforward = d_ff

        nhead = configs.nhead
        if configs.d_model % nhead != 0:
            raise ValueError(f"d_model ({configs.d_model}) must be divisible by nhead ({nhead}).")


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=configs.d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=configs.dropout,
            activation=configs.activation,
            layer_norm_eps=configs.layer_norm_eps,
            batch_first=configs.batch_first,
            norm_first=configs.norm_first,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, 
            num_layers=configs.num_layers,
        )

        # Maps the latent D dimension of each variate to the Horizon T
        self.output_layer = nn.Linear(configs.d_model, self.horizon)
        

    def forward(self, x):
        # x: [B, L, V]
        x = self.embedding(x)    # [B, V, D]
        x = self.encoder(x)      # [B, V, D]
        x = self.output_layer(x) # [B, V, T]
        return x.permute(0, 2, 1) # [B, T, V]
    




class TransformerModel(nn.Module):
    """
    Wrapper that selects Time vs Variate tokenization.

    Diagnostics are printed via utils.tools.print_diagnostics (shared with GRU/Mamba).
    """
    def __init__(self, configs, verbose: bool = True):
        super().__init__()
        self.token_style = configs.token_style
        
        if self.token_style == "time":
            self.model = TransformerTimeModel(configs=configs)
        else:
            self.model = TransformerVariateModel(configs=configs)
            
        # --- 🤖 Corposus Diagnostics ---
        if verbose:
            params, size_mb = count_parameters(self)
            print_diagnostics(self, configs, params, size_mb)

    def forward(self, x):
        # x: [B, L, V]
        return self.model(x) # [B, T, V]