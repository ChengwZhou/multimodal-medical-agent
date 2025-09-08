import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    
    def forward(self, query: torch.Tensor, key: Optional[torch.Tensor] = None, 
            value: Optional[torch.Tensor] = None, mask: torch.Tensor = None):
        if key is None:
            key = query
        if value is None:
            value = key
            
        B, Lq, D = query.shape
        Lk = key.shape[1]
        
        # project each separately
        q = self.q_proj(query).view(B, Lq, self.nhead, self.d_k).transpose(1, 2)
        k = self.k_proj(key).view(B, Lk, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_proj(value).view(B, Lk, self.nhead, self.d_k).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask[:, None, None, :], float('-inf'))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_out = torch.matmul(attn_probs, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, Lq, D)
        return self.o_proj(attn_out)


class CrossAttentionLayer(nn.Module):
    """
    Cross-attention from queries (Q) to a set of key/value memory tokens (K,V).
    Useful for attending current-window features to history features.

    Shapes:
      q: [B, Lq, D]
      kv: [B, Lkv, D]
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 4*256, dropout: float = 0.1):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = MultiheadSelfAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor,
                kv_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre norms
        qn = self.ln_q(q)
        kvn = self.ln_kv(kv)
        # attention
        out = self.attn(qn, kvn, kvn, mask=kv_mask)  # query, key, value, mask  # [B, Lq, D]
        q = q + self.dropout(out)
        # MLP
        q = q + self.mlp(q)
        return q


class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiheadSelfAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        # Self-attention
        x_tmp = self.self_attn(self.norm1(x), mask=x_mask)
        x = x_tmp + self.dropout(x)

        # Feedforward
        x = x + self.mlp(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int):
        super().__init__()

	    # Add debug code to see what attributes TransformerLayer has
	    # Add debug code to see what attributes TransformerLayer has
        print("TransformerLayer attributes:", [attr for attr in dir(encoder_layer) if not attr.startswith('_')])

        self.layers = nn.ModuleList([encoder_layer if i == 0 else type(encoder_layer)(
            encoder_layer.self_attn.d_model,
            encoder_layer.self_attn.nhead,
            encoder_layer.mlp[1].out_features,
            encoder_layer.dropout.p
        ) for i in range(num_layers)])

    def forward(self, src: torch.Tensor, mask: torch.Tensor = None):
        output = src
        for mod in self.layers:
            output = mod(output, x_mask=mask)
        return output


class TransformerEncoder(nn.Module):
    """Lightweight Transformer encoder stack for per-modality encoding."""
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int = 4*256, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   batch_first=True, norm_first=True)
        self.net = TransformerBlock(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.net(x, src_key_padding_mask=src_key_padding_mask)
