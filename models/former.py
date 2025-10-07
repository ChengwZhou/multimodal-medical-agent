import math
from typing import Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

from models.transformer_utils import CrossAttentionLayer, TransformerLayer, TransformerBlock

# -----------------------------
# Utility Modules (as before)
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, L, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        L = x.size(1)
        return x + self.pe[:, :L]


class ConvTokenizer1D(nn.Module):
    """
    Conv tokenizer that outputs a scalar feature per patch (out_channels=1).
    This matches the user's request to concatenate tokens into shape [B, L_total, 1].
    """
    def __init__(self, in_ch: int, out_ch: int = 1, patch_size: int = 10, norm: bool = True):
        super().__init__()
        # produce one scalar per patch per channel
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=patch_size, stride=patch_size, bias=False)
        self.norm = nn.LayerNorm(out_ch) if norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, T] (but we will pass single-channel per call: C_in==1)
        x = self.conv(x)          # [B, out_ch, L]
        x = x.transpose(1, 2)     # [B, L, out_ch]
        x = self.norm(x)
        return x                  # [B, L, out_ch]  (here out_ch==1)


class ModalityConfig:
    def __init__(self, name: str, in_ch: int, patch_size: int):
        self.name = name
        self.in_ch = in_ch
        self.patch_size = patch_size


class MultimodalActivityTransformer(nn.Module):
    def __init__(self,
        num_classes: int,
        model_dim: int = 256,
        nhead: int = 8,
        fusion_depth: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        modalities: Optional[List[ModalityConfig]] = None,
        use_modal_dropout: bool = True,
        modal_dropout_p: float = 0.0,
        max_len: int = 10000,
        return_mem: bool = False
    ):
        super().__init__()

        if modalities is None:
            modalities = [
                ModalityConfig('m0', 1, 10), ModalityConfig('m1', 1, 10),
                ModalityConfig('m2', 1, 10), ModalityConfig('m3', 1, 10),
                ModalityConfig('m4', 1, 10), ModalityConfig('m5', 1, 10),
                ModalityConfig('m6', 1, 10), ModalityConfig('m7', 1, 10),
                ModalityConfig('m8', 1, 10), ModalityConfig('m9', 1, 10),
                ModalityConfig('m10', 1, 10), ModalityConfig('m11', 1, 10),
                ModalityConfig('m12', 1, 10), ModalityConfig('m13', 1, 10),
            ]

        self.modalities = modalities
        self.model_dim = model_dim
        self.use_modal_dropout = use_modal_dropout
        self.modal_dropout_p = modal_dropout_p
        self.return_mem = return_mem

        # Tokenizers: each output 1 channel per patch (so final concat gives [B, 140, 1])
        self.tokenizers = nn.ModuleDict()
        for m in modalities:
            # out_ch=1 so each patch becomes scalar feature
            self.tokenizers[m.name] = ConvTokenizer1D(m.in_ch, out_ch=model_dim, patch_size=m.patch_size)

        # After concatenation, we will project token scalar -> model_dim
        # self.token_proj = nn.Linear(1, model_dim, bias=True)   # maps [B, L, 1] -> [B, L, model_dim]

        # positional encoding for projected tokens (length max_len should >= 140)
        self.positional = PositionalEncoding(model_dim, max_len=max_len)

        # CLS token for fusion
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # History projection: history expected as [B, 1, 140] or [B, 140, 1]
        # We will accept either shape, convert to [B, 140, 1] then project with this linear
        self.history_proj = nn.Linear(1, model_dim, bias=True)  # maps per-token dim 1 -> model_dim

        # Cross-attention layers: fused (Q) attends to mem (K,V)
        self.cross_attn = CrossAttentionLayer(model_dim, nhead, dim_feedforward=ff_dim, dropout=dropout)

        # Fusion transformer (final integration)
        self.fusion = TransformerBlock(
            TransformerLayer(d_model=model_dim, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout),
            num_layers=fusion_depth
        )

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, num_classes)
        )

    def _maybe_modal_dropout(self, x_tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        if not self.use_modal_dropout or self.modal_dropout_p <= 0.0 or not self.training:
            return x_tensors
        out = []
        for v in x_tensors:
            if torch.rand(1).item() < self.modal_dropout_p:
                out.append(torch.zeros_like(v))
            else:
                out.append(v)
        return out

    def forward(self,
        x: torch.Tensor,
        history: Optional[torch.Tensor] = None,
        history_mask: Optional[torch.Tensor] = None,
    ):
        """
        x: [B, C=14, T=100]
        history: expected [B, 1, 140] OR [B, 140, 1] (we handle both)
        """

        B = x.size(0)
        C = x.size(1)
        assert C == len(self.modalities), f"expected {len(self.modalities)} channels, got {C}"

        per_mod_tokens = []
        # iterate channels and tokenize each (we created tokenizers per modality)
        for i, m in enumerate(self.modalities):
            # take channel i as shape [B, 1, T]
            ch = x[:, i:i+1, :]                # [B, 1, T]
            t = self.tokenizers[m.name](ch)    # [B, L=10, out_ch=1]
            per_mod_tokens.append(t)           # keep [B, 10, 1]

        # optional modal dropout
        per_mod_tokens = self._maybe_modal_dropout(per_mod_tokens)

        # concatenate tokens along time dimension -> [B, 14*10=140, 1]
        concat_tokens = torch.cat(per_mod_tokens, dim=1)  # [B, L_total, 1]

        # Project scalar token -> model_dim: [B, L_total, model_dim]
        fused = self.positional(concat_tokens)        # add positional enc

        # (1) Cross-attention: fused (Q) attends to mem (K,V)
        if history is not None:
            fused = self.cross_attn(fused, history, kv_mask=history_mask)

        # (2) Fusion transformer
        out = self.fusion(fused)

        cls_out = out.mean(axis=1)
        logits = self.head(cls_out)
        if self.return_mem:
            return logits, out
        else:
            return logits


def build_former(num_modal=14, num_classes=10, model_dim=32,  return_mem=False):
    # define 14 modalities with patch_size=10
    modalities = [ModalityConfig(f'm{i}', 1, 10) for i in range(num_modal)]
    return MultimodalActivityTransformer(
        num_classes=num_classes,
        model_dim=model_dim,
        nhead=8,
        fusion_depth=1,
        ff_dim=4*model_dim,
        dropout=0.1,
        modalities=modalities,
        use_modal_dropout=True,
        modal_dropout_p=0.15,
        max_len=500,
        return_mem= return_mem
    )


if __name__ == "__main__":
    # test run according to your spec
    x = torch.randn(8, 14, 100)           # [B, 14, 100]
    # history in the form [B, 1, 140]
    history = torch.randn(8, 140, 32,)
    model = build_former(num_classes=12, model_dim=32)
    print(model)
    logits, aux = model(x, history=history)
    print("logits:", logits.shape)          # expected [8, 12]
    print("mem:", aux.shape)  # [8, 140, model_dim]
