import math
from typing import Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

from models.transformer_utils import CrossAttentionLayer, TransformerLayer, TransformerBlock, CrossModalAttention
from utils.modality_config import ModalityConfig
from collections import defaultdict


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


class MultimodalActivityTransformer(nn.Module):
    def __init__(self,
        num_classes: int,
        model_dim: int = 32,
        nhead: int = 8,
        fusion_depth: int = 4,
        ff_dim: int = 128,
        dropout: float = 0.1,
        modalities: Optional[List[ModalityConfig]] = None,
        use_modal_dropout: bool = True,
        modal_dropout_p: float = 0.0,
        max_len: int = 10000,
        return_mem: bool = False,
        modal_fusion: str = "concate"  # "concate" or "cross_atten",
    ):
        super().__init__()

        if modalities is None:
            modalities = [
                ModalityConfig('m0', 1, 10, 0)
            ]

        self.modalities = modalities
        self.model_dim = model_dim
        self.use_modal_dropout = use_modal_dropout
        self.modal_dropout_p = modal_dropout_p
        self.return_mem = return_mem
        self.modal_fusion = modal_fusion

        # Group modalities by device_idx
        sensor_mods = defaultdict(list)
        for m in self.modalities:
            sensor_mods[m.device_idx].append(m)

        # Tokenizers: one per sensor
        self.sensor_tokenizers = nn.ModuleDict()
        self.sensor_ranges = {}
        start_ch = 0
        for device_idx in sorted(sensor_mods.keys()):
            mods = sensor_mods[device_idx]
            total_in_ch = sum(m.in_ch for m in mods)
            patch_size = mods[0].patch_size  # assume all same per sensor
            self.sensor_tokenizers[str(device_idx)] = ConvTokenizer1D(total_in_ch, out_ch=model_dim, patch_size=patch_size)
            self.sensor_ranges[device_idx] = (start_ch, start_ch + total_in_ch)
            start_ch += total_in_ch

        # positional encoding for projected tokens (length max_len should >= 140)
        self.positional = PositionalEncoding(model_dim, max_len=max_len)

        # CLS token for fusion
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Cross-attention layers: fused (Q) attends to mem (K,V)
        self.cross_attn = CrossAttentionLayer(model_dim, nhead, dim_feedforward=ff_dim, dropout=dropout)

        # Cross-Modal attention
        if self.modal_fusion == "cross_atten":
            self.modal_cross_attn = CrossModalAttention(
                model_dim, nhead,
                dim_feedforward=ff_dim,
                dropout=dropout
            )

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
        per_sensor_tokens = []
        # iterate sensors and tokenize each
        for device_idx in sorted(self.sensor_ranges.keys()):
            start, end = self.sensor_ranges[device_idx]
            ch = x[:, start:end, :]                # [B, total_in_ch, T]
            t = self.sensor_tokenizers[str(device_idx)](ch)    # [B, L=10, out_ch=model_dim]
            per_sensor_tokens.append(t)           # keep [B, 10, model_dim]

        # optional modal dropout (now applied per sensor)
        per_sensor_tokens = self._maybe_modal_dropout(per_sensor_tokens)

        # concatenate tokens along time dimension -> [B, num_sensors*10, model_dim]
        if self.modal_fusion == "cross_atten":
            concat_tokens = self.modal_cross_attn(per_sensor_tokens, kv_masks=history_mask)
        else:
            concat_tokens = torch.cat(per_sensor_tokens, dim=1)  # [B, L_total, 1]

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


def build_former_device(num_modal=14, num_classes=10, model_dim=32,  return_mem=False, modalities=None, modal_fusion="concate"):
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
        max_len=5000,
        return_mem=return_mem,
        modal_fusion=modal_fusion,
    )


if __name__ == "__main__":
    # test run according to your spec
    x = torch.randn(8, 14, 1000)           # [B, 14, 100]
    # history in the form [B, 1, 140]
    history = torch.randn(8, 1400, 32)

    modalities = [
        ModalityConfig('acc_c', 3, 10, 0), ModalityConfig('ecg_g_c', 1, 10, 0),
        ModalityConfig('ecg_t_c', 1, 10, 0),
        ModalityConfig('eda_f', 1, 10, 1), ModalityConfig('ppg_f', 1, 10, 1),
        ModalityConfig('emg_f', 1, 10, 1),
        ModalityConfig('eda_w', 1, 10, 2), ModalityConfig('ppg_w', 1, 10, 2),
        ModalityConfig('temp_w', 1, 10, 2), ModalityConfig('acc_w', 3, 10, 2),
    ]

    model = build_former_device(num_classes=12, model_dim=32, return_mem=True, modalities=modalities, modal_fusion="cross_atten")
    # print(model)
    logits, aux = model(x, history=history)
    print("logits:", logits.shape)          # expected [8, 12]
    print("mem:", aux.shape)  # [8, 140, model_dim]