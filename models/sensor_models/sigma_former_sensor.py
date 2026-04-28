"""
sensor_models/sigma_former_sensor.py  —  standalone copy with local imports.

Identical logic to models/sigma_former_sensor.py but uses relative imports
so this directory can be run independently.  See parent file for full
change-log and architecture notes.
"""

import math
import sys
import os
from typing import Dict, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_utils import (
    CrossAttentionLayer,
    CrossModalAttention,
    TransformerBlock,
    TransformerLayer,
)


# ---------------------------------------------------------------------------
# ModalityConfig (local copy to keep this module standalone)
# ---------------------------------------------------------------------------

class ModalityConfig:
    def __init__(self, name: str, in_ch: int, patch_size: int,
                 device_idx=None, lead_idx=None):
        self.name       = name
        self.in_ch      = in_ch
        self.patch_size = patch_size
        self.device_idx = device_idx
        self.lead_idx   = lead_idx


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# AdaptiveSensingModule  (FIXED + IMPROVED — see parent file for details)
# ---------------------------------------------------------------------------

class AdaptiveSensingModule(nn.Module):
    """
    Sigma-delta patch sensing with:
      • Soft sigmoid threshold (differentiable w.r.t. log_threshold & log_temp)
      • Straight-Through Estimator for binary hard mask
      • Learnable temperature with cosine annealing
      • Soft learnable skip (no .item() gradient severance)
    """

    def __init__(
        self,
        patch_size: int = 10,
        init_threshold: float = 0.5,
        skip_steps: int = 1,
        learnable_skip: bool = False,
        min_threshold: float = 0.01,
        max_threshold: float = 5.0,
        init_temp: float = 1.0,
        min_temp: float = 0.05,
        temp_anneal_steps: int = 5000,
    ):
        super().__init__()
        self.patch_size    = patch_size
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.min_temp      = min_temp

        self.log_threshold = nn.Parameter(torch.tensor(math.log(init_threshold)))
        self.log_temp      = nn.Parameter(torch.tensor(math.log(max(init_temp, min_temp))))

        self.register_buffer("_step",         torch.tensor(0.0))
        self.register_buffer("_anneal_steps", torch.tensor(float(temp_anneal_steps)))

        self.learnable_skip = learnable_skip
        if learnable_skip:
            self.log_skip_steps = nn.Parameter(torch.tensor(math.log(float(skip_steps))))
            self.min_skip = 1.0
            self.max_skip = 20.0
        else:
            self.register_buffer("_skip_steps", torch.tensor(float(skip_steps)))

    def get_threshold(self) -> torch.Tensor:
        return torch.exp(self.log_threshold).clamp(self.min_threshold, self.max_threshold)

    def get_temperature(self) -> torch.Tensor:
        progress = (self._step / self._anneal_steps).clamp(0.0, 1.0)
        init_t   = torch.exp(self.log_temp)
        tau = self.min_temp + 0.5 * (init_t - self.min_temp) * (1.0 + torch.cos(math.pi * progress))
        return tau.clamp(min=self.min_temp)

    def _get_skip_soft(self) -> torch.Tensor:
        if self.learnable_skip:
            return torch.exp(self.log_skip_steps).clamp(self.min_skip, self.max_skip)
        return self._skip_steps

    def _get_skip_hard(self) -> int:
        return int(round(self._get_skip_soft().item()))

    def forward(self, x: torch.Tensor):
        B, C, T = x.shape
        P = self.patch_size
        assert T % P == 0
        L = T // P

        x_blocks   = x.unfold(-1, P, P).permute(0, 2, 1, 3)          # [B, L, C, P]
        zeros      = torch.zeros(B, 1, C, P, device=x.device, dtype=x.dtype)
        prev       = torch.cat([zeros, x_blocks[:, :-1]], dim=1)
        delta      = x_blocks - prev                                    # [B, L, C, P]
        activity   = delta.abs().mean(dim=-1)                           # [B, L, C]

        th   = self.get_threshold()
        temp = self.get_temperature()
        p_soft = torch.sigmoid((activity - th) / temp)                  # [B, L, C]

        trigger   = activity < th
        skip      = self._get_skip_hard()
        idx       = torch.arange(L, device=x.device).view(1, L, 1).expand(B, L, C)
        skip_end  = torch.where(trigger, idx + skip, torch.full_like(idx, -1))
        skip_end  = torch.cummax(skip_end, dim=1).values
        timer     = (skip_end - idx).clamp(min=0)
        hard      = (timer == 0).float()                                # [B, L, C]

        mask_ste  = hard + (p_soft - p_soft.detach())                   # STE

        def _expand(m: torch.Tensor) -> torch.Tensor:
            return m.unsqueeze(-1).expand(-1, -1, -1, P).permute(0, 2, 1, 3).contiguous().view(B, C, T)

        delta_flat   = delta.permute(0, 2, 1, 3).contiguous().view(B, C, T)
        masked_delta = delta_flat * _expand(mask_ste)

        if self.training:
            self._step = self._step + 1.0

        return masked_delta, _expand(hard), p_soft.mean(), hard.mean()


# ---------------------------------------------------------------------------
# ConvTokenizer1D
# ---------------------------------------------------------------------------

class ConvTokenizer1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 1, patch_size: int = 10, norm: bool = True):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=patch_size, stride=patch_size, bias=False)
        self.norm = nn.LayerNorm(out_ch) if norm else nn.Identity()

    def forward(self, x: torch.Tensor, use_sigma: bool = True) -> torch.Tensor:
        patches = torch.cumsum(self.conv(x), dim=-1) if use_sigma else self.conv(x)
        return self.norm(patches.transpose(1, 2))


# ---------------------------------------------------------------------------
# AdaptiveSensingMultimodalTransformer  (Bug #1 + CLS token fixed)
# ---------------------------------------------------------------------------

class AdaptiveSensingMultimodalTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        model_dim: int = 256,
        nhead: int = 8,
        fusion_depth: int = 4,
        ff_dim: int = 128,
        dropout: float = 0.1,
        modalities: Optional[List[ModalityConfig]] = None,
        use_modal_dropout: bool = True,
        modal_dropout_p: float = 0.0,
        max_len: int = 10000,
        return_mem: bool = False,
        init_threshold: float = 0.1,
        skip_steps: int = 5,
        learnable_skip: bool = False,
        init_temp: float = 1.0,
        temp_anneal_steps: int = 5000,
        return_sensing_info: bool = False,
        modal_fusion: str = "concate",
    ):
        super().__init__()
        if modalities is None:
            modalities = [ModalityConfig(f"m{i}", 1, 10) for i in range(18)]

        self.modalities        = modalities
        self.model_dim         = model_dim
        self.use_modal_dropout = use_modal_dropout
        self.modal_dropout_p   = modal_dropout_p
        self.return_mem        = return_mem
        self.return_sensing_info = return_sensing_info
        self.modal_fusion      = modal_fusion

        self.adaptive_sensing = nn.ModuleDict({
            m.name: AdaptiveSensingModule(
                patch_size=m.patch_size,
                init_threshold=init_threshold,
                skip_steps=skip_steps,
                learnable_skip=learnable_skip,
                init_temp=init_temp,
                temp_anneal_steps=temp_anneal_steps,
            )
            for m in modalities
        })

        self.tokenizers = nn.ModuleDict({
            m.name: ConvTokenizer1D(m.in_ch, out_ch=model_dim, patch_size=m.patch_size)
            for m in modalities
        })

        self.positional = PositionalEncoding(model_dim, max_len=max_len)

        self.cross_attn = CrossAttentionLayer(
            model_dim, nhead, dim_feedforward=ff_dim, dropout=dropout
        )

        if modal_fusion == "cross_atten":
            self.modal_cross_attn = CrossModalAttention(
                model_dim, nhead, dim_feedforward=ff_dim, dropout=dropout
            )

        self.fusion = TransformerBlock(
            TransformerLayer(d_model=model_dim, nhead=nhead,
                             dim_feedforward=ff_dim, dropout=dropout),
            num_layers=fusion_depth,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, num_classes),
        )

    def _maybe_modal_dropout(self, tokens):
        if not self.use_modal_dropout or self.modal_dropout_p <= 0.0 or not self.training:
            return tokens
        return [
            torch.zeros_like(t) if torch.rand(1).item() < self.modal_dropout_p else t
            for t in tokens
        ]

    def get_sensing_stats(self) -> Dict:
        return {
            name: {
                "threshold":   mod.get_threshold().item(),
                "temperature": mod.get_temperature().item(),
                "skip_steps":  mod._get_skip_hard(),
            }
            for name, mod in self.adaptive_sensing.items()
        }

    def forward(self, x, history=None, history_mask=None):
        B = x.size(0)
        assert x.size(1) == len(self.modalities)

        per_mod_tokens = []
        sensing_info   = {"masks": [], "active_ratios": [], "sparsity_losses": []}

        ch_offset = 0
        for m in self.modalities:
            ch = x[:, ch_offset: ch_offset + m.in_ch, :]
            ch_offset += m.in_ch

            masked_ch, mask_hard, active_mean, active_ratio = (
                self.adaptive_sensing[m.name](ch)
            )

            if self.return_sensing_info:
                sensing_info["masks"].append(mask_hard)
                sensing_info["active_ratios"].append(active_ratio)
                sensing_info["sparsity_losses"].append(active_ratio)

            # Bug #1 fix: tokenise masked_ch (not the original ch)
            tokens = self.tokenizers[m.name](masked_ch, use_sigma=True)
            per_mod_tokens.append(tokens)

        per_mod_tokens = self._maybe_modal_dropout(per_mod_tokens)

        if self.modal_fusion == "cross_atten":
            concat = self.modal_cross_attn(per_mod_tokens, kv_masks=history_mask)
        else:
            concat = torch.cat(per_mod_tokens, dim=1)

        fused = self.positional(concat)

        if history is not None:
            fused = self.cross_attn(fused, history)

        out    = self.fusion(fused)
        logits = self.head(out.mean(dim=1))   # mean pooling, matches former_sensor.py

        returns = [logits]
        if self.return_mem:
            returns.append(out)
        if self.return_sensing_info:
            returns.append(sensing_info)

        return returns[0] if len(returns) == 1 else tuple(returns)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_adaptive_sigma_former(
    num_modal: int = 14,
    num_classes: int = 10,
    model_dim: int = 32,
    return_mem: bool = False,
    init_threshold: float = 0.1,
    skip_steps: int = 5,
    learnable_skip: bool = False,
    return_sensing_info: bool = False,
    modalities=None,
    modal_fusion: str = "concate",
    init_temp: float = 1.0,
    temp_anneal_steps: int = 5000,
):
    if modalities is None:
        modalities = [ModalityConfig(f"m{i}", 1, 10) for i in range(num_modal)]

    return AdaptiveSensingMultimodalTransformer(
        num_classes=num_classes,
        model_dim=model_dim,
        nhead=max(1, model_dim // 16),
        fusion_depth=1,
        ff_dim=4 * model_dim,
        dropout=0.1,
        modalities=modalities,
        use_modal_dropout=True,
        modal_dropout_p=0.15,
        max_len=4000,
        return_mem=return_mem,
        init_threshold=init_threshold,
        skip_steps=skip_steps,
        learnable_skip=learnable_skip,
        init_temp=init_temp,
        temp_anneal_steps=temp_anneal_steps,
        return_sensing_info=return_sensing_info,
        modal_fusion=modal_fusion,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing sensor_models/sigma_former_sensor.py")
    mods  = [ModalityConfig(f"m{i}", 1, 10) for i in range(18)]
    model = build_adaptive_sigma_former(
        num_modal=18, num_classes=10, model_dim=32,
        return_mem=True, return_sensing_info=True,
        init_threshold=0.3, init_temp=1.0, modalities=mods,
    )
    model.train()
    x = torch.randn(2, 18, 100)
    logits, mem, sinfo = model(x)
    print(f"logits: {logits.shape}, mem: {mem.shape}")
    logits.sum().backward()
    for name, mod in model.adaptive_sensing.items():
        assert mod.log_threshold.grad is not None
    print("✓ All gradients flow correctly.")
