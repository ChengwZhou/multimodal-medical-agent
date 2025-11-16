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

class BypassMaskGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, delta_x, mask_pixel, prev_delta_x):
        ctx.save_for_backward(mask_pixel)
        ctx.prev_delta_x = prev_delta_x
        return delta_x * mask_pixel

    @staticmethod
    def backward(ctx, grad_output):
        mask_pixel, = ctx.saved_tensors
        prev_delta_x = ctx.prev_delta_x

        # Normal gradient：grad_delta_x = grad_output * mask_pixel
        grad_delta_x = grad_output * mask_pixel

        # The masked portion has its gradient “bypassed” to the previous patch.
        bypass_mask = (mask_pixel < 0.5).float()
        grad_bypass = grad_output * bypass_mask

        # Add the bypass gradient to the previous patch (dimensions must be aligned).
        if prev_delta_x is not None:
            grad_prev = grad_bypass  # [B, C, T]，Align to prev patch
            # If patches do not overlap, they can be added directly; if they overlap, they must be unfolded and aligned.
        else:
            grad_prev = None

        return grad_delta_x, None, grad_prev


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


class AdaptiveSensingModule(nn.Module):
    """
    Fully vectorized adaptive sensing (no Python loops over B or T).
    Skip is now performed **per patch** instead of per pixel.
    """

    def __init__(self,
                 patch_size: int = 10,
                 init_threshold: float = 0.5,
                 skip_steps: int = 1,          # now: number of patches to skip
                 learnable_skip: bool = False,
                 min_threshold: float = 0.01,
                 max_threshold: float = 5.0):
        super().__init__()
        self.patch_size = patch_size
        self.log_threshold = nn.Parameter(torch.tensor(math.log(init_threshold)))
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        self.learnable_skip = learnable_skip
        if learnable_skip:
            self.log_skip_steps = nn.Parameter(torch.tensor(math.log(float(skip_steps))))
            self.min_skip = 1
            self.max_skip = 20
        else:
            self.skip_steps = skip_steps

    @staticmethod
    def _unfold(x, size, step):
        try:
            # PyTorch >= 2.1
            return x.unfold(dimension=-1, size=size, step=step)
        except TypeError:
            # PyTorch <= 2.0
            return x.unfold(dimension=-1, size=size, stride=step)

    def get_threshold(self) -> torch.Tensor:
        th = torch.exp(self.log_threshold)
        return torch.clamp(th, self.min_threshold, self.max_threshold)

    def get_skip_steps(self) -> int:
        """Return the *integer* number of patches to skip."""
        if self.learnable_skip:
            s = torch.exp(self.log_skip_steps)
            s = torch.clamp(s, self.min_skip, self.max_skip)
            return int(torch.round(s).item())
        return self.skip_steps

    def forward(self, x: torch.Tensor):
        B, C, T = x.shape
        P = self.patch_size
        assert T % P == 0, f"T={T} must be divisible by patch_size={P}"
        L = T // P

        x_blocks = self._unfold(x, size=P, step=P)  # [B, P, L]
        x_blocks = x_blocks.transpose(1, 2).contiguous()  # [B, L, P]
        x_blocks = x_blocks.view(B, L, C, P)  # [B, L, C, P]

        # patch-level difference
        zeros = torch.zeros((B, 1, C, P), device=x.device, dtype=x.dtype)
        prev_blocks = torch.cat([zeros, x_blocks[:, :-1]], dim=1)  # [B, L, C, P]
        delta_blocks = x_blocks - prev_blocks  # [B, L, C, P]
        # # reshape fold to [B, C_in, T]
        delta_x = delta_blocks.transpose(1, 2).contiguous()  # [B, L, C_in, P] → [B, C_in, L, P]
        delta_x = delta_x.view(B, C, -1)  # [B, C_in, T]

        # threshold
        th = self.get_threshold()  # scalar
        abs_delta = torch.abs(delta_blocks)  # [B, L, C, P]

        # Calculate patch activity（mean over patch）
        patch_activity = abs_delta.mean(dim=-1)  # [B, L, C]

        # hard trigger: entire patch average < th → trigger
        trigger = patch_activity < th  # [B, L, C]  bool

        # skip logic (per patch, per channel)
        skip = self.get_skip_steps()  # int

        # patch index: [0,1,...,L-1]
        patch_idx = torch.arange(L, device=x.device)  # [L]
        patch_idx = patch_idx.view(1, L, 1).expand(B, L, C)  # [B, L, C]

        # skip_end: trigger → current_idx + skip
        skip_end = torch.where(
            trigger,
            patch_idx + skip,
            torch.full_like(patch_idx, -1)
        )  # [B, L, C]

        # cummax propagates along the L dimension (each (B,C) is independent)
        skip_end = torch.cummax(skip_end, dim=1).values  # [B, L, C]

        timer = torch.clamp(skip_end - patch_idx, min=0)  # [B, L, C]
        active_hard = (timer == 0).float()  # [B, L, C]

        active = active_hard  # [B, L, C]

        # expand to pixel level: [B, L, C] → [B, C, L, P] → [B, C, T]
        active_expanded = active_hard.unsqueeze(-1).expand(-1, -1, -1, P)  # [B, L, C, P]
        # [B, L, C, P] → [B, C, L, P] → [B, C, T]
        active_expanded = active_expanded.permute(0, 2, 1, 3).contiguous()  # [B, C, L, P]
        mask_pixel = active_expanded.view(B, C, T)  # [B, C, T]

        # Save prev_delta_x (pixel-level)
        prev_delta_x = prev_blocks.permute(0, 2, 1, 3).contiguous().view(B, C, T)

        # Using Custom Gradients
        masked_delta = BypassMaskGrad.apply(delta_x, mask_pixel, prev_delta_x)

        # hard mask (0/1) - Apply the same dimensionality transformation
        active_hard_expanded = active_hard.unsqueeze(-1).expand(-1, -1, -1, P)  # [B, L, C, P]
        active_hard_expanded = active_hard_expanded.permute(0, 2, 1, 3).contiguous()  # [B, C, L, P]
        mask_hard = active_hard_expanded.view(B, C, T)  # [B, C, T]

        # statistics
        active_mean = active.mean()
        active_ratio = active_hard.mean()

        # --------------------------------------------------------------- #
        # Debug print
        # --------------------------------------------------------------- #
        # print(f"active_hard shape: {active_hard.shape}")  # [B, L, C]
        # print(f"mask_pixel shape: {mask_pixel.shape}")  # [B, C, T]
        # print(f"mask_hard shape: {mask_hard.shape}")  # [B, C, T]
        # print(f"masked_delta shape: {masked_delta.shape}")  # [B, C, T]
        # print(f"trigger[0]:", trigger[0])  # [L, C]
        # print(f"active_hard[0]:", active_hard[0])  # [L, C]
        # print("masked_delta[0,0]:", masked_delta[0, 0])

        return masked_delta, mask_hard, active_mean, active_ratio

class ConvTokenizer1D(nn.Module):
    """
    Conv tokenizer that outputs a scalar feature per patch (out_channels=1).
    This matches the user's request to concatenate tokens into shape [B, L_total, 1].
    """
    def __init__(self, in_ch: int, out_ch: int = 1, patch_size: int = 10, norm: bool = True):
        super().__init__()
        # produce one scalar per patch per channel
        self.patch_size = patch_size
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=patch_size, stride=patch_size, bias=False)
        self.norm = nn.LayerNorm(out_ch) if norm else nn.Identity()

    def forward(self, x: torch.Tensor, use_patch_block_sigma: bool = True) -> torch.Tensor:
        # x: [B, C_in, T] (but we will pass single-channel per call: C_in==1)
        B, C_in, T = x.size()
        P = self.patch_size
        assert T % P == 0, "T must be divisible by patch_size"

        if not use_patch_block_sigma:
            patches = self.conv(x)  # [B, out_ch, L]
        else:
            # # x_unfold: [B, C_in, L, P]
            # x_blocks = x.unfold(dimension=-1, size=P, step=P).transpose(1, 2)  # [B, L, C_in, P]
            #
            # # Apply difference
            # zeros = torch.zeros((B, 1, C_in, P), device=x.device, dtype=x.dtype)
            # prev_blocks = torch.cat([zeros, x_blocks[:, :-1]], dim=1)  # [B, L, C_in, P]
            # delta_blocks = x_blocks - prev_blocks  # [B, L, C_in, P]
            #
            # # reshape fold to [B, C_in, T]
            # delta_x = delta_blocks.transpose(1, 2).contiguous()  # [B, L, C_in, P] → [B, C_in, L, P]
            # delta_x = delta_x.view(B, C_in, -1)  # [B, C_in, T]

            delta_x = x
            # Conv1d on diff
            delta_patches = self.conv(delta_x)  # [B, out_ch, L]
            patches = torch.cumsum(delta_patches, dim=-1)  # [B, out_ch, L]


            # # === test ===
            # original_patches = self.conv(x)
            # print("no_SD:", original_patches[0, 0, :10])
            # print("SD:   ", patches[0, 0, :10])
            # print("Max diff:", (original_patches - patches).abs().max().item())  # ~0

        patches = patches.transpose(1, 2)     # [B, L, out_ch]
        patches = self.norm(patches)
        return patches                  # [B, L, out_ch]  (here out_ch==1)


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
        # Adaptive sensing parameters
        init_threshold: float = 0.1,
        skip_steps: int = 5,
        learnable_skip: bool = False,
        return_sensing_info: bool = False,
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
        self.return_sensing_info = return_sensing_info
        self.modal_fusion = modal_fusion

        # Group modalities by device_idx
        sensor_mods = defaultdict(list)
        for m in self.modalities:
            sensor_mods[m.device_idx].append(m)

        # Tokenizers: one per sensor
        self.sensor_tokenizers = nn.ModuleDict()
        self.adaptive_sensing = nn.ModuleDict() # Adaptive sensing fairseq_signals_modules for each modality
        self.sensor_ranges = {}
        start_ch = 0
        for device_idx in sorted(sensor_mods.keys()):
            mods = sensor_mods[device_idx]
            total_in_ch = sum(m.in_ch for m in mods)
            patch_size = mods[0].patch_size  # assume all same per sensor
            self.sensor_tokenizers[str(device_idx)] = ConvTokenizer1D(total_in_ch, out_ch=model_dim, patch_size=patch_size)
            self.adaptive_sensing[str(device_idx)] = AdaptiveSensingModule(
                patch_size=patch_size,
                init_threshold=init_threshold,
                skip_steps=skip_steps,
                learnable_skip=learnable_skip
            )

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
        sensing_info = {'masks': [], 'active_st': [], 'active_ratios': []}

        # iterate sensors and tokenize each
        for device_idx in sorted(self.sensor_ranges.keys()):
            start, end = self.sensor_ranges[device_idx]
            ch = x[:, start:end, :]                # [B, total_in_ch, T]
            # print(str(device_idx))
            # Apply adaptive sensing
            masked_ch, mask, active_st, active_ratio = self.adaptive_sensing[str(device_idx)](ch)  # [B, 1, T], [B, 1, T]
            #
            # mask = torch.tensor(0).to(x.device)
            # active_st = torch.tensor(0).to(x.device)
            # active_ratio = torch.tensor(0).to(x.device)
            # # Store sensing info
            if self.return_sensing_info:
                sensing_info['masks'].append(mask)
                sensing_info['active_st'].append(active_st.float().mean())
                active_ratio = mask.float().mean()
                sensing_info['active_ratios'].append(active_ratio)

            # Tokenize masked input
            t = self.sensor_tokenizers[str(device_idx)](masked_ch)    # [B, L=10, out_ch=model_dim]

            # original_t = self.sensor_tokenizers[str(device_idx)](ch, use_patch_block_sigma=False)
            # print("no_SD:", original_t[0, :10, 0])
            # print("SD:   ", t[0, :10, 0])

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

        returns = [logits]

        if self.return_mem:
            returns.append(out)

        if self.return_sensing_info:
            returns.append(sensing_info)

        if len(returns) == 1:
            return returns[0]
        else:
            return tuple(returns)


def build_former_device(num_modal=14,
                        num_classes=10,
                        model_dim=32,
                        return_mem=False,
                        init_threshold: float = 0.1,
                        skip_steps: int = 5,
                        learnable_skip: bool = False,
                        return_sensing_info: bool = False,
                        modalities=None,
                        modal_fusion="concate"):
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
        init_threshold=init_threshold,
        skip_steps=skip_steps,
        learnable_skip=learnable_skip,
        return_sensing_info=return_sensing_info,
        modal_fusion=modal_fusion
    )


if __name__ == "__main__":
    # test run according to your spec
    x = torch.randn(8, 14, 100)           # [B, 14, 100]
    # history in the form [B, 1, 140]
    history = torch.randn(8, 140, 32)

    modalities = [
        ModalityConfig('acc_c', 3, 10, 0), ModalityConfig('ecg_g_c', 1, 10, 0),
        ModalityConfig('ecg_t_c', 1, 10, 0),
        ModalityConfig('eda_f', 1, 10, 1), ModalityConfig('ppg_f', 1, 10, 1),
        ModalityConfig('emg_f', 1, 10, 1),
        ModalityConfig('eda_w', 1, 10, 2), ModalityConfig('ppg_w', 1, 10, 2),
        ModalityConfig('temp_w', 1, 10, 2), ModalityConfig('acc_w', 3, 10, 2),
    ]

    model = build_former_device(num_classes=12, model_dim=32, return_mem=True, modalities=modalities,
                                modal_fusion="cross_atten", init_threshold=0.6, skip_steps=1)
    # print(model)
    logits, aux = model(x, history=history)
    print("logits:", logits.shape)          # expected [8, 12]
    print("mem:", aux.shape)  # [8, 140, model_dim]