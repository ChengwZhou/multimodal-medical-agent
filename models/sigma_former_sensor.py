import math
import sys
import os
from typing import Dict, Optional, List, Tuple
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer_utils import CrossAttentionLayer, TransformerLayer, TransformerBlock, CrossModalAttention
from utils.modality_config import ModalityConfig


# -----------------------------
# Utility Modules
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
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
            delta_x = x
            # Conv1d on diff
            delta_patches = self.conv(delta_x)  # [B, out_ch, L]
            patches = torch.cumsum(delta_patches, dim=-1)  # [B, out_ch, L]

        patches = patches.transpose(1, 2)     # [B, L, out_ch]
        patches = self.norm(patches)
        return patches                  # [B, L, out_ch]  (here out_ch==1)


class AdaptiveSensingMultimodalTransformer(nn.Module):
    """
    Multimodal transformer with adaptive sensing mechanism.
    Each modality has a learnable threshold that determines when to turn off the sensor.
    """

    def __init__(self,
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
                 # Adaptive sensing parameters
                 init_threshold: float = 0.1,
                 skip_steps: int = 5,
                 learnable_skip: bool = False,
                 return_sensing_info: bool = False,
                 modal_fusion: str = "concate"  # "concate" or "cross_atten",
                 ):
        super().__init__()

        if modalities is None:
            modalities = [ModalityConfig(f'm{i}', 1, 10) for i in range(2)]

        self.modalities = modalities
        self.model_dim = model_dim
        self.use_modal_dropout = use_modal_dropout
        self.modal_dropout_p = modal_dropout_p
        self.return_mem = return_mem
        self.return_sensing_info = return_sensing_info
        self.modal_fusion = modal_fusion

        # Adaptive sensing fairseq_signals_modules for each modality
        self.adaptive_sensing = nn.ModuleDict()
        for m in modalities:
            self.adaptive_sensing[m.name] = AdaptiveSensingModule(
                init_threshold=init_threshold,
                skip_steps=skip_steps,
                learnable_skip=learnable_skip
            )

        # Tokenizers for each modality
        self.tokenizers = nn.ModuleDict()
        for m in modalities:
            self.tokenizers[m.name] = ConvTokenizer1D(
                m.in_ch,
                out_ch=model_dim,
                patch_size=m.patch_size
            )

        # Positional encoding
        self.positional = PositionalEncoding(model_dim, max_len=max_len)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Cross-attention layer
        self.cross_attn = CrossAttentionLayer(
            model_dim, nhead,
            dim_feedforward=ff_dim,
            dropout=dropout
        )

        # Cross-Modal attention
        if self.modal_fusion == "cross_atten":
            self.modal_cross_attn = CrossModalAttention(
                model_dim, nhead,
                dim_feedforward=ff_dim,
                dropout=dropout
            )

        # Fusion transformer
        self.fusion = TransformerBlock(
            TransformerLayer(
                d_model=model_dim,
                nhead=nhead,
                dim_feedforward=ff_dim,
                dropout=dropout
            ),
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

    def get_sensing_stats(self) -> Dict[str, Dict[str, float]]:
        """Get current threshold and skip steps for each modality"""
        stats = {}
        for name, module in self.adaptive_sensing.items():
            stats[name] = {
                'threshold': module.get_threshold().item(),
                'skip_steps': module.get_skip_steps()
            }
        return stats

    def forward(self,
                x: torch.Tensor,
                history: Optional[torch.Tensor] = None,
                history_mask: Optional[torch.Tensor] = None,
                ):
        """
        Args:
            x: [B, C=14, T=100] - delta input: x0, x1-x0, x2-x1, ...
            history: [B, 140, model_dim] - optional history memory
            history_mask: [B, 140] - optional mask for history

        Returns:
            logits: [B, num_classes]
            (optional) mem: [B, L_total, model_dim] if return_mem=True
            (optional) sensing_info: Dict with masks and active ratios if return_sensing_info=True
        """
        B = x.size(0)
        C = x.size(1)
        assert C == len(self.modalities), f"expected {len(self.modalities)} channels, got {C}"

        per_mod_tokens = []
        sensing_info = {'masks': [], 'active_st': [], 'active_ratios': []}

        # Process each modality
        for i, m in enumerate(self.modalities):
            # Extract channel
            ch = x[:, i:i+m.in_ch, :]  # [B, m, T]
            # print(f"m{i}: {ch[0][0][:10]}")

            # # Apply adaptive sensing
            masked_ch, mask, active_st, active_ratio = self.adaptive_sensing[m.name](ch)  # [B, 1, T], [B, 1, T]

            # Store sensing info
            if self.return_sensing_info:
                sensing_info['masks'].append(mask)
                sensing_info['active_st'].append(active_st.float().mean())
                active_ratio = mask.float().mean()
                sensing_info['active_ratios'].append(active_ratio)

            # Tokenize masked input
            tokens = self.tokenizers[m.name](masked_ch)  # [B, L, model_dim]

            tokens = self.tokenizers[m.name](ch)
            per_mod_tokens.append(tokens)

        # Optional modal dropout
        per_mod_tokens = self._maybe_modal_dropout(per_mod_tokens)

        # modal cross attention or concatenate tokens along time dimension -> [B, 14*10=140, 1]
        if self.modal_fusion == "cross_atten":
            concat_tokens = self.modal_cross_attn(per_mod_tokens, kv_masks=history_mask)
        else:
            concat_tokens = torch.cat(per_mod_tokens, dim=1)  # [B, L_total, T]
        # print("c", concat_tokens.size())

        # Add positional encoding
        fused = self.positional(concat_tokens)

        # Cross-attention with history if provided
        if history is not None:
            fused = self.cross_attn(fused, history)

        # Fusion transformer
        out = self.fusion(fused)

        # Classification
        cls_out = out.mean(dim=1)
        logits = self.head(cls_out)

        # Prepare return values
        returns = [logits]

        if self.return_mem:
            returns.append(out)

        if self.return_sensing_info:
            returns.append(sensing_info)

        if len(returns) == 1:
            return returns[0]
        else:
            return tuple(returns)


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
        modal_fusion="concate"
):
    """
    Build adaptive sensing multimodal transformer.

    Args:
        num_modal: Number of modalities/sensors
        num_classes: Number of activity classes
        model_dim: Model dimension
        return_mem: Whether to return memory for temporal modeling
        init_threshold: Initial threshold for adaptive sensing
        skip_steps: Number of steps to skip when below threshold (if not learnable)
        learnable_skip: Whether skip_steps should be learnable
        return_sensing_info: Whether to return sensing statistics
    """
    # modalities = [ModalityConfig(f'm{i}', 1, 10) for i in range(num_modal)]
    return AdaptiveSensingMultimodalTransformer(
        num_classes=num_classes,
        model_dim=model_dim,
        nhead=8,
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
        return_sensing_info=return_sensing_info,
        modal_fusion=modal_fusion
    )


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Adaptive Sensing Multimodal Transformer")
    print("=" * 80)

    # Test with fixed skip steps
    print("\n1. Model with fixed skip steps:")
    model_fixed = build_adaptive_sigma_former(
        num_classes=12,
        model_dim=512,
        return_mem=True,
        init_threshold=0.6,
        skip_steps=1,
        learnable_skip=False,
        return_sensing_info=True
    )
    print(model_fixed)

    # x = torch.ones(8, 14, 100)  # [B, 14, T]
    # x = torch.zeros(8, 2, 100)
    # x[:, :, 0] = 1
    x = torch.randn(8, 2, 100)
    print(x.size())
    history = torch.randn(8, 140, 512)  # [B, L_total, model_dim]
    history_mask = torch.randn(8, 140, 512)  # [B, L_total, model_dim]

    logits, mem, sensing_info = model_fixed(x, history=history, history_mask=history_mask)
    print(f"  Logits shape: {logits.shape}")  # [8, 12]
    print(f"  Memory shape: {mem.shape}")  # [8, 140, 32]
    print(f"  Active ratios: {[f'{r:.2%}' for r in sensing_info['active_ratios'][:3]]}...")

    # # Test with learnable skip steps
    # print("\n2. Model with learnable skip steps:")
    # model_learnable = build_adaptive_sigma_former(
    #     num_classes=12,
    #     model_dim=32,
    #     return_mem=False,
    #     init_threshold=0.1,
    #     skip_steps=3,
    #     learnable_skip=True,
    #     return_sensing_info=True
    # )
    #
    # logits, sensing_info = model_learnable(x, history=history, history_mask=history_mask)
    # print(f"  Logits shape: {logits.shape}")
    # print(f"  Active ratios: {[f'{r:.2%}' for r in sensing_info['active_ratios'][:3]]}...")
    #
    # # Show sensing parameters
    # print("\n3. Sensing parameters for first 3 modalities:")
    # stats = model_learnable.get_sensing_stats()
    # for i, (name, params) in enumerate(list(stats.items())[:3]):
    #     print(f"  {name}: threshold={params['threshold']:.4f}, skip_steps={params['skip_steps']}")
    #
    # print("\n" + "=" * 80)
    # print("All tests passed!")
    # print("=" * 80)