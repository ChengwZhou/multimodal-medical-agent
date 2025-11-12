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


# class AdaptiveSensingModule(nn.Module):
#     """
#     Adaptive sensing module with learnable threshold for each modality.
#     When |delta_input| < threshold, the sensor is turned off for skip_steps.
#
#     Args:
#         init_threshold: Initial threshold value
#         skip_steps: Number of timesteps to skip when threshold is not met (can be int or 'learnable')
#         min_threshold: Minimum allowed threshold value
#         max_threshold: Maximum allowed threshold value
#     """
#
#     def __init__(self,
#                  init_threshold: float = 0.1,
#                  skip_steps: int = 5,
#                  learnable_skip: bool = False,
#                  min_threshold: float = 0.01,
#                  max_threshold: float = 1.0):
#         super().__init__()
#
#         # Learnable threshold (in log space for numerical stability)
#         self.log_threshold = nn.Parameter(torch.tensor(math.log(init_threshold)))
#         self.min_threshold = min_threshold
#         self.max_threshold = max_threshold
#
#         # Skip steps configuration
#         self.learnable_skip = learnable_skip
#         if learnable_skip:
#             # Use log space and round to nearest integer during forward
#             self.log_skip_steps = nn.Parameter(torch.tensor(math.log(float(skip_steps))))
#             self.min_skip = 1
#             self.max_skip = 20
#         else:
#             self.skip_steps = skip_steps
#
#     def get_threshold(self) -> torch.Tensor:
#         """Get the current threshold value with clamping"""
#         threshold = torch.exp(self.log_threshold)
#         return torch.clamp(threshold, self.min_threshold, self.max_threshold)
#
#     def get_skip_steps(self) -> int:
#         """Get the current skip steps value"""
#         if self.learnable_skip:
#             skip = torch.exp(self.log_skip_steps)
#             skip = torch.clamp(skip, self.min_skip, self.max_skip)
#             return int(torch.round(skip).item())
#         else:
#             return self.skip_steps
#
#     def forward(self, delta_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         B, C, T = delta_input.shape
#         assert C == 1
#         x = delta_input.squeeze(1)  # [B, T] ← delta = xt - x{t-1}
#
#         threshold = self.get_threshold()
#         skip_steps = self.get_skip_steps()
#
#         # Initialize per-batch state
#         timer = torch.zeros(B, dtype=torch.long, device=x.device)  # 剩余关闭步数
#         mask = torch.zeros_like(x)
#         masked = torch.zeros_like(x)  # ← 输出 masked delta
#
#         for t in range(T):
#             curr = x[:, t]  # 当前 delta
#             is_on = timer <= 0  # 是否可以采样
#
#             # === 关键：用 |curr| 判断是否触发关闭 ===
#             trigger_off = is_on & (torch.abs(curr) < threshold)
#             # update timer
#             timer = torch.where(
#                 trigger_off,
#                 torch.full_like(timer, skip_steps),  # 触发 → 关闭 skip_steps
#                 torch.clamp(timer - 1, min=0)  # 否则倒计时
#             )
#             # decide to active
#             active = is_on & ~trigger_off
#             mask[:, t] = active.float()
#
#             masked[:, t] = torch.where(active, curr, torch.zeros_like(curr))
#
#         active_ratio = mask.float().mean()
#         return masked.unsqueeze(1), mask.unsqueeze(1), active_ratio


class AdaptiveSensingModule(nn.Module):
    """
    Fully vectorized adaptive sensing (no Python loops over B or T).
    """

    def __init__(self,
                 init_threshold: float = 0.1,
                 skip_steps: int = 5,
                 learnable_skip: bool = False,
                 min_threshold: float = 0.01,
                 max_threshold: float = 5):
        super().__init__()
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

    # --------------------------------------------------------------------- #
    #  工具函数
    # --------------------------------------------------------------------- #
    def get_threshold(self) -> torch.Tensor:
        th = torch.exp(self.log_threshold)
        return torch.clamp(th, self.min_threshold, self.max_threshold)

    def get_skip_steps(self) -> int:
        if self.learnable_skip:
            s = torch.exp(self.log_skip_steps)
            s = torch.clamp(s, self.min_skip, self.max_skip)
            return int(torch.round(s).item())
        return self.skip_steps

    def forward(self, delta_input: torch.Tensor):
        B, C, T = delta_input.shape
        assert C == 1
        x = delta_input.squeeze(1)  # [B, T]

        th = self.get_threshold()
        skip = self.get_skip_steps()

        # 1. 计算软权重（用于梯度）
        abs_x = torch.abs(x)
        # 使用 sigmoid 让阈值附近有平滑的梯度
        soft_weight = torch.sigmoid((abs_x - th) / (th * 0.1))  # 0.1 是温度参数

        # 2. 计算硬触发（用于前向传播）
        trigger = abs_x < th

        # 3. 传播 skip_end
        t = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        skip_end = torch.where(trigger, t + skip, torch.full_like(t, -1))
        skip_end = torch.cummax(skip_end, dim=1).values

        # 4. 硬激活
        timer = torch.clamp(skip_end - t, min=0)
        active_hard = (timer == 0).float()

        # 5. Straight-Through Estimator
        active = active_hard + soft_weight - soft_weight.detach()

        # 6. 应用 mask
        masked_delta = x * active

        mask = active_hard
        active_ratio = mask.mean()

        return masked_delta.unsqueeze(1), mask.unsqueeze(1), active_ratio

        # B, C, T = delta_input.shape
        # assert C == 1
        # x = delta_input.squeeze(1)
        #
        # th = self.get_threshold()
        # skip = self.get_skip_steps()
        #
        # trigger = torch.abs(x) < th
        # t = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        # skip_end = torch.where(trigger, t + skip, torch.full_like(t, -1))
        # skip_end = torch.cummax(skip_end, dim=1).values
        # timer = torch.clamp(skip_end - t, min=0)
        # active = timer == 0
        #
        # masked_delta = x * active.float()  # ← 仍是 delta！
        # mask = active.float()
        # active_ratio = mask.mean()
        #
        # return masked_delta.unsqueeze(1), mask.unsqueeze(1), active_ratio


class ConvTokenizer1D(nn.Module):
    """
    Conv tokenizer that outputs features per patch.
    Converts differential features back to cumulative features after conv.
    """

    def __init__(self, in_ch: int, out_ch: int = 1, patch_size: int = 10, norm: bool = True):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=patch_size, stride=patch_size, bias=False)
        self.norm = nn.LayerNorm(out_ch) if norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, T] - differential features
        Returns:
            [B, L, out_ch] - cumulative features after conv and cumsum
        """
        x = self.conv(x)  # [B, out_ch, L]
        x = x.transpose(1, 2)  # [B, L, out_ch]

        # Convert differential to cumulative
        x = torch.cumsum(x, dim=1)

        x = self.norm(x)
        return x


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
            modalities = [ModalityConfig(f'm{i}', 1, 10) for i in range(14)]

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
        sensing_info = {'masks': [], 'active_ratios': []}

        # Process each modality
        for i, m in enumerate(self.modalities):
            # Extract channel
            ch = x[:, i:i+m.in_ch, :]  # [B, m, T]
            # print(f"m{i}: {ch[0][0][:10]}")

            # Apply adaptive sensing
            masked_ch, mask, active_ratio = self.adaptive_sensing[m.name](ch)  # [B, 1, T], [B, 1, T]

            # Store sensing info
            if self.return_sensing_info:
                sensing_info['masks'].append(mask)
                active_ratio = mask.float().mean()
                sensing_info['active_ratios'].append(active_ratio)

            # Tokenize masked input
            tokens = self.tokenizers[m.name](masked_ch)  # [B, L, model_dim]
            per_mod_tokens.append(tokens)

        # Optional modal dropout
        per_mod_tokens = self._maybe_modal_dropout(per_mod_tokens)

        # modal cross attention or concatenate tokens along time dimension -> [B, 14*10=140, 1]
        if self.modal_fusion == "cross_atten":
            concat_tokens = self.modal_cross_attn(per_mod_tokens, kv_masks=history_mask)
        else:
            concat_tokens = torch.cat(per_mod_tokens, dim=1)  # [B, L_total, 1]


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
        max_len=500,
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
        model_dim=32,
        return_mem=True,
        init_threshold=0.4,
        skip_steps=5,
        learnable_skip=False,
        return_sensing_info=True
    )

    x = torch.randn(8, 14, 100)  # [B, 14, T]
    history = torch.randn(8, 140, 32)  # [B, L_total, model_dim]
    history_mask = torch.randn(8, 140, 32)  # [B, L_total, model_dim]

    logits, mem, sensing_info = model_fixed(x, history=history, history_mask=history_mask)
    print(f"  Logits shape: {logits.shape}")  # [8, 12]
    print(f"  Memory shape: {mem.shape}")  # [8, 140, 32]
    print(f"  Active ratios: {[f'{r:.2%}' for r in sensing_info['active_ratios'][:3]]}...")

    # Test with learnable skip steps
    print("\n2. Model with learnable skip steps:")
    model_learnable = build_adaptive_sigma_former(
        num_classes=12,
        model_dim=32,
        return_mem=False,
        init_threshold=0.1,
        skip_steps=3,
        learnable_skip=True,
        return_sensing_info=True
    )

    logits, sensing_info = model_learnable(x, history=history, history_mask=history_mask)
    print(f"  Logits shape: {logits.shape}")
    print(f"  Active ratios: {[f'{r:.2%}' for r in sensing_info['active_ratios'][:3]]}...")

    # Show sensing parameters
    print("\n3. Sensing parameters for first 3 modalities:")
    stats = model_learnable.get_sensing_stats()
    for i, (name, params) in enumerate(list(stats.items())[:3]):
        print(f"  {name}: threshold={params['threshold']:.4f}, skip_steps={params['skip_steps']}")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)