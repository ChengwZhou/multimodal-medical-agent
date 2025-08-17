import math
from typing import Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_utils import CrossAttentionLayer, TransformerLayer, TransformerBlock


# -----------------------------
# Utility Modules
# -----------------------------

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for 1D token sequences."""
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
    def __init__(self, in_ch: int, embed_dim: int, patch_size: int = 16, norm: bool = True):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.norm = nn.LayerNorm(embed_dim) if norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)          # [B, D, L]
        x = x.transpose(1, 2)     # [B, L, D]
        x = self.norm(x)
        return x


# -----------------------------
# Multimodal Model
# -----------------------------

class ModalityConfig:
    def __init__(self, name: str, in_ch: int, patch_size: int):
        self.name = name
        self.in_ch = in_ch
        self.patch_size = patch_size


class MultimodalActivityTransformer(nn.Module):
    """
    Multimodal Activity Classification Transformer.

    This model encodes multiple physiological modalities into token sequences,
    enriches them with modality embeddings and positional encodings, and fuses
    them with a Transformer encoder. Optionally, it incorporates historical
    context via cross-attention before classification.

    Design:
        - Each modality -> ConvTokenizer1D -> PositionalEncoding + ModalityEmbedding
        - Concatenate all modality tokens + CLS token
        - (Optional) CrossAttention: current tokens (Q) attend to history tokens (K,V)
        - Fusion Transformer to integrate multi-modal interactions
        - Classification head on CLS token

    Args:
        num_classes (int): Number of activity classes.
        model_dim (int): Hidden dimension of Transformer embeddings. Default=256.
        nhead (int): Number of attention heads. Default=8.
        fusion_depth (int): Number of Transformer layers in the fusion encoder. Default=4.
        ff_dim (int): Feedforward hidden dimension. Default=1024.
        dropout (float): Dropout rate. Default=0.1.
        modalities (List[ModalityConfig]): Configurations for input modalities.
        use_modal_dropout (bool): Whether to randomly drop modalities during training.
        modal_dropout_p (float): Dropout probability for each modality. Default=0.0.
        max_len (int): Maximum sequence length for positional encoding. Default=10000.
        history_dim (int, optional): Dimension of input history features.
        cross_attn_layers (int): Number of cross-attention layers. Default=1.
    """

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
        history_dim: Optional[int] = None,
        cross_attn_layers: int = 1,
    ):
        super().__init__()

        if modalities is None:
            modalities = [
                ModalityConfig('ecg', 1, 32),
                ModalityConfig('ppg', 1, 32),
                ModalityConfig('eda', 1, 16),
                ModalityConfig('emg', 1, 16),
                ModalityConfig('temp', 1, 16),
                ModalityConfig('c_acc', 3, 16),
                ModalityConfig('w_acc', 3, 16),
            ]
        self.modalities = modalities
        self.model_dim = model_dim
        self.use_modal_dropout = use_modal_dropout
        self.modal_dropout_p = modal_dropout_p

        # Per-modality tokenizers, embeddings, positional encodings
        self.tokenizers = nn.ModuleDict()
        self.positional = nn.ModuleDict()
        self.modality_embeddings = nn.ParameterDict()
        for m in modalities:
            self.tokenizers[m.name] = ConvTokenizer1D(m.in_ch, model_dim, patch_size=m.patch_size)
            self.positional[m.name] = PositionalEncoding(model_dim, max_len=max_len)
            self.modality_embeddings[m.name] = nn.Parameter(torch.randn(1, 1, model_dim))

        # Fusion CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Optional history cross-attention (applied BEFORE fusion)
        self.history_proj = None
        if history_dim is not None:
            self.history_proj = nn.Linear(history_dim, model_dim)
        self.cross_attn = nn.ModuleList([
            CrossAttentionLayer(model_dim, nhead, dim_feedforward=ff_dim, dropout=dropout)
            for _ in range(cross_attn_layers)
        ])

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

    def _maybe_modal_dropout(self, x_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Randomly drop some modalities during training."""
        if not self.use_modal_dropout or self.modal_dropout_p <= 0.0 or not self.training:
            return x_tokens
        out = {}
        for k, v in x_tokens.items():
            if torch.rand(1).item() < self.modal_dropout_p:
                out[k] = torch.zeros_like(v)
            else:
                out[k] = v
        return out

    def forward(self,
        x_dict: Dict[str, torch.Tensor],
        history: Optional[torch.Tensor] = None,
        history_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass.

        Args:
            x_dict (dict[str, Tensor]): Dictionary of modality -> [B, C_in, T].
            history (Tensor, optional): Historical memory tokens [B, L_h, H].
            history_mask (Tensor, optional): Boolean mask for history tokens [B, L_h].

        Returns:
            logits (Tensor): Classification logits [B, num_classes].
            aux (dict): Extra outputs, e.g. fused tokens.
        """
        B = None
        tokens = []

        for m in self.modalities:
            if m.name not in x_dict:
                raise KeyError(f"Missing modality '{m.name}' in x_dict")
            x = x_dict[m.name]  # [B, C_in, T]
            if B is None:
                B = x.size(0)
            t = self.tokenizers[m.name](x)  # [B, L, D]
            t = self.positional[m.name](t)
            t = t + self.modality_embeddings[m.name]  # add modality embedding
            tokens.append(t)

        tokens_per_mod = {m.name: t for m, t in zip(self.modalities, tokens)}
        tokens_per_mod = self._maybe_modal_dropout(tokens_per_mod)

        concat_tokens = [tokens_per_mod[m.name] for m in self.modalities]
        fused = torch.cat(concat_tokens, dim=1)  # [B, L_total, D]

        cls = self.cls_token.expand(B, -1, -1)
        fused = torch.cat([cls, fused], dim=1)  # [B, 1+L_total, D]

        # (1) Cross-attention with history first
        if history is not None:
            if self.history_proj is not None:
                mem = self.history_proj(history)
            else:
                mem = history
            for block in self.cross_attn:
                fused = block(fused, mem, kv_mask=history_mask)

        # (2) Fusion transformer
        out = self.fusion(fused)

        cls_out = out[:, 0, :]
        logits = self.head(cls_out)

        return logits, fused



def build_baseline_model(num_classes: int, model_dim: int = 256, history_dim: Optional[int] = 256):
    modalities = [
        ModalityConfig('ecg', 1, 32),
        ModalityConfig('ppg', 1, 32),
        ModalityConfig('eda', 1, 16),
        ModalityConfig('emg', 1, 16),
        ModalityConfig('temp', 1, 16),
        ModalityConfig('c_acc', 3, 16),
        ModalityConfig('w_acc', 3, 16),
    ]
    return MultimodalActivityTransformer(
        num_classes=num_classes,
        model_dim=model_dim,
        nhead=8,
        fusion_depth=4,
        ff_dim=4*model_dim,
        dropout=0.1,
        modalities=modalities,
        use_modal_dropout=True,
        modal_dropout_p=0.15,
        max_len=500,
        history_dim=history_dim,
        cross_attn_layers=1,
    )


if __name__ == "__main__":
    B, T = 8, 1024
    x_dict = {
        'ecg': torch.randn(B, 1, T),
        'ppg': torch.randn(B, 1, T),
        'eda': torch.randn(B, 1, T),
        'emg': torch.randn(B, 1, T),
        'temp': torch.randn(B, 1, T),
        'c_acc': torch.randn(B, 3, T),
        'w_acc': torch.randn(B, 3, T),
    }
    history = torch.randn(B, 8, 256)
    model = build_baseline_model(num_classes=12, model_dim=256, history_dim=256)
    logits, aux = model(x_dict, history=history)
    print(logits.shape)
