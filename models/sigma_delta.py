# sd_transformer.py
import math
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer_utils import CrossAttentionLayer, TransformerLayer, TransformerBlock


# -----------------------------
# Sigma-Delta core module
# -----------------------------
class SigmaDeltaQuantizer(nn.Module):
    """
    Stateful Sigma-Delta quantizer that supports:
      - stateful streaming mode (maintains buffer across calls)
      - stateless batch mode (process an explicit time dimension)
    Behavior:
      - threshold theta (can be scalar or per-feature)
      - forward(x, stateful=True/False)
        if stateful: uses/updates internal buffer 'sigma_state'
        if not: expects input with time axis and runs sequentially internally
    Implementation note:
      - We use d = round((x - sigma_prev) / theta)
      - sigma_new = sigma_prev + d * theta
      - returns (d, sigma_new)
    """
    def __init__(self, feature_shape: Tuple[int, ...], theta_init: float = 1e-2, device=None):
        """
        feature_shape: shape of features per sample excluding batch and time, e.g. (L, D) or (D,) depending usage.
        We'll store theta per-last-dim (per feature) if possible.
        """
        super().__init__()
        # create a learnable log-theta (positive) per feature dimension if feasible
        # For simplicity we parameterize a single scalar theta per feature vector (last dim)
        # feature_shape[-1] is assumed to be feature dimension D
        D = feature_shape[-1] if len(feature_shape) > 0 else 1
        self.log_theta = nn.Parameter(torch.log(torch.ones(D) * theta_init))
        # sigma_state buffer will be initialized on the first call with batch size
        self.register_buffer('_sigma_state', None, persistent=False)

    def theta(self):
        return torch.exp(self.log_theta)

    def reset_state(self, batch_size: int, device=None, shape_prefix: Optional[Tuple[int,...]]=None):
        """
        Initialize sigma_state to zeros with shape: [batch_size, *shape_prefix, D]
        shape_prefix is e.g. (L,) if we maintain per-patch positions, else None.
        """
        D = self.log_theta.shape[0]
        if shape_prefix is None:
            s = torch.zeros(batch_size, D, device=device)
        else:
            s = torch.zeros(batch_size, *shape_prefix, D, device=device)
        self._sigma_state = s
        return self._sigma_state

    def forward_stateful(self, x: torch.Tensor):
        """
        x: [B, ..., D]  (one time step)
        returns:
            d: integer delta tensor same shape as x  (float dtype but integer-valued)
            s_new: updated sigma state [B,...,D]  (same shape)
        updates internal buffer
        """
        if self._sigma_state is None:
            # infer batch and prefix shapes
            B = x.shape[0]
            prefix = x.shape[1:-1]  # may be empty
            device = x.device
            self.reset_state(B, device=device, shape_prefix=prefix if len(prefix)>0 else None)

        s_prev = self._sigma_state
        theta = self.theta().view(*([1] * (x.dim()-1)), -1)  # broadcast to x shape
        # compute delta integers
        diff = x - s_prev
        d = torch.round(diff / theta)            # integer pulses (float dtype)
        s_new = s_prev + d * theta
        # update buffer
        self._sigma_state = s_new.detach().clone()  # detach to avoid backprop through state history
        return d, s_new

    def forward_stateless(self, x_seq: torch.Tensor):
        """
        Process a full sequence along time dim:
        x_seq: [B, T, ..., D]  assume time is dim=1
        returns:
            d_seq: [B, T, ..., D]
            s_last: last sigma state [B, ..., D]
        NOTE: no internal buffer updated (stateless).
        """
        # Assume x_seq dims: [B, T, *prefix, D] ; we'll iterate over T
        B = x_seq.shape[0]
        T = x_seq.shape[1]
        prefix = x_seq.shape[2:-1]
        D = x_seq.shape[-1]
        device = x_seq.device
        theta = self.theta().view(*([1] * (x_seq.dim()-2)), -1)  # shape fits broadcasting over time
        s_prev = torch.zeros(B, *prefix, D, device=device)
        d_list = []
        for t in range(T):
            xt = x_seq[:, t]              # [B, *prefix, D]
            diff = xt - s_prev
            d = torch.round(diff / theta)
            s_prev = s_prev + d * theta
            d_list.append(d)
        d_seq = torch.stack(d_list, dim=1)    # [B, T, *prefix, D]
        return d_seq, s_prev

    def forward(self, x: torch.Tensor, stateful: bool = True):
        """
        If x has time dim (>=3 dims) and stateful=False, use stateless path.
        If stateful=True, expects x to be one time step: shape [B, ..., D].
        """
        if stateful:
            return self.forward_stateful(x)
        else:
            # assume x has shape [B, T, ..., D]
            return self.forward_stateless(x)


# -----------------------------
# Adapted Tokenizer -> SD version
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

class SDConvTokenizer1D(nn.Module):
    """
    Conv tokenizer wrapped with Sigma-Delta quantizer operating on token (patch) dimension.
    Behavior:
      - compute conv -> tokens [B, L, out_ch]
      - optionally run sigma-delta in stateless or stateful mode along L as 'time'
    Note: we treat patch index as temporal steps for ΣΔ; this approximates streaming patches.
    """
    def __init__(self, in_ch: int, out_ch: int = 1, patch_size: int = 10, norm: bool = True, use_stateful: bool=False):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=patch_size, stride=patch_size, bias=False)
        self.norm = nn.LayerNorm(out_ch) if norm else nn.Identity()
        # We'll create a SigmaDeltaQuantizer that works on features of shape (out_ch,)
        self.sd = SigmaDeltaQuantizer(feature_shape=(out_ch,), theta_init=1e-2)
        self.use_stateful = use_stateful

    def reset_state(self, batch_size: int, device=None):
        # tokens are shape [B,L,out_ch] -> we want sigma state per patch index: prefix = (L,)
        # but we don't yet know L until a forward; we'll lazily init on first forward if needed.
        self.sd.reset_state(batch_size, device=device, shape_prefix=None)

    def forward(self, x: torch.Tensor, stateful: bool=False):
        # x: [B, C_in, T]
        x = self.conv(x)          # [B, out_ch, L]
        x = x.transpose(1, 2)     # [B, L, out_ch]
        x = self.norm(x)
        # Now apply SD quantization along the patch (L) axis
        if stateful or self.use_stateful:
            # We'll treat each patch as sequential step and update internal buffer;
            # BUT forward() receives all patches at once, so we iterate to update buffer per step
            d_list = []
            s_list = []
            for t in range(x.size(1)):
                xt = x[:, t]        # [B, out_ch]
                d, s = self.sd(xt, stateful=True)
                d_list.append(d)
                s_list.append(s)
            d_seq = torch.stack(d_list, dim=1)   # [B, L, out_ch]
            s_last = s_list[-1]
            # Return both delta pulses and reconstructed sigma if needed. For simplicity return sigma (recon)
            # recon = cumulative sigma values = s_last broadcast? But we will return the sequence recon at each step:
            recon_seq = torch.stack(s_list, dim=1)  # [B, L, out_ch]
            return recon_seq, d_seq
        else:
            # stateless processing: treat patch axis as time and do stateless SD on the whole seq
            # input shape -> [B, L, out_ch] -> we call sd.forward with stateful=False
            d_seq, s_last = self.sd(x, stateful=False)   # returns d_seq: [B, L, out_ch], s_last: [B, out_ch]
            recon_seq = torch.cumsum(d_seq * self.sd.theta().view(1,1,-1), dim=1)  # approximate recon per step
            # but more correct is to reconstruct incremental s: do running sum starting at 0:
            # We'll compute s_t = sum_{tau<=t} d_tau * theta
            recon_seq = torch.cumsum(d_seq * self.sd.theta().view(1,1,-1), dim=1)
            return recon_seq, d_seq


# -----------------------------
# SD Wrapper for modules (Linear / CrossAttention / TransformerBlock)
# -----------------------------
class SDWrapper(nn.Module):
    """
    Generic wrapper: receives inputs that may be delta-encoded.
    Expected usage: input_pass is either:
      - tensor of sigma values (floating), or
      - delta pulses (integer-valued floats) together with a SigmaDeltaQuantizer to reconstruct sigma.
    This wrapper will:
      - if 'delta' provided: reconstruct sigma via sigma_prev + delta * theta
      - then call wrapped module with reconstructed sigma.
    For simplicity the wrapper owns an internal SigmaDeltaQuantizer (optional) to decode delta inputs,
    or it can be configured to just pass through sigma inputs.
    """
    def __init__(self, module: nn.Module, feature_dim: int):
        super().__init__()
        self.module = module
        # wrapper has its own SD quantizer used for decoding delta streams if needed
        self.sd = SigmaDeltaQuantizer(feature_shape=(feature_dim,), theta_init=1e-2)

    def reset_state(self, batch_size: int, device=None, shape_prefix: Optional[Tuple[int,...]] = None):
        self.sd.reset_state(batch_size, device=device, shape_prefix=shape_prefix)

    def forward(self, x, input_is_delta: bool=False):
        """
        x:
          - if input_is_delta==False: x is sigma (continuous) -> pass through to module
          - if input_is_delta==True: x is delta pulses sequence for current step: we decode to sigma then call module
        Expected shapes vary by module:
          - For Linear-like / Transformer: x [B, L, D] (sigma)
          - If delta: x same shape but integer pulses to be decoded
        """
        if input_is_delta:
            # decode using stateful mode (we assume x is current time step or sequence with time)
            # Here we support x being [B, L, D] (one step) -> decode via stateful forward
            d, s = self.sd(x, stateful=True)
            # call module with decoded sigma
            return self.module(s)
        else:
            return self.module(x)


# -----------------------------
# Sigma-Delta Multimodal Transformer
# -----------------------------
class ModalityConfig:
    def __init__(self, name: str, in_ch: int, patch_size: int):
        self.name = name
        self.in_ch = in_ch
        self.patch_size = patch_size

class SigmaDeltaMultimodalActivityTransformer(nn.Module):
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
        return_mem: bool = False,
        sd_stateful: bool = False,
    ):
        super().__init__()

        if modalities is None:
            modalities = [ModalityConfig(f'm{i}', 1, 10) for i in range(14)]

        self.modalities = modalities
        self.model_dim = model_dim
        self.use_modal_dropout = use_modal_dropout
        self.modal_dropout_p = modal_dropout_p
        self.return_mem = return_mem
        self.sd_stateful = sd_stateful

        # Tokenizers: SDConvTokenizer per modality -> outputs [B, L, out_ch]
        self.tokenizers = nn.ModuleDict()
        for m in modalities:
            # out_ch == model_dim? We'll keep out_ch= model_dim to avoid extra proj later
            self.tokenizers[m.name] = SDConvTokenizer1D(m.in_ch, out_ch=model_dim, patch_size=m.patch_size, norm=True, use_stateful=sd_stateful)

        # positional
        self.positional = PositionalEncoding(model_dim, max_len=max_len)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # History projection (wrapped by SD wrapper)
        self.history_proj = SDWrapper(nn.Linear(model_dim, model_dim), feature_dim=model_dim)

        # Cross-attention layer (wrapped)
        raw_cross = CrossAttentionLayer(model_dim, nhead, dim_feedforward=ff_dim, dropout=dropout)
        self.cross_attn = SDWrapper(raw_cross, feature_dim=model_dim)

        # Fusion transformer block (wrap each block?)
        # We'll create a standard TransformerBlock and wrap it: for simplicity we'll wrap the whole block
        raw_fusion = TransformerBlock(TransformerLayer(d_model=model_dim, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout), num_layers=fusion_depth)
        self.fusion = SDWrapper(raw_fusion, feature_dim=model_dim)

        # head
        self.head_norm = nn.LayerNorm(model_dim)
        self.head_lin = nn.Linear(model_dim, num_classes)

    def reset_sd_state(self, batch_size: int, device=None):
        # propagate reset to tokenizers and wrappers
        for name, tok in self.tokenizers.items():
            tok.sd.reset_state(batch_size, device=device, shape_prefix=None)
        self.history_proj.reset_state(batch_size, device=device, shape_prefix=None)
        self.cross_attn.reset_state(batch_size, device=device, shape_prefix=None)
        self.fusion.reset_state(batch_size, device=device, shape_prefix=None)

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
        stateful: bool = False,
    ):
        """
        x: [B, C, T]
        history: [B, L_total, D]  or None
        stateful: if True, advance ΣΔ stateful buffers; otherwise run stateless path.
        Returns logits (and optionally mem if return_mem True).
        """
        B = x.size(0)
        C = x.size(1)
        assert C == len(self.modalities)

        per_mod_tokens = []
        per_mod_deltas = []

        # Tokenize per modality (each tokenizers returns recon_seq [B, L, D], d_seq)
        for i, m in enumerate(self.modalities):
            ch = x[:, i:i+1, :]                # [B, 1, T]
            recon_seq, d_seq = self.tokenizers[m.name](ch, stateful=stateful)
            # recon_seq: [B, L, D] ; d_seq: [B, L, D]
            # We'll use recon_seq as sigma inputs to downstream (i.e. decoded continuous tokens)
            per_mod_tokens.append(recon_seq)

        # optional modal dropout (applies to each per-mod token sequence)
        per_mod_tokens = self._maybe_modal_dropout(per_mod_tokens)

        # Now concatenate tokens along patch/time dimension: each is [B, L, D]. We concatenate over L.
        concat_tokens = torch.cat(per_mod_tokens, dim=1)   # [B, L_total, D]
        # positional
        fused = self.positional(concat_tokens)

        # (1) Cross-attention: fused (Q) attends to mem (K,V)
        if history is not None:
            # Project history via history_proj wrapper (it expects sigma inputs -> input_is_delta=False)
            # history may be provided as [B, L_total, D]
            history_proj = self.history_proj(history, input_is_delta=False)  # returns module output
            # cross-attn wrapper call: it expects sigma inputs; set input_is_delta=False
            cross_out = self.cross_attn.module(fused, history_proj)   # residual-like addition is optional
            fused = fused + cross_out
            # Instead of adding, we call cross_attn as in original: cross_attn(fused, history_proj)
            # But our SDWrapper was built to call its inner module with a single arg; to preserve semantics,
            # we'll call the wrapped cross-attn module directly using reconstructed history:
            fused = self.cross_attn.module(fused, history_proj)  # call underlying module directly

        # (2) Fusion transformer (wrapped)
        out = self.fusion(fused, input_is_delta=False)  # pass sigma data through wrapper -> calls module

        cls_out = out.mean(axis=1)
        logits = self.head_lin(self.head_norm(cls_out))
        if self.return_mem:
            return logits, out
        else:
            return logits

# Helper to build
def build_sd_former(num_modal=14, num_classes=10, model_dim=32, return_mem=False):
    modalities = [ModalityConfig(f'm{i}', 1, 10) for i in range(num_modal)]
    return SigmaDeltaMultimodalActivityTransformer(
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
        return_mem= return_mem,
        sd_stateful=False
    )

# Quick test run
if __name__ == "__main__":
    x = torch.randn(8, 14, 100)           # [B, 14, 100]
    history = torch.randn(8, 140, 32)
    model = build_sd_former(num_classes=12, model_dim=32)
    logits = model(x, history=history, stateful=False)
    print("logits:", logits.shape)  # expected [8, 12]
