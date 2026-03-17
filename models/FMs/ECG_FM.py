import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging

from models.FMs.fairseq_signals_modules import (
    TransformerEncoder,
    ConvFeatureExtraction,
    ConvPositionalEncoding,
    LayerNorm,
    GradMultiply,
    GatherLayer,
)

logger = logging.getLogger(__name__)


@dataclass
class ECGTransformerClassificationConfig:
    """Complete configuration for ECG Transformer Classification"""
    # Classification
    num_labels: int = field(
        default=None,
        metadata={"help": "number of labels to be classified"}
    )

    # Convolution Feature Extractor
    conv_feature_layers: str = field(
        default="[(256, 2, 2)] * 4",
        metadata={"help": "convolutional layers: [(dim, kernel_size, stride), ...]"}
    )
    in_d: int = field(default=12, metadata={"help": "input dimension (channels)"})
    conv_bias: bool = field(default=False, metadata={"help": "use bias in conv"})
    extractor_mode: str = field(default="default", metadata={"help": "conv mode"})
    feature_grad_mult: float = field(
        default=0.0,
        metadata={"help": "multiply feature extractor grads by this (0=frozen)"}
    )

    # Positional Encoding
    conv_pos: int = field(default=128, metadata={"help": "conv pos embedding filters"})
    conv_pos_groups: int = field(default=16, metadata={"help": "conv pos groups"})

    # Transformer Encoder
    encoder_layers: int = field(default=12, metadata={"help": "transformer layers"})
    encoder_embed_dim: int = field(default=768, metadata={"help": "embedding dim"})
    encoder_ffn_embed_dim: int = field(default=3072, metadata={"help": "FFN dim"})
    encoder_attention_heads: int = field(default=12, metadata={"help": "attention heads"})
    layer_norm_first: bool = field(default=False, metadata={"help": "pre-norm"})

    # Dropouts
    dropout: float = field(default=0.0, metadata={"help": "general dropout"})
    attention_dropout: float = field(default=0.0, metadata={"help": "attention dropout"})
    activation_dropout: float = field(default=0.0, metadata={"help": "activation dropout"})
    encoder_layerdrop: float = field(default=0.0, metadata={"help": "layer drop"})
    dropout_input: float = field(default=0.0, metadata={"help": "input dropout"})
    dropout_features: float = field(default=0.0, metadata={"help": "feature dropout"})
    final_dropout: float = field(default=0.0, metadata={"help": "final dropout"})

    # Training
    freeze_finetune_updates: int = field(
        default=0,
        metadata={"help": "freeze encoder for first N updates"}
    )
    model_path: str = field(default="", metadata={"help": "pretrained model path"})
    no_pretrained_weights: bool = field(
        default=False,
        metadata={"help": "don't load pretrained weights"}
    )

    # Advanced
    saliency: bool = field(default=False, metadata={"help": "extract attention weights"})
    all_gather: bool = field(default=False, metadata={"help": "gather across GPUs"})
    apply_mask: bool = field(default=False, metadata={"help": "apply masking"})


class ECGTransformerClassificationModel(nn.Module):
    """
    Standalone ECG Transformer Classification Model

    Architecture Flow:
    1. Input ECG [B, C, T]
       ↓
    2. ConvFeatureExtraction: multi-layer 1D conv → [B, embed, T']
       ↓
    3. LayerNorm + Projection (if needed) → [B, T', encoder_embed_dim]
       ↓
    4. ConvPositionalEncoding: add positional info
       ↓
    5. TransformerEncoder: self-attention layers → [B, T', encoder_embed_dim]
       ↓
    6. Global Average Pooling (with padding mask) → [B, encoder_embed_dim]
       ↓
    7. Classification Head: Linear → [B, num_labels]
    """

    def __init__(self, cfg: ECGTransformerClassificationConfig):
        super().__init__()
        self.cfg = cfg

        # ============ 1. Convolutional Feature Extractor ============
        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtraction(
            conv_layers=feature_enc_layers,
            in_d=cfg.in_d,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias
        )

        self.feature_grad_mult = cfg.feature_grad_mult
        self.layer_norm = LayerNorm(self.embed)

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim else None
        )

        # ============ 2. Positional Encoding ============
        self.conv_pos = ConvPositionalEncoding(cfg)

        # ============ 3. Dropouts ============
        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)
        self.final_dropout = nn.Dropout(cfg.final_dropout)

        # ============ 4. Transformer Encoder ============
        self.encoder = TransformerEncoder(cfg)

        # ============ 5. Classification Head ============
        self.proj = nn.Linear(cfg.encoder_embed_dim, cfg.num_labels)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.0)

        # ============ Training State ============
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """Compute output length after convolutions"""

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)

        conv_cfg_list = eval(self.cfg.conv_feature_layers)
        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(
                input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2]
            )
        return input_lengths.to(torch.long)

    def _extract_conv_features(self, source, padding_mask):
        """Step 1: Extract convolutional features"""
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)

        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if padding_mask is not None and padding_mask.any():
            input_lengths = (1 - padding_mask.long()).sum(-1)
            if input_lengths.dim() > 1:
                for input_len in input_lengths:
                    assert (input_len == input_len[0]).all()
                input_lengths = input_lengths[:, 0]

            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                features.shape[:2], dtype=features.dtype, device=features.device
            )
            padding_mask[
                (torch.arange(padding_mask.shape[0], device=padding_mask.device),
                 output_lengths - 1)
            ] = 1
            padding_mask[torch.where(output_lengths == 0)] = 0
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        else:
            padding_mask = None

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)

        return features, padding_mask

    def _add_positional_encoding(self, x):
        """Step 2: Add positional encoding"""
        x_conv = self.conv_pos(x, channel_first=False)
        x = x + x_conv
        return x

    def _apply_transformer(self, x, padding_mask):
        """Step 3: Apply transformer encoder"""
        res = self.encoder(x, padding_mask=padding_mask)
        return res

    def _global_average_pooling(self, x, padding_mask):
        """Step 4: Global average pooling over time dimension"""
        x = self.final_dropout(x)
        if padding_mask is not None and padding_mask.any():
            x[padding_mask] = 0
        x = torch.div(x.sum(dim=1), (x != 0).sum(dim=1))
        return x

    def forward(self, source, padding_mask=None, return_embeddings=False, **kwargs):
        """
        Forward pass

        Args:
            source: [B, in_d, T] - input ECG signals
            padding_mask: [B, T] - True for padded positions
            return_embeddings: if True, return embeddings; if False, return logits

        Returns:
            if return_embeddings=True: [B, T', encoder_embed_dim] embeddings
            if return_embeddings=False: dict with 'out' (logits), 'encoder_out', etc.
        """
        freeze_encoder = self.num_updates < self.freeze_finetune_updates

        with torch.no_grad() if freeze_encoder else contextlib.ExitStack():
            x, padding_mask = self._extract_conv_features(source, padding_mask)
            x = self._add_positional_encoding(x)
            res = self._apply_transformer(x, padding_mask)
            x = res["x"]  # [B, T', encoder_embed_dim]
            saliency = res["saliency"]

        # if return_embeddings:
        #     # Return sequence embeddings

        x = x.transpose(1, 2)  # [B, encoder_embed_dim, T']
        return x

        # # Continue with pooling and classification
        # pooled = self._global_average_pooling(x, padding_mask)
        # logits = self.proj(pooled)
        #
        # return {
        #     "out": logits,
        #     "encoder_out": x,
        #     "padding_mask": padding_mask,
        #     "saliency": None if saliency is None else saliency,
        # }

    @staticmethod
    def _remap_checkpoint_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Remap checkpoint keys from old fairseq_signals format"""
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if key.startswith("encoder.feature_extractor."):
                new_key = key.replace("encoder.feature_extractor.", "feature_extractor.")
            elif key.startswith("encoder.post_extract_proj."):
                new_key = key.replace("encoder.post_extract_proj.", "post_extract_proj.")
            elif key.startswith("encoder.conv_pos."):
                new_key = key.replace("encoder.conv_pos.", "conv_pos.")
            elif key.startswith("encoder.encoder."):
                new_key = key.replace("encoder.encoder.", "encoder.")
            elif key == "encoder.layer_norm.weight" or key == "encoder.layer_norm.bias":
                new_key = key.replace("encoder.layer_norm.", "layer_norm.")

            if "weight_g" in new_key:
                new_key = new_key.replace("weight_g", "parametrizations.weight.original0")
            elif "weight_v" in new_key:
                new_key = new_key.replace("weight_v", "parametrizations.weight.original1")

            new_state_dict[new_key] = value
        return new_state_dict

    @classmethod
    def from_pretrained(cls, cfg, checkpoint_path, strict=False, map_location="cpu"):
        """Load model from pretrained checkpoint"""
        model = cls(cfg)
        logger.info(f"Loading pretrained weights from {checkpoint_path}")

        state = torch.load(checkpoint_path, map_location=torch.device(map_location), weights_only=False)

        if "model" in state:
            state_dict = state["model"]
        elif "state_dict" in state:
            state_dict = state["state_dict"]
        else:
            state_dict = state

        state_dict = cls._remap_checkpoint_keys(state_dict)

        if not strict:
            model_dict = model.state_dict()
            pretrained_dict = {
                k: v for k, v in state_dict.items()
                if k in model_dict and model_dict[k].shape == v.shape
            }
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            logger.info(f"Loaded {len(pretrained_dict)}/{len(model_dict)} parameters")
        else:
            model.load_state_dict(state_dict, strict=True)
            logger.info("Loaded all weights (strict mode)")

        return model


# ============================================================================
# Single-Lead Adaptation Utilities
# ============================================================================

def adapt_single_lead_to_12lead(single_lead_signal, lead_index=0, method="zero_pad"):
    """
    Adapt single-lead ECG to 12-lead format

    Args:
        single_lead_signal: [B, 1, T] - single lead ECG
        lead_index: which lead position (0=Lead I, 1=Lead II, etc.)
        method: "zero_pad" or "repeat"

    Returns:
        [B, 12, T] - adapted to 12-lead format
    """
    B, C, T = single_lead_signal.shape
    assert C == 1, f"Expected 1 channel, got {C}"

    if method == "zero_pad":
        output = torch.zeros(B, 12, T, device=single_lead_signal.device, dtype=single_lead_signal.dtype)
        output[:, lead_index:lead_index + 1, :] = single_lead_signal
    elif method == "repeat":
        output = single_lead_signal.repeat(1, 12, 1)
    else:
        raise ValueError(f"Unknown method: {method}")

    return output


class SingleLeadAdapter(nn.Module):
    """Learnable adapter: single-lead → 12-lead prediction"""

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=15, padding=7)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=15, padding=7)
        self.conv3 = nn.Conv1d(hidden_dim, 12, kernel_size=15, padding=7)
        self.relu = nn.ReLU()

    def forward(self, x):
        """[B, 1, T] -> [B, 12, T]"""
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class ECG12LeadModelWithSingleLeadSupport(nn.Module):
    """
    Wrapper for 12-lead model that supports single-lead input

    Use this when you have a 12-lead pretrained model but single-lead data
    """

    def __init__(self, ecg_12lead_model, adaptation_method="zero_pad", lead_index=0, learnable_adapter=None):
        """
        Args:
            ecg_12lead_model: pretrained 12-lead ECGTransformerClassificationModel
            adaptation_method: "zero_pad", "repeat", or "learnable"
            lead_index: which lead position (0=I, 1=II, 2=III, etc.)
            learnable_adapter: SingleLeadAdapter instance (required if method="learnable")
        """
        super().__init__()
        self.ecg_model = ecg_12lead_model
        self.adaptation_method = adaptation_method
        self.lead_index = lead_index

        if adaptation_method == "learnable":
            self.adapter = learnable_adapter if learnable_adapter else SingleLeadAdapter()
            # Freeze ECG model by default when using learnable adapter
            for param in self.ecg_model.parameters():
                param.requires_grad = False
        else:
            self.adapter = None

    def unfreeze_ecg_model(self):
        """Unfreeze ECG model for end-to-end training"""
        for param in self.ecg_model.parameters():
            param.requires_grad = True

    def forward(self, single_lead_input, padding_mask=None, return_embeddings=False, **kwargs):
        """
        Args:
            single_lead_input: [B, 1, T] - single lead ECG

        Returns:
            same as ECGTransformerClassificationModel
        """
        if self.adaptation_method == "learnable":
            twelve_lead_input = self.adapter(single_lead_input)
        else:
            twelve_lead_input = adapt_single_lead_to_12lead(
                single_lead_input,
                lead_index=self.lead_index,
                method=self.adaptation_method
            )

        return self.ecg_model(twelve_lead_input, padding_mask=padding_mask, return_embeddings=return_embeddings,
                              **kwargs)


# ============================================================================
# Convenience Functions
# ============================================================================

def load_checkpoint_and_config(checkpoint_path: str, map_location: str = "cpu") -> tuple:
    """Load checkpoint and extract configuration"""
    state = torch.load(checkpoint_path, map_location=torch.device(map_location), weights_only=False)

    if "model" in state:
        state_dict = state["model"]
    elif "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state

    if "cfg" in state:
        cfg_dict = state["cfg"]
        if "model" in cfg_dict:
            cfg_dict = cfg_dict["model"]
    elif "config" in state:
        cfg_dict = state["config"]
    else:
        cfg_dict = None

    return state_dict, cfg_dict


def get_ecg_fm(pretrained=True, local_path="/Users/chengweizhou/PycharmProjects/data/mimic_iv_ecg_finetuned.pt",
               return_embeddings=False):
    """
    Get ECG Foundation Model (12-lead)

    Args:
        pretrained: load pretrained weights
        local_path: path to checkpoint
        return_embeddings: if True, model returns embeddings; if False, returns logits

    Returns:
        model configured for embeddings or classification
    """
    state_dict, cfg_dict = load_checkpoint_and_config(local_path)
    cfg = ECGTransformerClassificationConfig(
        num_labels=cfg_dict.get("num_labels", 5),
        encoder_embed_dim=cfg_dict.get("encoder_embed_dim", 768),
        encoder_layers=cfg_dict.get("encoder_layers", 12),
    )

    model = ECGTransformerClassificationModel.from_pretrained(
        cfg=cfg,
        checkpoint_path=local_path,
        strict=False
    )

    return model


def get_single_lead_ecg_model(pretrained_12lead_path, adaptation_method="zero_pad", lead_index=0):
    """
    Get single-lead ECG model using 12-lead pretrained weights

    Args:
        pretrained_12lead_path: path to 12-lead checkpoint
        adaptation_method: "zero_pad", "repeat", or "learnable"
        lead_index: which lead (0=I, 1=II, 2=III, ...)

    Returns:
        model that accepts [B, 1, T] single-lead input
    """
    # Load 12-lead model
    ecg_12lead_model = get_ecg_fm(pretrained=True, local_path=pretrained_12lead_path)

    # Wrap with single-lead support
    model = ECG12LeadModelWithSingleLeadSupport(
        ecg_12lead_model=ecg_12lead_model,
        adaptation_method=adaptation_method,
        lead_index=lead_index
    )

    print(f"✅ Single-lead ECG model ready (method={adaptation_method}, lead={lead_index})")
    return model


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    local_path = "/Users/chengweizhou/PycharmProjects/data/mimic_iv_ecg_finetuned.pt"

    # print("=" * 70)
    # print("Load 12-lead model and use with 12-lead data")
    # print("=" * 70)
    # model_12lead = get_ecg_fm(local_path=local_path)
    # ecg_12lead_data = torch.randn(2, 12, 1500)
    # output = model_12lead(ecg_12lead_data, return_embeddings=True)
    # print(f"Input: {ecg_12lead_data.shape} -> Output: {output.shape}\n")

    print("=" * 70)
    print("Single-lead with zero-padding (fastest, no training)")
    print("=" * 70)
    model_single_zeropad = get_single_lead_ecg_model(
        local_path,
        adaptation_method="zero_pad",
        lead_index=0  # Lead I
    )
    print(model_single_zeropad)
    single_lead_data = torch.randn(2, 1, 1600)  # Lead I data
    output = model_single_zeropad(single_lead_data, return_embeddings=True)
    print(f"Input: {single_lead_data.shape} -> Output: {output.shape}")
    print("Lead I at position 0, rest are zeros\n")  # Input: torch.Size([2, 1, 1600]) -> Output: torch.Size([2, 768, 100])
