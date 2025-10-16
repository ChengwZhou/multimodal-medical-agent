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

    Usage:
        cfg = ECGTransformerClassificationConfig(num_labels=5)
        model = ECGTransformerClassificationModel(cfg)
        # Or load pretrained weights
        model = ECGTransformerClassificationModel.from_pretrained(cfg, checkpoint_path)
        output = model(source=ecg_data, padding_mask=mask)
        logits = output["out"]  # [B, num_labels]
    """

    def __init__(self, cfg: ECGTransformerClassificationConfig):
        super().__init__()
        self.cfg = cfg

        # ============ 1. Convolutional Feature Extractor ============
        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]  # output dim of conv layers

        self.feature_extractor = ConvFeatureExtraction(
            conv_layers=feature_enc_layers,
            in_d=cfg.in_d,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias
        )

        self.feature_grad_mult = cfg.feature_grad_mult
        self.layer_norm = LayerNorm(self.embed)

        # Project conv output to transformer dim if needed
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
        # Apply conv with optional gradient scaling
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)

        # [B, embed, T] -> [B, T, embed]
        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        # Update padding mask based on conv output length
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

        # Project to transformer dimension
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
        # Apply dropout
        x = self.final_dropout(x)

        # Zero out padded positions
        if padding_mask is not None and padding_mask.any():
            x[padding_mask] = 0

        # Average pooling: sum / count of non-zero positions
        x = torch.div(x.sum(dim=1), (x != 0).sum(dim=1))

        return x

    def forward(self, source, padding_mask=None, **kwargs):
        """
        Forward pass

        Args:
            source: [B, in_d, T] - input ECG signals
            padding_mask: [B, T] - True for padded positions

        Returns:
            dict:
                - out: [B, num_labels] classification logits
                - encoder_out: [B, T', encoder_embed_dim] transformer output
                - padding_mask: [B, T'] updated padding mask
                - saliency: attention weights (if enabled)
        """
        # Determine if encoder should be frozen
        freeze_encoder = self.num_updates < self.freeze_finetune_updates

        with torch.no_grad() if freeze_encoder else contextlib.ExitStack():
            # Step 1: Conv feature extraction
            x, padding_mask = self._extract_conv_features(source, padding_mask)

            # Step 2: Add positional encoding
            x = self._add_positional_encoding(x)

            # Step 3: Transformer encoding
            res = self._apply_transformer(x, padding_mask)
            x = res["x"]
            saliency = res["saliency"]

        # Step 4: Global average pooling
        pooled = self._global_average_pooling(x, padding_mask)

        # Step 5: Classification
        logits = self.proj(pooled)

        return {
            "out": logits,
            "encoder_out": x.detach(),
            "padding_mask": padding_mask,
            "saliency": None if saliency is None else saliency.detach(),
        }

    def get_logits(self, net_output, normalize=False, **kwargs):
        """Get classification logits (optionally normalized)"""
        logits = net_output["out"]
        if normalize:
            logits = F.log_softmax(logits.float(), dim=-1)
        return logits

    def get_targets(self, sample, net_output, **kwargs):
        """Extract targets from sample"""
        if isinstance(sample["label"], torch.Tensor):
            return sample["label"].float()
        else:
            return sample["label"]

    def get_normalized_probs(self, net_output, log_probs=True):
        """Get normalized probabilities"""
        logits = self.get_logits(net_output)
        if log_probs:
            return F.log_softmax(logits.float(), dim=-1)
        else:
            return F.softmax(logits.float(), dim=-1)

    def set_num_updates(self, num_updates):
        """Update training step counter"""
        self.num_updates = num_updates

    @staticmethod
    def _remap_checkpoint_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remap checkpoint keys from old fairseq_signals format to new format

        Old format has nested structure:
            encoder.feature_extractor.* -> feature_extractor.*
            encoder.encoder.layers.* -> encoder.layers.*
            encoder.layer_norm.* -> layer_norm.*
            encoder.post_extract_proj.* -> post_extract_proj.*
            encoder.conv_pos.* -> conv_pos.*

        Args:
            state_dict: original checkpoint state dict

        Returns:
            remapped state dict
        """
        new_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Remove "encoder." prefix for feature extractor components
            if key.startswith("encoder.feature_extractor."):
                new_key = key.replace("encoder.feature_extractor.", "feature_extractor.")
            elif key.startswith("encoder.post_extract_proj."):
                new_key = key.replace("encoder.post_extract_proj.", "post_extract_proj.")
            elif key.startswith("encoder.conv_pos."):
                new_key = key.replace("encoder.conv_pos.", "conv_pos.")
            # Handle transformer encoder layers (double encoder prefix)
            elif key.startswith("encoder.encoder."):
                new_key = key.replace("encoder.encoder.", "encoder.")
            # Handle layer_norm (but not the one inside encoder)
            elif key == "encoder.layer_norm.weight" or key == "encoder.layer_norm.bias":
                new_key = key.replace("encoder.layer_norm.", "layer_norm.")

            # Handle weight normalization: weight_g and weight_v -> parametrizations
            if "weight_g" in new_key:
                new_key = new_key.replace("weight_g", "parametrizations.weight.original0")
            elif "weight_v" in new_key:
                new_key = new_key.replace("weight_v", "parametrizations.weight.original1")

            new_state_dict[new_key] = value

        return new_state_dict

    @classmethod
    def from_pretrained(
            cls,
            cfg: ECGTransformerClassificationConfig,
            checkpoint_path: str,
            strict: bool = False,
            map_location: str = "cpu"
    ):
        """
        Load model from pretrained checkpoint

        Args:
            cfg: model configuration
            checkpoint_path: path to checkpoint file
            strict: whether to strictly enforce key matching
            map_location: device to map tensors to

        Returns:
            ECGTransformerClassificationModel instance with loaded weights
        """
        model = cls(cfg)

        logger.info(f"Loading pretrained weights from {checkpoint_path}")

        # Load checkpoint
        state = torch.load(checkpoint_path, map_location=torch.device(map_location), weights_only=False)

        # Extract model state dict
        if "model" in state:
            state_dict = state["model"]
        elif "state_dict" in state:
            state_dict = state["state_dict"]
        else:
            state_dict = state

        # Remap keys from old format to new format
        state_dict = cls._remap_checkpoint_keys(state_dict)

        if strict:
            # Strict mode: load all matching keys
            model.load_state_dict(state_dict, strict=True)
            logger.info("Loaded all weights (strict mode)")
        else:
            # Non-strict mode: load only matching keys
            model_dict = model.state_dict()
            pretrained_dict = {
                k: v for k, v in state_dict.items()
                if k in model_dict and model_dict[k].shape == v.shape
            }

            # Show what's being loaded
            missing_keys = set(model_dict.keys()) - set(pretrained_dict.keys())
            unexpected_keys = set(state_dict.keys()) - set(model_dict.keys())

            # Filter out projection head keys from missing (they're new)
            missing_keys = {k for k in missing_keys if not k.startswith("proj.")}

            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

            logger.info(f"Loaded {len(pretrained_dict)}/{len(model_dict)} parameters")
            if missing_keys:
                logger.info(f"Missing keys (will use random init): {len(missing_keys)}")
                logger.debug(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logger.info(f"Unexpected keys (ignored): {len(unexpected_keys)}")
                logger.debug(f"Unexpected keys: {unexpected_keys}")

        return model

    @classmethod
    def build_model(cls, cfg: ECGTransformerClassificationConfig):
        """
        Build model with optional pretrained weight loading

        Args:
            cfg: model configuration

        Returns:
            ECGTransformerClassificationModel instance
        """
        # If pretrained weights specified, load them
        if cfg.model_path and not cfg.no_pretrained_weights:
            return cls.from_pretrained(cfg, cfg.model_path, strict=False)
        else:
            return cls(cfg)


def load_checkpoint_and_config(checkpoint_path: str, map_location: str = "cpu") -> tuple:
    """
    Helper function to load checkpoint and extract configuration

    Args:
        checkpoint_path: path to checkpoint file
        map_location: device to map tensors to

    Returns:
        tuple: (state_dict, config_dict)
    """
    state = torch.load(checkpoint_path, map_location=torch.device(map_location), weights_only=False)

    # Extract model state dict
    if "model" in state:
        state_dict = state["model"]
    elif "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state

    # Extract config
    if "cfg" in state:
        cfg_dict = state["cfg"]
        if "model" in cfg_dict:
            cfg_dict = cfg_dict["model"]
    elif "config" in state:
        cfg_dict = state["config"]
    else:
        cfg_dict = None

    return state_dict, cfg_dict


if __name__ == "__main__":
    # Example usage
    local_path = "/Users/chengweizhou/PycharmProjects/data/mimic_iv_ecg_finetuned.pt"

    # Load checkpoint and config
    state_dict, cfg_dict = load_checkpoint_and_config(local_path)

    print("Config keys:", cfg_dict.keys() if cfg_dict else "No config found")
    print("Model state dict keys:", len(state_dict.keys()))

    # Create config from checkpoint
    if cfg_dict:
        # Convert namespace or dict to config
        cfg = ECGTransformerClassificationConfig(
            num_labels=cfg_dict.get("num_labels", 5),
            encoder_embed_dim=cfg_dict.get("encoder_embed_dim", 768),
            encoder_layers=cfg_dict.get("encoder_layers", 12),
            # in_d=1
            # Add other fields as needed
        )

        # Load model
        model = ECGTransformerClassificationModel.from_pretrained(
            cfg=cfg,
            checkpoint_path=local_path,
            strict=True
        )

        print(f"Model loaded successfully with {sum(p.numel() for p in model.parameters())} parameters")

        ecg_data = torch.randn(2, 12, 5000)  # [batch=2, channels=12, time=5000]

        output = model(ecg_data)

        output = model(ecg_data, verbose=True)
        print(output.keys())
        print(output["out"].shape, output["encoder_out"].shape)  #torch.Size([2, 17]) torch.Size([2, 312, 768])