import os
import sys
import time
import csv

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pynvml
from models.former_device import build_former_device
from models.agent_sensor_masking import SensorGatingAgent
from models.FMformer import ModalityConfig
from models.sigma_former_sensor import build_adaptive_sigma_former



def load_from_checkpoint(
    checkpoint_path: str,
    num_classes: int,
    num_modalities: int,
    modalities,
    skip_steps=5,
    device: str = "cuda",
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model = build_adaptive_sigma_former(
        num_classes=num_classes,
        model_dim=512,
        return_mem=True,
        return_sensing_info=True,
        skip_steps=skip_steps,
        init_threshold=0.1,
        modalities=modalities,
        modal_fusion="cross_atten",
    ).to(device)

    agent = SensorGatingAgent(
        num_modalities=num_modalities,
        feature_dim=512
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    agent.load_state_dict(ckpt["agent_state_dict"])

    model.eval()
    agent.eval()

    return model, agent, device


class AgentONNXWrapper(nn.Module):
    def __init__(self, agent: SensorGatingAgent):
        super().__init__()
        self.agent = agent

    def forward(self, mem, sensor_history):
        out = self.agent(
            represent_features=mem,
            sensor_history=sensor_history,
            use_straight_through=True,
        )
        return out["p_st"]  # [B, M]


class TokenizerONNXWrapper(nn.Module):
    def __init__(self, model, sensor_history_length, num_modalities: int, batch_size: int):
        super().__init__()
        self.model = model
        self.num_modalities = num_modalities
        self.sensor_history_length = sensor_history_length

        self.register_buffer(
            "sensor_history",
            torch.ones(batch_size, sensor_history_length, num_modalities)
        )

    def forward(self, window, p_st):
        device = window.device
        window = window.to(device)
        p_st = p_st.to(device)
        self.sensor_history = self.sensor_history.to(device)

        B, C, T = window.shape
        M = p_st.shape[1]

        channels_per_modality = C // M

        # p_st -> [B, M, 1, 1] -> [B, M, C_per_mod, T]
        p_st_expanded = p_st.unsqueeze(-1).unsqueeze(-1)
        p_st_expanded = p_st_expanded.repeat(1, 1, channels_per_modality, T)
        mask = p_st_expanded.view(B, C, T)
        window_masked = window * mask

        self.sensor_history = torch.cat([
            self.sensor_history[:, 1:, :],
            torch.round(p_st).unsqueeze(1)
        ], dim=1)

        tokens_list = []
        for i, m in enumerate(self.model.modalities):
            x_m = window_masked[:, i : i + m.in_ch, :]
            masked_ch, mask_m, active_st, _ = self.model.adaptive_sensing[m.name](x_m)
            tok_m = self.model.tokenizers[m.name](x_m)  # [B, L_m, D]
            tokens_list.append(tok_m)

        per_mod_tokens = self.model._maybe_modal_dropout(tokens_list)
        return per_mod_tokens

class TransformerONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, per_mod_tokens, history, history_mask):
        concat_tokens = self.model.modal_cross_attn(per_mod_tokens, history_mask)
        fused = self.model.positional(concat_tokens)          # [B, L_total, D]
        fused = self.model.cross_attn(fused, history)
        out = self.model.fusion(fused)                 # [B, L_total, D]

        cls_out = out.mean(dim=1)                      # [B, D]
        logits = self.model.head(cls_out)              # [B, num_classes]

        return logits, out



def export_three_blocks_to_onnx(
    model,
    agent,
    num_modal,
    eff_patch,
    T,
    onnx_dir,
    device="cuda",
):
    os.makedirs(onnx_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model = model.to(device).eval()
    agent = agent.to(device).eval()

    B = 1


    # dummy sensor_history
    mem_dummy = torch.randn(
        B, T//eff_patch*num_modal, model.model_dim, device=device
    )
    sensor_history_dummy = torch.randn(
        B, agent.history_length, num_modal, device=device
    )
    dummy_window_for_tok = torch.randn(
        B, model.mod_cnt, T, device=device
    )

    agent_wrapper = AgentONNXWrapper(agent).to(device).eval()
    tokenizer_wrapper = TokenizerONNXWrapper(
        model=model,
        num_modalities=num_modal,
        sensor_history_length=agent.history_length,
        batch_size=B,
    ).to(device).eval()
    transformer_wrapper = TransformerONNXWrapper(model).to(device).eval()


    with torch.no_grad():
        p_st_dummy = agent_wrapper(mem_dummy, sensor_history_dummy)
        tokens_dummy = tokenizer_wrapper(dummy_window_for_tok, p_st_dummy)
        _, _= transformer_wrapper(tokens_dummy, mem_dummy, mem_dummy)


    agent_onnx_path = os.path.join(
        onnx_dir,
        f"act_{eff_patch}_mod_{num_modal}_agent.onnx",
    )



    tokenizer_onnx_path = os.path.join(
        onnx_dir,
        f"act_{eff_patch}_mod_{num_modal}_tokenizer.onnx",
    )



    transformer_onnx_path = os.path.join(
        onnx_dir,
        f"act_{eff_patch}_mod_{num_modal}_transformer.onnx",
    )




    torch.onnx.export(
        agent_wrapper,
        (mem_dummy, sensor_history_dummy),
        agent_onnx_path,
        input_names=["mem", "sensor_history"],
        output_names=["p_st"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            "mem": {0: "batch_size"},
            "sensor_history": {0: "batch_size"},
            "p_st": {0: "batch_size"},
        }
    )
    print(f"  -> Exported agent block to {agent_onnx_path}")



    torch.onnx.export(
        tokenizer_wrapper,
        (dummy_window_for_tok, p_st_dummy),
        tokenizer_onnx_path,
        input_names=["window", "p_st"],
        output_names=["tokens"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            "window": {0: "batch_size"},
            "p_st": {0: "batch_size"},
            "tokens": {0: "batch_size"},
        }
    )
    print(f"  -> Exported tokenizer block to {tokenizer_onnx_path}")



    torch.onnx.export(
        transformer_wrapper,
        (tokens_dummy, mem_dummy, mem_dummy),
        transformer_onnx_path,
        input_names=["tokens", "mem_context", "mem_context_mask"],
        output_names=["logits", "mem_new"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            "tokens": {0: "batch_size"},
            "mem_context": {0: "batch_size"},
            "mem_context_mask": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "mem_new": {0: "batch_size"},
        }
    )
    print(f"  -> Exported transformer block to {transformer_onnx_path}")



if __name__ == "__main__":
    num_classes = 12
    B = 1
    patch_size = 10
    tokens = 300
    skip_steps = 5

    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_modal_list = [4, 8, 12]
    activate_ratio_list = [0.4, 0.6, 0.8, 1.0]

    onnx_dir = "onnx_blocks"
    os.makedirs(onnx_dir, exist_ok=True)

    for num_modal in num_modal_list:
        for activate_ratio in activate_ratio_list:
            eff_patch = int(patch_size * activate_ratio)
            T = eff_patch * tokens

            modalities = [
                ModalityConfig(f"m{i}", 1, eff_patch, 0)
                for i in range(num_modal)
            ]

            ckpt_name = f"checkpoints/act_{eff_patch}_mod_{num_modal}.pth"
            print(f"\n=== Config: num_modal={num_modal}, "
                  f"activate_ratio={activate_ratio} (eff_patch={eff_patch}) ===")
            print(f"Loading checkpoint: {ckpt_name}")

            model, agent, device_used = load_from_checkpoint(
                checkpoint_path=ckpt_name,
                num_classes=num_classes,
                num_modalities=num_modal,
                modalities=modalities,
                skip_steps=skip_steps,
                device=device,
            )

            export_three_blocks_to_onnx(
                model=model,
                agent=agent,
                num_modal=num_modal,
                eff_patch=eff_patch,
                T=T,
                onnx_dir=onnx_dir,
                device=device,
            )

