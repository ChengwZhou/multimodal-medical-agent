import os
import torch
import torch.nn as nn
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.former_device import build_former_device
from models.agent_device_masking import DeviceGatingAgent
from models.FMformer import ModalityConfig

CKPT_PATH = "./checkpoints/best_model.pth"
ONNX_PATH = "./performance/exported/exported_model.onnx"

def load_from_checkpoint(device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    num_classes = 12
    num_modal = 12
    modal_fusion = "cross_atten"

    modalities = [
        ModalityConfig('al', 3, 10, 0),
        ModalityConfig('gl', 3, 10, 0),
        ModalityConfig('ar', 3, 10, 1),
        ModalityConfig('gr', 3, 10, 1),
    ]

    checkpoint = torch.load(CKPT_PATH, map_location=device)

    # backbone：device-wise former
    model = build_former_device(
        num_classes=num_classes,
        model_dim=512,
        return_mem=True,
        modalities=modalities,
        modal_fusion=modal_fusion
    )

    # agent：DeviceGatingAgent
    agent = DeviceGatingAgent(
        num_modalities=num_modal,
        modalities=modalities,
        feature_dim=512
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    agent.load_state_dict(checkpoint["agent_state_dict"])

    model.to(device).eval()
    agent.to(device).eval()

    print("Loaded model & agent from checkpoint.")

    return model, agent, modalities, num_classes


class CombinedGatedModel(nn.Module):
    def __init__(self, backbone, agent, mem_context_cache_length=10, use_soft_gating=True):
        super().__init__()
        self.backbone = backbone
        self.agent = agent
        self.mem_context_cache_length = mem_context_cache_length
        self.use_soft_gating = use_soft_gating

    def forward(self, sequences: torch.Tensor):
        # sequences: [B, L, C, T]
        B, L, C, T = sequences.shape
        device = sequences.device

        M = self.agent.num_modalities
        history_len = self.agent.history_length

        sensor_history = torch.ones(B, history_len, M, device=device)

        mem = None
        mem_cache = []
        mem_running_context = None

        logits_list = []
        gating_list = []

        for i in range(L):
            window = sequences[:, i, :, :]      # [B, C, T]

            if mem is not None:
                agent_out = self.agent(
                    represent_features=mem,
                    sensor_history=sensor_history,
                    use_straight_through=False if self.use_soft_gating else True
                )

                p_soft = agent_out["p_soft"]    # [B, M]

                channels_per_mod = C // M
                p_soft_exp = p_soft.unsqueeze(-1).unsqueeze(-1)
                p_soft_exp = p_soft_exp.expand(B, M, channels_per_mod, T)
                p_soft_exp = p_soft_exp.reshape(B, C, T)

                window_masked = window * p_soft_exp

                # 更新 history
                sensor_history = torch.cat(
                    [sensor_history[:, 1:, :], torch.round(p_soft).unsqueeze(1)],
                    dim=1
                )
            else:
                window_masked = window
                p_soft = torch.ones(B, M, device=device)

            logits, mem = self.backbone(window_masked, mem_running_context)

            if mem_running_context is None:
                mem_running_context = mem.detach()
            else:
                mem_cache.append(mem.detach())
                if len(mem_cache) > self.mem_context_cache_length:
                    mem_cache.pop(0)
                mem_running_context = torch.stack(mem_cache, dim=0).mean(0)

            logits_list.append(logits.unsqueeze(1))   # [B, 1, num_cls]
            gating_list.append(p_soft.unsqueeze(1))   # [B, 1, M]

        logits_all = torch.cat(logits_list, dim=1)    # [B, L, num_cls]
        gating_all = torch.cat(gating_list, dim=1)    # [B, L, M]

        return logits_all, gating_all


def export_to_onnx():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, agent, modalities, num_classes = load_from_checkpoint(device=device)
    combined = CombinedGatedModel(
        backbone=model,
        agent=agent,
        mem_context_cache_length=10,
        use_soft_gating=True,
    ).to(device).eval()

    total_channels = 12

    batch_size = 8
    seq_len = 10
    T = 100

    dummy_input = torch.randn(batch_size, seq_len, total_channels, T, device=device)

    os.makedirs(os.path.dirname(ONNX_PATH), exist_ok=True)

    torch.onnx.export(
        combined,
        dummy_input,
        ONNX_PATH,
        input_names=["sequences"],
        output_names=["logits", "gating"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            "sequences": {0: "batch_size", 1: "seq_len"},
            "logits":    {0: "batch_size", 1: "seq_len"},
            "gating":    {0: "batch_size", 1: "seq_len"},
        },
    )

    print(f"Exported ONNX model to: {ONNX_PATH}")


if __name__ == "__main__":
    export_to_onnx()
