import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from visualization.style import apply_neurips_style, DOUBLE_COL
apply_neurips_style()
sns.set_theme(style="ticks", rc={"axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.35})
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.sequential_dataset import SequentialDataset, collate_sequential_batch
from dataset.mHEALTH_loader import MHealthDataset
from models.sigma_former_device import build_former_device
from models.agent_device_masking import DeviceGatingAgent
from utils.modality_config import ModalityConfig


CHECKPOINT_PATH = "/home/cxz760/multimodal-biomedical-agent/checkpoints/mhealth_100e_full_SD_lambda2_5e-2_ecg/best_model.pth"
DATA_ROOT = "/usr/homes/cxz760/data/MHEALTHDATASET"

NUM_CLASSES = 12
NUM_DEVICES = 3
REAL_MODALITIES = 13
TIME_STEPS = 100
BATCH_SIZE = 16
MODEL_DIM = 512
SKIP_STEPS = 1
INIT_THRESHOLD = 0.1
MODAL_FUSION = "cross_atten"
MEM_CONTEXT_CACHE_LENGTH = 10
HISTORY_LENGTH = 10

modalities = [
                ModalityConfig('ecg', 1, 10, 2),
                ModalityConfig('al', 3, 10, 0), ModalityConfig('gl', 3, 10, 0),
                ModalityConfig('ar', 3, 10, 1), ModalityConfig('gr', 3, 10, 1),
            ]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = MHealthDataset(
    data_root=DATA_ROOT,
    subjects=list(range(1, 11)),
    time_steps=100,
    step=50,
    balance=False,
    remove_zero_activity=True
)

val_dataset = SequentialDataset(dataset, subject_ids=[8, 9, 10])
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_sequential_batch,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

model = build_former_device(
    num_classes=NUM_CLASSES,
    model_dim=MODEL_DIM,
    return_mem=True,
    return_sensing_info=True,
    skip_steps=SKIP_STEPS,
    init_threshold=INIT_THRESHOLD,
    modalities=modalities,
    modal_fusion=MODAL_FUSION
).to(device)

agent = DeviceGatingAgent(
    num_modalities=13,
    modalities=modalities,
    feature_dim=MODEL_DIM,
    history_length=HISTORY_LENGTH
).to(device)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
agent.load_state_dict(checkpoint["agent_state_dict"], strict=False)
model.eval()
agent.eval()

active_sum = np.zeros((REAL_MODALITIES, TIME_STEPS))
total_instances = 0

device_sensor_ranges = []
start = 0
for modal in modalities:
    device_sensor_ranges.append((start, start + modal.in_ch))
    start += modal.in_ch

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Collecting fine-grained sensor masks"):
        sequences = batch.sequences.to(device)   # [B, S, C, T]
        labels = batch.labels.to(device)
        B, max_seq_len, C, T = sequences.shape

        mem = None
        mem_cache = []
        mem_running_context = None
        device_history = torch.ones(B, HISTORY_LENGTH, 13).to(device)

        for step in range(max_seq_len):
            window = sequences[:, step, :, :]                    # [B, C, 100]

            # ====== Agent（device-level）======
            if mem is not None:
                # print(mem.size())
                # print(device_history.size())
                out = agent(represent_features=mem,
                            sensor_history=device_history,
                            use_straight_through=False)
                p_device = out['p_soft']                       # [B, 2]

                # device-level masking
                window_masked = torch.zeros_like(window)
                ch_idx = 0
                for d in range(NUM_DEVICES):
                    ch = modalities[d].in_ch
                    mask_d = p_device[:, d].view(B, 1, 1)        # [B,1,1]
                    window_masked[:, ch_idx:ch_idx+ch, :] = window[:, ch_idx:ch_idx+ch, :] * mask_d
                    ch_idx += ch

                # print(device_history.size())
                device_history = torch.cat([device_history[:, 1:, :],
                                           torch.round(p_device).unsqueeze(1)], dim=1)
            else:
                window_masked = window

            # ====== Model forward ======
            logits, mem, sensing_info = model(window_masked, mem_running_context)

            masks_list = sensing_info['masks']    # list of length=2, each [B, 6, 100]

            for dev_idx, mask_dev in enumerate(masks_list):
                print(mask_dev.size())
                if dev_idx == 2:
                    print(mask_dev.sum(dim=0).cpu().size())
                    active_sum[0:1] += mask_dev.sum(dim=0).cpu().numpy()
                else:
                    active_sum[dev_idx*6+1:(dev_idx+1)*6+1] += mask_dev.sum(dim=0).cpu().numpy()   # [6,100]

            total_instances += B

            # memory cache update
            if mem_running_context is None:
                mem_running_context = mem.detach()
            else:
                mem_cache.append(mem.detach())
                if len(mem_cache) > MEM_CONTEXT_CACHE_LENGTH:
                    mem_cache.pop(0)
                mem_running_context = torch.stack(mem_cache).mean(0)

activation_rate = active_sum / total_instances          # [12, 100], ∈ [0,1]

print("total_instances", total_instances)

fig, ax = plt.subplots(figsize=(DOUBLE_COL * 1.3, 4.5))
sns.heatmap(
    activation_rate,
    ax=ax,
    cmap="RdYlBu_r",
    annot=False,
    cbar_kws={"label": "Activation Rate", "shrink": 0.8},
    xticklabels=10,
    yticklabels=[f"Sensor {i}" for i in range(13)],
    linewidths=0,
)
ax.set_xlabel("Time Step")
ax.set_ylabel("Sensor")
ax.set_title("Per-Sensor Activation Rate — Device-Level Gating (mHEALTH Val Set)")
plt.tight_layout()
plt.savefig("mhealth_fine_grained_12x100_sensor_skipping_heatmap.png")
plt.show()

print("\n=== Average activation rate per sensor (lower = more aggressively skipped) ===")
for i in range(13):
    rate = activation_rate[i].mean()
    print(f"Sensor {i:2d}: {rate:.4f}  →  {rate*100:6.2f}% active  |  Skip ratio: {(1-rate)*100:5.2f}%")

print(activation_rate)
np.save("mhealth_13x100_activation_rate.npy", activation_rate)