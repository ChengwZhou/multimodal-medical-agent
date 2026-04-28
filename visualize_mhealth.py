"""
mHEALTH Dataset Visualization
Plots raw signal waveforms for each modality with color-coded activity labels.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_ROOT = "/Users/chengweizhou/PycharmProjects/data/MHEALTHDATASET"
SUBJECT_ID = 1
SAMPLE_RATE = 50          # Hz (original)
KEEP_NULL   = False       # if False, null (activity=0) segments are dropped
SAVE_PATH = "save/mhealth_visualization.png"

# ─── Activity labels ──────────────────────────────────────────────────────────
ACTIVITY_NAMES = {
    0: "Null",
    1: "Standing",
    2: "Sitting",
    3: "Lying down",
    4: "Walking",
    5: "Climbing stairs",
    6: "Waist bend fwd",
    7: "Arms elevation",
    8: "Knees bending",
    9: "Cycling",
    10: "Jogging",
    11: "Running",
    12: "Jump fwd & back",
}

# ─── Modality groupings ───────────────────────────────────────────────────────
# Each entry: (modality_name, [channel_names], y_unit)
MODALITIES = [
    ("ECG",                  ["ECG"],          "mV"),
    ("Left Ankle\nAccel",    ["alx","aly","alz"], "m/s²"),
    ("Left Ankle\nGyro",     ["glx","gly","glz"], "deg/s"),
    ("Right Arm\nAccel",     ["arx","ary","arz"], "m/s²"),
    ("Right Arm\nGyro",      ["grx","gry","grz"], "deg/s"),
]

# ─── Load data ────────────────────────────────────────────────────────────────
path = os.path.join(DATA_ROOT, f"mHealth_subject{SUBJECT_ID}.log")
df = pd.read_csv(path, header=None, sep="\t")
df = df.loc[:, [1, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 19, 23]]
df = df.rename(columns={
    1:  "ECG",
    5:  "alx", 6:  "aly", 7:  "alz",
    8:  "glx", 9:  "gly", 10: "glz",
    14: "arx", 15: "ary", 16: "arz",
    17: "grx", 18: "gry", 19: "grz",
    23: "Activity",
})
df = df.dropna().reset_index(drop=True)

# ─── Drop null (Activity=0) segments to show diverse activities ──────────────
if not KEEP_NULL:
    df = df[df["Activity"] != 0].reset_index(drop=True)

# Build a pseudo-continuous time axis (preserves per-sample spacing = 1/SR)
time_axis = np.arange(len(df)) / SAMPLE_RATE   # seconds
labels = df["Activity"].values.astype(int)

# Compute segment boundaries to draw vertical dividers between activities
_boundaries = np.where(np.diff(labels) != 0)[0] + 1

# ─── Color palette for activities ────────────────────────────────────────────
all_acts = sorted(ACTIVITY_NAMES.keys())          # 0-12
n_acts   = len(all_acts)
cmap_acts = plt.colormaps["tab20"].resampled(n_acts)
act_colors = {a: cmap_acts(i) for i, a in enumerate(all_acts)}

# ─── Build figure ─────────────────────────────────────────────────────────────
n_modalities = len(MODALITIES)
fig_height   = 3.5 * n_modalities + 1.5   # extra space for legend
fig, axes = plt.subplots(n_modalities, 1,
                         figsize=(22, fig_height),
                         sharex=True,
                         gridspec_kw={"hspace": 0.45})

channel_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]   # up to 3 lines per panel

for ax, (modal_name, channels, unit) in zip(axes, MODALITIES):
    # ── Background shading per activity segment ──
    # Find contiguous runs of the same label
    boundaries = np.where(np.diff(labels) != 0)[0] + 1
    seg_starts = np.concatenate([[0], boundaries])
    seg_ends   = np.concatenate([boundaries, [len(labels)]])

    for s, e in zip(seg_starts, seg_ends):
        act = labels[s]
        ax.axvspan(time_axis[s], time_axis[e - 1],
                   alpha=0.25, color=act_colors[act], linewidth=0)

    # ── Signal waveforms ──
    for ch, color in zip(channels, channel_colors):
        ax.plot(time_axis, df[ch].values,
                color=color, linewidth=0.6, label=ch, alpha=0.9)

    # ── Vertical dividers between activity segments ──
    for b in _boundaries:
        ax.axvline(x=time_axis[b], color="black", linewidth=0.8,
                   linestyle="--", alpha=0.5)

    ax.set_ylabel(f"{modal_name}\n({unit})", fontsize=9)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.6)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)

axes[-1].set_xlabel("Time (s)", fontsize=10)
fig.suptitle(f"mHEALTH Dataset – Subject {SUBJECT_ID} (null segments removed)",
             fontsize=13, fontweight="bold", y=0.995)

# ── Activity name labels on the top panel ──
ax_top = axes[0]
seg_starts = np.concatenate([[0], _boundaries])
seg_ends   = np.concatenate([_boundaries, [len(labels)]])
for s, e in zip(seg_starts, seg_ends):
    act = labels[s]
    mid_t = (time_axis[s] + time_axis[e - 1]) / 2
    ax_top.text(mid_t, ax_top.get_ylim()[1],
                ACTIVITY_NAMES.get(act, str(act)),
                ha="center", va="bottom", fontsize=7,
                color="black", rotation=0,
                bbox=dict(facecolor=act_colors[act], alpha=0.5,
                          boxstyle="round,pad=0.2", edgecolor="none"))

# ─── Shared legend for activity colors ───────────────────────────────────────
present_acts = sorted(set(labels))
patches = [mpatches.Patch(color=act_colors[a], alpha=0.6,
                           label=f"{a}: {ACTIVITY_NAMES[a]}")
           for a in present_acts]
fig.legend(handles=patches,
           loc="lower center",
           ncol=min(len(present_acts), 7),
           fontsize=8,
           title="Activity",
           title_fontsize=9,
           bbox_to_anchor=(0.5, 0.0),
           framealpha=0.8)

plt.tight_layout(rect=[0, 0.05, 1, 1])

# ─── Save & show ──────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
print(f"Saved to {SAVE_PATH}")
plt.show()
