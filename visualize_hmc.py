"""
HMC Sleep Dataset Visualization
Plots PSG signal waveforms for each modality with color-coded sleep stage labels.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pyedflib
from scipy import signal as scipy_signal

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_ROOT   = "/Users/chengweizhou/PycharmProjects/data/hmc"
SUBJECT_ID  = "SN029"
TARGET_FS   = 100.0          # Hz (loader resamples to this)
PLOT_MIN    = 30             # minutes of recording to visualize
SAVE_PATH   = "save/hmc_visualization.png"

# ─── Sleep stage labels ───────────────────────────────────────────────────────
STAGE_MAP = {
    "Sleep stage W":  0,
    "Sleep stage N1": 1,
    "Sleep stage N2": 2,
    "Sleep stage N3": 3,
    "Sleep stage R":  4,
}
STAGE_NAMES  = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
STAGE_COLORS = {0: "#e8e8e8", 1: "#aec6f5", 2: "#4e91e3", 3: "#1a3c8a", 4: "#f4a43c"}

# ─── Channel groupings ────────────────────────────────────────────────────────
# (panel_name, [channel_names], y_unit)
REQUIRED_CHANNELS = ['EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1', 'EEG C3-M2',
                     'EMG chin', 'EOG E1-M2', 'EOG E2-M2', 'ECG']

MODALITIES = [
    ("EEG Frontal\nF4-M1",          ["EEG F4-M1"],                      "µV"),
    ("EEG Central\nC4-M1 / C3-M2",  ["EEG C4-M1", "EEG C3-M2"],        "µV"),
    ("EEG Occipital\nO2-M1",        ["EEG O2-M1"],                      "µV"),
    ("EOG / EMG",                    ["EMG chin", "EOG E1-M2", "EOG E2-M2"], "µV"),
    ("ECG",                          ["ECG"],                             "mV"),
]
CHANNEL_LINE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]

# ─── Load PSG (EDF) ───────────────────────────────────────────────────────────
psg_path = os.path.join(DATA_ROOT, f"{SUBJECT_ID}.edf")
hyp_path = os.path.join(DATA_ROOT, f"{SUBJECT_ID}_sleepscoring.edf")

print(f"Loading {SUBJECT_ID} ...")
with pyedflib.EdfReader(psg_path) as f:
    labels    = f.getSignalLabels()
    orig_fs   = f.getSampleFrequency(labels.index(REQUIRED_CHANNELS[0]))
    ch_signals = {}
    for ch in REQUIRED_CHANNELS:
        if ch not in labels:
            raise RuntimeError(f"Channel '{ch}' not found in {psg_path}")
        ch_signals[ch] = f.readSignal(labels.index(ch))

# ─── Resample to TARGET_FS ────────────────────────────────────────────────────
resampled = {}
for ch, sig in ch_signals.items():
    if abs(orig_fs - TARGET_FS) < 1e-3:
        resampled[ch] = sig
    else:
        n_out = int(len(sig) * TARGET_FS / orig_fs)
        resampled[ch] = scipy_signal.resample_poly(
            sig, up=int(TARGET_FS), down=int(orig_fs))[:n_out]

n_samples_total = len(resampled[REQUIRED_CHANNELS[0]])

# ─── Load hypnogram ───────────────────────────────────────────────────────────
with pyedflib.EdfReader(hyp_path) as f:
    ann = f.readAnnotations()
    onsets, durations, descriptions = ann

stage_array = np.full(n_samples_total, -1, dtype=np.int8)
for onset, dur, desc in zip(onsets, durations, descriptions):
    if desc not in STAGE_MAP:
        continue
    lbl   = STAGE_MAP[desc]
    i_s   = int(onset * TARGET_FS)
    i_e   = int((onset + dur) * TARGET_FS)
    i_e   = min(i_e, n_samples_total)
    stage_array[i_s:i_e] = lbl

# ─── Trim to PLOT_MIN minutes ─────────────────────────────────────────────────
n_plot   = int(PLOT_MIN * 60 * TARGET_FS)
n_plot   = min(n_plot, n_samples_total)
stages   = stage_array[:n_plot]
time_sec = np.arange(n_plot) / TARGET_FS / 60.0   # minutes

# ─── Build segment boundaries ─────────────────────────────────────────────────
_boundaries = np.where(np.diff(stages) != 0)[0] + 1
seg_starts  = np.concatenate([[0], _boundaries])
seg_ends    = np.concatenate([_boundaries, [n_plot]])

# ─── Plot ─────────────────────────────────────────────────────────────────────
n_panels   = len(MODALITIES)
fig_height = 3.0 * n_panels + 2.0
fig, axes  = plt.subplots(n_panels, 1,
                           figsize=(22, fig_height),
                           sharex=True,
                           gridspec_kw={"hspace": 0.45})

for ax, (panel_name, channels, unit) in zip(axes, MODALITIES):
    # Background shading per stage
    for s, e in zip(seg_starts, seg_ends):
        lbl = stages[s]
        if lbl < 0:
            continue
        ax.axvspan(time_sec[s], time_sec[e - 1],
                   alpha=0.30, color=STAGE_COLORS[lbl], linewidth=0)

    # Signals
    for ch, color in zip(channels, CHANNEL_LINE_COLORS):
        sig = resampled[ch][:n_plot]
        ax.plot(time_sec, sig, color=color, linewidth=0.5, label=ch, alpha=0.9)

    # Vertical dividers at stage transitions
    for b in _boundaries:
        ax.axvline(x=time_sec[b], color="black", linewidth=0.7,
                   linestyle="--", alpha=0.4)

    ax.set_ylabel(f"{panel_name}\n({unit})", fontsize=9)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.6)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)

axes[-1].set_xlabel("Time (min)", fontsize=10)

# ─── Activity annotations on top panel ───────────────────────────────────────
ax_top  = axes[0]
ylim_top = ax_top.get_ylim()
for s, e in zip(seg_starts, seg_ends):
    lbl = stages[s]
    if lbl < 0:
        continue
    mid_t = (time_sec[s] + time_sec[e - 1]) / 2
    ax_top.text(mid_t, ylim_top[1],
                STAGE_NAMES[lbl],
                ha="center", va="bottom", fontsize=7.5,
                bbox=dict(facecolor=STAGE_COLORS[lbl], alpha=0.7,
                          boxstyle="round,pad=0.2", edgecolor="none"))

fig.suptitle(f"HMC Sleep Dataset – {SUBJECT_ID}  (first {PLOT_MIN} min)",
             fontsize=13, fontweight="bold", y=0.998)

# ─── Legend ───────────────────────────────────────────────────────────────────
present_stages = sorted(set(stages[stages >= 0]))
patches = [mpatches.Patch(color=STAGE_COLORS[s], alpha=0.7,
                           label=f"{s}: {STAGE_NAMES[s]}")
           for s in present_stages]
fig.legend(handles=patches,
           loc="lower center", ncol=len(present_stages),
           fontsize=9, title="Sleep Stage", title_fontsize=10,
           bbox_to_anchor=(0.5, 0.0), framealpha=0.8)

plt.tight_layout(rect=[0, 0.04, 1, 1])

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
fig.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
print(f"Saved to {SAVE_PATH}")
plt.show()
