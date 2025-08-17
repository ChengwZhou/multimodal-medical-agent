# =====================================
# File: load.py
# ScientISST MOVE — EDF multi-device, multi-modal windowed loader (subject-sequence version)
#   - Aligns scientisst_chest.edf, scientisst_forearm.edf, empatica.edf per subject
#   - Builds fixed-length, stride-based windows in absolute time per subject
#   - Extracts modality tensors expected by the baseline model
#   - (Optional) derives labels from EDF annotations by majority overlap
# Requirements: pyEDFlib, numpy, torch
# =====================================

from __future__ import annotations
import os
import re
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import pyedflib
except Exception as e:
    raise ImportError("This loader requires 'pyEDFlib'. Install via: pip install pyEDFlib")


# -----------------------------
# EDF helpers
# -----------------------------

@dataclass
class EDFSignal:
    label: str
    fs: float
    data: np.ndarray  # shape [T]


@dataclass
class EDFDevice:
    path: str
    start_time_epoch: float
    duration_sec: float
    signals: Dict[str, EDFSignal]
    annotations: List[Tuple[float, float, str]]  # (onset_sec, duration_sec, label)


def _read_edf_signals(path: str) -> EDFDevice:
    f = pyedflib.EdfReader(path)
    try:
        n_signals = f.signals_in_file
        labels = [f.getLabel(i).strip() for i in range(n_signals)]
        fs = [f.getSampleFrequency(i) for i in range(n_signals)]
        sigs = {}
        for i, lab in enumerate(labels):
            data = f.readSignal(i)
            sigs[lab] = EDFSignal(label=lab, fs=float(fs[i]), data=data.astype(np.float32))
        dt = f.getStartdatetime()
        start_epoch = dt.timestamp() if hasattr(dt, 'timestamp') else 0.0
        file_dur = float(f.getFileDuration())
        try:
            anns = f.readAnnotations()
            onsets = anns[0]
            durations = anns[1]
            ann_labels = [a.decode('utf-8') if isinstance(a, (bytes, bytearray)) else str(a) for a in anns[2]]
            annotations = [(float(o), float(d), s) for o, d, s in zip(onsets, durations, ann_labels)]
        except Exception:
            annotations = []
    finally:
        f.close()
    return EDFDevice(path=path, start_time_epoch=start_epoch, duration_sec=file_dur, signals=sigs, annotations=annotations)


# -----------------------------
# Channel selection rules
# -----------------------------

DEFAULT_CHANNEL_PATTERNS = {
    'ecg': [r'ECG'],
    'ppg': [r'PPG', r'BVP'],
    'eda': [r'EDA', r'GSR'],
    'emg': [r'EMG'],
    'temp': [r'TEMP', r'Temperature'],
    'c_acc': [r'CHEST.*ACC.*X', r'CHEST.*ACC.*Y', r'CHEST.*ACC.*Z'],
    'w_acc': [r'WRIST.*ACC.*X', r'WRIST.*ACC.*Y', r'WRIST.*ACC.*Z'],
}

EMPATICA_HINTS = {
    'ppg': [r'BVP', r'PPG'],
    'eda': [r'EDA'],
    'temp': [r'TEMP', r'Temperature'],
    'w_acc': [r'ACC.*X', r'ACC.*Y', r'ACC.*Z'],
}
SCIENTISST_CHEST_HINTS = {
    'ecg': [r'ECG'],
    'c_acc': [r'ACC.*X', r'ACC.*Y', r'ACC.*Z'],
}
SCIENTISST_FOREARM_HINTS = {
    'emg': [r'EMG'],
}


def _match_first(label_list: List[str], patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        regex = re.compile(pat, re.IGNORECASE)
        for lab in label_list:
            if regex.search(lab):
                return lab
    return None


# -----------------------------
# WindowIndex
# -----------------------------

@dataclass
class WindowIndex:
    subject_id: str
    t0: float
    dur: float
    chest: EDFDevice
    forearm: EDFDevice
    wrist: EDFDevice
    y: Optional[int]


# -----------------------------
# Dataset
# -----------------------------

class MoveEDFWindowDataset(Dataset):
    def __init__(
        self,
        root: str,
        window_sec: float = 3.0,
        stride_sec: Optional[float] = None,
        label_map: Optional[Dict[str, int]] = None,
        min_coverage: float = 0.95,
        channel_patterns: Optional[Dict[str, List[str]]] = None,
        none_policy: str = "extra_class",   # "ignore" | "extra_class"
        ignore_index: int = -100,      # for none_policy = "ignore"
    ):
        super().__init__()
        self.root = root
        self.window_sec = float(window_sec)
        self.stride_sec = float(stride_sec) if stride_sec is not None else float(window_sec)
        self.min_coverage = float(min_coverage)
        self.patterns = channel_patterns or DEFAULT_CHANNEL_PATTERNS
        self.none_policy = none_policy
        self.ignore_index = ignore_index

        # scan subjects
        subs = []
        for sid in sorted(os.listdir(root)):
            subj_dir = os.path.join(root, sid)
            if not os.path.isdir(subj_dir):
                continue
            if all(os.path.exists(os.path.join(subj_dir, f)) for f in [
                'empatica.edf', 'scientisst_chest.edf', 'scientisst_forearm.edf'
            ]):
                subs.append(sid)
        if not subs:
            raise FileNotFoundError(f"No subjects found under {root}.")
        self.subjects = subs

        # read devices
        self.devices: Dict[str, Tuple[EDFDevice, EDFDevice, EDFDevice]] = {}
        all_label_strings: List[str] = []
        for sid in self.subjects:
            sdir = os.path.join(root, sid)
            wrist = _read_edf_signals(os.path.join(sdir, 'empatica.edf'))
            chest = _read_edf_signals(os.path.join(sdir, 'scientisst_chest.edf'))
            fore = _read_edf_signals(os.path.join(sdir, 'scientisst_forearm.edf'))
            self.devices[sid] = (chest, fore, wrist)
            for dev in (chest, fore, wrist):
                for (_, _, lab) in dev.annotations:
                    all_label_strings.append(lab)

        if label_map is None:
            uniq = sorted(set(self._normalize_label(s) for s in all_label_strings))
            self.label_map = {s: i for i, s in enumerate(uniq)}
        else:
            self.label_map = label_map

        # Deal with "extra_class"
        if self.none_policy == "extra_class" and "__none__" not in self.label_map:
            self.label_map["__none__"] = len(self.label_map)

        # build per-subject window indices
        self.index_per_subject: Dict[str, List[WindowIndex]] = {}
        for sid in self.subjects:
            chest, fore, wrist = self.devices[sid]
            start_max = max(chest.start_time_epoch, fore.start_time_epoch, wrist.start_time_epoch)
            end_min = min(
                chest.start_time_epoch + chest.duration_sec,
                fore.start_time_epoch + fore.duration_sec,
                wrist.start_time_epoch + wrist.duration_sec,
            )
            total = max(0.0, end_min - start_max)
            if total < self.window_sec * self.min_coverage:
                continue
            n = int(math.floor((total - self.window_sec) / self.stride_sec) + 1)
            windows = []
            for i in range(n):
                t0 = start_max + i * self.stride_sec
                y = self._majority_label((chest, fore, wrist), t0, self.window_sec)
                # None label policy
                if y is None:
                    if self.none_policy == "ignore":
                        y = self.ignore_index
                    elif self.none_policy == "extra_class":
                        y = self.label_map["__none__"]
                windows.append(WindowIndex(subject_id=sid, t0=t0, dur=self.window_sec,
                                           chest=chest, forearm=fore, wrist=wrist, y=y))
            self.index_per_subject[sid] = windows

        # flatten for __len__ & __getitem__ if needed
        self.flatten_index: List[Tuple[str, int]] = [(sid, i) for sid, wlist in self.index_per_subject.items() for i in range(len(wlist))]

    def _normalize_label(self, lab: str) -> str:
        # "lift-1", "lift-2" → "lift"
        if lab.startswith("lift"):
            return "lift"
        # " walk_before", "walk_before_downstairs", "walk_before_elevatordown", "walk_before_elevatorup" → "walk_before"
        if lab.startswith("walk_before"):
            return "walk_before"
        return lab

    # -----------------------------
    # Label resolving
    # -----------------------------
    def _majority_label(self, devices: Tuple[EDFDevice, EDFDevice, EDFDevice], t0: float, dur: float) -> Optional[int]:
        votes: Dict[str, float] = {}
        for dev in devices:
            for onset, length, lab in dev.annotations:
                abs_onset = dev.start_time_epoch + float(onset)
                abs_end = abs_onset + float(length if length > 0 else 0.0)
                w0, w1 = t0, t0 + dur
                inter = max(0.0, min(abs_end, w1) - max(abs_onset, w0))
                if inter > 0:
                    norm_lab = self._normalize_label(lab)
                    votes[norm_lab] = votes.get(norm_lab, 0.0) + inter
        if not votes:
            return None
        lab = max(votes.items(), key=lambda kv: kv[1])[0]
        return self.label_map.get(lab, None)

    # -----------------------------
    # Core window slicing
    # -----------------------------
    def _slice_device(self, dev: EDFDevice, t0_abs: float, dur: float, sel: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        all_labels = list(dev.signals.keys())

        def get_segment(lab: str) -> np.ndarray:
            sig = dev.signals[lab]
            fs = sig.fs
            rel_start = max(0.0, t0_abs - dev.start_time_epoch)
            i0 = int(round(rel_start * fs))
            i1 = int(round((rel_start + dur) * fs))
            x = sig.data[i0:i1]
            need = int(round(dur * fs)) - x.shape[0]
            if need > 0:
                x = np.pad(x, (0, need), mode='constant')
            return x.astype(np.float32)

        for key, patterns in sel.items():
            found = _match_first(all_labels, patterns)
            if found is None:
                fs_guess = max([s.fs for s in dev.signals.values()]) if dev.signals else 1.0
                length = int(round(dur * fs_guess))
                out[key] = np.zeros((length,), dtype=np.float32)
            else:
                out[key] = get_segment(found)
        return out

    def _stack_axes(self, d: Dict[str, np.ndarray], order: List[str]) -> np.ndarray:
        arrs = [d[k] if k in d else np.zeros_like(next(iter(d.values()))) for k in order]
        return np.stack(arrs, axis=0)

    def _get_window_data(self, wi: WindowIndex):
        """get single window data( x_dict & label)"""
        t0 = wi.t0
        dur = wi.dur

        wrist_sel = {
            'ppg': EMPATICA_HINTS.get('ppg', self.patterns['ppg']),
            'eda': EMPATICA_HINTS.get('eda', self.patterns['eda']),
            'temp': EMPATICA_HINTS.get('temp', self.patterns['temp']),
            'ax': EMPATICA_HINTS.get('w_acc', self.patterns['w_acc'][0:1]),
            'ay': EMPATICA_HINTS.get('w_acc', self.patterns['w_acc'][1:2]),
            'az': EMPATICA_HINTS.get('w_acc', self.patterns['w_acc'][2:3]),
        }
        w_parts = self._slice_device(wi.wrist, t0, dur, wrist_sel)
        w_acc = self._stack_axes({'x': w_parts.get('ax', np.array([])),
                                  'y': w_parts.get('ay', np.array([])),
                                  'z': w_parts.get('az', np.array([]))}, ['x','y','z'])

        chest_sel = {
            'ecg': SCIENTISST_CHEST_HINTS.get('ecg', self.patterns['ecg']),
            'cx': SCIENTISST_CHEST_HINTS.get('c_acc', self.patterns['c_acc'][0:1]),
            'cy': SCIENTISST_CHEST_HINTS.get('c_acc', self.patterns['c_acc'][1:2]),
            'cz': SCIENTISST_CHEST_HINTS.get('c_acc', self.patterns['c_acc'][2:3]),
        }
        c_parts = self._slice_device(wi.chest, t0, dur, chest_sel)
        c_acc = self._stack_axes({'x': c_parts.get('cx', np.array([])),
                                  'y': c_parts.get('cy', np.array([])),
                                  'z': c_parts.get('cz', np.array([]))}, ['x','y','z'])

        fore_sel = {
            'emg': SCIENTISST_FOREARM_HINTS.get('emg', self.patterns['emg'])
        }
        f_parts = self._slice_device(wi.forearm, t0, dur, fore_sel)

        x_dict_np = {
            'ecg': np.expand_dims(c_parts['ecg'], 0),
            'ppg': np.expand_dims(w_parts['ppg'], 0),
            'eda': np.expand_dims(w_parts['eda'], 0),
            'emg': np.expand_dims(f_parts['emg'], 0),
            'temp': np.expand_dims(w_parts['temp'], 0),
            'c_acc': c_acc,
            'w_acc': w_acc,
        }
        x_dict = {k: torch.from_numpy(v.copy()) for k, v in x_dict_np.items()}
        return x_dict, wi.y

    # -----------------------------
    # Dataset API
    # -----------------------------
    def __len__(self):
        return len(self.flatten_index)

    def __getitem__(self, idx: int):
        sid, wi_idx = self.flatten_index[idx]
        wi = self.index_per_subject[sid][wi_idx]
        return self._get_window_data(wi)

    # -----------------------------
    # Subject-level sequence access
    # -----------------------------
    def get_subject_sequence(self, subject_id: str) -> List[Tuple[Dict[str, torch.Tensor], Optional[int]]]:
        seq = []
        for wi in self.index_per_subject[subject_id]:
            seq.append(self._get_window_data(wi))
        return seq


# -----------------------------
# Convenience: split subjects
# -----------------------------

def split_by_subject(dataset: MoveEDFWindowDataset, val_ratio: float = 0.2):
    rng = np.random.default_rng(42)
    sub_ids = dataset.subjects
    n_val = max(1, int(round(len(sub_ids) * val_ratio)))
    val_subs = set(rng.choice(sub_ids, size=n_val, replace=False).tolist())
    train_idx, val_idx = [], []
    for i, (sid, wi_idx) in enumerate(dataset.flatten_index):
        (val_idx if sid in val_subs else train_idx).append(i)
    return train_idx, val_idx
