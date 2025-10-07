# =====================================
# File: load_optimized.py
# ScientISST MOVE — Optimized EDF multi-device loader
#   - Pre-processes and caches all windows during initialization
#   - Ensures slice-first, then filter/resample workflow
#   - Much faster __getitem__ access
# =====================================

from __future__ import annotations
import os
import re
import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.distributed import get_rank, is_initialized

try:
    import pyedflib
except Exception as e:
    raise ImportError("This loader requires 'pyEDFlib'. Install via: pip install pyEDFlib")

from utils.filters import apply_filter, resample_to, zscore


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_info(msg):
    if not is_initialized() or get_rank() == 0:
        logger.info(msg)


# -----------------------------
# EDF helpers (unchanged)
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
    return EDFDevice(path=path, start_time_epoch=start_epoch, duration_sec=file_dur, signals=sigs,
                     annotations=annotations)


# -----------------------------
# Channel selection rules (unchanged)
# -----------------------------

DEFAULT_CHANNEL_PATTERNS = {
    'ecg': [r'ECG'],
    'ppg': [r'PPG', r'BVP'],
    'eda': [r'EDA', r'GSR'],
    'emg': [r'EMG'],
    'temp': [r'temp', r'Temperature'],
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
    'ecg_gel': [r'ECG.*gel'],  # 匹配 gel 电极
    'ecg_textile': [r'ECG.*dry'],
    'c_acc': [r'ACC.*X', r'ACC.*Y', r'ACC.*Z'],
}
SCIENTISST_FOREARM_HINTS = {
    'emg': [r'EMG'],
    'eda': [r'EDA'],  # S3
    'ppg': [r'PPG'],  # S5
}


def _match_first(label_list: List[str], patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        regex = re.compile(pat, re.IGNORECASE)
        for lab in label_list:
            if regex.search(lab):
                return lab
    return None


# -----------------------------
# Pre-processed Window Data
# -----------------------------

@dataclass
class ProcessedWindow:
    """Stores pre-processed window data as integrated tensor"""
    subject_id: str
    window_idx: int
    x_tensor: torch.Tensor  # Integrated tensor [C=14, T]
    y: Optional[int]

    def __getstate__(self):
        # Convert tensor to numpy for pickling
        state = self.__dict__.copy()
        state['x_tensor'] = self.x_tensor.numpy()
        return state

    def __setstate__(self, state):
        # Convert numpy back to tensor
        self.__dict__.update(state)
        self.x_tensor = torch.from_numpy(self.x_tensor)


# -----------------------------
# Optimized Dataset
# -----------------------------

class ScientISSTMOVEDataset(Dataset):
    def __init__(
            self,
            root: str,
            window_sec: float = 1.0,
            stride_sec: Optional[float] = 0.5,
            label_map: Optional[Dict[str, int]] = None,
            min_coverage: float = 0.95,
            channel_patterns: Optional[Dict[str, List[str]]] = None,
            none_policy: str = "extra_class",  # "ignore" | "extra_class"
            ignore_index: int = -100,
            cache_dir: Optional[str] = None,  # Directory to cache processed windows
            force_reprocess: bool = False,  # Force reprocessing even if cache exists
            norm_mode: Optional[str] = 'dataset',
            # "dataset" for dataset-level normalization, None for no normalization
    ):
        super().__init__()
        self.root = root
        self.window_sec = float(window_sec)
        self.stride_sec = float(stride_sec) if stride_sec is not None else float(window_sec)
        self.min_coverage = float(min_coverage)
        self.patterns = channel_patterns or DEFAULT_CHANNEL_PATTERNS
        self.none_policy = none_policy
        self.ignore_index = ignore_index
        self.norm_mode = norm_mode

        # Setup cache
        self.cache_dir = cache_dir or os.path.join(root, '.cache')
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        cache_key = f"w{window_sec}_s{stride_sec}_c{min_coverage}_{none_policy}_{norm_mode}"
        self.cache_file = os.path.join(self.cache_dir, f"processed_windows_{cache_key}.pkl")

        log_info(f"Initializing dataset from {root}")
        log_info(f"Cache file: {self.cache_file}")

        # Try to load from cache first
        if not force_reprocess and os.path.exists(self.cache_file):
            log_info("Loading from cache...")
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.subjects = cache_data['subjects']
                    self.label_map = cache_data['label_map']
                    self.processed_windows = cache_data['processed_windows']
                    self.subject_to_windows = cache_data['subject_to_windows']
                    if norm_mode == 'dataset':
                        self.channel_mean = cache_data['channel_mean']
                        self.channel_std = cache_data['channel_std']
                    log_info(f"Loaded {len(self.processed_windows)} windows from cache")
                    return
            except Exception as e:
                log_info(f"Cache loading failed: {e}, reprocessing...")

        # Scan subjects
        log_info("Scanning subjects...")
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
        log_info(f"Found {len(subs)} subjects")

        # Read devices and collect labels
        log_info("Reading EDF files...")
        self.devices: Dict[str, Tuple[EDFDevice, EDFDevice, EDFDevice]] = {}
        all_label_strings: List[str] = []
        for i, sid in enumerate(self.subjects):
            if i % 5 == 0:
                log_info(f"Processing subject {i + 1}/{len(self.subjects)}: {sid}")
            sdir = os.path.join(root, sid)
            wrist = _read_edf_signals(os.path.join(sdir, 'empatica.edf'))
            chest = _read_edf_signals(os.path.join(sdir, 'scientisst_chest.edf'))
            fore = _read_edf_signals(os.path.join(sdir, 'scientisst_forearm.edf'))
            self.devices[sid] = (chest, fore, wrist)
            for dev in (chest, fore, wrist):
                for (_, _, lab) in dev.annotations:
                    all_label_strings.append(lab)

        # Setup label mapping
        if label_map is None:
            uniq = sorted(set(self._normalize_label(s) for s in all_label_strings))
            self.label_map = {s: i for i, s in enumerate(uniq)}
        else:
            self.label_map = label_map

        if self.none_policy == "extra_class" and "__none__" not in self.label_map:
            self.label_map["__none__"] = len(self.label_map)

        log_info(f"Label mapping: {self.label_map}")

        # Pre-process all windows
        log_info("Pre-processing all windows...")
        self._preprocess_all_windows()

        # Compute channel-wise statistics for normalization if needed
        if self.norm_mode == 'dataset':
            log_info("Computing channel-wise mean and std...")
            self.channel_mean, self.channel_std = self._compute_channel_stats()
            log_info(f"CHANNEL_MEAN shape: {self.channel_mean.shape}")
            log_info(f"CHANNEL_STD shape: {self.channel_std.shape}")

            # Apply normalization to all processed windows
            log_info("Applying dataset-level normalization...")
            self._apply_normalization()

        # Save to cache
        log_info("Saving to cache...")
        try:
            cache_data = {
                'subjects': self.subjects,
                'label_map': self.label_map,
                'processed_windows': self.processed_windows,
                'subject_to_windows': self.subject_to_windows,
            }
            if self.norm_mode == 'dataset':
                cache_data['channel_mean'] = self.channel_mean
                cache_data['channel_std'] = self.channel_std

            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            log_info(f"Cache saved to {self.cache_file}")
        except Exception as e:
            log_info(f"Cache saving failed: {e}")

    def _normalize_label(self, lab: str) -> str:
        # "lift-1", "lift-2" → "lift"
        if lab.startswith("lift"):
            return "lift"
        # " walk_before", "walk_before_downstairs", etc. → "walk_before"
        if lab.startswith("walk_before"):
            return "walk_before"
        return lab

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

    def _slice_and_process_signal(self, dev: EDFDevice, signal_key: str, t0_abs: float, dur: float,
                                  target_fs: float = 100, signal_type: str = "default") -> np.ndarray:
        """Slice signal from device, then apply filtering and resampling"""
        if signal_key not in dev.signals:
            # Return zeros if signal not found
            length = int(round(dur * target_fs))
            return np.zeros((length,), dtype=np.float32)

        sig = dev.signals[signal_key]
        fs = sig.fs

        # Step 1: Slice the signal
        rel_start = max(0.0, t0_abs - dev.start_time_epoch)
        i0 = int(round(rel_start * fs))
        i1 = int(round((rel_start + dur) * fs))
        x = sig.data[i0:i1]

        # Pad if needed
        expected_len = int(round(dur * fs))
        if x.shape[0] < expected_len:
            x = np.pad(x, (0, expected_len - x.shape[0]), mode='constant')

        # Step 2: Apply filtering (after slicing)
        if signal_type == "ecg":
            x = apply_filter(x, fs, "ecg")
        elif signal_type == "ppg":
            x = apply_filter(x, fs, "ppg")
        elif signal_type == "eda":
            x = apply_filter(x, fs, "eda")
        elif signal_type == "acc":
            x = apply_filter(x, fs, "acc")

        # Step 3: Resample to target frequency
        x = resample_to(x, fs, target_fs)

        return x.astype(np.float32)

    def _integrate_signals_to_tensor(self, wrist_signals: dict, chest_signals: dict,
                                     fore_signals: dict) -> torch.Tensor:
        """Integrate all signals into a single tensor [C=14, T]"""
        # Single-channel signals (8 channels)
        parts = []

        # Add single-channel signals in order
        signal_keys = [
            ('ecg_chest_gel', chest_signals.get('ecg_gel', np.zeros(100))),
            ('ecg_chest_textile', chest_signals.get('ecg_textile', np.zeros(100))),
            ('eda_forearm_scientisst', fore_signals.get('eda', np.zeros(100))),
            ('eda_wrist_e4', wrist_signals.get('eda', np.zeros(100))),
            ('ppg_forearm_scientisst', fore_signals.get('ppg', np.zeros(100))),
            ('ppg_wrist_e4', wrist_signals.get('ppg', np.zeros(100))),
            ('emg_forearm', fore_signals.get('emg', np.zeros(100))),
            ('temp_wrist', wrist_signals.get('temp', np.zeros(100))),
        ]

        for _, signal in signal_keys:
            # Ensure signal is 1D, then convert to tensor and add channel dim
            if signal.ndim == 1:
                parts.append(torch.from_numpy(signal).unsqueeze(0))  # [1, T]
            else:
                parts.append(torch.from_numpy(signal))  # Should already be [1, T]

        # Multi-channel accelerometer signals (6 channels total)
        c_acc = chest_signals.get('c_acc', np.zeros((3, 100)))  # [3, 100]
        w_acc = wrist_signals.get('w_acc', np.zeros((3, 100)))  # [3, 100]

        # Convert to tensors
        c_acc_tensor = torch.from_numpy(c_acc)  # [3, T]
        w_acc_tensor = torch.from_numpy(w_acc)  # [3, T]

        # Ensure correct dimensions
        if c_acc_tensor.dim() == 1:
            c_acc_tensor = c_acc_tensor.unsqueeze(0)  # [1, T] -> will be split later
        if w_acc_tensor.dim() == 1:
            w_acc_tensor = w_acc_tensor.unsqueeze(0)  # [1, T] -> will be split later

        # Split accelerometer channels and add to parts
        c_chs = list(torch.unbind(c_acc_tensor, dim=0))  # List of [T] tensors
        c_chs = [ch.unsqueeze(0) if ch.dim() == 1 else ch for ch in c_chs]  # Ensure [1, T]

        w_chs = list(torch.unbind(w_acc_tensor, dim=0))  # List of [T] tensors
        w_chs = [ch.unsqueeze(0) if ch.dim() == 1 else ch for ch in w_chs]  # Ensure [1, T]

        # Concatenate all parts: 8 single + 3 chest_acc + 3 wrist_acc = 14 channels
        all_parts = parts + c_chs + w_chs
        x_integrated = torch.cat(all_parts, dim=0)  # [14, T]

        return x_integrated

    def _preprocess_single_window(self, subject_id: str, t0: float, dur: float) -> ProcessedWindow:
        """Process a single window and return ProcessedWindow object with integrated tensor"""
        chest, fore, wrist = self.devices[subject_id]

        # Get label for this window
        y = self._majority_label((chest, fore, wrist), t0, dur)
        if y is None:
            if self.none_policy == "ignore":
                y = self.ignore_index
            elif self.none_policy == "extra_class":
                y = self.label_map["__none__"]

        # Process wrist signals
        wrist_signals = {}
        for key, patterns in EMPATICA_HINTS.items():
            matched_label = _match_first(list(wrist.signals.keys()), patterns)
            if matched_label:
                if key == 'ppg':
                    wrist_signals[key] = self._slice_and_process_signal(wrist, matched_label, t0, dur, 100, "ppg")
                elif key == 'eda':
                    wrist_signals[key] = self._slice_and_process_signal(wrist, matched_label, t0, dur, 100, "eda")
                elif key == 'temp':
                    wrist_signals[key] = self._slice_and_process_signal(wrist, matched_label, t0, dur, 100, "default")
                elif key == 'w_acc':
                    # Handle 3-axis accelerometer
                    acc_labels = [_match_first(list(wrist.signals.keys()), [p]) for p in patterns]
                    acc_data = []
                    for acc_label in acc_labels:
                        if acc_label:
                            acc_data.append(self._slice_and_process_signal(wrist, acc_label, t0, dur, 100, "acc"))
                        else:
                            acc_data.append(np.zeros(100, dtype=np.float32))
                    wrist_signals['w_acc'] = np.stack(acc_data, axis=0)  # [3, 100]
            else:
                if key == 'w_acc':
                    wrist_signals[key] = np.zeros((3, 100), dtype=np.float32)
                else:
                    wrist_signals[key] = np.zeros(100, dtype=np.float32)

        # Process chest signals
        chest_signals = {}
        for key, patterns in SCIENTISST_CHEST_HINTS.items():
            matched_label = _match_first(list(chest.signals.keys()), patterns)
            if matched_label:
                if key in ['ecg_gel', 'ecg_textile']:
                    chest_signals[key] = self._slice_and_process_signal(chest, matched_label, t0, dur, 100, "ecg")
                elif key == 'c_acc':
                    # Handle 3-axis accelerometer
                    acc_labels = [_match_first(list(chest.signals.keys()), [p]) for p in patterns]
                    acc_data = []
                    for acc_label in acc_labels:
                        if acc_label:
                            acc_data.append(self._slice_and_process_signal(chest, acc_label, t0, dur, 100, "acc"))
                        else:
                            acc_data.append(np.zeros(100, dtype=np.float32))
                    chest_signals['c_acc'] = np.stack(acc_data, axis=0)  # [3, 100]
            else:
                if key == 'c_acc':
                    chest_signals[key] = np.zeros((3, 100), dtype=np.float32)
                else:
                    chest_signals[key] = np.zeros(100, dtype=np.float32)

        # Process forearm signals
        fore_signals = {}
        for key, patterns in SCIENTISST_FOREARM_HINTS.items():
            matched_label = _match_first(list(fore.signals.keys()), patterns)
            if matched_label:
                if key == 'emg':
                    fore_signals[key] = self._slice_and_process_signal(fore, matched_label, t0, dur, 100, "default")
                elif key == 'eda':
                    fore_signals[key] = self._slice_and_process_signal(fore, matched_label, t0, dur, 100, "eda")
                elif key == 'ppg':
                    fore_signals[key] = self._slice_and_process_signal(fore, matched_label, t0, dur, 100, "ppg")
            else:
                fore_signals[key] = np.zeros(100, dtype=np.float32)

        # Integrate all signals into single tensor
        x_tensor = self._integrate_signals_to_tensor(wrist_signals, chest_signals, fore_signals)

        return ProcessedWindow(
            subject_id=subject_id,
            window_idx=-1,  # Will be set later
            x_tensor=x_tensor,  # [14, 100]
            y=y
        )

    def _preprocess_all_windows(self):
        """Pre-process all windows for all subjects"""
        self.processed_windows: List[ProcessedWindow] = []
        self.subject_to_windows: Dict[str, List[int]] = {}  # subject_id -> list of window indices

        total_windows = 0
        for sid in self.subjects:
            chest, fore, wrist = self.devices[sid]
            start_max = max(chest.start_time_epoch, fore.start_time_epoch, wrist.start_time_epoch)
            end_min = min(
                chest.start_time_epoch + chest.duration_sec,
                fore.start_time_epoch + fore.duration_sec,
                wrist.start_time_epoch + wrist.duration_sec,
            )
            total_duration = max(0.0, end_min - start_max)
            if total_duration < self.window_sec * self.min_coverage:
                continue
            n_windows = int(math.floor((total_duration - self.window_sec) / self.stride_sec) + 1)
            total_windows += n_windows

        log_info(f"Total windows to process: {total_windows}")

        processed_count = 0
        for sid in self.subjects:
            log_info(f"Processing subject {sid}...")
            chest, fore, wrist = self.devices[sid]
            start_max = max(chest.start_time_epoch, fore.start_time_epoch, wrist.start_time_epoch)
            end_min = min(
                chest.start_time_epoch + chest.duration_sec,
                fore.start_time_epoch + fore.duration_sec,
                wrist.start_time_epoch + wrist.duration_sec,
            )
            total_duration = max(0.0, end_min - start_max)
            if total_duration < self.window_sec * self.min_coverage:
                continue

            n_windows = int(math.floor((total_duration - self.window_sec) / self.stride_sec) + 1)
            subject_window_indices = []

            for i in range(n_windows):
                t0 = start_max + i * self.stride_sec
                processed_window = self._preprocess_single_window(sid, t0, self.window_sec)
                processed_window.window_idx = len(self.processed_windows)

                self.processed_windows.append(processed_window)
                subject_window_indices.append(processed_window.window_idx)

                processed_count += 1
                if processed_count % 10000 == 0:
                    log_info(f"Processed {processed_count}/{total_windows} windows")

            self.subject_to_windows[sid] = subject_window_indices

        log_info(f"Pre-processing complete! Total windows: {len(self.processed_windows)}")

    def _compute_channel_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-channel mean and std across all windows
        Returns:
            channel_mean: torch.Tensor of shape [C]
            channel_std: torch.Tensor of shape [C]
        """
        if not self.processed_windows:
            raise ValueError("No processed windows available for computing statistics")

        # Get first window to determine shape
        first_tensor = self.processed_windows[0].x_tensor  # [C, T]
        C, T = first_tensor.shape

        # Accumulate statistics
        sum_vals = torch.zeros(C, dtype=torch.float64)
        sum_squares = torch.zeros(C, dtype=torch.float64)
        count = 0

        for window in self.processed_windows:
            tensor = window.x_tensor.float()  # [C, T]
            # Compute mean and variance per channel
            channel_means = tensor.mean(dim=1)  # [C]
            channel_vars = tensor.var(dim=1, unbiased=False)  # [C]

            sum_vals += channel_means.double()
            sum_squares += (channel_vars + channel_means.pow(2)).double()
            count += 1

        # Compute final statistics
        channel_mean = (sum_vals / count).float()
        channel_std = torch.sqrt(sum_squares / count - channel_mean.pow(2).double()).float()

        # Clamp std to avoid division by zero
        channel_std = channel_std.clamp(min=1e-6)

        return channel_mean, channel_std

    def _apply_normalization(self):
        """Apply dataset-level normalization to all processed windows"""
        if not hasattr(self, 'channel_mean') or not hasattr(self, 'channel_std'):
            raise ValueError("Channel statistics not computed")

        log_info("Applying normalization to all windows...")

        # Expand dimensions for broadcasting: [C] -> [C, 1]
        mean = self.channel_mean.view(-1, 1)  # [C, 1]
        std = self.channel_std.view(-1, 1)  # [C, 1]

        for i, window in enumerate(self.processed_windows):
            # Apply normalization: (x - mean) / std
            window.x_tensor = (window.x_tensor.float() - mean) / std

            if i % 10000 == 0:
                log_info(f"Normalized {i}/{len(self.processed_windows)} windows")

    def __len__(self):
        return len(self.processed_windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Fast access to pre-processed and integrated data"""
        window = self.processed_windows[idx]
        y = -100 if window.y is None else int(window.y)
        return window.x_tensor, y  # [C=14, T], int

    def get_subject_sequence(self, subject_id: str) -> List[Tuple[torch.Tensor, int]]:
        """Get all windows for a specific subject as a sequence"""
        if subject_id not in self.subject_to_windows:
            return []

        sequence = []
        for window_idx in self.subject_to_windows[subject_id]:
            window = self.processed_windows[window_idx]
            y = -100 if window.y is None else int(window.y)
            sequence.append((window.x_tensor, y))

        return sequence


# -----------------------------
# Utility functions (unchanged)
# -----------------------------

def split_by_subject(dataset: ScientISSTMOVEDataset, val_ratio: float = 0.2):
    rng = np.random.default_rng(42)
    sub_ids = dataset.subjects
    n_val = max(1, int(round(len(sub_ids) * val_ratio)))
    val_subs = set(rng.choice(sub_ids, size=n_val, replace=False).tolist())

    train_idx, val_idx = [], []
    for i, window in enumerate(dataset.processed_windows):
        if window.subject_id in val_subs:
            val_idx.append(i)
        else:
            train_idx.append(i)

    return train_idx, val_idx


def filter_labels(dataset: ScientISSTMOVEDataset, remove_labels):
    """Remove specified labels and update the dataset"""
    remove_set = set(remove_labels)
    to_remove = [lbl for lbl in dataset.label_map if lbl in remove_set]

    if not to_remove:
        return dataset

    remove_ids = {dataset.label_map[lbl] for lbl in to_remove}

    new_label_map = {}
    new_id = 0
    id_map = {}
    for lbl, old_id in dataset.label_map.items():
        if lbl in remove_set:
            continue
        new_label_map[lbl] = new_id
        id_map[old_id] = new_id
        new_id += 1

    dataset.label_map = new_label_map

    filtered_windows = []
    new_subject_to_windows = {}

    for window in dataset.processed_windows:
        if window.y is not None and window.y in remove_ids:
            continue

        if window.y is None or window.y == dataset.ignore_index:
            pass
        else:
            window.y = id_map[window.y]

        window.window_idx = len(filtered_windows)
        filtered_windows.append(window)

        if window.subject_id not in new_subject_to_windows:
            new_subject_to_windows[window.subject_id] = []
        new_subject_to_windows[window.subject_id].append(window.window_idx)

    dataset.processed_windows = filtered_windows
    dataset.subject_to_windows = new_subject_to_windows

    return dataset