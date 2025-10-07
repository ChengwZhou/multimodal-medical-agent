# =====================================
# File: load_optimized.py
# ScientISST MOVE — Optimized EDF multi-device loader with differencing
#   - Pre-processes and caches all windows during initialization
#   - Ensures slice-first, then resample/difference workflow
#   - Much faster __getitem__ access
#   - Applies differencing instead of filtering within windows
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

from utils.filters import resample_to, zscore  # Only import resample_to, not apply_filter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_info(msg):
    if not is_initialized() or get_rank() == 0:
        logger.info(msg)


def apply_differencing(x: np.ndarray) -> np.ndarray:
    """
    Apply differencing to signal: [t0, t1-t0, t2-t1, t3-t2, ...]
    First value remains unchanged, subsequent values are differences

    Args:
        x: Input signal array of shape [T]

    Returns:
        Differenced signal of same shape [T]
    """
    if len(x) <= 1:
        return x.copy()

    diff_x = np.zeros_like(x)
    diff_x[0] = x[0]  # First value unchanged: t0
    diff_x[1:] = x[1:] - x[:-1]  # Differences: t1-t0, t2-t1, t3-t2, ...
    return diff_x


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
        cache_key = f"w{window_sec}_s{stride_sec}_c{min_coverage}_{none_policy}_{norm_mode}_diff"
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

            # Apply normalization first, then differencing to all processed windows
            log_info("Applying dataset-level normalization then differencing...")
            self._apply_normalization_then_differencing()

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

    def _slice_resample_and_normalize_signal(self, dev: EDFDevice, signal_key: str, t0_abs: float, dur: float,
                                             target_fs: float = 100) -> np.ndarray:
        """Slice signal from device, then apply resampling (no filtering, no differencing yet)"""
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

        # Step 2: Resample to target frequency (no filtering)
        x = resample_to(x, fs, target_fs)

        # Step 3: No differencing here - will be done after normalization

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
                if key == 'w_acc':
                    # Handle 3-axis accelerometer
                    acc_labels = [_match_first(list(wrist.signals.keys()), [p]) for p in patterns]
                    acc_data = []
                    for acc_label in acc_labels:
                        if acc_label:
                            acc_data.append(self._slice_resample_and_normalize_signal(wrist, acc_label, t0, dur, 100))
                        else:
                            acc_data.append(np.zeros(100, dtype=np.float32))
                    wrist_signals['w_acc'] = np.stack(acc_data, axis=0)  # [3, 100]
                else:
                    # Single channel signals
                    wrist_signals[key] = self._slice_resample_and_normalize_signal(wrist, matched_label, t0, dur, 100)
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
                if key == 'c_acc':
                    # Handle 3-axis accelerometer
                    acc_labels = [_match_first(list(chest.signals.keys()), [p]) for p in patterns]
                    acc_data = []
                    for acc_label in acc_labels:
                        if acc_label:
                            acc_data.append(self._slice_resample_and_normalize_signal(chest, acc_label, t0, dur, 100))
                        else:
                            acc_data.append(np.zeros(100, dtype=np.float32))
                    chest_signals['c_acc'] = np.stack(acc_data, axis=0)  # [3, 100]
                else:
                    # Single channel signals
                    chest_signals[key] = self._slice_resample_and_normalize_signal(chest, matched_label, t0, dur, 100)
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
                fore_signals[key] = self._slice_resample_and_normalize_signal(fore, matched_label, t0, dur, 100)
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

    def _apply_normalization_then_differencing(self):
        """Apply dataset-level normalization first, then differencing to all processed windows"""
        if not hasattr(self, 'channel_mean') or not hasattr(self, 'channel_std'):
            raise ValueError("Channel statistics not computed")

        log_info("Applying normalization then differencing to all windows...")

        # Expand dimensions for broadcasting: [C] -> [C, 1]
        mean = self.channel_mean.view(-1, 1)  # [C, 1]
        std = self.channel_std.view(-1, 1)  # [C, 1]

        for i, window in enumerate(self.processed_windows):
            # Step 1: Apply normalization: (x - mean) / std
            normalized_tensor = (window.x_tensor.float() - mean) / std

            # Step 2: Apply differencing to each channel
            differenced_tensor = torch.zeros_like(normalized_tensor)
            for c in range(normalized_tensor.shape[0]):
                channel_data = normalized_tensor[c, :].numpy()  # [T]
                differenced_channel = apply_differencing(channel_data)
                differenced_tensor[c, :] = torch.from_numpy(differenced_channel)

            window.x_tensor = differenced_tensor

            if i % 10000 == 0:
                log_info(f"Processed {i}/{len(self.processed_windows)} windows (normalize + difference)")

            # Debug: Show first few windows
            if i < 3:
                log_info(f"DEBUG Window {i} channel 0:")
                log_info(f"  After normalization: {normalized_tensor[0, :5]}")
                log_info(f"  After differencing: {differenced_tensor[0, :5]}")

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


# -----------------------------
# Test code
# -----------------------------

if __name__ == "__main__":
    import time

    # Test configuration
    data_root = "/Users/chengweizhou/PycharmProjects/data/scientisst-move-annotated-wearable-multimodal-biosignals-recorded-during-everyday-life-activities-in-naturalistic-environments-1.0.1"

    print("=" * 60)
    print("Testing ScientISST MOVE Dataset with Differencing")
    print("=" * 60)

    try:
        # Initialize dataset
        print("Initializing dataset...")
        start_time = time.time()

        dataset = ScientISSTMOVEDataset(
            root=data_root,
            window_sec=1.0,
            stride_sec=10,
            min_coverage=0.95,
            none_policy="extra_class",
            norm_mode='dataset',
            force_reprocess=True  # Force reprocessing to use new differencing logic
        )

        init_time = time.time() - start_time
        print(f"Dataset initialization took: {init_time:.2f} seconds")
        print()

        # Basic dataset info
        print("Dataset Information:")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Number of subjects: {len(dataset.subjects)}")
        print(f"  Subjects: {dataset.subjects[:5]}{'...' if len(dataset.subjects) > 5 else ''}")
        print(f"  Label mapping: {dataset.label_map}")
        print()

        # Test data loading
        print("Testing data loading...")
        sample_indices = [0, len(dataset) // 4, len(dataset) // 2, len(dataset) - 1]

        for i, idx in enumerate(sample_indices):
            if idx < len(dataset):
                x, y = dataset[idx]
                window = dataset.processed_windows[idx]
                print(f"Sample {i + 1} (index {idx}):")
                print(f"  Subject: {window.subject_id}")
                print(f"  Tensor shape: {x.shape}")
                print(f"  Label: {y}")
                print(f"  Tensor dtype: {x.dtype}")
                print(f"  Tensor range: [{x.min():.4f}, {x.max():.4f}]")

                # Check differencing - first few values of first channel
                first_channel = x[0, :10].numpy()
                print(f"  First 10 values of channel 0: {first_channel}")

                # Test differencing logic with a simple example
                if i == 0:  # Only for first sample
                    print("  Testing differencing logic:")
                    test_signal = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0])
                    diff_result = apply_differencing(test_signal)
                    print(f"    Original: {test_signal}")
                    print(f"    Differenced: {diff_result}")
                    expected = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
                    print(f"    Expected: {expected}")
                    print(f"    Match: {np.allclose(diff_result, expected)}")

                # Verify differencing property (for constant signals)
                if np.allclose(first_channel[1:], 0.0, atol=1e-6):
                    print(f"  ✓ Constant signal correctly differenced (differences are ~0)")
                else:
                    # Check if differencing property holds for non-constant signals
                    differences = first_channel[1:]
                    reconstructed_diffs = np.diff(x[0, :len(differences) + 1].numpy())
                    if len(reconstructed_diffs) > 0 and np.allclose(differences, reconstructed_diffs, atol=1e-5):
                        print(f"  ✓ Differencing property verified")
                    else:
                        print(f"  ⚠ Differencing property check inconclusive")
                print()

        # Test subject sequence
        if dataset.subjects:
            test_subject = dataset.subjects[0]
            sequence = dataset.get_subject_sequence(test_subject)
            print(f"Subject '{test_subject}' sequence:")
            print(f"  Number of windows: {len(sequence)}")
            if sequence:
                x_seq, y_seq = sequence[0]
                print(f"  First window shape: {x_seq.shape}")
                print(f"  First window label: {y_seq}")
            print()

        # Test train/validation split
        print("Testing train/validation split...")
        train_idx, val_idx = split_by_subject(dataset, val_ratio=0.2)
        print(f"  Training samples: {len(train_idx)}")
        print(f"  Validation samples: {len(val_idx)}")
        print(f"  Split ratio: {len(val_idx) / (len(train_idx) + len(val_idx)):.3f}")
        print()

        # Test label filtering
        print("Testing label filtering...")
        original_labels = set(dataset.label_map.keys())
        print(f"  Original labels: {original_labels}")

        # Create a copy for testing filtering
        try:
            import copy

            test_dataset = copy.deepcopy(dataset)

            # Remove a label if available (e.g., remove 'lift' if it exists)
            labels_to_remove = []
            for label in ['lift', 'walk_before', '__none__']:
                if label in test_dataset.label_map:
                    labels_to_remove.append(label)
                    break

            if labels_to_remove:
                filtered_dataset = filter_labels(test_dataset, labels_to_remove)
                if filtered_dataset is not None:
                    print(f"  Removed labels: {labels_to_remove}")
                    print(f"  Remaining labels: {set(filtered_dataset.label_map.keys())}")
                    print(f"  Samples after filtering: {len(filtered_dataset)}")
                else:
                    print(f"  Label filtering returned None")
            else:
                print("  No suitable labels found for filtering test")
        except Exception as e:
            print(f"  Label filtering test failed: {e}")
        print()

        # Performance test
        print("Performance test (100 random samples)...")
        import random

        random_indices = random.sample(range(len(dataset)), min(100, len(dataset)))

        start_time = time.time()
        for idx in random_indices:
            x, y = dataset[idx]
        load_time = time.time() - start_time

        print(f"  Loading 100 samples took: {load_time:.4f} seconds")
        print(f"  Average time per sample: {load_time / len(random_indices):.6f} seconds")
        print()

        # Channel statistics (if normalization is enabled)
        if hasattr(dataset, 'channel_mean') and hasattr(dataset, 'channel_std'):
            print("Channel-wise statistics:")
            print(f"  Mean shape: {dataset.channel_mean.shape}")
            print(f"  Std shape: {dataset.channel_std.shape}")
            print(f"  Mean range: [{dataset.channel_mean.min():.4f}, {dataset.channel_mean.max():.4f}]")
            print(f"  Std range: [{dataset.channel_std.min():.4f}, {dataset.channel_std.max():.4f}]")
            print()

        print("=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"Error: Data directory not found - {e}")
        print("Please check the data path and make sure the dataset is available.")

    except Exception as e:
        print(f"Error during testing: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

