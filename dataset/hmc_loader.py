import os
import numpy as np
import logging
from torch.utils.data import Dataset
import torch
from collections import Counter
from typing import List, Tuple, Optional, Dict

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch.utils.data import DataLoader
import pyedflib
from scipy import signal
from scipy.signal import butter, lfilter, iirnotch
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_info(msg):
    from torch.distributed import is_initialized, get_rank
    if not is_initialized() or get_rank() == 0:
        logger.info(msg)

class HMCSleepDataset(Dataset):
    """
    Dataset for Haaglanden Medisch Centrum (HMC) Sleep Staging Database
    - Uses SNXXX.edf (PSG) and SNXXX_sleepscoring.edf (hypnogram)
    - Preprocessing: 100 Hz resampling, 30s epochs, global channel-wise Gaussian normalization
    - Optional: notch (50/60 Hz), EMG HPF (15 Hz), ECG artifact removal
    - Fully compatible with DeltaDataset and SequentialDataset
    """
    def __init__(self, data_root: str, subjects: List[str] = None,
                 time_steps: int = 3000, step: int = 3000,
                 balance: bool = False, majority_n: int = 30000,
                 remove_wake: bool = False,
                 apply_notch: bool = False, notch_freq: float = 50.0,
                 apply_emg_hp: bool = False, emg_hp_cutoff: float = 15.0,
                 apply_ecg_filter: bool = False,
                 global_stats: Optional[Dict[str, np.ndarray]] = None):
        """
        Args:
            data_root: Root directory containing SNXXX.edf and SNXXX_sleepscoring.edf
            subjects: List of subject IDs as strings, e.g., ["SN001", "SN002"]
            time_steps: 3000 (30s * 100Hz)
            step: 3000 (non-overlapping)
            balance: Downsample majority class
            remove_wake: Remove Wake (0) epochs
            apply_notch: Apply notch filter at notch_freq
            apply_emg_hp: High-pass EMG at 15 Hz
            apply_ecg_filter: Remove ECG artifacts if ECG channel exists
            global_stats: Dict with 'mean' and 'std' for channel-wise normalization (from train set)
        """
        self.data_root = data_root
        if subjects is not None:
            self.subjects = sorted(subjects)
        else:
            self.subjects = self.get_all_subjects_name()
        self.time_steps = time_steps
        self.step = step
        self.balance = balance
        self.majority_n = majority_n
        self.remove_wake = remove_wake

        # Preprocessing flags
        self.apply_notch = apply_notch
        self.notch_freq = notch_freq
        self.apply_emg_hp = apply_emg_hp
        self.emg_hp_cutoff = emg_hp_cutoff
        self.apply_ecg_filter = apply_ecg_filter
        self.global_stats = global_stats

        # Required channels (must match EDF labels)
        self.required_channels = ['EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1', 'EEG C3-M2', 'E in', 'EOG E1-M2', 'EOG E2-M2', 'ECG']
        self.ecg_channel_name = "ECG"

        # Stage mapping from EDF+ standard
        self.stage_map = {
            "Sleep stage W": 0,
            "Sleep stage N1": 1,
            "Sleep stage N2": 2,
            "Sleep stage N3": 3,
            "Sleep stage R": 4
        }

        self.target_fs = 100.0

        self.subject_to_windows = {}
        self.cache_dir = Path(data_root) / "hmc_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.preloaded = False
        self.X = None
        self.y = None

    def get_all_subjects_name(self) -> List[str]:
        """
        Scan the data_root directory and return all valid subject IDs in the format
        ['SN001', 'SN002', ..., 'SN154'] based on existing .edf files.
        """
        subjects = set()
        if not os.path.isdir(self.data_root):
            log_info(f"Data root {self.data_root} is not a valid directory")
            return []

        for filename in os.listdir(self.data_root):
            if filename.endswith(".edf") and "_sleepscoring" not in filename:
                # Extract SNXXX from SNXXX.edf
                if filename.startswith("SN") and filename[2:5].isdigit():
                    subj_id = filename[:5]  # e.g., SN001
                    psg_path = os.path.join(self.data_root, f"{subj_id}.edf")
                    hyp_path = os.path.join(self.data_root, f"{subj_id}_sleepscoring.edf")
                    if os.path.exists(psg_path) and os.path.exists(hyp_path):
                        subjects.add(subj_id)

        valid_subjects = sorted(subjects)
        log_info(f"Found {len(valid_subjects)} valid subjects: {valid_subjects[:5]}...{valid_subjects[-5:]}")
        return valid_subjects

    def _load_one_subject(self, subj):
        psg_path = os.path.join(self.data_root, f"{subj}.edf")
        hyp_path = os.path.join(self.data_root, f"{subj}_sleepscoring.edf")

        if not (os.path.exists(psg_path) and os.path.exists(hyp_path)):
            log_info(f"Subject {subj}: Missing PSG or hypnogram file, skipping")
            return None, None

        # Load PSG
        with pyedflib.EdfReader(psg_path) as f_psg:
            labels = f_psg.getSignalLabels()
            ch_idx = {}
            missing = False
            for ch in self.required_channels:
                if ch in labels:
                    ch_idx[ch] = labels.index(ch)
                else:
                    log_info(f"Subject {subj}: Missing channel {ch}, skipping")
                    missing = True
                    break
            if missing:
                return None, None

            ecg_idx = labels.index(self.ecg_channel_name) if self.ecg_channel_name in labels else None
            orig_fs = f_psg.getSampleFrequency(ch_idx[self.required_channels[0]])
            if not all(f_psg.getSampleFrequency(ch_idx[c]) == orig_fs for c in self.required_channels):
                log_info(f"Subject {subj}: Inconsistent fs, skipping")
                return None, None

            raw_sig = {ch: f_psg.readSignal(idx) for ch, idx in ch_idx.items()}
            if ecg_idx is not None and self.apply_ecg_filter:
                raw_sig[self.ecg_channel_name] = f_psg.readSignal(ecg_idx)

        # Load hypnogram
        with pyedflib.EdfReader(hyp_path) as f_hyp:
            ann = f_hyp.readAnnotations()
            onset = ann[0]  # seconds
            duration = ann[1]
            description = ann[2]

        # Build stage array
        stages = []
        for d, desc in zip(duration, description):
            if desc not in self.stage_map:
                continue
            label = self.stage_map[desc]
            num_samples = int(d * self.target_fs)
            stages.extend([label] * num_samples)

        y = np.array(stages, dtype=np.int64)

        # Apply filtering at original fs
        filtered = {}
        for ch, sig in raw_sig.items():
            if self.apply_notch and ch != self.ecg_channel_name:
                b, a = iirnotch(self.notch_freq / (orig_fs / 2), 30.0)
                sig = lfilter(b, a, sig)

            if self.apply_emg_hp and ch == "EMG chin":
                b, a = butter(1, self.emg_hp_cutoff / (orig_fs / 2), btype='high')
                sig = lfilter(b, a, sig)

            if self.apply_ecg_filter and ch != self.ecg_channel_name and ecg_idx is not None:
                ecg_ref = raw_sig[self.ecg_channel_name]
                sig = self._adaptive_ecg_filter(sig, ecg_ref, orig_fs)

            filtered[ch] = sig

        # Resample
        resampled = {}
        for ch, sig in filtered.items():
            if ch in self.required_channels:
                if abs(orig_fs - self.target_fs) < 1e-3:
                    resampled[ch] = sig
                else:
                    num_samples = int(len(sig) * self.target_fs / orig_fs)
                    resampled[ch] = signal.resample_poly(sig, up=int(self.target_fs), down=int(orig_fs))[:num_samples]

        # Stack: [T, 8]
        X = np.stack([resampled[ch] for ch in self.required_channels], axis=1)

        # Trim
        min_len = min(X.shape[0], y.shape[0])
        X, y = X[:min_len], y[:min_len]

        # Remove wake
        if self.remove_wake:
            mask = y > 0
            X, y = X[mask], y[mask]
            unique = sorted(np.unique(y))
            remap = {old: new for new, old in enumerate(unique)}
            y = np.array([remap[yy] for yy in y])
            log_info(f"Subject {subj}: Wake removed, remap: {remap}")

        # Windowing
        subject_Xs, subject_ys = [], []
        for i in range(0, len(X) - self.time_steps + 1, self.step):
            x_win = X[i:i + self.time_steps]
            subject_Xs.append(x_win)

            # Label
            y_win = y[i:i + self.time_steps]
            unique_labels = np.unique(y_win)
            if len(unique_labels) == 1:
                label = unique_labels[0]
            else:
                counts = Counter(y_win)
                most = counts.most_common(1)[0]
                if most[1] / len(y_win) < 0.8:
                    continue
                label = most[0]

            subject_ys.append(label)

        subject_Xs = np.array(subject_Xs, dtype=np.float32)

        # Apply global normalization if provided
        if self.global_stats is not None:
            mean = self.global_stats['mean']
            std = self.global_stats['std']
            subject_Xs = (subject_Xs - mean) / std

        return subject_Xs, np.array(subject_ys, dtype=np.int64)

    def _adaptive_ecg_filter(self, contaminated, ecg_ref, fs, mu=0.01, n_taps=32):
        w = np.zeros(n_taps)
        filtered = np.zeros_like(contaminated)
        for i in range(n_taps, len(contaminated)):
            x = ecg_ref[i-n_taps:i][::-1]
            y = np.dot(w, x)
            e = contaminated[i] - y
            filtered[i] = e
            w = w + mu * e * x
        return filtered

    def _load_and_preprocess(self):
        all_Xs = []
        all_ys = []
        self.subject_to_windows = {}

        # Try to load full cache first
        cache_hash = str(hash(tuple(sorted(self.subjects)) + (self.time_steps, self.step, self.balance, self.remove_wake,
                                                             self.apply_notch, self.notch_freq, self.apply_emg_hp,
                                                             self.emg_hp_cutoff, self.apply_ecg_filter)))
        full_cache_file = self.cache_dir / f"full_{cache_hash}.pkl"
        # if full_cache_file.exists():
        #     self.X, self.y, self.subject_to_windows = joblib.load(full_cache_file)
        #     log_info("Loaded full dataset from cache!")
        #     return

        for subj in tqdm(self.subjects, desc="Loading HMC Subjects"):
            subj_cache_file = self.cache_dir / f"{subj}.pkl"
            if subj_cache_file.exists():
                X_subj, y_subj = joblib.load(subj_cache_file)
                log_info(f"Loaded {subj} from cache")
            else:
                X_subj, y_subj = self._load_one_subject(subj)
                if X_subj is None:
                    continue
                joblib.dump((X_subj, y_subj), subj_cache_file)
                log_info(f"Saved {subj} to cache")

            self.subject_to_windows[subj] = list(zip(list(X_subj), list(y_subj)))
            all_Xs.extend(X_subj)
            all_ys.extend(y_subj)

        self.X = np.array(all_Xs, dtype=np.float32)
        self.y = np.array(all_ys, dtype=np.int64)

        # Balance
        if self.balance:
            unique, counts = np.unique(self.y, return_counts=True)
            log_info(f"Before balance: {dict(zip(unique, counts))}")
            maj_class = unique[np.argmax(counts)]
            maj_mask = self.y == maj_class
            Xs_maj, ys_maj = self.X[maj_mask], self.y[maj_mask]
            Xs_min, ys_min = self.X[~maj_mask], self.y[~maj_mask]

            if len(Xs_maj) > self.majority_n:
                np.random.seed(42)
                idx = np.random.choice(len(Xs_maj), self.majority_n, replace=False)
                Xs_maj, ys_maj = Xs_maj[idx], ys_maj[idx]

            self.X = np.concatenate([Xs_maj, Xs_min])
            self.y = np.concatenate([ys_maj, ys_min])
            perm = np.random.permutation(len(self.X))
            self.X, self.y = self.X[perm], self.y[perm]
            self.subject_to_windows = {s: [] for s in self.subjects}  # Reset since balanced
            log_info("Balancing applied")

        log_info(f"Final: {dict(zip(*np.unique(self.y, return_counts=True)))} | Total: {len(self.X)}")

        # # Save full cache
        # joblib.dump((self.X, self.y, self.subject_to_windows), full_cache_file)
        # log_info(f"Saved full dataset to cache {full_cache_file}")

    def __len__(self) -> int:
        if not self.preloaded:
            self._load_and_preprocess()
            self.preloaded = True
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.preloaded:
            self._load_and_preprocess()
            self.preloaded = True
        x = torch.from_numpy(self.X[idx])  # [T, C]
        x = x.transpose(0, 1)  # [C, T]
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

    def get_subject_sequence(self, subject_id: str) -> List[Tuple[torch.Tensor, int]]:
        if self.balance:
            raise RuntimeError("get_subject_sequence not supported with balance=True")
        if not self.preloaded:
            self._load_and_preprocess()
            self.preloaded = True
        if subject_id not in self.subject_to_windows:
            return []
        seq = []
        for x_np, y in self.subject_to_windows[subject_id]:
            x = torch.from_numpy(x_np).float().transpose(0, 1)
            seq.append((x, int(y)))
        return seq

def compute_channel_stats(dataset: HMCSleepDataset):
    all_X = []
    for subj in tqdm(dataset.subjects, desc="Computing global stats"):
        X_subj, _ = dataset._load_one_subject(subj)
        if X_subj is not None:
            all_X.append(X_subj)
    if not all_X:
        raise ValueError("No data for stats")
    all_X = np.concatenate(all_X, axis=0)  # [N, T, C]
    all_X_reshaped = all_X.reshape(-1, all_X.shape[-1])  # [N*T, C]
    mean = all_X_reshaped.mean(axis=0)
    std = all_X_reshaped.std(axis=0) + 1e-8
    return mean, std

def get_dataloaders(data_root: str, batch_size: int = 64,
                    balance: bool = False, remove_wake: bool = False,
                    apply_diff: bool = False,
                    apply_notch: bool = False, notch_freq: float = 50.0,
                    apply_emg_hp: bool = False, apply_ecg_filter: bool = False) -> Tuple[DataLoader, DataLoader]:
    # Example split (151 subjects total)
    train_subjects = [f"SN{str(i).zfill(3)}" for i in range(1, 121)]
    test_subjects = [f"SN{str(i).zfill(3)}" for i in range(121, 152)]

    # Compute global stats from train set (without normalization)
    train_base_for_stats = HMCSleepDataset(
        data_root, train_subjects,
        time_steps=3000, step=3000,
        balance=False, remove_wake=remove_wake,
        apply_notch=apply_notch, notch_freq=notch_freq,
        apply_emg_hp=apply_emg_hp, apply_ecg_filter=apply_ecg_filter,
        global_stats=None
    )
    mean, std = compute_channel_stats(train_base_for_stats)
    global_stats = {'mean': mean, 'std': std}

    train_base = HMCSleepDataset(
        data_root, train_subjects,
        time_steps=3000, step=3000,
        balance=balance, remove_wake=remove_wake,
        apply_notch=apply_notch, notch_freq=notch_freq,
        apply_emg_hp=apply_emg_hp, apply_ecg_filter=apply_ecg_filter,
        global_stats=global_stats
    )
    test_base = HMCSleepDataset(
        data_root, test_subjects,
        time_steps=3000, step=3000,
        balance=False, remove_wake=remove_wake,
        apply_notch=apply_notch, notch_freq=notch_freq,
        apply_emg_hp=apply_emg_hp, apply_ecg_filter=apply_ecg_filter,
        global_stats=global_stats
    )

    if apply_diff:
        train_base = DeltaDataset(train_base, axis=-1)
        test_base = DeltaDataset(test_base, axis=-1)

    train_ds = SequentialDataset(train_base, subject_ids=train_subjects)
    test_ds = SequentialDataset(test_base, subject_ids=test_subjects)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_sequential_batch)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_sequential_batch)

    return train_loader, test_loader


# -----------------------------
# Test code
# -----------------------------

if __name__ == "__main__":
    from dataset.delta_dataset import DeltaDataset
    from trainer.sequential_trainer import SequentialDataset, collate_sequential_batch
    import time

    data_root = "/Users/chengweizhou/PycharmProjects/data/hmc/"  # UPDATE THIS

    print("=" * 60)
    print("Testing HMC Sleep Dataset (SNXXX format)")
    print("=" * 60)

    try:
        test_subjects = ["SN030", "SN029"]
        # test_subjects = ["SN030"]
        print(f"Loading {test_subjects} with full preprocessing...")
        start = time.time()

        dataset = HMCSleepDataset(
            data_root=data_root,
            subjects=None,
            balance=False,
            remove_wake=False,
            apply_notch=True, notch_freq=50.0,
            apply_emg_hp=True,
            apply_ecg_filter=False
        )

        mean, std = compute_channel_stats(dataset)
        global_stats = {'mean': mean, 'std': std}
        print(global_stats)

        # dataset.get_all_subjects_name()
        seq_ds = SequentialDataset(dataset, ["SN030"])

        print(f"Init time: {time.time() - start:.2f}s")
        print(f"Subjects loaded: {len(seq_ds)} | Total epochs: {len(seq_ds)}")

        if len(seq_ds) > 0:
            sid, seq = seq_ds[0]
            print(f"Subject {sid}: {len(seq)} epochs")
            x, y = seq[0]
            print(f"  Input shape: {x.shape} | Label: {y} | Range: [{x.min():.3f}, {x.max():.3f}]")

        print("All tests passed!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()