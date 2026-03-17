import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample, butter, filtfilt, firwin, lfilter
import scipy.signal as scisig
import logging
from typing import List, Tuple, Optional

# Configure logging (same as mHEALTH)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_info(msg: str):
    from torch.distributed import is_initialized, get_rank
    if not is_initialized() or get_rank() == 0:
        logger.info(msg)

# -------------------- Butter & FIR Filter --------------------
def butter_filter(sig, fs, ftype="band", low=None, high=None, order=4):
    sig = np.asarray(sig).flatten()
    n = len(sig)
    if n < 2:
        return sig

    nyq = 0.5 * fs
    if ftype == "band":
        b, a = butter(order, [low / nyq, high / nyq], btype='band')
    elif ftype == "low":
        b, a = butter(order, low / nyq, btype='low')
    elif ftype == "high":
        b, a = butter(order, high / nyq, btype='high')
    else:
        raise ValueError("ftype must be 'low', 'high', or 'band'")

    padlen = min(n - 1, 3 * max(len(a), len(b)))
    if padlen < 1:
        return sig
    return filtfilt(b, a, sig, padtype='odd', padlen=padlen)


class WESADDataset(Dataset):
    """
    WESAD dataset loader that is fully compatible with SequentialDataset (exactly like MHealthDataset).
    Sliding window + per-subject sequence preservation + optional global balancing.
    """
    def __init__(
        self,
        root: str,
        subject_ids: Optional[List[int]] = None,
        window_sec: float = 10.0,
        step_sec: Optional[float] = None,
        target_fs: int = 64,
        balance: bool = False,
        majority_n: int = 30000
    ):
        self.root = root
        self.window_sec = window_sec
        self.step_sec = step_sec if step_sec is not None else window_sec / 2  # default 50% overlap
        self.target_fs = target_fs
        self.balance = balance
        self.majority_n = majority_n

        if subject_ids is None:
            self.subject_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
        else:
            self.subject_ids = sorted(subject_ids)

        self.chest_fs = 700
        self.wrist_fs = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4}

        self.target_len = int(self.window_sec * self.target_fs)
        self.win_size_chest = int(self.window_sec * self.chest_fs)
        self.step_chest = int(self.step_sec * self.chest_fs)

        self.subject_to_windows = {}  # sid -> list of (x_np (T, C), label_int)
        all_Xs = []
        all_ys = []

        for sid in self.subject_ids:
            fname = os.path.join(root, f"S{sid}", f"S{sid}.pkl")
            if not os.path.exists(fname):
                log_info(f"Missing {fname}, skipping...")
                continue

            with open(fname, "rb") as f:
                data = pickle.load(f, encoding="latin1")

            chest = data["signal"]["chest"]
            wrist = data["signal"]["wrist"]
            label = data["label"]  # (T_chest,)

            n_samples_chest = len(label)
            subject_windows = []

            start_idx = 0
            while start_idx + self.win_size_chest <= n_samples_chest:
                s_chest = start_idx
                e_chest = start_idx + self.win_size_chest

                # -------------------- Chest signals --------------------
                chest_win = []
                for k, sig in chest.items():
                    seg = sig[s_chest:e_chest]
                    if seg.shape[0] < self.win_size_chest:
                        pad_width = self.win_size_chest - seg.shape[0]
                        seg = np.pad(seg, ((0, pad_width), (0, 0)), 'constant') if seg.ndim == 2 else np.pad(seg, (0, pad_width), 'constant')
                    seg = self._apply_filter(k, seg, self.chest_fs)
                    seg_res = resample(seg, self.target_len)
                    if seg_res.ndim == 1:
                        seg_res = seg_res[:, None]
                    chest_win.append(seg_res)
                chest_win = np.column_stack(chest_win)  # (target_len, C_chest)

                # -------------------- Wrist signals --------------------
                wrist_win = []
                for k, sig in wrist.items():
                    fs_w = self.wrist_fs[k]
                    s_wrist = int(start_idx * fs_w / self.chest_fs)
                    e_wrist = int((start_idx + self.win_size_chest) * fs_w / self.chest_fs)
                    seg = sig[s_wrist:e_wrist]
                    expected = int(round(self.window_sec * fs_w))
                    if len(seg) < expected:
                        pad_width = expected - len(seg)
                        seg = np.pad(seg, ((0, pad_width), (0, 0)), 'constant') if seg.ndim == 2 else np.pad(seg, (0, pad_width), 'constant')
                    seg = self._apply_filter(k, seg, fs_w)
                    seg_res = resample(seg, self.target_len)
                    if seg_res.ndim == 1:
                        seg_res = seg_res[:, None]
                    wrist_win.append(seg_res)
                wrist_win = np.column_stack(wrist_win)  # (target_len, C_wrist)

                # Combine
                x_win = np.concatenate([chest_win, wrist_win], axis=1).astype(np.float32)  # (T, C)

                # Label (majority vote among non-transient labels)
                label_slice = label[s_chest:e_chest]
                non_transient = np.round(label_slice).astype(int)
                non_transient = non_transient[non_transient > 0]
                if len(non_transient) == 0:
                    start_idx += self.step_chest
                    continue
                y_raw = np.bincount(non_transient).argmax()
                if y_raw not in [1, 2, 3]:
                    start_idx += self.step_chest
                    continue
                y = int(y_raw - 1)  # → 0, 1, 2

                subject_windows.append((x_win, y))
                start_idx += self.step_chest

            # Store per subject
            if subject_windows:
                self.subject_to_windows[sid] = subject_windows
                all_Xs.extend([x for x, _ in subject_windows])
                all_ys.extend([y for _, y in subject_windows])

        if len(all_Xs) == 0:
            raise ValueError("No valid windows found in the dataset!")

        self.samples = np.array(all_Xs, dtype=np.float32)  # (N, T, C)
        self.labels = np.array(all_ys, dtype=np.int64)  # (N,)

        # -------------------- Optional global balancing --------------------
        if self.balance:
            unique, counts = np.unique(self.labels, return_counts=True)
            log_info(f"Before balancing: {dict(zip(unique.tolist(), counts.tolist()))}")
            majority_class = unique[np.argmax(counts)]
            majority_mask = self.labels == majority_class
            Xs_major = self.samples[majority_mask]
            ys_major = self.labels[majority_mask]
            Xs_minor = self.samples[~majority_mask]
            ys_minor = self.labels[~majority_mask]

            if len(Xs_major) > self.majority_n:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(Xs_major), self.majority_n, replace=False)
                Xs_major = Xs_major[idx]
                ys_major = ys_major[idx]

            self.samples = np.concatenate([Xs_major, Xs_minor])
            self.labels = np.concatenate([ys_major, ys_minor])

            # Shuffle after balancing
            perm = np.random.permutation(len(self.samples))
            self.samples = self.samples[perm]
            self.labels = self.labels[perm]

            self.subject_to_windows = {}
            log_info("Global balancing applied → subject_to_windows invalidated (cannot use sequential mode)")

        log_info(f"Final distribution: {dict(zip(*np.unique(self.labels, return_counts=True)))}")
        log_info(f"Total windows: {len(self.samples)} | Shape: {self.samples.shape}")
        self.num_channels = self.samples.shape[-1]
        self.label_map = {0: "baseline", 1: "stress", 2: "amusement"}

    def _apply_filter(self, k: str, seg: np.ndarray, fs: float):
        if k == "ECG":
            return butter_filter(seg, fs, "band", low=1, high=40)
        elif k == "EMG":
            return butter_filter(seg, fs, "band", low=10, high=100)
        elif k == "EDA":
            return butter_filter(seg, fs, "low", low=1)
        elif k == "Resp":
            return butter_filter(seg, fs, "band", low=0.1, high=1)
        elif k == "Temp" or k == "TEMP":
            return butter_filter(seg, fs, "low", low=0.1)
        elif k == "ACC":
            if fs > 100:  # chest
                cutoff = 20 / (fs / 2.0)
            else:  # wrist
                cutoff = 10 / (fs / 2.0)
            fir_coeff = firwin(numtaps=64, cutoff=cutoff, pass_zero='lowpass')
            return lfilter(fir_coeff, 1.0, seg)
        elif k == "BVP":
            return butter_filter(seg, fs, "band", low=0.5, high=8)
        return seg  # no filter

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        x_np = self.samples[idx]  # (T, C)
        x = torch.from_numpy(x_np).float().transpose(0, 1)  # (C, T)
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

    def get_subject_sequence(self, subject_id: int) -> List[Tuple[torch.Tensor, int]]:
        """Return the full temporal sequence of a subject (used by SequentialDataset)"""
        if self.balance:
            raise RuntimeError("get_subject_sequence is not supported when balance=True (global shuffle)")
        windows = self.subject_to_windows.get(subject_id, [])
        if not windows:
            return []

        sequence = []
        for x_np, y in windows:
            x = torch.from_numpy(x_np).float().transpose(0, 1)  # (C, T)
            x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)
            sequence.append((x, int(y)))
        return sequence


# Example usage (you can also write a get_dataloaders similar to mhealth)
if __name__ == "__main__":
    dataset_dir = "/path/to/your/WESAD"  # change or use get_wesad_path()

    train_set = WESADDataset(dataset_dir, subject_ids=[2,3,4,5,6,7,8,9,10],
                             window_sec=10.0, step_sec=5.0, target_fs=64, balance=False)

    seq_dataset = SequentialDataset(train_set, subject_ids=train_set.subject_ids)

    loader = DataLoader(seq_dataset, batch_size=4, shuffle=True, collate_fn=collate_sequential_batch)

    batch = next(iter(loader))
    print(batch.sequences.shape)  # [B, max_seq_len, C, T]
    print(batch.labels.shape)
    print(batch.seq_lengths)