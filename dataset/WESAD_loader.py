import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import kagglehub
import scipy.signal as scisig
from scipy.signal import resample, butter, filtfilt

def get_wesad_path():
    print("Downloading WESAD dataset from KaggleHub...")
    path = kagglehub.dataset_download("orvile/wesad-wearable-stress-affect-detection-dataset")
    print("Dataset downloaded to:", path)
    return os.path.join(path, "WESAD")

# -------------------- Butter Filter --------------------
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
        raise ValueError("ftype must be 'low','high','band'")

    padlen = min(n - 1, 3 * max(len(a), len(b)))
    if padlen < 1:
        return sig
    '''
    if np.all(sig == 0):
        sig = sig + 1e-6
    '''
    return filtfilt(b, a, sig, padtype='odd', padlen=padlen)


class MultiModalWESADDataset(Dataset):
    """
    Return:
        x:       [C, T]
        y_seq:   [T]，{0,1,2, ignore_index}
                 1/2/3 -> 0/1/2；{0,4,5,6,7} -> ignore_index
        loss_mask(optional): [T]，float32，1=loss，0=ignore
    Usage:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        loss = criterion(logits.transpose(1,2), y_seq)  # logits: [B, T, num_classes]
    """
    def __init__(self, root, subject_ids=None, window_sec=1, step_sec=1, target_fs=100,
                 ignore_index: int = -100, return_loss_mask: bool = True):
        self.samples = []
        self.y_seqs = []
        self.subject_per_window = []

        self.subject_ids = subject_ids if subject_ids is not None else [
            2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17
        ]

        self.window_sec = window_sec
        self.step_sec = step_sec if step_sec is not None else window_sec
        self.target_fs = target_fs
        self.ignore_index = ignore_index
        self.return_loss_mask = return_loss_mask

        self.chest_fs = 700
        self.wrist_fs = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4}

        self.label_remap = {1: 0, 2: 1, 3: 2}
        self.keep_labels = {1, 2, 3}

        # Initialize subject_to_windows dictionary
        self.subject_to_windows = {}
        target_len = int(window_sec * target_fs)

        for sid in self.subject_ids:
            fname = os.path.join(root, f"S{sid}", f"S{sid}.pkl")
            if not os.path.exists(fname):
                print(f"Missing {fname}, skipping...")
                continue

            with open(fname, "rb") as f:
                data = pickle.load(f, encoding="latin1")

            chest = data["signal"]["chest"]
            wrist = data["signal"]["wrist"]
            label = np.asarray(data["label"], dtype=int)  # 700Hz 对齐

            n_samples_chest = len(label)
            win_size_chest = int(self.window_sec * self.chest_fs)
            step_chest = int(self.step_sec * self.chest_fs)

            start_idx = 0
            self.subject_to_windows[sid] = []

            while start_idx + win_size_chest <= n_samples_chest / 100:
                s_chest = start_idx
                e_chest = start_idx + win_size_chest

                # -------- Chest --------
                chest_win = []
                for k, sig in chest.items():
                    seg = sig[s_chest:e_chest]
                    if len(seg) < win_size_chest:
                        seg = np.pad(seg, (0, win_size_chest - len(seg)), mode='constant')
                    seg = self.apply_filter(k, seg, self.chest_fs)
                    seg_res = resample(seg, target_len)
                    chest_win.append(seg_res)
                if len(chest_win) == 0:
                    start_idx += step_chest
                    continue
                chest_win = np.column_stack(chest_win)  # [T, C_chest]

                # -------- Wrist --------
                wrist_win = []
                for k, sig in wrist.items():
                    fs = self.wrist_fs.get(k, None)
                    if fs is None:
                        continue
                    s_wrist = int(round(start_idx * fs / self.chest_fs))
                    e_wrist = int(round((start_idx + win_size_chest) * fs / self.chest_fs))
                    seg = sig[s_wrist:e_wrist]
                    expected_len = int(round(self.window_sec * fs))
                    if len(seg) < expected_len:
                        seg = np.pad(seg, (0, expected_len - len(seg)), mode='constant')
                    seg = self.apply_filter(k, seg, fs)
                    seg_res = resample(seg, target_len)
                    wrist_win.append(seg_res)
                if len(wrist_win) == 0:
                    start_idx += step_chest
                    continue
                wrist_win = np.column_stack(wrist_win)  # [T, C_wrist]

                # -------- Combine --------
                sig_win = np.concatenate([chest_win, wrist_win], axis=1)  # [T, C_total]

                # -------- Label --------
                label_slice = label[s_chest:e_chest]
                y_seq = self._make_y_seq(label_slice, target_len)  # [T]
                unique, counts = np.unique(y_seq, return_counts=True)
                label_counts = dict(zip(unique, counts))

                # --- most labels ---
                y_label = max(label_counts, key=label_counts.get)

                # --- save ---
                self.samples.append(sig_win.astype(np.float32))
                self.y_seqs.append(np.int64(y_label))
                self.subject_to_windows[sid].append((
                    sig_win.astype(np.float32),
                    np.int64(y_label)
                ))

                start_idx += step_chest
                start_idx += step_chest

        self.samples = np.array(self.samples, dtype=np.float32)   # [N, T, C]
        self.y_seqs  = np.array(self.y_seqs,  dtype=np.int64)     # [N, T]
        self.num_modal = self.samples.shape[-1]
        self.label_map = {0: "baseline", 1: "stress", 2: "amusement"}

    # ---------------- helpers ----------------
    def _make_y_seq(self, label_slice: np.ndarray, target_len: int) -> np.ndarray:
        src_len = len(label_slice)
        if target_len == src_len:
            labels_aligned = label_slice
        else:
            src_idx = np.linspace(0, src_len - 1, num=target_len)
            labels_aligned = label_slice[np.round(src_idx).astype(int)]

        y_seq = np.full(target_len, self.ignore_index, dtype=np.int64)

        mask1 = (labels_aligned == 1)
        mask2 = (labels_aligned == 2)
        mask3 = (labels_aligned == 3)
        y_seq[mask1] = 0
        y_seq[mask2] = 1
        y_seq[mask3] = 2

        return y_seq
    def apply_filter(self, k, seg, fs):
        if k == "ECG":
            return butter_filter(seg, fs, ftype="band", low=1, high=40)
        elif k == "EMG":
            return butter_filter(seg, fs, ftype="band", low=10, high=100)
        elif k == "EDA":
            return butter_filter(seg, fs, ftype="low", low=1)
        elif k == "Resp":
            return butter_filter(seg, fs, ftype="band", low=0.1, high=1)
        elif k in ("Temp", "TEMP"):
            return butter_filter(seg, fs, ftype="low", low=0.1)
        elif k == "ACC":
            cutoff = 0.4 / (fs / 2.0)
            FIR_coeff = scisig.firwin(numtaps=64, cutoff=cutoff)
            return scisig.lfilter(FIR_coeff, 1, seg)
        elif k == "BVP":
            return butter_filter(seg, fs, ftype="band", low=0.5, high=8)
        else:
            return seg

    # --------------- Dataset API ---------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # x: [C, T]
        x = torch.tensor(self.samples[idx].T, dtype=torch.float32)

        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)

        y_seq = torch.tensor(self.y_seqs[idx], dtype=torch.long)  # [T]
        if self.return_loss_mask:
            loss_mask = (y_seq != self.ignore_index).float()      # [T], 1=计入损失
            return x, y_seq, loss_mask
        else:
            return x, y_seq
    def get_subject_sequence(self, subject_id: int):
        """
        Get all windows for a specific subject as a sequence.

        Args:
            subject_id: The subject ID (int)

        Returns:
            List of (x_tensor, y_label) for the subject's windows

        Raises:
            RuntimeError: If balance=True, as subject sequences are not reliable after global shuffling
        """
        if subject_id not in self.subject_to_windows:
            log_info(f"Subject {subject_id} has no data")
            return []

        sequence = []
        for x_np, y in self.subject_to_windows[subject_id]:
            x = torch.from_numpy(x_np).float()  # [T, C]
            x = x.transpose(0, 1)  # [C, T]
            y = int(y)  # Ensure y is a Python int for compatibility
            sequence.append((x, y))

        return sequence



if __name__ == "__main__":
    dataset_dir = "./WESAD/WESAD/"

    train_subjects = [2,3]
    test_subjects  = [11]

    train_set = MultiModalWESADDataset(dataset_dir, train_subjects, window_sec=10, target_fs=64)
    test_set  = MultiModalWESADDataset(dataset_dir, test_subjects, window_sec=10, target_fs=64)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    test_loader  = DataLoader(test_set, batch_size=8, shuffle=False)

    for xb, yb, _ in train_loader:
        print("Batch X:", xb.shape, "Batch y:", yb.shape)
        break
