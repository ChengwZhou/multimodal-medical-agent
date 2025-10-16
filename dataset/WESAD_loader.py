import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import kagglehub
import scipy.signal as scisig
from scipy.signal import resample, butter, filtfilt

'''
WRIST_FS = {
    "ACC": 32,
    "BVP": 64,
    "EDA": 4,
    "TEMP": 4
}

CHEST_FS = 700  # chest 
'''
def get_wesad_path():
    print("Downloading WESAD dataset from KaggleHub...")
    path = kagglehub.dataset_download("orvile/wesad-wearable-stress-affect-detection-dataset")
    print("Dataset downloaded to:", path)
    return os.path.join(path, "WESAD")
'''
class MultiModalWESADDataset(Dataset):
    def __init__(self, root, subject_ids=None, window_sec=10, target_fs=100):
        self.samples = []
        self.labels = []
        self.subject_per_window = []  

        self.subject_ids = subject_ids if subject_ids is not None else [
            2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17
        ]

        self.window_size = int(window_sec * target_fs)
        self.chest_fs = 700
        self.wrist_fs = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4}

        for sid in self.subject_ids:
            fname = os.path.join(root, f"S{sid}", f"S{sid}.pkl")
            if not os.path.exists(fname):
                print(f"Missing {fname}, skipping...")
                continue

            with open(fname, "rb") as f:
                data = pickle.load(f, encoding="latin1")

            chest = data["signal"]["chest"]
            wrist = data["signal"]["wrist"]
            label = data["label"]

            chest_sig = []
            for k, sig in chest.items():
                if k == "ECG":
                    sig = butter_filter(sig, self.chest_fs, ftype="band", low=1, high=40)
                elif k == "EMG":
                    sig = butter_filter(sig, self.chest_fs, ftype="band", low=10, high=100)
                elif k == "EDA":
                    sig = butter_filter(sig, self.chest_fs, ftype="low", low=1)
                elif k == "Resp":
                    sig = butter_filter(sig, self.chest_fs, ftype="band", low=0.1, high=1)
                elif k == "Temp":
                    sig = butter_filter(sig, self.chest_fs, ftype="low", low=1)
                elif k == "ACC":
                    sig = butter_filter(sig, self.chest_fs, ftype="low", low=10)

                sig_res = resample(sig, int(sig.shape[0] * target_fs / self.chest_fs))
                chest_sig.append(sig_res if sig_res.ndim > 1 else sig_res[:, None])
            chest_sig = np.column_stack(chest_sig)

            wrist_sig = []
            for k, sig in wrist.items():
                fs = self.wrist_fs[k]
                if k == "BVP":
                    sig = butter_filter(sig, fs, ftype="band", low=0.5, high=8)
                elif k == "EDA":
                    sig = butter_filter(sig, fs, ftype="low", low=1)
                elif k == "TEMP":
                    sig = butter_filter(sig, fs, ftype="low", low=1)
                elif k == "ACC":
                    sig = butter_filter(sig, fs, ftype="low", low=10)

                sig_res = resample(sig, int(sig.shape[0] * target_fs / fs))
                wrist_sig.append(sig_res if sig_res.ndim > 1 else sig_res[:, None])
            wrist_sig = np.column_stack(wrist_sig)

            min_len = min(chest_sig.shape[0], wrist_sig.shape[0])
            chest_sig = chest_sig[:min_len]
            wrist_sig = wrist_sig[:min_len]
            sig = np.concatenate([chest_sig, wrist_sig], axis=1)  # (T, C)

            label_res = resample(label, int(len(label) * target_fs / self.chest_fs))
            label_res = np.round(label_res).astype(int)[:min_len]

            n_win = int(min_len // self.window_size)
            for i in range(n_win):
                s, e = i * self.window_size, (i + 1) * self.window_size
                x = sig[s:e]

                window_labels = label_res[s:e]
                y = np.bincount(window_labels[window_labels > 0]).argmax() if np.any(window_labels > 0) else 0

                if y in [1, 2, 3]:
                    self.samples.append(x)
                    self.labels.append(y)
                    self.subject_per_window.append(sid)

        self.samples = np.array(self.samples)
        self.labels = np.array(self.labels)
        self.subject_per_window = np.array(self.subject_per_window)
        self.num_modal = self.samples.shape[-1]
        self.label_map = {1: "baseline", 2: "stress", 3: "amusement"}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 转成 (C, T) 格式
        x = torch.tensor(self.samples[idx].T, dtype=torch.float32)
        y = torch.tensor(self.labels[idx] - 1, dtype=torch.long)  # [0,1,2]
        return x, y
'''

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import resample, butter, filtfilt

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


# -------------------- Dataset --------------------
class MultiModalWESADDataset(Dataset):
    def __init__(self, root, subject_ids=None, window_sec=1, step_sec=1, target_fs=700):
        self.samples = []
        self.labels = []
        self.subject_per_window = []

        self.subject_ids = subject_ids if subject_ids is not None else [
            2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17
        ]

        self.window_sec = window_sec
        self.step_sec = step_sec if step_sec is not None else window_sec
        self.target_fs = target_fs

        self.chest_fs = 700
        self.wrist_fs = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4}

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
            label = data["label"]

            n_samples_chest = len(label)
            win_size_chest = int(window_sec * self.chest_fs)
            step_chest = int(self.step_sec * self.chest_fs)

            start_idx = 0
            while start_idx + win_size_chest <= n_samples_chest:
                s_chest = start_idx
                e_chest = start_idx + win_size_chest

                # -------- Chest --------
                chest_win = []
                for k, sig in chest.items():
                    seg = sig[s_chest:e_chest]
                    # pad
                    if len(seg) < win_size_chest:
                        seg = np.pad(seg, (0, win_size_chest - len(seg)), mode='constant')
                    seg = self.apply_filter(k, seg, self.chest_fs)
                    seg_res = resample(seg, target_len)
                    chest_win.append(seg_res)
                if len(chest_win) == 0:
                    start_idx += step_chest
                    continue
                chest_win = np.column_stack(chest_win)

                # -------- Wrist --------
                wrist_win = []
                for k, sig in wrist.items():
                    fs = self.wrist_fs[k]
                    s_wrist = int(start_idx * fs / self.chest_fs)
                    e_wrist = int((start_idx + win_size_chest) * fs / self.chest_fs)
                    seg = sig[s_wrist:e_wrist]
                    expected_len = int(round(window_sec * fs))
                    if len(seg) < expected_len:
                        seg = np.pad(seg, (0, expected_len - len(seg)), mode='constant')
                    seg = self.apply_filter(k, seg, fs)
                    seg_res = resample(seg, target_len)
                    wrist_win.append(seg_res)
                if len(wrist_win) == 0:
                    start_idx += step_chest
                    continue
                wrist_win = np.column_stack(wrist_win)

                # -------- Combine --------
                sig_win = np.concatenate([chest_win, wrist_win], axis=1)

                label_slice = label[s_chest:e_chest]
                label_win = np.round(label_slice).astype(int)
                label_win = label_win[label_win > 0]
                if len(label_win) == 0:
                    start_idx += step_chest
                    continue
                y = np.bincount(label_win).argmax()
                if y in [1, 2, 3]:
                    self.samples.append(sig_win)
                    self.labels.append(y)
                    self.subject_per_window.append(sid)

                start_idx += step_chest

        self.samples = np.array(self.samples, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        self.subject_per_window = np.array(self.subject_per_window)
        self.num_modal = self.samples.shape[-1]
        self.label_map = {1: "baseline", 2: "stress", 3: "amusement"}

    '''
    def apply_filter(self, k, seg, fs):
        if k == "ECG":
            return butter_filter(seg, fs, ftype="band", low=1, high=40)
        elif k == "EMG":
            return butter_filter(seg, fs, ftype="band", low=10, high=100)
        elif k == "EDA":
            return butter_filter(seg, fs, ftype="low", low=1)
        elif k == "Resp":
            return butter_filter(seg, fs, ftype="band", low=0.1, high=1)
        elif k == "Temp":
            return butter_filter(seg, fs, ftype="low", low=0.1)
        elif k == "ACC":
            if fs > 100:
                return butter_filter(seg, fs, ftype="band", low=0.1, high=20)
            else:
                return butter_filter(seg, fs, ftype="band", low=0.1, high=10)
        elif k == "BVP":
            return butter_filter(seg, fs, ftype="band", low=0.5, high=8)
        else:
            return seg
    '''
    def apply_filter(self, k, seg, fs):
        if k == "ECG":
            return butter_filter(seg, fs, ftype="band", low=1, high=40)
        elif k == "EMG":
            return butter_filter(seg, fs, ftype="band", low=10, high=100)
        elif k == "EDA":
            return butter_filter(seg, fs, ftype="low", low=1)
        elif k == "Resp":
            return butter_filter(seg, fs, ftype="band", low=0.1, high=1)
        elif k == "Temp":
            return butter_filter(seg, fs, ftype="low", low=0.1)
        elif k == "ACC":
            cutoff = 0.4 / (fs / 2.0)   
            FIR_coeff = scisig.firwin(numtaps=64, cutoff=cutoff)  
            return scisig.lfilter(FIR_coeff, 1, seg)
        elif k == "BVP":
            return butter_filter(seg, fs, ftype="band", low=0.5, high=8)
        else:
            return seg

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx].T, dtype=torch.float32)
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)
        y = torch.tensor(self.labels[idx] - 1, dtype=torch.long)
        return x, y


def wesad_split_by_subject(dataset, val_ratio=0.1):
    rng = np.random.default_rng(42)
    unique_subs = dataset.subject_ids
    n_val = max(1, int(round(len(unique_subs) * val_ratio)))
    val_subs = set(rng.choice(unique_subs, size=n_val, replace=False).tolist())

    train_idx, val_idx = [], []
    for i, sid in enumerate(dataset.subject_per_window):
        if sid in val_subs:
            val_idx.append(i)
        else:
            train_idx.append(i)

    return train_idx, val_idx


if __name__ == "__main__":
    dataset_dir = get_wesad_path()

    train_subjects = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    test_subjects  = [11]

    train_set = MultiModalWESADDataset(dataset_dir, train_subjects, window_sec=10, target_fs=64)
    test_set  = MultiModalWESADDataset(dataset_dir, test_subjects, window_sec=10, target_fs=64)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_set, batch_size=32, shuffle=False)

    for xb, yb in train_loader:
        print("Batch X:", xb.shape, "Batch y:", yb.shape)
        break
