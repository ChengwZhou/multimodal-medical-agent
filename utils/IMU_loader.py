import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, resample

def extract_features(x):
    feats = []
    num_channels = x.shape[1]
    for c in range(num_channels):
        sig = x[:, c]
        mean = np.mean(sig)
        std = np.std(sig)
        auc = np.trapz(np.abs(sig))
        peak = np.max(np.abs(sig))
        feats.append([mean, std, auc, peak])
    return np.array(feats)   # (C, 4)

class IMUDataset(Dataset):
    def __init__(self, root="Data", subject_numbers=None, pert_window=6000,
                 num_imus=5, label_encoder=None, fs=100,
                 lowcut=0.1, highcut=20, apply_filter=True):

        self.X, self.Y = [], []
        self.pert_window = pert_window
        self.num_imus = num_imus
        self.num_modal = num_imus * 6
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.apply_filter = apply_filter

        if subject_numbers is None:
            subject_numbers = []

        for subj in subject_numbers:
            subj_folder = os.path.join(root, f"HAB-{subj}", "Processed")
            if not os.path.isdir(subj_folder):
                continue

            for file in os.listdir(subj_folder):
                if not file.endswith(".mat"):
                    continue

                filename = os.path.splitext(file)[0]
                perturbation_type = re.sub(r"\d+", "", filename).rstrip("_")

                mat = loadmat(os.path.join(subj_folder, file))
                perturbations = mat['perturbation_data'][0] 
                #print(f"Subject {subj}, file {file}, perturbations: {len(perturbations)}")
                for pert in perturbations:
                    keep_idx = []
                    for i in range(self.num_imus):
                        base = i * 10  
                        keep_idx.extend([base, base+1, base+2, base+3, base+4, base+5])
                    data_selected = pert[:, keep_idx]  # shape: (T, num_imus*6)

                    if data_selected.shape[0] != pert_window:
                        data_selected = resample(data_selected, pert_window, axis=0)

                    if apply_filter:
                        data_selected = self._butter_bandpass_filter(data_selected, lowcut, highcut, fs)

                    self.X.append(data_selected)
                    self.Y.append(perturbation_type)
                    
        self.X = np.array(self.X, dtype=np.float32)
        if label_encoder is None:
            self.le = LabelEncoder()
            self.Y = self.le.fit_transform(self.Y)
        else:
            self.le = label_encoder
            self.Y = self.le.transform(self.Y)

        self.Y = np.array(self.Y, dtype=np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.Y[idx], dtype=torch.long)
        return x, y

    def _butter_bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
        filtered = filtfilt(b, a, data, axis=0)
        return filtered

class IMUFeatureDataset(Dataset):
    def __init__(self, root="Data", subject_numbers=None, num_imus=5, label_encoder=None):
        self.X, self.Y = [], []
        self.num_imus = num_imus
        self.num_modal = num_imus * 10   # 每个 IMU 10 个通道

        if subject_numbers is None:
            subject_numbers = []

        for subj in subject_numbers:
            subj_folder = os.path.join(root, f"HAB-{subj}", "Processed")
            if not os.path.isdir(subj_folder):
                continue

            for file in os.listdir(subj_folder):
                if not file.endswith(".mat"):
                    continue

                filename = os.path.splitext(file)[0]
                perturbation_type = re.sub(r"\d+", "", filename).rstrip("_")

                mat = loadmat(os.path.join(subj_folder, file))
                perturbations = mat['perturbation_data'][0]
                #print(np.shape(mat['perturbation_data'][1]))

                for pert in perturbations:
                    data = pert[:, :self.num_modal]

                    feats = []
                    for imu_idx in range(num_imus):
                        start_col = imu_idx * 10
                        accel = data[:, start_col:start_col+3]  # Ax, Ay, Az
                        gyro  = data[:, start_col+3:start_col+6] # Gx, Gy, Gz

                        for sig in [*accel.T, *gyro.T]:
                            mean = np.mean(sig)
                            std = np.std(sig)
                            auc = np.trapz(sig)
                            peak = np.max(np.abs(sig))
                            feats.extend([mean, std, auc, peak])

                    self.X.append(feats)
                    self.Y.append(perturbation_type)

        self.X = np.array(self.X, dtype=np.float32)   # (N, F)

        if label_encoder is None:
            self.le = LabelEncoder()
            self.Y = self.le.fit_transform(self.Y)
        else:
            self.le = label_encoder
            self.Y = self.le.transform(self.Y)

        self.Y = np.array(self.Y, dtype=np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)   # (F,)
        y = torch.tensor(self.Y[idx], dtype=torch.long)
        return x, y

if __name__ == "__main__":
    train_subjects = ['15','16','17','18','19','20','21']
    test_subjects  = ['22','23']

    print("Building train dataset ...")
    train_set = IMUDataset(root="../IMU_Data", subject_numbers=train_subjects, pert_window=2000)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    #train_set = IMUFeatureDataset(root="../IMU_Data", subject_numbers=train_subjects)
    #train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

    print("Building test dataset ...")
    test_set = IMUDataset(root="../IMU_Data", subject_numbers=test_subjects, pert_window=2000, label_encoder=train_set.le)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    #test_set = IMUFeatureDataset(root="../IMU_Data", subject_numbers=test_subjects)
    #test_loader = DataLoader(train_set, batch_size=32, shuffle=True)

    for xb, yb in train_loader:
        print("X batch shape:", xb.shape)   
        print("Y batch shape:", yb.shape)   
        print("First 10 labels:", yb[:10])
        print("Classes:", train_set.le.classes_)
        break