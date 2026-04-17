import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, resample
from typing import List, Tuple

class IMUKinematicsDataset(Dataset):
    """
    Dataset loader for IMU + VICON kinematics perturbation data.
    
    Data structure per .mat file:
    - IMU_perturbation_data: cell array of perturbations, each shape (T, 70)
      Columns 0-49: First 5 sensors (each sensor: AccX, AccY, AccZ, GyroX, GyroY, GyroZ, +4 orientation cols)
      We keep only first 6 columns per sensor (acc + gyro) -> 5 sensors × 6 = 30 columns
    - VICON_perturbation_data: cell array of perturbations, each shape (T, 9)
      Columns: COMx, COMy, COMz, L_hip_angle, L_knee_angle, L_ankle_angle,
               R_hip_angle, R_knee_angle, R_ankle_angle
    
    Sensor locations:
    1 - Right thigh
    2 - Right shank
    3 - Left shank
    4 - Left thigh
    5 - Torso
    """
    
    # Perturbation type mapping based on README
    PERTURBATION_TYPES = [
        'QS_F',          # quiet standing, falls forward
        'QS_B',          # quiet standing, falls backward
        'HS_B_Left',     # heel-strike, left leg pulled forward
        'HS_B_Right',    # heel-strike, right leg pulled forward
        'TO_F_Left',     # toe-off, left leg pulled backward
        'TO_F_Right',    # toe-off, right leg pulled backward
        'MS_F_Left',     # mid-stance, left leg falls forward
        'MS_F_Right',    # mid-stance, right leg falls forward
        'MS_B_Left',     # mid-stance, left leg falls backward
        'MS_B_Right'     # mid-stance, right leg falls backward
    ]
    
    def __init__(self, root="Data", subject_numbers=None, pert_window=6000,
                 num_imus=5, label_encoder=None, fs=100,
                 lowcut=0.1, highcut=20, apply_filter=True,
                 use_imu=True, use_kinematics=True, normalize=True):
        """
        Args:
            root: Root directory containing HAB-XX folders
            subject_numbers: List of subject IDs as strings (e.g., ['15','16','17'])
            pert_window: Target window length for resampling perturbations
            num_imus: Number of IMU sensors to use (max 5)
            label_encoder: Optional fitted LabelEncoder for consistent label encoding
            fs: Sampling frequency in Hz
            lowcut, highcut: Bandpass filter cutoff frequencies
            apply_filter: Whether to apply bandpass filter to IMU data
            use_imu: Include IMU data in the output
            use_kinematics: Include VICON kinematics data in the output
            normalize: Apply z-score normalization per perturbation
        """
        self.root = root
        self.pert_window = pert_window
        self.num_imus = num_imus
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.apply_filter = apply_filter
        self.use_imu = use_imu
        self.use_kinematics = use_kinematics
        self.normalize = normalize
        
        # Calculate channel dimensions
        self.imu_channels = num_imus * 6  # 6 channels per IMU (AccX,Y,Z + GyroX,Y,Z)
        self.kinematics_channels = 9  # 9 VICON channels
        self.num_channels = 0
        if use_imu:
            self.num_channels += self.imu_channels
        if use_kinematics:
            self.num_channels += self.kinematics_channels
        
        self.X_imu = []
        self.X_kinematics = []
        self.Y = []
        self.subject_ids = []
        self.perturbation_ids = []
        
        if subject_numbers is None:
            subject_numbers = []
        
        # Statistics for normalization
        self.imu_mean = None
        self.imu_std = None
        self.kinematics_mean = None
        self.kinematics_std = None
        
        # Load all data
        self._load_data(subject_numbers)
        
        # Convert to arrays
        if use_imu:
            self.X_imu = np.array(self.X_imu, dtype=np.float32)
        if use_kinematics:
            self.X_kinematics = np.array(self.X_kinematics, dtype=np.float32)
        
        # Handle label encoding
        if label_encoder is None:
            self.le = LabelEncoder()
            self.Y = self.le.fit_transform(self.Y)
        else:
            self.le = label_encoder
            self.Y = self.le.transform(self.Y)
        
        self.Y = np.array(self.Y, dtype=np.int64)
        
        # Compute normalization statistics if requested
        if normalize and len(self.X_imu) > 0:
            self._compute_normalization_stats()
            self._apply_normalization()

        self._build_subject_index()
    
    def _load_data(self, subject_numbers):
        """Load all perturbation data from subject folders."""
        for subj in subject_numbers:
            subj_folder = os.path.join(self.root, f"HAB-{subj}")
            if not os.path.isdir(subj_folder):
                print(f"Warning: Subject folder {subj_folder} not found")
                continue
            
            print(f"Loading subject {subj}...")
            
            for file in sorted(os.listdir(subj_folder)):
                if not file.endswith(".mat"):
                    continue
                
                filepath = os.path.join(subj_folder, file)
                filename = os.path.splitext(file)[0]
                
                # Extract perturbation type from filename
                perturbation_type = self._extract_perturbation_type(filename)
                
                # Load .mat file
                mat = loadmat(filepath)
                
                # Check for required data
                if 'IMU_perturbation_data' not in mat:
                    print(f"  Warning: No IMU_perturbation_data in {file}")
                    continue
                
                imu_perts = mat['IMU_perturbation_data'][0]
                
                # VICON data is optional
                if self.use_kinematics and 'VICON_perturbation_data' not in mat:
                    print(f"  Warning: No VICON_perturbation_data in {file}")
                    vicon_perts = [None] * len(imu_perts)
                elif self.use_kinematics:
                    vicon_perts = mat['VICON_perturbation_data'][0]
                else:
                    vicon_perts = [None] * len(imu_perts)
                
                print(f"  File: {file} | Type: {perturbation_type} | "
                      f"Perturbations: {len(imu_perts)}")
                
                # Process each perturbation
                for pert_idx, (imu_pert, vicon_pert) in enumerate(zip(imu_perts, vicon_perts)):
                    # Process IMU data
                    if self.use_imu:
                        imu_processed = self._process_imu_perturbation(imu_pert)
                        self.X_imu.append(imu_processed)
                    
                    # Process VICON data
                    if self.use_kinematics and vicon_pert is not None:
                        kinematics_processed = self._process_kinematics_perturbation(vicon_pert)
                        self.X_kinematics.append(kinematics_processed)
                    
                    self.Y.append(perturbation_type)
                    self.subject_ids.append(subj)
                    self.perturbation_ids.append(f"{filename}_pert{pert_idx+1}")
    
    def _extract_perturbation_type(self, filename):
        """Extract perturbation type from filename, handling trial numbers."""
        # Remove trial numbers (e.g., "T1", "T2") and any digits
        base_name = re.sub(r'_?T\d+', '', filename)
        base_name = re.sub(r'\d+', '', base_name).strip('_')
        
        # Check against known perturbation types
        for pert_type in self.PERTURBATION_TYPES:
            if pert_type in base_name:
                return pert_type
        
        # Fallback: use cleaned filename
        return base_name
    
    def _process_imu_perturbation(self, imu_data):
        """
        Process IMU perturbation data.
        
        Args:
            imu_data: shape (T, 70) - raw IMU data
        
        Returns:
            processed: shape (pert_window, num_imus*6)
        """
        # Select columns for first num_imus sensors (6 columns each: AccX,Y,Z + GyroX,Y,Z)
        keep_idx = []
        for i in range(self.num_imus):
            base = i * 10
            # Keep first 6 columns per sensor (skip orientation columns 6-9)
            keep_idx.extend([base, base+1, base+2, base+3, base+4, base+5])
        
        data_selected = imu_data[:, keep_idx]  # shape: (T, num_imus*6)
        
        # Resample to fixed window length
        if data_selected.shape[0] != self.pert_window:
            data_selected = resample(data_selected, self.pert_window, axis=0)
        
        # Apply bandpass filter if requested
        if self.apply_filter:
            data_selected = self._butter_bandpass_filter(
                data_selected, self.lowcut, self.highcut, self.fs
            )
        
        return data_selected
    
    def _process_kinematics_perturbation(self, vicon_data):
        """
        Process VICON kinematics perturbation data.
        
        Args:
            vicon_data: shape (T, 9) - COM + joint angles
        
        Returns:
            processed: shape (pert_window, 9)
        """
        # VICON data already has the right shape, just resample
        if vicon_data.shape[0] != self.pert_window:
            vicon_data = resample(vicon_data, self.pert_window, axis=0)
        
        return vicon_data
    
    def _butter_bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        """Apply Butterworth bandpass filter."""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        
        # Handle case where highcut >= Nyquist
        if high >= 1.0:
            high = 0.99
        
        b, a = butter(order, [low, high], btype='band')
        filtered = filtfilt(b, a, data, axis=0)
        return filtered
    
    def _compute_normalization_stats(self):
        """Compute mean and std for z-score normalization."""
        if self.use_imu and len(self.X_imu) > 0:
            # Compute across all perturbations and time steps
            all_imu = np.concatenate(self.X_imu, axis=0)  # (N*T, C)
            self.imu_mean = np.mean(all_imu, axis=0, keepdims=True)
            self.imu_std = np.std(all_imu, axis=0, keepdims=True)
            self.imu_std[self.imu_std < 1e-6] = 1.0
        
        if self.use_kinematics and len(self.X_kinematics) > 0:
            all_kinematics = np.concatenate(self.X_kinematics, axis=0)
            self.kinematics_mean = np.mean(all_kinematics, axis=0, keepdims=True)
            self.kinematics_std = np.std(all_kinematics, axis=0, keepdims=True)
            self.kinematics_std[self.kinematics_std < 1e-6] = 1.0
    
    def _apply_normalization(self):
        """Apply z-score normalization to all data."""
        if self.use_imu and self.imu_mean is not None:
            for i in range(len(self.X_imu)):
                self.X_imu[i] = (self.X_imu[i] - self.imu_mean) / self.imu_std
        
        if self.use_kinematics and self.kinematics_mean is not None:
            for i in range(len(self.X_kinematics)):
                self.X_kinematics[i] = (self.X_kinematics[i] - self.kinematics_mean) / self.kinematics_std
    
    def get_combined_data(self, idx):
        """Combine IMU and kinematics data for a given index."""
        data_parts = []
        
        if self.use_imu:
            data_parts.append(self.X_imu[idx])
        
        if self.use_kinematics:
            data_parts.append(self.X_kinematics[idx])
        
        if len(data_parts) == 1:
            return data_parts[0]
        else:
            return np.concatenate(data_parts, axis=1)  # (T, imu_channels + kinematics_channels)
    
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, idx):
        x = self.get_combined_data(idx)
        y = self.Y[idx]
        
        # Convert to torch tensors
        # Shape: (T, C) -> (C, T) for Conv1D models
        x_tensor = torch.tensor(x, dtype=torch.float32).permute(1, 0)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        return x_tensor, y_tensor
    
    def get_metadata(self, idx):
        """Return metadata for a given index."""
        return {
            'subject': self.subject_ids[idx],
            'perturbation_id': self.perturbation_ids[idx],
            'label': self.le.inverse_transform([self.Y[idx]])[0]
        }
    
    def get_channel_info(self):
        """Return information about the channels."""
        channels = []
        
        if self.use_imu:
            sensor_names = ['R_Thigh', 'R_Shank', 'L_Shank', 'L_Thigh', 'Torso']
            signal_types = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']
            
            for i in range(self.num_imus):
                for sig in signal_types:
                    channels.append(f"IMU_{sensor_names[i]}_{sig}")
        
        if self.use_kinematics:
            kin_channels = [
                'COM_X', 'COM_Y', 'COM_Z',
                'L_Hip_Angle', 'L_Knee_Angle', 'L_Ankle_Angle',
                'R_Hip_Angle', 'R_Knee_Angle', 'R_Ankle_Angle'
            ]
            channels.extend([f"VICON_{ch}" for ch in kin_channels])
        
        return channels

    def _build_subject_index(self):
        """
        Build subject_to_windows mapping for SequentialDataset compatibility.
        This groups all perturbations by subject and maintains temporal order.
        """
        self.subject_to_windows = {}
        
        for idx in range(len(self.Y)):
            subj = self.subject_ids[idx]
            if subj not in self.subject_to_windows:
                self.subject_to_windows[subj] = []
            
            # Get the data and label for this perturbation
            x = self.get_combined_data(idx)  # (T, C)
            x_tensor = torch.tensor(x, dtype=torch.float32).permute(1, 0)  # (C, T)
            y = self.Y[idx]
            
            # Store with perturbation ID for ordering
            pert_id = self.perturbation_ids[idx]
            self.subject_to_windows[subj].append((x_tensor, y, pert_id))
        
        # Sort each subject's windows by perturbation ID (temporal order)
        for subj in self.subject_to_windows:
            self.subject_to_windows[subj].sort(key=lambda item: item[2])
            # Remove pert_id after sorting
            self.subject_to_windows[subj] = [(x, y) for x, y, _ in self.subject_to_windows[subj]]
    
    def get_subject_sequence(self, subject_id: str) -> List[Tuple[torch.Tensor, int]]:
        """
        Get the sequence of windows for a specific subject in temporal order.
        
        Args:
            subject_id: Subject identifier (e.g., '15')
            
        Returns:
            List of (x_tensor, y_label) tuples in temporal order
        """
        return self.subject_to_windows.get(subject_id, [])
    
    @property
    def subjects(self) -> List[str]:
        """Return list of unique subject IDs."""
        return list(self.subject_to_windows.keys())    


class IMUFeatureDataset(Dataset):
    """
    Feature-based dataset that extracts statistical features from both
    IMU and VICON data (mean, std, AUC, peak).
    
    This replicates the feature extraction approach from network_training.m
    """
    
    def __init__(self, root="Data", subject_numbers=None, num_imus=5, 
                 label_encoder=None, use_imu=True, use_kinematics=True):
        self.root = root
        self.num_imus = num_imus
        self.use_imu = use_imu
        self.use_kinematics = use_kinematics
        
        self.X = []
        self.Y = []
        
        if subject_numbers is None:
            subject_numbers = []
        
        self._load_and_extract_features(subject_numbers)
        
        self.X = np.array(self.X, dtype=np.float32)
        
        if label_encoder is None:
            self.le = LabelEncoder()
            self.Y = self.le.fit_transform(self.Y)
        else:
            self.le = label_encoder
            self.Y = self.le.transform(self.Y)
        
        self.Y = np.array(self.Y, dtype=np.int64)
    
    def _load_and_extract_features(self, subject_numbers):
        """Load data and extract statistical features."""
        for subj in subject_numbers:
            subj_folder = os.path.join(self.root, f"HAB-{subj}")
            if not os.path.isdir(subj_folder):
                continue
            
            for file in os.listdir(subj_folder):
                if not file.endswith(".mat"):
                    continue
                
                filename = os.path.splitext(file)[0]
                perturbation_type = re.sub(r"[_T]?\d+", "", filename).strip('_')
                
                mat = loadmat(os.path.join(subj_folder, file))
                
                if 'IMU_perturbation_data' not in mat:
                    continue
                
                imu_perts = mat['IMU_perturbation_data'][0]
                vicon_perts = mat.get('VICON_perturbation_data', [None] * len(imu_perts))[0]
                
                for imu_pert, vicon_pert in zip(imu_perts, vicon_perts):
                    feats = []
                    
                    if self.use_imu:
                        imu_feats = self._extract_imu_features(imu_pert)
                        feats.extend(imu_feats)
                    
                    if self.use_kinematics and vicon_pert is not None:
                        kin_feats = self._extract_kinematics_features(vicon_pert)
                        feats.extend(kin_feats)
                    
                    if feats:  # Only add if we have features
                        self.X.append(feats)
                        self.Y.append(perturbation_type)
    
    def _extract_features_from_signal(self, sig):
        """Extract mean, std, AUC, and peak from a signal."""
        return [
            np.mean(sig),
            np.std(sig),
            np.trapz(np.abs(sig)),
            np.max(np.abs(sig))
        ]
    
    def _extract_imu_features(self, imu_data):
        """Extract features from IMU data."""
        feats = []
        for i in range(self.num_imus):
            base = i * 10
            # AccX, AccY, AccZ
            for j in range(3):
                feats.extend(self._extract_features_from_signal(imu_data[:, base + j]))
            # GyroX, GyroY, GyroZ
            for j in range(3, 6):
                feats.extend(self._extract_features_from_signal(imu_data[:, base + j]))
        return feats
    
    def _extract_kinematics_features(self, vicon_data):
        """Extract features from VICON kinematics data."""
        feats = []
        for j in range(vicon_data.shape[1]):
            feats.extend(self._extract_features_from_signal(vicon_data[:, j]))
        return feats
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.Y[idx], dtype=torch.long)
        return x, y


# -----------------------
# Usage Example
# -----------------------
if __name__ == "__main__":
    # Test the dataset loader
    train_subjects = ['15', '16', '17', '18', '19', '20', '21']
    test_subjects = ['22', '23']
    
    print("=" * 60)
    print("Testing IMU + Kinematics Dataset Loader")
    print("=" * 60)
    
    # Create dataset with both IMU and kinematics
    print("\n1. Loading dataset with IMU + VICON kinematics...")
    train_set = IMUKinematicsDataset(
        root="../IMU_Data",
        subject_numbers=train_subjects,
        pert_window=2000,
        use_imu=True,
        use_kinematics=True,
        normalize=True,
        apply_filter=True
    )
    
    print(f"   Dataset size: {len(train_set)} perturbations")
    print(f"   Number of classes: {len(train_set.le.classes_)}")
    print(f"   Classes: {train_set.le.classes_}")
    print(f"   Total channels: {train_set.num_channels}")
    print(f"   Channel info: {train_set.get_channel_info()[:5]}...")
    
    # Test data loader
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    
    for xb, yb in train_loader:
        print(f"\n   Batch X shape: {xb.shape}")  # Expected: (B, C, T)
        print(f"   Batch Y shape: {yb.shape}")
        print(f"   First 10 labels: {yb[:10]}")
        
        # Check metadata
        print(f"\n   Sample metadata:")
        for i in range(min(3, len(train_set))):
            meta = train_set.get_metadata(i)
            print(f"     {i}: {meta}")
        break
    
    # Test feature-based dataset
    print("\n" + "=" * 60)
    print("Testing Feature-based Dataset Loader")
    print("=" * 60)
    
    feature_set = IMUFeatureDataset(
        root="../IMU_Data",
        subject_numbers=train_subjects[:2],  # Use fewer subjects for quick test
        use_imu=True,
        use_kinematics=True
    )
    
    print(f"   Dataset size: {len(feature_set)} perturbations")
    print(f"   Feature dimension: {feature_set.X.shape[1]}")
    
    feature_loader = DataLoader(feature_set, batch_size=32, shuffle=True)
    
    for xb, yb in feature_loader:
        print(f"\n   Batch X shape: {xb.shape}")  # Expected: (B, F)
        print(f"   Batch Y shape: {yb.shape}")
        break