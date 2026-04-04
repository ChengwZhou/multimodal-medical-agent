import os
import numpy as np
import scipy.io
import logging
from torch.utils.data import Dataset
import torch
from collections import Counter
from typing import List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KinematicsDataset(Dataset):
    """
    Dataset for HAB perturbation data with kinematics.
    
    Data structure:
    - Each .mat file contains 10 perturbation events (cells in IMU_perturbation_data and VICON_perturbation_data)
    - IMU: 1000 Hz, 70 signals (10 signals × 7 sensors) 
    - VICON: 100 Hz, 9 signals (COM xyz, L/R hip/knee/ankle angles)
    """
    
    def __init__(
        self,
        data_root: str,
        subjects: List[int],
        time_steps: int = 100,  
        step: int = 50,          
        balance: bool = False,
        majority_n: int = 30000,
        remove_short_perturbations: bool = True,
        min_length: int = 100,   # minimum perturbation length in samples at 100 Hz
    ):
        """
        Args:
            data_root: Data directory path
            subjects: List of subject IDs (e.g., [1, 2, 3])
            time_steps: Sliding window length
            step: Sliding window step size
            balance: Whether to downsample majority class
            majority_n: Number of samples after downsampling
            remove_short_perturbations: Remove perturbations shorter than min_length
            min_length: Minimum length required for a perturbation (after downsampling)
        """
        self.data_root = data_root
        self.subjects = sorted(subjects)
        self.time_steps = time_steps
        self.step = step
        self.balance = balance
        self.majority_n = majority_n
        self.remove_short_perturbations = remove_short_perturbations
        self.min_length = min_length
        
        # Mapping from perturbation type to label
        self.label_map = self._create_label_map()
        self.inverse_label_map = {v: k for k, v in self.label_map.items()}
        
        # Store subject to windows mapping
        self.subject_to_windows = {}
        
        # Load all data
        self.X, self.y = self._load_and_window()
        
    def _create_label_map(self) -> dict:
        """Create mapping from perturbation type to integer label."""
        perturbation_types = [
            "QS_F",      # 0: quiet standing, fall forwards
            "QS_B",      # 1: quiet standing, fall backwards
            "HS_B_Left", # 2: heel-strike, left leg
            "HS_B_Right",# 3: heel-strike, right leg
            "TO_F_Left", # 4: toe-off, left leg
            "TO_F_Right",# 5: toe-off, right leg
            "MS_F_Left", # 6: mid-stance, left leg, fall forwards
            "MS_F_Right",# 7: mid-stance, right leg, fall forwards
            "MS_B_Left", # 8: mid-stance, left leg, fall backwards
            "MS_B_Right",# 9: mid-stance, right leg, fall backwards
        ]
        return {ptype: idx for idx, ptype in enumerate(perturbation_types)}
    


    def _extract_label_from_filename(self, filename: str) -> str:
        """
        Extract perturbation type from filename.
        Examples:
            "HS_B_Left_1.mat" -> "HS_B_Left"
            "TO_F_Right_2.mat" -> "TO_F_Right"
            "QS_F_1.mat" -> "QS_F"
            "QS_B_1.mat" -> "QS_B"
        """
        name = os.path.splitext(filename)[0]  # Remove .mat
        parts = name.split('_')
        
        if parts[-1].isdigit():
            perturbation_parts = parts[:-1]
        else:
            perturbation_parts = parts
        
        return '_'.join(perturbation_parts)

    
    def _downsample_imu(self, imu_data: np.ndarray, target_length: int) -> np.ndarray:
        """
        Downsample IMU data from 1000 Hz to 100 Hz.
        
        Args:
            imu_data: [N_imu, 30] where N_imu ≈ 10 * target_length
            target_length: Desired length after downsampling (matches VICON length)
        
        Returns:
            downsampled: [target_length, 30]
        """
        # every 10th sample 
        step = imu_data.shape[0] // target_length
        indices = np.linspace(0, imu_data.shape[0] - 1, target_length, dtype=int)
        return imu_data[indices, :]
    
    def _process_mat_file(self, mat_path: str, subject_id: int) -> List[Tuple[np.ndarray, int]]:
        """
        Process a single .mat file, extracting all perturbations.
        
        Args:
            mat_path: Path to .mat file
            subject_id: Subject ID (for logging)
        
        Returns:
            List of (features, label) where features shape = [time, 39]
        """

        mat_data = scipy.io.loadmat(mat_path)
        
        imu_cells = mat_data['IMU_perturbation_data'][0]
        vicon_cells = mat_data['VICON_perturbation_data'][0]
        
        # get label
        label_str = self._extract_label_from_filename(os.path.basename(mat_path))
        if label_str not in self.label_map:
            logger.warning(f"Unknown label '{label_str}' in {mat_path}, skipping")
            return []
        
        label_id = self.label_map[label_str]
        
        samples = []
        
        for pert_idx in range(len(imu_cells)):
            imu_raw = imu_cells[pert_idx]  # [N_imu, 70]
            vicon_raw = vicon_cells[pert_idx]  # [N_vicon, 9]
            
            imu_30 = imu_raw[:, :30]  # [N_imu, 30]
            
            if subject_id == 17:
                pass
            
            # downsample IMU to match VICON length
            target_len = vicon_raw.shape[0]
            imu_downsampled = self._downsample_imu(imu_30, target_len)  # [N_vicon, 30]
            
            combined = np.concatenate([imu_downsampled, vicon_raw], axis=1)  # [N_vicon, 39]
            
            # check min length
            if self.remove_short_perturbations and combined.shape[0] < self.min_length:
                logger.debug(f"Skipping short perturbation ({combined.shape[0]} < {self.min_length}) in {mat_path}")
                continue
            
            # apply sliding windows
            for start in range(0, combined.shape[0] - self.time_steps + 1, self.step):
                window = combined[start:start + self.time_steps, :]  #[time_steps, 39]
                samples.append((window, label_id))
        
        return samples
    
    def _load_and_window(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load all .mat files for all subjects and create windows.
        
        Returns:
            X: [N, time_steps, 39] array of windows
            y: [N] array of labels
        """
        all_X = []
        all_y = []
        
        for subject_id in self.subjects:
            subject_folder = os.path.join(self.data_root, f"HAB-{subject_id}")
            
            if not os.path.exists(subject_folder):
                logger.warning(f"Subject folder {subject_folder} does not exist, skipping")
                continue
            
            # get all .mat files
            mat_files = [f for f in os.listdir(subject_folder) if f.endswith('.mat')]
            
            subject_samples = []
            
            for mat_file in mat_files:
                mat_path = os.path.join(subject_folder, mat_file)
                try:
                    samples = self._process_mat_file(mat_path, subject_id)
                    subject_samples.extend(samples)
                except Exception as e:
                    logger.error(f"Error processing {mat_path}: {e}")
                    continue
            
            self.subject_to_windows[subject_id] = subject_samples

            for window, label in subject_samples:
                all_X.append(window)
                all_y.append(label)
            
            logger.info(f"Subject {subject_id}: {len(subject_samples)} windows from {len(mat_files)} files")
        
        if not all_X:
            raise ValueError(f"No data loaded for subjects {self.subjects}")
        
        X = np.array(all_X, dtype=np.float32)  # [N, T, C]
        y = np.array(all_y, dtype=np.int64)    # [N]
        
        if self.balance:
            X, y = self._balance_dataset(X, y)
        
        logger.info(f"Total windows: {len(X)}")
        logger.info(f"Feature shape: {X.shape}")
        logger.info(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        return X, y
    
    def _balance_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Downsample majority class to majority_n samples."""
        unique_labels, counts = np.unique(y, return_counts=True)
        logger.info(f"Label distribution before balancing: {dict(zip(unique_labels, counts))}")
        
        # majority class
        majority_idx = np.argmax(counts)
        majority_class = unique_labels[majority_idx]
        
        majority_mask = (y == majority_class)
        minority_mask = ~majority_mask
        
        X_majority = X[majority_mask]
        y_majority = y[majority_mask]
        X_minority = X[minority_mask]
        y_minority = y[minority_mask]
        
        if len(X_majority) > self.majority_n:
            np.random.seed(42)
            indices = np.random.choice(len(X_majority), self.majority_n, replace=False)
            X_majority = X_majority[indices]
            y_majority = y_majority[indices]
        
        # Combine and shuffle
        X_balanced = np.concatenate([X_majority, X_minority], axis=0)
        y_balanced = np.concatenate([y_majority, y_minority], axis=0)
        
        shuffle_idx = np.random.permutation(len(X_balanced))
        X_balanced = X_balanced[shuffle_idx]
        y_balanced = y_balanced[shuffle_idx]
        
        logger.info(f"Label distribution after balancing: {dict(zip(*np.unique(y_balanced, return_counts=True)))}")
        
        return X_balanced, y_balanced
    
    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (features, label) where features shape = [C, T] (channels first)."""
        x = torch.from_numpy(self.X[idx])  # [T, C]
        x = x.transpose(0, 1)  # [C, T]
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y
    
    def get_subject_sequence(self, subject_id: int) -> List[Tuple[torch.Tensor, int]]:
        """Get all windows for a subject in temporal order."""
        if self.balance:
            raise RuntimeError("get_subject_sequence not supported when balance=True")
        
        if subject_id not in self.subject_to_windows:
            return []
        
        sequence = []
        for x_np, y_label in self.subject_to_windows[subject_id]:
            x = torch.from_numpy(x_np).float()  # [T, C]
            x = x.transpose(0, 1)  # [C, T]
            sequence.append((x, y_label))
        
        return sequence


def get_kinematics_dataloaders(
    data_root: str,
    batch_size: int = 64,
    time_steps: int = 100,
    step: int = 50,
    balance: bool = False,
    train_subjects: List[int] = None,
    val_subjects: List[int] = None,
) -> Tuple:
    """
    Create train and validation dataloaders for kinematics dataset.
    
    Args:
        data_root: Path to Data_with_Kinematics directory
        batch_size: Batch size
        time_steps: Window length in samples (at 100 Hz)
        step: Window step
        balance: Whether to balance training set
        train_subjects: List of subject IDs for training (default: [15-22])
        val_subjects: List of subject IDs for validation (default: [23, 24])
    
    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader
    from dataset.sequential_dataset import SequentialDataset, collate_sequential_batch
    
    if train_subjects is None:
        train_subjects = list(range(15, 23))  # HAB-15 to HAB-22
    if val_subjects is None:
        val_subjects = [23, 24]  # HAB-23 to HAB-24
    
    train_base = KinematicsDataset(
        data_root=data_root,
        subjects=train_subjects,
        time_steps=time_steps,
        step=step,
        balance=balance,
    )
    
    val_base = KinematicsDataset(
        data_root=data_root,
        subjects=val_subjects,
        time_steps=time_steps,
        step=step,
        balance=False,  
    )
    
    train_dataset = SequentialDataset(train_base, subject_ids=train_subjects)
    val_dataset = SequentialDataset(val_base, subject_ids=val_subjects)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_sequential_batch,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_sequential_batch,
        num_workers=4,
        pin_memory=True,
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
    else:
        data_root = "/home/bkl46/CSE_MSE_RXF131/cradle-members/mds3/bkl46/multimodal-medical-agent/data/Data_with_Kinematics" 
    
    print("Testing KinematicsDataset...")
    
    # Test with one subject
    dataset = KinematicsDataset(
        data_root=data_root,
        subjects=[15],
        time_steps=100,
        step=50,
        balance=False,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        x, y = dataset[0]
        print(f"Sample shape: {x.shape} (channels, time)")
        print(f"Sample label: {y}")
        print(f"Label mapping: {dataset.label_map}")

    hab17_dataset = KinematicsDataset(
        data_root=data_root,
        subjects=[17],
        time_steps=100,
        step=50,
        balance=False,
    )
    print(f"HAB-17 dataset size: {len(hab17_dataset)}")
    if len(hab17_dataset) > 0:
        x, y = hab17_dataset[0]

        print(f"HAB-17 sample stats: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}")
