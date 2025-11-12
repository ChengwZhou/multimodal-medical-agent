import os
import numpy as np
import pandas as pd
import logging
from torch.utils.data import Dataset
import torch
from collections import Counter
from typing import List, Tuple, Optional
from dataset.delta_dataset import DeltaDataset  # Assuming delta_dataset.py is available

from torch.utils.data import DataLoader
from dataset.sequential_dataset import SequentialDataset, collate_sequential_batch

# Configure logging (aligned with sequential_trainer.py)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_info(msg):
    from torch.distributed import is_initialized, get_rank
    if not is_initialized() or get_rank() == 0:
        logger.info(msg)

class MHealthDataset(Dataset):
    """
    Dataset for mHealth data, adapted for sequential processing compatible with sequential_trainer.py
    Groups data by subject and maintains temporal order
    """
    def __init__(self, data_root: str, subjects: List[int], time_steps: int = 100, step: int = 50,
                 balance: bool = False, majority_n: int = 30000, remove_zero_activity: bool = True):
        """
        Args:
            data_root: Data directory path
            subjects: List of subject IDs (e.g., [1, 2, 3])
            time_steps: Sliding window length
            step: Sliding window step size
            balance: Whether to downsample the majority class
            majority_n: Number of samples after downsampling
            remove_zero_activity: Whether to remove Activity=0 samples
        """
        self.data_root = data_root
        self.subjects = sorted(subjects)  # Store subjects list
        self.time_steps = time_steps
        self.step = step
        self.balance = balance
        self.majority_n = majority_n
        self.remove_zero_activity = remove_zero_activity

        # Initialize subject_to_windows dictionary
        self.subject_to_windows = {}
        self.X, self.y = self._load_and_window()

    def _load_and_window(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and process mHealth data, creating windows for each subject
        Returns:
            Xs: Numpy array of all windows [N, T, C]
            ys: Numpy array of labels [N]
        """
        all_Xs, all_ys = [], []

        for subject_id in self.subjects:
            path = os.path.join(self.data_root, f"mHealth_subject{subject_id}.log")
            if not os.path.exists(path):
                log_info(f"Subject {subject_id}: Data file not found at {path}, skipping")
                continue

            # Load data
            df = pd.read_csv(path, header=None, sep="\t")
            # Select feature columns (same as original)
            df = df.loc[:, [5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 19, 23]]
            df = df.rename(columns={
                5: "alx", 6: "aly", 7: "alz",
                8: "glx", 9: "gly", 10: "glz",
                14: "arx", 15: "ary", 16: "arz",
                17: "grx", 18: "gry", 19: "grz",
                23: "Activity"
            })

            # Remove missing values
            df = df.dropna()

            # Handle Activity=0
            if self.remove_zero_activity:
                df = df[df['Activity'] > 0]
                unique_activities = sorted(df['Activity'].unique())
                activity_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_activities)}
                df['Activity'] = df['Activity'].map(activity_mapping)
                log_info(f"Subject {subject_id}: Activity mapping after removing zero activity: {activity_mapping}")
            else:
                log_info(f"Subject {subject_id}: Keeping Activity=0, original labels preserved")

            # Apply sliding window
            X = df.drop(['Activity'], axis=1)
            y = df['Activity']
            subject_Xs, subject_ys = [], []

            for i in range(0, len(X) - self.time_steps + 1, self.step):
                x_window = X.iloc[i:(i + self.time_steps)].values  # [T, C]
                y_window = y.iloc[i:(i + self.time_steps)]

                # Check label consistency
                unique_labels = y_window.unique()
                if len(unique_labels) == 1:
                    label = unique_labels[0]
                else:
                    label_counts = Counter(y_window)
                    most_common = label_counts.most_common(1)[0]
                    if most_common[1] / len(y_window) < 0.8:
                        continue
                    label = most_common[0]

                subject_Xs.append(x_window)
                subject_ys.append(label)

            # Store windows for this subject
            self.subject_to_windows[subject_id] = list(zip(subject_Xs, subject_ys))
            all_Xs.extend(subject_Xs)
            all_ys.extend(subject_ys)

        # Convert to numpy arrays
        Xs = np.array(all_Xs, dtype=np.float32)  # [N, T, C]
        ys = np.array(all_ys, dtype=np.int64)  # [N]

        # Balance at window level (global balancing)
        if self.balance:
            unique_labels, counts = np.unique(ys, return_counts=True)
            log_info(f"Sample distribution before balancing: {dict(zip(unique_labels, counts))}")

            max_count_idx = np.argmax(counts)
            majority_class = unique_labels[max_count_idx]

            majority_mask = (ys == majority_class)
            minority_mask = ~majority_mask

            Xs_majority = Xs[majority_mask]
            ys_majority = ys[majority_mask]
            Xs_minority = Xs[minority_mask]
            ys_minority = ys[minority_mask]

            if len(Xs_majority) > self.majority_n:
                np.random.seed(42)
                downsample_indices = np.random.choice(
                    len(Xs_majority),
                    size=self.majority_n,
                    replace=False
                )
                Xs_majority = Xs_majority[downsample_indices]
                ys_majority = ys_majority[downsample_indices]

            Xs = np.concatenate([Xs_majority, Xs_minority], axis=0)
            ys = np.concatenate([ys_majority, ys_minority], axis=0)

            # Shuffle data order
            shuffle_indices = np.random.permutation(len(Xs))
            Xs = Xs[shuffle_indices]
            ys = ys[shuffle_indices]

            # Invalidate subject_to_windows since balancing shuffles globally
            self.subject_to_windows = {sid: [] for sid in self.subjects}
            log_info("Balancing applied, subject_to_windows invalidated")

        unique_labels, counts = np.unique(ys, return_counts=True)
        log_info(f"Final sample distribution: {dict(zip(unique_labels, counts))}")
        log_info(f"Total number of windows: {len(Xs)}")
        log_info(f"Feature shape: {Xs.shape}")

        return Xs, ys

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.X[idx])  # [T, C]
        x = x.transpose(0, 1)  # [C, T]
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

    def get_subject_sequence(self, subject_id: int) -> List[Tuple[torch.Tensor, int]]:
        """
        Get all windows for a specific subject as a sequence.

        Args:
            subject_id: The subject ID (int)

        Returns:
            List of (x_tensor, y_label) for the subject's windows

        Raises:
            RuntimeError: If balance=True, as subject sequences are not reliable after global shuffling
        """
        if self.balance:
            raise RuntimeError("get_subject_sequence is not supported when balance=True, as global shuffling disrupts subject-specific sequences")
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


def get_dataloaders(data_root: str, batch_size: int = 64, time_steps: int = 100, step: int = 50,
                    balance: bool = False, remove_zero_activity: bool = True, apply_diff: bool = False) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test data loaders with optional differencing.

    Args:
        data_root: Data directory path
        batch_size: Batch size for data loaders
        time_steps: Sliding window length
        time_steps: Sliding window length
        step: Sliding window step size
        balance: Whether to balance the dataset
        remove_zero_activity: Whether to remove Activity=0 samples
        apply_diff: Whether to apply differencing

    Returns:
        train_loader, test_loader: PyTorch DataLoader objects
    """

    train_subjects = [i for i in range(1, 9)]
    test_subjects = [9, 10]

    # Initialize base datasets
    train_base_dataset = MHealthDataset(
        data_root,
        train_subjects,
        time_steps,
        step,
        balance=balance,
        remove_zero_activity=remove_zero_activity
    )
    test_base_dataset = MHealthDataset(
        data_root,
        test_subjects,
        time_steps,
        step,
        balance=False,  # Usually don't balance test set
        remove_zero_activity=remove_zero_activity
    )

    # Apply differencing if requested
    if apply_diff:
        train_base_dataset = DeltaDataset(train_base_dataset, axis=-1)
        test_base_dataset = DeltaDataset(test_base_dataset, axis=-1)

    # Wrap in SequentialDataset for sequential processing
    train_dataset = SequentialDataset(train_base_dataset, subject_ids=train_subjects)
    test_dataset = SequentialDataset(test_base_dataset, subject_ids=test_subjects)
    print(train_dataset.max_lengths)

    # Create data loaders with sequential collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_sequential_batch
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_sequential_batch
    )

    return train_loader, test_loader


# -----------------------------
# Test code
# -----------------------------

if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    # Test configuration
    data_root = "/Users/chengweizhou/PycharmProjects/data/MHEALTHDATASET"  # Update with your actual path

    print("=" * 80)
    print("Testing MHealth + DeltaDataset + SequentialDataset")
    print("=" * 80)

    try:
        # ========================================
        # 1. Test without DeltaDataset (baseline)
        # ========================================
        print("\n1. Testing without DeltaDataset...")
        start_time = time.time()

        base_dataset = MHealthDataset(
            data_root=data_root,
            subjects=[1],  # 只测一个 subject 加快速度
            time_steps=100,
            step=50,
            balance=False,
            remove_zero_activity=True
        )
        seq_dataset = SequentialDataset(base_dataset, subject_ids=[1])
        loader = DataLoader(seq_dataset, batch_size=1, shuffle=False, collate_fn=collate_sequential_batch)

        # 取第一个 batch
        batch = next(iter(loader))
        x_orig = batch.sequences[0]  # [C, max_T]
        y_orig = batch.labels[0]
        lengths = batch.seq_lengths[0].item()

        print(f"   Original x shape: {x_orig.shape}")
        print(f"   Sequence length: {lengths}")
        print(f"   Label sequence: {y_orig[:10].tolist()}...")

        # 取第一个模态第一个时间步
        signal_orig = x_orig[0, 0, :10]  # [T]
        print(signal_orig)
        print(f"   First modality range: [{signal_orig.min():.4f}, {signal_orig.max():.4f}]")

        init_time = time.time() - start_time
        print(f"   Init time: {init_time:.2f}s")

        # ========================================
        # 2. Test WITH DeltaDataset
        # ========================================
        print("\n2. Testing WITH DeltaDataset...")
        start_time = time.time()

        delta_dataset = DeltaDataset(base_dataset, axis=-1)  # 沿时间轴差分
        seq_delta_dataset = SequentialDataset(delta_dataset, subject_ids=[1])
        delta_loader = DataLoader(seq_delta_dataset, batch_size=1, shuffle=False, collate_fn=collate_sequential_batch)

        delta_batch = next(iter(delta_loader))
        x_delta = delta_batch.sequences[0]  # [C, max_T]
        print(f"   Delta x shape: {x_delta.shape}")
        print(f"   Delta sequence length: {delta_batch.seq_lengths[0].item()}")

        signal_delta = x_delta[0, 0, :10]  # [T]
        print(signal_delta)
        print(f"   First modality delta range: [{signal_delta.min():.6f}, {signal_delta.max():.6f}]")

        # ========================================
        # 3. Verify differencing is correct: cumsum(delta) == original
        # ========================================
        print("\n3. Verifying cumsum(delta) == original...")
        signal_recovered = torch.cumsum(signal_delta, dim=0)
        diff = torch.abs(signal_recovered - signal_orig).max().item()
        print(f"   Max recovery error: {diff:.8f}")
        assert diff < 1e-6, "Differencing is not reversible!"
        print("   Recovery verified!")

        # ========================================
        # 4. Test get_dataloaders with apply_diff=True
        # ========================================
        # print("\n4. Testing get_dataloaders(apply_diff=True)...")
        # train_loader, test_loader = get_dataloaders(
        #     data_root=data_root,
        #     batch_size=2,
        #     time_steps=100,
        #     step=50,
        #     balance=False,
        #     remove_zero_activity=True,
        #     apply_diff=True
        # )
        #
        # batch = next(iter(train_loader))
        # print(f"   Train batch sequences shape: {batch.sequences.shape}")
        # print(f"   Train batch labels shape: {batch.labels.shape}")
        # print(f"   Train batch seq_lengths: {batch.seq_lengths.tolist()}")
        # print(f"   First sequence first modality delta[0]: {batch.sequences[0, 0, 0].item():.6f}")
        #
        # print("\n" + "=" * 80)
        # print("All DeltaDataset tests passed!")
        # print("=" * 80)

    except FileNotFoundError as e:
        print(f"Error: Data directory not found - {e}")
        print("Please check the data path and make sure the dataset is available.")
    except Exception as e:
        print(f"Error during testing: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()