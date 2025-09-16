import os
import numpy as np
import pandas as pd
import logging
from scipy import stats
from torch.utils.data import Dataset, DataLoader
from torch.distributed import get_rank, is_initialized
import torch
from sklearn.utils import resample
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_info(msg):
    if not is_initialized() or get_rank() == 0:
        logger.info(msg)


class MHealthDataset(Dataset):
    def __init__(self, data_root, subjects, time_steps=100, step=50, balance=False, majority_n=30000,
                 remove_zero_activity=True):
        """
        Args:
            data_root: Data directory path
            subjects: List of subjects to load (e.g. [1,2,3])
            time_steps: Sliding window length
            step: Sliding window step size
            balance: Whether to downsample Activity=0 samples (or the majority class)
            majority_n: Number of samples after downsampling
            remove_zero_activity: Whether to remove Activity=0 samples. If True, labels start from 0; if False, labels include 0
        """
        self.remove_zero_activity = remove_zero_activity
        self.X, self.y = self._load_and_window(data_root, subjects, time_steps, step, balance, majority_n)

    def _load_and_window(self, data_root, subjects, time_steps, step, balance, majority_n):
        all_Xs, all_ys = [], []

        # Process each subject separately
        for subject_id in subjects:
            path = os.path.join(data_root, f"mHealth_subject{subject_id}.log")
            df = pd.read_csv(path, header=None, sep="\t")

            # Select feature columns
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

            # Handle Activity=0 based on parameter
            if self.remove_zero_activity:
                df = df[df['Activity'] > 0]  # Remove Activity=0
                # Relabel activities to start from 0
                unique_activities = sorted(df['Activity'].unique())
                activity_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_activities)}
                df['Activity'] = df['Activity'].map(activity_mapping)
                if subject_id == 0:
                    log_info(f"Subject {subject_id}: Activity mapping after removing zero activity: {activity_mapping}")
            else:
                if subject_id == 0:
                    # Keep Activity=0, no relabeling needed
                    log_info(f"Subject {subject_id}: Keeping Activity=0, original labels preserved")

            # Apply sliding window to current subject
            X = df.drop(['Activity'], axis=1)
            y = df['Activity']

            subject_Xs, subject_ys = [], []
            for i in range(0, len(X) - time_steps + 1, step):
                x_window = X.iloc[i:(i + time_steps)].values
                y_window = y.iloc[i:(i + time_steps)]

                # Check label consistency within the window
                unique_labels = y_window.unique()
                if len(unique_labels) == 1:
                    # Only one activity label in the window, use it directly
                    label = unique_labels[0]
                else:
                    # Multiple activity labels in the window, use majority vote or skip
                    label_counts = Counter(y_window)
                    most_common = label_counts.most_common(1)[0]

                    # Skip window if majority vote ratio is too low
                    if most_common[1] / len(y_window) < 0.8:
                        continue
                    label = most_common[0]

                subject_Xs.append(x_window)
                subject_ys.append(label)

            all_Xs.extend(subject_Xs)
            all_ys.extend(subject_ys)

        # Convert to numpy arrays
        Xs = np.array(all_Xs, dtype=np.float32)
        ys = np.array(all_ys, dtype=np.int64)

        # Balance at window level
        if balance:
            # Count samples for each class
            unique_labels, counts = np.unique(ys, return_counts=True)
            log_info(f"Sample distribution before balancing: {dict(zip(unique_labels, counts))}")

            # Find the class that needs downsampling (usually the one with most samples)
            max_count_idx = np.argmax(counts)
            majority_class = unique_labels[max_count_idx]

            # Separate majority and minority classes
            majority_mask = (ys == majority_class)
            minority_mask = ~majority_mask

            Xs_majority = Xs[majority_mask]
            ys_majority = ys[majority_mask]
            Xs_minority = Xs[minority_mask]
            ys_minority = ys[minority_mask]

            # Downsample majority class
            if len(Xs_majority) > majority_n:
                np.random.seed(42)
                downsample_indices = np.random.choice(
                    len(Xs_majority),
                    size=majority_n,
                    replace=False
                )
                Xs_majority = Xs_majority[downsample_indices]
                ys_majority = ys_majority[downsample_indices]

            # Recombine data
            Xs = np.concatenate([Xs_majority, Xs_minority], axis=0)
            ys = np.concatenate([ys_majority, ys_minority], axis=0)

            # Shuffle data order
            shuffle_indices = np.random.permutation(len(Xs))
            Xs = Xs[shuffle_indices]
            ys = ys[shuffle_indices]

        # Print final class distribution
        unique_labels, counts = np.unique(ys, return_counts=True)
        log_info(f"Final sample distribution: {dict(zip(unique_labels, counts))}")
        log_info(f"Total number of windows: {len(Xs)}")
        log_info(f"Feature shape: {Xs.shape}")

        return Xs, ys

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        x = x.transpose(0, 1)  # Transpose to (features, time_steps)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


def get_dataloaders(data_root, batch_size=64, time_steps=100, step=50, balance=False, remove_zero_activity=True):
    """
    Create train and test data loaders

    Args:
        data_root: Data directory path
        batch_size: Batch size for data loaders
        time_steps: Sliding window length
        step: Sliding window step size
        balance: Whether to balance the dataset
        remove_zero_activity: Whether to remove Activity=0 samples

    Returns:
        train_loader, test_loader: PyTorch DataLoader objects
    """
    train_subjects = [i for i in range(1, 9)]  # Subjects 1-8 for training
    test_subjects = [9, 10]  # Subjects 9-10 for testing

    train_dataset = MHealthDataset(
        data_root,
        train_subjects,
        time_steps,
        step,
        balance=balance,
        remove_zero_activity=remove_zero_activity
    )
    test_dataset = MHealthDataset(
        data_root,
        test_subjects,
        time_steps,
        step,
        balance=False,  # Usually don't balance test set
        remove_zero_activity=remove_zero_activity
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader