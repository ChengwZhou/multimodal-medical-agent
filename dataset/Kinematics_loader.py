import os
import numpy as np
import scipy.io
import logging
from torch.utils.data import Dataset, DataLoader
import torch
from collections import Counter
from typing import List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_info(msg: str) -> None:
    """Rank-0-only logging, compatible with DDP (mirrors mHEALTH_loader.py)."""
    from torch.distributed import is_initialized, get_rank
    if not is_initialized() or get_rank() == 0:
        logger.info(msg)

# ---------------------------------------------------------------------------
# IMU column layout
# ---------------------------------------------------------------------------
# The raw IMU array has 70 columns: 7 sensors x 10 signals each.
# Signal order per sensor: Accx Accy Accz Gyrox Gyroy Gyroz Ori_i Ori_j Ori_k <unused>
# We keep only acc+gyro (first 6 per sensor) for sensors 1-5 (indices 0-4).
# Sensors 6-7 (columns 50-69) are never used and are always omitted.
#
# This gives the correct 30 columns:
#   Sensor 1 (right thigh)  -> raw cols  0- 5
#   Sensor 2 (right shank)  -> raw cols 10-15
#   Sensor 3 (left shank)   -> raw cols 20-25
#   Sensor 4 (left thigh)   -> raw cols 30-35
#   Sensor 5 (torso)        -> raw cols 40-45
#
# IMPORTANT: naively slicing [:, :30] would grab cols 0-29, which mixes
# orientation data from sensors 1-3 and misses sensors 4 and 5 entirely.
# The correct approach is to stride-select 6 columns from each sensor block.

IMU_SENSOR_COLS: List[int] = []
for _s in range(5):          # sensors 1-5 (0-indexed)
    _base = _s * 10
    IMU_SENSOR_COLS.extend(range(_base, _base + 6))   # acc + gyro only
# Result: [0,1,2,3,4,5, 10,11,12,13,14,15, 20,21,22,23,24,25, 30,31,32,33,34,35, 40,41,42,43,44,45]

NUM_IMU_CHANNELS   = len(IMU_SENSOR_COLS)              # 30
NUM_VICON_CHANNELS = 9                                  # COMx/y/z + 6 joint angles
NUM_ALL_CHANNELS   = NUM_IMU_CHANNELS + NUM_VICON_CHANNELS  # 39

# Sensors 4 & 5 are the ones that failed on HAB-17.
# After column selection their positions in the 30-channel output are:
#   sensor 4 (left thigh) -> output cols 18-23
#   sensor 5 (torso)      -> output cols 24-29
HAB17_BAD_SENSOR_COLS = list(range(18, 30))

# Subjects with known sensor faults -> map to which output cols to zero-fill.
# Extend this dict if you discover other subjects with hardware issues.
SUBJECTS_WITH_MISSING_SENSORS = {
    17: HAB17_BAD_SENSOR_COLS,
}


class KinematicsDataset(Dataset):
    """
    Dataset for the HAB perturbation study ("Data with Kinematics").

    Directory layout:

        data_root/
            HAB-15/
                QS_F_1.mat
                QS_B_1.mat
                HS_B_Left_1.mat
                ...
            HAB-16/
                ...

    Each .mat file contains MATLAB cell arrays:
        IMU_perturbation_data   - (1, n_reps), each cell [N_imu  x 70]  @ 1000 Hz
        VICON_perturbation_data - (1, n_reps), each cell [N_vicon x  9]  @  100 Hz

    IMU is downsampled to 100 Hz to match VICON, then the two are concatenated
    into a 39-channel feature vector [acc+gyro (30) | COM+angles (9)].

    Windows are labelled by perturbation type derived from the filename.
    The dataset preserves per-subject temporal order via subject_to_windows
    so that SequentialDataset + BPTT in agent_trainer.py work correctly.

    Args:
        data_root:                   Root directory (one sub-folder per participant).
        subjects:                    Integer subject IDs, e.g. [15, 16, 17].
        time_steps:                  Sliding window length in samples (at 100 Hz).
        step:                        Sliding window stride in samples.
        balance:                     Downsample majority class to majority_n windows.
        majority_n:                  Target size for majority-class downsampling.
        remove_short_perturbations:  Drop individual repetitions shorter than min_length.
        min_length:                  Minimum repetition length (samples at 100 Hz).
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
        min_length: int = 100,
    ):
        self.data_root   = data_root
        self.subjects    = sorted(subjects)
        self.time_steps  = time_steps
        self.step        = step
        self.balance     = balance
        self.majority_n  = majority_n
        self.remove_short_perturbations = remove_short_perturbations
        self.min_length  = min_length

        self.label_map         = self._create_label_map()
        self.inverse_label_map = {v: k for k, v in self.label_map.items()}

        # subject_id (int) -> [(window_np [T, C], label_int), ...]
        # Populated during _load_and_window; used by get_subject_sequence().
        self.subject_to_windows: dict = {}

        self.X, self.y = self._load_and_window()

    # ------------------------------------------------------------------
    # Label helpers
    # ------------------------------------------------------------------

    def _create_label_map(self) -> dict:
        """Fixed integer label for each perturbation type (0-indexed, stable)."""
        perturbation_types = [
            "QS_F",       # 0 - quiet standing, fall forwards
            "QS_B",       # 1 - quiet standing, fall backwards
            "HS_B_Left",  # 2 - heel-strike, left leg pulled forward
            "HS_B_Right", # 3 - heel-strike, right leg pulled forward
            "TO_F_Left",  # 4 - toe-off, left leg pulled backward
            "TO_F_Right", # 5 - toe-off, right leg pulled backward
            "MS_F_Left",  # 6 - mid-stance, left, fall forwards
            "MS_F_Right", # 7 - mid-stance, right, fall forwards
            "MS_B_Left",  # 8 - mid-stance, left, fall backwards
            "MS_B_Right", # 9 - mid-stance, right, fall backwards
        ]
        return {ptype: idx for idx, ptype in enumerate(perturbation_types)}

    def _extract_label_from_filename(self, filename: str) -> str:
        """
        Strip the trial-number suffix and extension to get the perturbation key.

        Examples:
            "HS_B_Left_1.mat"   -> "HS_B_Left"
            "TO_F_Right_2.mat"  -> "TO_F_Right"
            "QS_F_1.mat"        -> "QS_F"
            "MS_B_Right.mat"    -> "MS_B_Right"   (no suffix - also handled)
        """
        name  = os.path.splitext(filename)[0]   # drop .mat
        parts = name.split('_')
        if parts[-1].isdigit():
            parts = parts[:-1]
        return '_'.join(parts)

    # ------------------------------------------------------------------
    # Signal processing
    # ------------------------------------------------------------------

    @staticmethod
    def _select_imu_channels(imu_raw: np.ndarray) -> np.ndarray:
        """
        Extract acc+gyro (6 channels) from each of the 5 used sensors.

        Parameters
        ----------
        imu_raw : np.ndarray  [N, 70]  (or [N, 50] if sensors 6-7 already absent)

        Returns
        -------
        np.ndarray  [N, 30]  ordered as:
            [s1_acc(3), s1_gyro(3), s2_acc(3), s2_gyro(3), ..., s5_acc(3), s5_gyro(3)]
        """
        return imu_raw[:, IMU_SENSOR_COLS].astype(np.float32)

    @staticmethod
    def _downsample_imu(imu_data: np.ndarray, target_length: int) -> np.ndarray:
        """
        Downsample IMU from 1000 Hz to 100 Hz by uniform index selection.

        Uses np.linspace rather than a fixed stride so it is robust when the
        recorded IMU length is not exactly 10x the VICON length (which can
        happen at trial boundaries or if the collection was interrupted).

        Parameters
        ----------
        imu_data      : np.ndarray  [N_imu, 30]
        target_length : int         desired output rows (== VICON length)

        Returns
        -------
        np.ndarray  [target_length, 30]
        """
        indices = np.linspace(0, imu_data.shape[0] - 1, target_length, dtype=int)
        return imu_data[indices, :]

    @staticmethod
    def _zero_fill_missing_sensors(
        imu_30: np.ndarray,
        bad_cols: List[int],
    ) -> np.ndarray:
        """
        Zero-fill channels that belong to non-reporting sensors.

        Keeps the channel count fixed at 30 so the model architecture does not
        change between subjects.  The gating agent can learn to ignore
        these channels (constant-zero signal carries no information).

        Parameters
        ----------
        imu_30   : np.ndarray  [N, 30]
        bad_cols : list of column indices (into the 30-ch output) to zero-fill
        """
        out = imu_30.copy()
        out[:, bad_cols] = 0.0
        return out

    # ------------------------------------------------------------------
    # Core loading logic
    # ------------------------------------------------------------------

    def _process_mat_file(
        self, mat_path: str, subject_id: int
    ) -> List[Tuple[np.ndarray, int]]:
        """
        Load one .mat file and return a list of (window_array, label) pairs.

        Each .mat file represents one perturbation type, repeated n_reps times
        (typically >= 10).  Every window within the file gets the same label,
        derived from the filename.

        Parameters
        ----------
        mat_path   : full path to the .mat file
        subject_id : integer subject ID (used for logging and sensor fix lookup)

        Returns
        -------
        List of (np.ndarray [time_steps, 39], int)
        """
        mat_data    = scipy.io.loadmat(mat_path)
        imu_cells   = mat_data['IMU_perturbation_data'][0]    # object array of cells
        vicon_cells = mat_data['VICON_perturbation_data'][0]

        label_str = self._extract_label_from_filename(os.path.basename(mat_path))
        if label_str not in self.label_map:
            logger.warning(
                f"Subject {subject_id}: unknown label '{label_str}' in "
                f"'{os.path.basename(mat_path)}', skipping file."
            )
            return []

        label_id = self.label_map[label_str]
        bad_cols  = SUBJECTS_WITH_MISSING_SENSORS.get(subject_id, [])
        samples: List[Tuple[np.ndarray, int]] = []

        for rep_idx in range(len(imu_cells)):
            imu_raw   = np.array(imu_cells[rep_idx],  dtype=np.float64)  # [N_imu,  70]
            vicon_raw = np.array(vicon_cells[rep_idx], dtype=np.float64)  # [N_vicon, 9]

            # Replace any NaN values with 0 before processing
            imu_raw   = np.where(np.isnan(imu_raw),  0.0, imu_raw)
            vicon_raw = np.where(np.isnan(vicon_raw), 0.0, vicon_raw)

            # --- Correct column selection (stride, not naive slice) ---
            imu_30 = self._select_imu_channels(imu_raw)   # [N_imu, 30]

            # --- Zero-fill known bad sensors (e.g. HAB-17 sensors 4 & 5) ---
            if bad_cols:
                imu_30 = self._zero_fill_missing_sensors(imu_30, bad_cols)

            # Downsample IMU (1000 Hz) to match VICON rate (100 Hz)
            target_len      = vicon_raw.shape[0]
            imu_downsampled = self._downsample_imu(imu_30, target_len)  # [N_vicon, 30]

            combined = np.concatenate(
                [imu_downsampled, vicon_raw.astype(np.float32)], axis=1
            ).astype(np.float32)  # [N_vicon, 39]

            # Drop repetitions too short to yield even one window
            if self.remove_short_perturbations and combined.shape[0] < self.min_length:
                logger.debug(
                    f"Subject {subject_id}: rep {rep_idx + 1} of '{label_str}' "
                    f"too short ({combined.shape[0]} < {self.min_length}), skipping."
                )
                continue

            # Sliding window
            for start in range(
                0, combined.shape[0] - self.time_steps + 1, self.step
            ):
                window = combined[start: start + self.time_steps, :]  # [T, 39]
                samples.append((window, label_id))

        return samples

    def _load_and_window(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Iterate over all subjects and .mat files, build the flat window arrays.

        Returns
        -------
        X : np.ndarray  [N, T, C]
        y : np.ndarray  [N]
        """
        all_X: List[np.ndarray] = []
        all_y: List[int]        = []

        for subject_id in self.subjects:
            # Try zero-padded (HAB-01) and non-padded (HAB-1) folder names
            subject_folder = os.path.join(self.data_root, f"HAB-{subject_id:02d}")
            if not os.path.exists(subject_folder):
                subject_folder = os.path.join(self.data_root, f"HAB-{subject_id}")
            if not os.path.exists(subject_folder):
                logger.warning(
                    f"Subject {subject_id}: folder not found "
                    f"(tried HAB-{subject_id:02d} and HAB-{subject_id}), skipping."
                )
                continue

            mat_files = sorted(
                f for f in os.listdir(subject_folder) if f.endswith('.mat')
            )
            if not mat_files:
                logger.warning(
                    f"Subject {subject_id}: no .mat files in {subject_folder}, skipping."
                )
                continue

            if subject_id in SUBJECTS_WITH_MISSING_SENSORS:
                bad = SUBJECTS_WITH_MISSING_SENSORS[subject_id]
                log_info(
                    f"Subject {subject_id}: output cols {bad} will be zero-filled "
                    f"(known hardware fault on sensors 4 & 5)."
                )

            subject_samples: List[Tuple[np.ndarray, int]] = []

            for mat_file in mat_files:
                mat_path = os.path.join(subject_folder, mat_file)
                try:
                    file_samples = self._process_mat_file(mat_path, subject_id)
                    subject_samples.extend(file_samples)
                except Exception as e:
                    logger.error(
                        f"Subject {subject_id}: error processing '{mat_file}': {e}"
                    )
                    continue

            self.subject_to_windows[subject_id] = subject_samples

            for window, label in subject_samples:
                all_X.append(window)
                all_y.append(label)

            log_info(
                f"Subject {subject_id}: {len(subject_samples)} windows "
                f"from {len(mat_files)} file(s)."
            )

        if not all_X:
            raise ValueError(
                f"KinematicsDataset: no windows loaded for subjects {self.subjects}. "
                f"Check data_root='{self.data_root}'."
            )

        X = np.array(all_X, dtype=np.float32)   # [N, T, C]
        y = np.array(all_y, dtype=np.int64)      # [N]


                # ========== ADD NORMALIZATION CODE HERE ==========
        # Normalize per channel across all data
        from sklearn.preprocessing import StandardScaler
        
        N, T, C = X.shape
        X_flat = X.reshape(-1, C)  # [N*T, C]
        
        # Fit scaler on the current data (training or validation)
        # Note: In production, you should fit only on training data and transform validation
        self.scaler = StandardScaler()
        X_flat_normalized = self.scaler.fit_transform(X_flat)
        
        # Reshape back to original dimensions
        X = X_flat_normalized.reshape(N, T, C).astype(np.float32)
        
        # Store normalization stats for reference
        self.channel_means = self.scaler.mean_
        self.channel_stds = self.scaler.scale_
        
        if self.is_main_process if hasattr(self, 'is_main_process') else True:
            log_info(f"Data normalized: mean range [{self.channel_means.min():.2f}, {self.channel_means.max():.2f}], "
                     f"std range [{self.channel_stds.min():.2f}, {self.channel_stds.max():.2f}]")
        # ========== END NORMALIZATION CODE ==========

        if self.balance:
            X, y = self._balance_dataset(X, y)

        unique_labels, counts = np.unique(y, return_counts=True)
        log_info(f"Total windows : {len(X)}")
        log_info(f"Feature shape : {X.shape}")
        log_info(f"Label dist.   : {dict(zip(unique_labels.tolist(), counts.tolist()))}")

        return X, y

    # ------------------------------------------------------------------
    # Balancing
    # ------------------------------------------------------------------

    def _balance_dataset(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Downsample the majority class to self.majority_n windows."""
        unique_labels, counts = np.unique(y, return_counts=True)
        log_info(
            f"Label distribution before balancing: "
            f"{dict(zip(unique_labels.tolist(), counts.tolist()))}"
        )

        majority_class = unique_labels[np.argmax(counts)]
        maj_mask = y == majority_class
        min_mask = ~maj_mask

        X_maj, y_maj = X[maj_mask], y[maj_mask]
        X_min, y_min = X[min_mask], y[min_mask]

        if len(X_maj) > self.majority_n:
            np.random.seed(42)
            idx   = np.random.choice(len(X_maj), self.majority_n, replace=False)
            X_maj = X_maj[idx]
            y_maj = y_maj[idx]

        X_out = np.concatenate([X_maj, X_min], axis=0)
        y_out = np.concatenate([y_maj, y_min], axis=0)

        shuf  = np.random.permutation(len(X_out))
        X_out = X_out[shuf]
        y_out = y_out[shuf]

        # Subject sequences are now meaningless after global shuffle
        self.subject_to_windows = {sid: [] for sid in self.subjects}
        log_info("Balancing applied; subject_to_windows invalidated.")

        return X_out, y_out

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (x [C, T], y) - channels-first, matching mHEALTH convention."""
        x = torch.from_numpy(self.X[idx])  # [T, C]
        x = x.transpose(0, 1)             # [C, T]
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

    def get_subject_sequence(
        self, subject_id: int
    ) -> List[Tuple[torch.Tensor, int]]:
        """
        Return all windows for one subject in temporal order.

        Used by SequentialDataset and the BPTT loop in agent_trainer.py.
        Raises RuntimeError if balance=True was used (order is destroyed).
        """
        if self.balance:
            raise RuntimeError(
                "get_subject_sequence is not supported when balance=True "
                "because global shuffling destroys temporal ordering."
            )
        if subject_id not in self.subject_to_windows:
            log_info(f"Subject {subject_id} has no data in this dataset split.")
            return []

        sequence = []
        for x_np, lbl in self.subject_to_windows[subject_id]:
            x = torch.from_numpy(x_np).float()  # [T, C]
            x = x.transpose(0, 1)               # [C, T]
            sequence.append((x, int(lbl)))
        return sequence


# ---------------------------------------------------------------------------
# Convenience dataloader factory
# ---------------------------------------------------------------------------

def get_kinematics_dataloaders(
    data_root: str,
    batch_size: int = 64,
    time_steps: int = 100,
    step: int = 50,
    balance: bool = False,
    train_subjects: Optional[List[int]] = None,
    val_subjects:   Optional[List[int]] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders for the kinematics dataset.

    Args:
        data_root:       Path to the Data_with_Kinematics root directory.
        batch_size:      DataLoader batch size.
        time_steps:      Sliding window length (samples at 100 Hz).
        step:            Sliding window stride.
        balance:         Downsample majority class in the training set.
        train_subjects:  Subject IDs for training   (default: HAB-15 to HAB-22).
        val_subjects:    Subject IDs for validation (default: HAB-23, HAB-24).

    Returns:
        train_loader, val_loader
    """
    from dataset.sequential_dataset import SequentialDataset, collate_sequential_batch

    if train_subjects is None:
        train_subjects = list(range(15, 23))   # HAB-15 to HAB-22
    if val_subjects is None:
        val_subjects = [23, 24]                # HAB-23, HAB-24

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
        balance=False,   # never balance the validation set
    )

    train_dataset = SequentialDataset(train_base, subject_ids=train_subjects)
    val_dataset   = SequentialDataset(val_base,   subject_ids=val_subjects)

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


# ---------------------------------------------------------------------------
# Smoke test   python kinematics_loader.py /path/to/Data_with_Kinematics
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    data_root = (
        sys.argv[1] if len(sys.argv) > 1
        else "/home/bkl46/CSE_MSE_RXF131/cradle-members/mds3/bkl46/"
             "multimodal-medical-agent/data/Data_with_Kinematics"
    )

    print("=" * 60)
    print("KinematicsDataset smoke test")
    print("=" * 60)

    # ---- Basic single-subject test ----
    print("\n[1] Single subject (HAB-15) ...")
    ds = KinematicsDataset(
        data_root=data_root,
        subjects=[15],
        time_steps=100,
        step=50,
        balance=False,
    )
    print(f"    Dataset size : {len(ds)}")
    if len(ds) > 0:
        x, y = ds[0]
        print(f"    x shape      : {x.shape}  (expected [39, 100])")
        print(f"    y            : {y.item()} = '{ds.inverse_label_map[y.item()]}'")
        print(f"    x[0] range   : [{x[0].min():.4f}, {x[0].max():.4f}]")
    print(f"    Label map    : {ds.label_map}")

    # ---- Verify column selection correctness ----
    print("\n[2] Verifying IMU column selection ...")
    print(f"    Selected raw cols : {IMU_SENSOR_COLS}")
    print(f"    Num IMU channels  : {NUM_IMU_CHANNELS}  (expected 30)")
    print(f"    Total channels    : {NUM_ALL_CHANNELS}  (expected 39)")

    # ---- Subject sequence (BPTT compatibility) ----
    print("\n[3] Subject sequence for BPTT ...")
    seq = ds.get_subject_sequence(15)
    print(f"    Sequence length : {len(seq)} windows")
    if seq:
        sx, sy = seq[0]
        print(f"    First window    : x={sx.shape}, y={sy}")

    # ---- HAB-17 zero-fill test ----
    print("\n[4] HAB-17 (missing sensors 4 & 5) ...")
    try:
        ds17 = KinematicsDataset(
            data_root=data_root,
            subjects=[17],
            time_steps=100,
            step=50,
        )
        print(f"    Dataset size : {len(ds17)}")
        if len(ds17) > 0:
            x17, _ = ds17[0]
            # Output channels 18-29 should be exactly 0 after zero-fill
            bad_ch_max = x17[18:30].abs().max().item()
            status = "OK (all zero)" if bad_ch_max == 0.0 else f"WARNING max={bad_ch_max:.4f}"
            print(f"    Sensor 4-5 output cols (18-29): {status}")
    except Exception as e:
        print(f"    HAB-17 not available or error: {e}")
