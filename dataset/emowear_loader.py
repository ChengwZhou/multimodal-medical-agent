"""
EmoWear Dataset Loader
======================
Zenodo: https://zenodo.org/records/10407279
Paper:  https://doi.org/10.1038/s41597-024-03429-3

Directory layout expected::

    {data_root}/
      {code}-{ID}/            e.g. 01-9TZK, 27-9VUW
        signals-e4-bvp.csv    timestamp [s], value [bits]        @ 64  Hz
        signals-e4-eda.csv    timestamp [s], value [µS]          @  4  Hz
        signals-e4-skt.csv    timestamp [s], value [°C]          @  4  Hz
        signals-e4-acc.csv    timestamp [s], x, y, z [mg]        @ 32  Hz
        signals-bh3-ecg.csv   timestamp [s], value [6.25 µV]     @250  Hz
        signals-bh3-rsp.csv   timestamp [s], value [bits]        @ 25  Hz
        signals-bh3-acc.csv   timestamp [s], x, y, z [mg]        @100  Hz
        signals-front-acc.csv timestamp [s], x1,y1,z1, x2,y2,z2, x3,y3,z3 [mg]
        signals-front-gyro.csv timestamp [s], x, y, z [mdps]    @208  Hz
        signals-back-acc.csv  (same columns as front-acc)
        signals-back-gyro.csv (same columns as front-gyro)
        markers-phase2.csv    seq, exp, preB, vidB, postB, ...   (times in [s])
        surveys.csv           seq, exp, valence, arousal, dominance, liking, familiarity

All timestamps are relative seconds from the synchronisation moment.

Target output
-------------
Each ``__getitem__`` returns ``(x [C, T] float32, y int64)`` where
``T = time_steps`` samples at ``FS = 64 Hz``.  Signals from different
devices are resampled to ``FS`` using linear interpolation on their
irregular timestamp grids.

Default channel layout (13 channels, configurable via *modalities* arg):

    ch 0        : ECG          (BH3)          device_idx = 0
    ch 1        : RSP          (BH3)          device_idx = 0
    ch 2, 3, 4  : BVP,EDA,SKT (E4)           device_idx = 1
    ch 5, 6, 7  : ACC x/y/z   (E4)           device_idx = 1
    ch 8, 9, 10 : ACC₃ x/y/z  (front STb)    device_idx = 2
    ch 11,12,13 : GYRO x/y/z  (front STb)    device_idx = 2

Label modes
-----------
* ``'valence'``  – binary  0 (low, SAM <  5) / 1 (high, SAM >= 5)  [default]
* ``'arousal'``  – binary  0 / 1  same threshold
* ``'quadrant'`` – 4-class: 0=LALV, 1=HALV, 2=LAHV, 3=HAHV
"""

import os
import re
import glob
import hashlib
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_info(msg: str):
    from torch.distributed import is_initialized, get_rank
    if not is_initialized() or get_rank() == 0:
        logger.info(msg)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FS = 64          # target sampling frequency (Hz)
TRIAL_DUR = 60   # video stimulus duration (seconds)

# Timestamp column name used in all signal CSV files
_TS = 'timestamp [s]'

# Native sampling rates (Hz)
_FS_BH3_ECG  = 250
_FS_BH3_RSP  = 25
_FS_BH3_ACC  = 100
_FS_E4_BVP   = 64
_FS_E4_EDA   = 4
_FS_E4_SKT   = 4
_FS_E4_ACC   = 32
_FS_STB_ACC  = 208   # LSM6DSOX (acc3 columns)
_FS_STB_GYRO = 208

# ACC files have 3 sub-sensors; we use acc3 (LSM6DSOX) columns
_STB_ACC3_COLS = ['x3', 'y3', 'z3']
_STB_GYRO_COLS = ['x', 'y', 'z']
_E4_ACC_COLS   = ['x', 'y', 'z']
_E4_SCALAR     = ['value']
_BH3_SCALAR    = ['value']
_BH3_ACC_COLS  = ['x', 'y', 'z']


# ---------------------------------------------------------------------------
# Signal config: name → (csv_filename, value_columns, native_fs, device_idx)
# ---------------------------------------------------------------------------

SIGNAL_CONFIG: Dict[str, Tuple[str, List[str], float, int]] = {
    'ecg':        ('signals-bh3-ecg.csv',   _BH3_SCALAR,   _FS_BH3_ECG,  0),
    'rsp':        ('signals-bh3-rsp.csv',   _BH3_SCALAR,   _FS_BH3_RSP,  0),
    'bvp':        ('signals-e4-bvp.csv',    _E4_SCALAR,    _FS_E4_BVP,   1),
    'eda':        ('signals-e4-eda.csv',     _E4_SCALAR,    _FS_E4_EDA,   1),
    'skt':        ('signals-e4-skt.csv',    _E4_SCALAR,    _FS_E4_SKT,   1),
    'acc_e4':     ('signals-e4-acc.csv',    _E4_ACC_COLS,  _FS_E4_ACC,   1),
    'acc_front':  ('signals-front-acc.csv', _STB_ACC3_COLS,_FS_STB_ACC,  2),
    'gyro_front': ('signals-front-gyro.csv',_STB_GYRO_COLS,_FS_STB_GYRO, 2),
    'acc_back':   ('signals-back-acc.csv',  _STB_ACC3_COLS,_FS_STB_ACC,  3),
    'gyro_back':  ('signals-back-gyro.csv', _STB_GYRO_COLS,_FS_STB_GYRO, 3),
}

DEFAULT_MODALITIES = ('ecg', 'rsp', 'bvp', 'eda', 'skt', 'acc_e4',
                      'acc_front', 'gyro_front')


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _resample_to_grid(ts: np.ndarray, vals: np.ndarray,
                      t_start: float, t_end: float) -> np.ndarray:
    """Interpolate irregularly-sampled signal onto a uniform 64-Hz grid.

    Args:
        ts:      raw timestamp array (seconds, relative)
        vals:    raw signal array  [N, C]  or [N]
        t_start: window start time (seconds)
        t_end:   window end time (seconds)

    Returns:
        ndarray [n_samples, C]  where n_samples = round((t_end-t_start)*FS)
    """
    n = max(1, round((t_end - t_start) * FS))
    t_grid = np.linspace(t_start, t_end, n, endpoint=False)
    if vals.ndim == 1:
        vals = vals[:, None]
    out = np.zeros((n, vals.shape[1]), dtype=np.float32)
    for c in range(vals.shape[1]):
        out[:, c] = np.interp(t_grid, ts, vals[:, c].astype(float))
    return out


def _read_signal_raw(csv_path: str, value_cols: List[str]
                     ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Read a signal CSV once, return (ts [N], vals [N, C]) without slicing.

    Used by the caching path in ``_load_subject`` to avoid re-reading the
    same file for every trial.  Returns ``None`` on any read / parse failure.
    """
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    # Flexible timestamp column matching
    ts_col = None
    for c in df.columns:
        if 'timestamp' in c.lower():
            ts_col = c
            break
    if ts_col is None:
        return None

    # Flexible value column matching
    resolved = []
    for vc in value_cols:
        if vc in df.columns:
            resolved.append(vc)
        else:
            matches = [c for c in df.columns if c.lower().startswith(vc.lower())]
            if matches:
                resolved.append(matches[0])
            else:
                return None  # required column absent

    ts   = df[ts_col].to_numpy(dtype=np.float64)
    vals = df[resolved].to_numpy(dtype=np.float32)
    return ts, vals


def _read_signal(csv_path: str, value_cols: List[str],
                 t_start: float, t_end: float) -> Optional[np.ndarray]:
    """Read one signal CSV and interpolate to the target 64-Hz grid.

    Returns [n_samples, len(value_cols)] float32, or None on failure.
    """
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    # Flexible timestamp column matching (handles minor naming variations)
    ts_col = None
    for c in df.columns:
        if 'timestamp' in c.lower():
            ts_col = c
            break
    if ts_col is None:
        return None

    # Flexible value column matching
    resolved = []
    for vc in value_cols:
        if vc in df.columns:
            resolved.append(vc)
        else:
            # Try case-insensitive / partial match
            matches = [c for c in df.columns if c.lower().startswith(vc.lower())]
            if matches:
                resolved.append(matches[0])
            else:
                return None

    ts   = df[ts_col].to_numpy(dtype=np.float64)
    vals = df[resolved].to_numpy(dtype=np.float32)

    # Trim to window + small margin to avoid edge extrapolation artefacts
    margin = 1.0
    mask = (ts >= t_start - margin) & (ts <= t_end + margin)
    if mask.sum() < 2:
        return None
    return _resample_to_grid(ts[mask], vals[mask], t_start, t_end)


# ---------------------------------------------------------------------------
# Main dataset class
# ---------------------------------------------------------------------------

class EmoWearDataset(Dataset):
    """PyTorch Dataset for EmoWear (Zenodo 10407279).

    Each item is a sliding-window excerpt from one video trial:
    ``(x [C, T] float32, label int64)``.

    Args:
        data_root:   Root directory containing per-participant sub-folders.
        subjects:    List of subject folder names, e.g. ``['01-9TZK', '02-...']``.
                     Pass ``None`` to auto-discover all folders.
        time_steps:  Window length in samples at 64 Hz.  Default 384 = 6 s.
        step:        Sliding-window step.  Default 64 = 1 s.
        modalities:  Subset of ``SIGNAL_CONFIG`` keys to include.
        label_mode:  ``'valence'``, ``'arousal'``, or ``'quadrant'``.
        threshold:   SAM mid-point for binary split (default 5.0 on 1-9 scale).
        balance:     Downsample majority class to ``balance_ratio × minority``.
        balance_ratio: Used only when ``balance=True``.
        global_stats:  Optional ``{'mean': [C], 'std': [C]}`` for z-score.
                       If ``None``, per-window z-score is applied.
        cache_dir:     Directory for per-subject pickle caches.  Defaults to
                       ``{data_root}/.emowear_cache``.  Pass ``False``/an
                       empty string to disable caching (not recommended).
                       Cache filenames embed a hash of all preprocessing
                       parameters, so changing any param auto-invalidates.
    """

    FS: int = FS

    def __init__(
        self,
        data_root: str,
        subjects: Optional[List[str]] = None,
        time_steps: int = 384,           # 6 s × 64 Hz
        step: int = 64,                  # 1 s
        modalities: Tuple[str, ...] = DEFAULT_MODALITIES,
        label_mode: str = 'valence',     # 'valence' | 'arousal' | 'quadrant'
        threshold: float = 5.0,
        balance: bool = False,
        balance_ratio: int = 3,
        global_stats: Optional[Dict[str, np.ndarray]] = None,
        cache_dir: Optional[str] = None,  # None → {data_root}/.emowear_cache
    ):
        self.data_root   = data_root
        self.modalities  = list(modalities)
        self.time_steps  = time_steps
        self.step        = step
        self.label_mode  = label_mode
        self.threshold   = threshold
        self.balance     = balance
        self.balance_ratio = balance_ratio
        self.global_stats  = global_stats
        self.cache_dir   = cache_dir if cache_dir is not None \
                           else os.path.join(data_root, '.emowear_cache')

        # Discover subjects
        if subjects is None:
            subjects = self._discover_subjects()
        self.subjects = sorted(subjects)

        # subject_to_windows: required by SequentialDataset
        self.subject_to_windows: Dict[str, List] = {}

        # Build flat index
        self._windows: List[Tuple[np.ndarray, int]] = []   # (x [C,T], label)
        self._build_index()

        if balance:
            self._apply_balance()

        counts = defaultdict(int)
        for _, lbl in self._windows:
            counts[lbl] += 1
        log_info(f"EmoWearDataset: {len(self._windows)} windows, "
                 f"distribution={dict(sorted(counts.items()))}")

    # ------------------------------------------------------------------
    # Subject discovery
    # ------------------------------------------------------------------

    def _discover_subjects(self) -> List[str]:
        """Return all ``NN-XXXX`` sub-folders found in data_root."""
        pattern = re.compile(r'^\d{2}-[A-Z0-9]+$')
        found = [d for d in os.listdir(self.data_root)
                 if os.path.isdir(os.path.join(self.data_root, d))
                 and pattern.match(d)]
        if not found:
            raise ValueError(f"No subject folders found in {self.data_root}. "
                             "Expected folders named like '01-9TZK'.")
        return sorted(found)

    # ------------------------------------------------------------------
    # Cache key
    # ------------------------------------------------------------------

    def _cache_key(self) -> str:
        """Short hash that encodes all preprocessing parameters.

        A change to any of time_steps / step / modalities / label_mode /
        threshold will produce a different key, ensuring stale caches are
        never re-used.
        """
        sig = (
            self.time_steps,
            self.step,
            tuple(sorted(self.modalities)),
            self.label_mode,
            round(self.threshold, 6),
        )
        h = hashlib.md5(str(sig).encode()).hexdigest()[:10]
        return h

    # ------------------------------------------------------------------
    # Index build
    # ------------------------------------------------------------------

    def _build_index(self):
        """Load all trials for all subjects, apply sliding window, build index.

        Per-subject windows are persisted to ``{cache_dir}/{subj}_{key}.pkl``
        so that subsequent runs skip all CSV parsing and signal resampling.
        The cache is keyed by preprocessing parameters (time_steps, step,
        modalities, label_mode, threshold); changing any parameter
        automatically invalidates the cache.
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        key = self._cache_key()

        for subj in self.subjects:
            subj_dir = os.path.join(self.data_root, subj)
            if not os.path.isdir(subj_dir):
                log_info(f"  {subj}: folder not found, skipping")
                continue

            cache_path = os.path.join(self.cache_dir, f"{subj}_{key}.pkl")
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        wins = pickle.load(f)
                    log_info(f"  {subj}: {len(wins)} windows (from cache)")
                except Exception as e:
                    log_info(f"  {subj}: cache load failed ({e}), reprocessing")
                    wins = None
            else:
                wins = None

            if wins is None:
                wins = self._load_subject(subj, subj_dir)
                try:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(wins, f, protocol=pickle.HIGHEST_PROTOCOL)
                    log_info(f"  {subj}: {len(wins)} windows (cached → {cache_path})")
                except Exception as e:
                    log_info(f"  {subj}: {len(wins)} windows (cache write failed: {e})")

            self.subject_to_windows[subj] = wins
            self._windows.extend(wins)

    def _load_subject(self, subj: str, subj_dir: str
                      ) -> List[Tuple[np.ndarray, int]]:
        """Load markers + surveys + signals for one subject.

        Speed: read each signal CSV exactly once per subject, cache as raw
        numpy arrays, then slice per-trial.  Avoids the original pattern of
        38 trials × 8 modalities = 304 redundant full-file CSV parses.
        """
        # --- Markers ---
        mk_path = os.path.join(subj_dir, 'markers-phase2.csv')
        if not os.path.exists(mk_path):
            log_info(f"  {subj}: markers-phase2.csv missing, skipping")
            return []
        try:
            markers = pd.read_csv(mk_path)
        except Exception as e:
            log_info(f"  {subj}: markers read error ({e}), skipping")
            return []

        # --- Surveys ---
        sv_path = os.path.join(subj_dir, 'surveys.csv')
        if not os.path.exists(sv_path):
            log_info(f"  {subj}: surveys.csv missing, skipping")
            return []
        try:
            surveys = pd.read_csv(sv_path)
        except Exception as e:
            log_info(f"  {subj}: surveys read error ({e}), skipping")
            return []

        # Merge markers + surveys on 'seq'
        if 'seq' not in markers.columns or 'seq' not in surveys.columns:
            log_info(f"  {subj}: missing 'seq' column in markers or surveys, "
                     f"markers cols={list(markers.columns)}, surveys cols={list(surveys.columns)}")
            return []
        merged = markers.merge(surveys, on='seq', how='inner')
        if merged.empty:
            log_info(f"  {subj}: markers/surveys merge is empty (check 'seq' values match)")
            return []

        # vidB column check
        if 'vidB' not in merged.columns:
            log_info(f"  {subj}: 'vidB' column missing from markers, "
                     f"available={list(merged.columns)}")
            return []

        # ------------------------------------------------------------------
        # Pre-load every signal CSV once and cache as (ts [N], vals [N,C]).
        # Original _read_signal re-opened the file for each of the 38 trials
        # (304 full reads per subject).  Now we read each CSV once.
        # ------------------------------------------------------------------
        sig_cache: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]] = {}
        for mod in self.modalities:
            csv_file, val_cols, _, _ = SIGNAL_CONFIG[mod]
            csv_path = os.path.join(subj_dir, csv_file)
            sig_cache[mod] = _read_signal_raw(csv_path, val_cols)

        # Diagnose missing modalities once (not once per trial)
        missing = [m for m in self.modalities if sig_cache[m] is None]
        if missing:
            log_info(f"  {subj}: signal(s) missing or unreadable: {missing} — all trials skipped")
            return []

        windows = []
        n_skip_label = 0
        n_skip_signal = 0
        for _, row in merged.iterrows():
            if pd.isna(row['vidB']):
                continue
            t_start = float(row['vidB'])
            t_end   = t_start + TRIAL_DUR

            label = self._make_label(row)
            if label is None:
                n_skip_label += 1
                continue

            # Slice pre-loaded signals for this trial window
            channels = self._slice_signals(sig_cache, t_start, t_end)
            if channels is None:
                n_skip_signal += 1
                continue

            T_total = channels.shape[0]
            for s in range(0, T_total - self.time_steps + 1, self.step):
                seg = channels[s : s + self.time_steps, :]
                x   = seg.T.astype(np.float32)
                x   = self._normalize(x)
                windows.append((x, label))

        if not windows:
            log_info(f"  {subj}: 0 windows — "
                     f"trials={len(merged)}, skipped(label)={n_skip_label}, "
                     f"skipped(signal)={n_skip_signal}")
        return windows

    def _make_label(self, row) -> Optional[int]:
        """Convert SAM ratings to integer label per label_mode."""
        try:
            val = float(row['valence'])
            aro = float(row['arousal'])
        except (KeyError, ValueError, TypeError):
            return None

        if self.label_mode == 'valence':
            return int(val >= self.threshold)
        elif self.label_mode == 'arousal':
            return int(aro >= self.threshold)
        elif self.label_mode == 'quadrant':
            hi_v = val >= self.threshold
            hi_a = aro >= self.threshold
            # 0=LALV, 1=HALV, 2=LAHV, 3=HAHV
            return int(hi_a) * 2 + int(hi_v)
        else:
            raise ValueError(f"Unknown label_mode: {self.label_mode}")

    def _load_signals(self, subj_dir: str,
                      t_start: float, t_end: float) -> Optional[np.ndarray]:
        """Read all selected modalities and stack into [T_target, C] float32."""
        n_target = round(TRIAL_DUR * FS)
        channel_arrays = []
        for mod in self.modalities:
            if mod not in SIGNAL_CONFIG:
                raise ValueError(f"Unknown modality '{mod}'. "
                                 f"Choose from {list(SIGNAL_CONFIG.keys())}")
            csv_file, val_cols, _, _ = SIGNAL_CONFIG[mod]
            csv_path = os.path.join(subj_dir, csv_file)
            arr = _read_signal(csv_path, val_cols, t_start, t_end)  # [T, C_mod]
            if arr is None:
                return None   # required modality missing for this trial
            # Trim / pad to exactly n_target samples
            if arr.shape[0] >= n_target:
                arr = arr[:n_target]
            else:
                pad = np.zeros((n_target - arr.shape[0], arr.shape[1]),
                               dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=0)
            channel_arrays.append(arr)

        return np.concatenate(channel_arrays, axis=1)  # [T, C_total]

    def _slice_signals(self, sig_cache: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]],
                       t_start: float, t_end: float) -> Optional[np.ndarray]:
        """Slice pre-cached signal arrays to a trial window and stack into [T, C].

        Args:
            sig_cache: ``{modality: (ts [N], vals [N,C])}`` built once per subject.
            t_start:   Trial window start (seconds).
            t_end:     Trial window end (seconds).

        Returns:
            ``[T_target, C_total] float32`` or ``None`` if any modality is missing
            / has insufficient coverage.
        """
        n_target = round(TRIAL_DUR * FS)
        margin = 1.0
        channel_arrays = []
        for mod in self.modalities:
            cached = sig_cache[mod]
            if cached is None:
                return None
            ts, vals = cached
            # Trim to window + small margin to avoid edge extrapolation artefacts
            mask = (ts >= t_start - margin) & (ts <= t_end + margin)
            if mask.sum() < 2:
                return None
            arr = _resample_to_grid(ts[mask], vals[mask], t_start, t_end)  # [T, C_mod]
            # Trim / pad to exactly n_target samples
            if arr.shape[0] >= n_target:
                arr = arr[:n_target]
            else:
                pad = np.zeros((n_target - arr.shape[0], arr.shape[1]), dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=0)
            channel_arrays.append(arr)
        return np.concatenate(channel_arrays, axis=1)  # [T, C_total]

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Channel-wise z-score.  x: [C, T]."""
        if self.global_stats is not None:
            mean = self.global_stats['mean'].reshape(-1, 1)
            std  = self.global_stats['std'].reshape(-1, 1) + 1e-8
        else:
            mean = x.mean(axis=1, keepdims=True)
            std  = x.std(axis=1, keepdims=True) + 1e-8
        return (x - mean) / std

    # ------------------------------------------------------------------
    # Balancing
    # ------------------------------------------------------------------

    def _apply_balance(self):
        labels  = np.array([lbl for _, lbl in self._windows])
        classes, counts = np.unique(labels, return_counts=True)
        min_cls = classes[np.argmin(counts)]
        n_min   = counts.min()
        n_max   = n_min * self.balance_ratio
        log_info(f"Balancing: minority class {min_cls} has {n_min} samples; "
                 f"capping majority at {n_max}")
        kept = []
        np.random.seed(42)
        by_class = defaultdict(list)
        for item in self._windows:
            by_class[item[1]].append(item)
        for cls, items in by_class.items():
            if len(items) > n_max:
                idxs  = np.random.choice(len(items), n_max, replace=False)
                items = [items[i] for i in idxs]
            kept.extend(items)
        np.random.shuffle(kept)
        self._windows = kept

    # ------------------------------------------------------------------
    # SequentialDataset compatibility
    # ------------------------------------------------------------------

    @property
    def max_length(self) -> int:
        """Longest subject sequence length (for SequentialDataset)."""
        if not self.subject_to_windows:
            return 1
        return max(len(v) for v in self.subject_to_windows.values())

    def get_subject_sequence(self, subject_id: str
                             ) -> List[Tuple[torch.Tensor, int]]:
        wins = self.subject_to_windows.get(subject_id, [])
        return [(torch.from_numpy(x).float(), lbl) for x, lbl in wins]

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, lbl = self._windows[idx]
        return torch.from_numpy(x), torch.tensor(lbl, dtype=torch.long)


# ---------------------------------------------------------------------------
# Modality → ModalityConfig helper
# ---------------------------------------------------------------------------

def emowear_modality_configs(modalities: Tuple[str, ...], patch_size: int = 8):
    """Return a list of ModalityConfig objects matching the chosen modalities.

    Import ModalityConfig from utils.modality_config before calling this.
    """
    from utils.modality_config import ModalityConfig
    configs = []
    for mod in modalities:
        _, val_cols, _, dev_idx = SIGNAL_CONFIG[mod]
        configs.append(ModalityConfig(mod, len(val_cols), patch_size, dev_idx))
    return configs


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    data_root: str,
    train_subjects: Optional[List[str]] = None,
    val_subjects:   Optional[List[str]] = None,
    batch_size: int = 64,
    time_steps: int = 384,
    step: int = 64,
    modalities: Tuple[str, ...] = DEFAULT_MODALITIES,
    label_mode: str = 'valence',
    balance: bool = False,
    num_workers: int = 4,
):
    from torch.utils.data import DataLoader
    from dataset.sequential_dataset import SequentialDataset, collate_sequential_batch

    if train_subjects is None or val_subjects is None:
        all_subs = EmoWearDataset(data_root)._discover_subjects()
        np.random.seed(42)
        perm   = np.random.permutation(len(all_subs))
        split  = int(len(all_subs) * 0.8)
        train_subjects = [all_subs[i] for i in perm[:split]]
        val_subjects   = [all_subs[i] for i in perm[split:]]

    train_ds = EmoWearDataset(data_root, train_subjects, time_steps, step,
                              modalities, label_mode, balance=balance)
    val_ds   = EmoWearDataset(data_root, val_subjects,   time_steps, step,
                              modalities, label_mode, balance=False)

    train_seq = SequentialDataset(train_ds, subject_ids=train_subjects)
    val_seq   = SequentialDataset(val_ds,   subject_ids=val_subjects)

    train_loader = DataLoader(train_seq, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers,
                              collate_fn=collate_sequential_batch)
    val_loader   = DataLoader(val_seq,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers,
                              collate_fn=collate_sequential_batch)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    data_root = sys.argv[1] if len(sys.argv) > 1 else './emowear_data'

    print('Discovering subjects ...')
    ds = EmoWearDataset(
        data_root=data_root,
        time_steps=384,   # 6 s
        step=192,         # 3 s hop
        modalities=('ecg', 'bvp', 'eda'),
        label_mode='valence',
    )
    print(f'Total windows: {len(ds)}')
    x, y = ds[0]
    print(f'  x.shape={x.shape}, y={y}')
    print(f'  x range: [{x.min():.3f}, {x.max():.3f}]')
