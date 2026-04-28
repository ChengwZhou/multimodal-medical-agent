import os
import sys
import glob
import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.delta_dataset import DeltaDataset
from dataset.sequential_dataset import SequentialDataset, collate_sequential_batch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_info(msg):
    from torch.distributed import is_initialized, get_rank
    if not is_initialized() or get_rank() == 0:
        logger.info(msg)


# ---------------------------------------------------------------------------
# Per-worker EDF file-handle LRU cache
# ---------------------------------------------------------------------------
# Each DataLoader worker is a separate process (fork/spawn), so this dict is
# private to each worker.  We keep at most _EDF_CACHE_SIZE EDF handles open
# simultaneously.  When the cache is full, the least-recently-used entry is
# closed before the new file is opened, bounding OS file-descriptor usage to
#   num_workers × _EDF_CACHE_SIZE   per GPU rank.
# With 4 workers and size=8: 32 fds per rank — well within the OS default
# of 1024 open files.
# ---------------------------------------------------------------------------
from collections import OrderedDict

_EDF_CACHE_SIZE: int = 8          # max open EDF handles per worker process
_worker_edf_cache: "OrderedDict[str, Any]" = OrderedDict()


def _get_edf_reader(path: str) -> "Any":
    """Return a cached EdfReader for *path*, opening it on first access.

    Implements an LRU eviction policy: when the cache is at capacity, the
    least-recently-used handle is closed and removed before the new file
    is opened.
    """
    if path in _worker_edf_cache:
        # Move to end (most-recently-used)
        _worker_edf_cache.move_to_end(path)
        return _worker_edf_cache[path]

    try:
        import pyedflib
    except ImportError:
        raise ImportError("pyedflib is required: pip install pyedflib")

    # Evict LRU entry if at capacity
    while len(_worker_edf_cache) >= _EDF_CACHE_SIZE:
        _, evicted = _worker_edf_cache.popitem(last=False)  # oldest
        try:
            evicted._close()
        except Exception:
            pass

    reader = pyedflib.EdfReader(path)
    _worker_edf_cache[path] = reader
    return reader


# -----------------------------------------------------------------------
# Channel classification helpers
# -----------------------------------------------------------------------

_ECG_KW  = ['ecg', 'ekg', 'cardiac', 'heart']
_EMG_KW  = ['emg', 'muscle']
_ACC_KW  = ['acc', 'accel']
_GYRO_KW = ['gyro', 'gyr', 'angular']

_SEIZURE_KW = [
    'sz', 'seizure', 'seiz', 'ictal',
    'focal aware', 'focal impaired', 'tonic-clonic', 'tonic clonic',
    'fbtc', 'fa', 'fia',
]


def _classify_channel(ch_name: str) -> str:
    """Map an EDF channel label to one of: 'eeg', 'ecg', 'emg', 'acc', 'gyro'."""
    v = ch_name.lower().strip()
    for kw in _ECG_KW:
        if kw in v:
            return 'ecg'
    for kw in _EMG_KW:
        if kw in v:
            return 'emg'
    for kw in _GYRO_KW:   # check before acc to avoid 'gyro' matching 'acc' substring
        if kw in v:
            return 'gyro'
    for kw in _ACC_KW:
        if kw in v:
            return 'acc'
    return 'eeg'


# -----------------------------------------------------------------------
# Lightweight index structures (no signal data)
# -----------------------------------------------------------------------

@dataclass
class _RecordingMeta:
    """Header-only metadata for one recording run.  No signal arrays stored.

    A run may span multiple EDF files (one per modality).
    ``sel_ch_edfs[i]`` gives the EDF file that contains ``sel_ch_indices[i]``.
    When all channels come from the same file (legacy), ``sel_ch_edfs`` is empty
    and ``edf_path`` is used for all channels.
    """
    subject_id:     str
    edf_path:       str             # primary EDF (EEG); also used for events TSV
    sel_ch_indices: List[int]       # per-channel index within its source EDF
    sel_ch_fs:      List[float]     # per-channel native sampling rate
    n_target:       int             # recording length in target-FS samples
    seizure_ivs:    List[Tuple[int, int]] = field(default_factory=list)
    sel_ch_edfs:    List[str]       = field(default_factory=list)  # per-channel EDF path


@dataclass
class _WindowRecord:
    """One sliding-window entry in the flat index."""
    rec_idx:      int   # index into SeizeIT2Dataset._metas
    start_sample: int   # window start in target-FS samples
    label:        int   # 0 = background, 1 = seizure


# -----------------------------------------------------------------------
# Main dataset class — lazy / index-only init
# -----------------------------------------------------------------------

class SeizeIT2Dataset(Dataset):
    """
    Lazy loader for SeizeIT2 (OpenNeuro ds005873) – wearable multimodal
    seizure detection in patients with focal epilepsy.

    **Lazy loading strategy**
    -------------------------
    ``__init__`` only reads EDF *headers* (channel names, sampling rates,
    total sample counts) and ``_events.tsv`` files.  No signal data is
    loaded at construction time, making init near-instantaneous even for
    the full 125-subject / ~11 000 h dataset.

    Signal data is read on demand:

    * ``__getitem__`` — opens the EDF, seeks to the window start with
      ``pyedflib.fseek``, reads exactly ``time_steps`` samples per channel,
      then closes the file.  Suitable for ``DataLoader`` with random access.

    * ``get_subject_sequence`` — opens each EDF *once* and reads all of
      that subject's windows sequentially.  Suitable for
      ``SequentialDataset``.

    BIDS layout expected::

        {data_root}/
          sub-001/ses-01/eeg/sub-001_ses-01_..._eeg.edf
          sub-001/ses-01/eeg/sub-001_ses-01_..._events.tsv
          ...

    Modalities:
      - **eeg**  – behind-the-ear EEG       (250 Hz, ≥2 ch)
      - **ecg**  – electrocardiography      (250 Hz, 1 ch)
      - **emg**  – electromyography         (250 Hz, 1 ch)
      - **imu**  – accelerometer + gyroscope (25 Hz → resampled to 250 Hz, 6 ch)

    Binary label: 0 = background, 1 = seizure.
    A window is labelled 1 when ``seizure_fraction ≥ seizure_boundary``.

    Fully compatible with :class:`DeltaDataset` and :class:`SequentialDataset`.
    Each item is returned as a ``[C, T]`` float32 tensor.
    """

    FS: int     = 250   # target sampling frequency (Hz)
    FS_IMU: int = 25    # original IMU sampling frequency (Hz)

    def __init__(
        self,
        data_root: str,
        subjects: List[str],
        time_steps: int = 500,           # 2 s × 250 Hz
        step: int = 250,                 # 50 % overlap
        modalities: Tuple[str, ...] = ('eeg', 'ecg', 'emg', 'imu'),
        balance: bool = False,
        balance_ratio: int = 5,          # max bg : seizure when balance=True
        seizure_boundary: float = 0.5,
        global_stats: Optional[Dict[str, np.ndarray]] = None,
        cache_dir: Optional[str] = None,  # local dir for index cache (avoids NFS re-scan)
    ):
        """
        Args:
            data_root:        Root BIDS directory (contains ``sub-XXX/`` folders).
            subjects:         Subject IDs, e.g. ``['sub-001', 'sub-002']``.
            time_steps:       Window length in samples at ``FS`` = 250 Hz.
            step:             Sliding-window step in samples.
            modalities:       Which channel types to include (subset of
                              ``('eeg', 'ecg', 'emg', 'imu')``).
            balance:          Downsample background windows so that
                              ``n_bg / n_seiz ≤ balance_ratio``.
            balance_ratio:    Used only when ``balance=True``.
            seizure_boundary: Fraction threshold for window-level seizure label.
            global_stats:     Optional ``{'mean': ndarray[C], 'std': ndarray[C]}``
                              for channel-wise z-score normalisation.
                              If ``None``, per-window z-score is applied.
        """
        self.data_root        = data_root
        self.subjects         = sorted(subjects)
        self.time_steps       = time_steps
        self.step             = step
        self.modalities       = list(modalities)
        self.balance          = balance
        self.balance_ratio    = balance_ratio
        self.seizure_boundary = seizure_boundary
        self.global_stats     = global_stats
        self.cache_dir        = cache_dir

        # Internal index structures — built by _build_index(), no signal data
        self._metas: List[_RecordingMeta] = []
        self._index: List[_WindowRecord]  = []

        # subject_id → [window_record, ...]  (used by get_subject_sequence)
        # Kept separate from _index so that balance does not invalidate it.
        self._subj_wins: Dict[str, List[_WindowRecord]] = {}

        # subject_to_windows is required by SequentialDataset (.keys()) and
        # DeltaDataset (__getattr__ delegation).  Values are empty lists;
        # only the keys matter for SequentialDataset's subject discovery.
        self.subject_to_windows: Dict[str, List] = {}

        # --- Fast init: scan headers + TSV only (no signal reads) ---
        self._build_index()

        if balance:
            self._apply_balance()

        n_seiz = sum(1 for r in self._index if r.label == 1)
        n_bg   = len(self._index) - n_seiz
        log_info(f"SeizeIT2 index ready: {len(self._index)} windows "
                 f"(seiz={n_seiz}, bg={n_bg}) across {len(self._metas)} recordings")

    # ------------------------------------------------------------------
    # File discovery  (unchanged from eager version)
    # ------------------------------------------------------------------

    def _find_recording_groups(self, subject_id: str):
        """Yield one group per recording run for *subject_id*.

        SeizeIT2 stores each modality in a separate sub-directory and EDF file::

            ses-01/eeg/*_eeg.edf   ← primary (EEG channels + events TSV)
            ses-01/ecg/*_ecg.edf   ← ECG channels
            ses-01/mov/*_mov.edf   ← motion channels (acc + gyro)
            ses-01/emg/*_emg.edf   ← EMG channels (optional)

        Yields dicts with keys:
            ``eeg_edf``    – path to the EEG EDF (required)
            ``events_tsv`` – path to the events TSV, or None
            ``modal_edfs`` – {modality: edf_path} for non-EEG modalities found
        """
        ses_roots = [
            os.path.join(self.data_root, subject_id, 'ses-01'),
            os.path.join(self.data_root, subject_id),
        ]
        for ses_root in ses_roots:
            eeg_dir = os.path.join(ses_root, 'eeg')
            if not os.path.isdir(eeg_dir):
                continue
            for eeg_edf in sorted(glob.glob(os.path.join(eeg_dir, '*_eeg.edf'))):
                base_fname   = os.path.basename(eeg_edf)[:-len('_eeg.edf')]
                events_tsv   = os.path.join(eeg_dir, base_fname + '_events.tsv')
                modal_edfs: Dict[str, str] = {}
                for mod in ('ecg', 'emg', 'mov'):
                    p = os.path.join(ses_root, mod, base_fname + f'_{mod}.edf')
                    if os.path.exists(p):
                        modal_edfs[mod] = p
                yield {
                    'eeg_edf':    eeg_edf,
                    'events_tsv': events_tsv if os.path.exists(events_tsv) else None,
                    'modal_edfs': modal_edfs,
                }
            break  # found ses-01, skip fallback roots

    @staticmethod
    def _read_channels_tsv(channels_tsv: str) -> List[str]:
        """Parse a BIDS *_channels.tsv and return an ordered list of channel types.

        The list is in the same row-order as the EDF channels, so element i gives
        the type of EDF channel i.  Types are lower-cased ('eeg', 'ecg', 'mov', …).
        Returns an empty list on any failure.
        """
        try:
            df = pd.read_csv(channels_tsv, sep='\t')
        except Exception:
            return []
        if 'type' not in df.columns:
            return []
        return [str(t).lower() for t in df['type']]

    # ------------------------------------------------------------------
    # Header-only EDF scan  (fast — no signal data read)
    # ------------------------------------------------------------------

    @staticmethod
    def _read_edf_header(edf_path: str):
        """Read EDF header only.

        Returns:
            ch_names  – list of channel label strings
            ch_fs     – list of per-channel sampling frequencies
            n_samples – list of total sample counts per channel
        """
        try:
            import pyedflib
        except ImportError:
            raise ImportError("pyedflib is required: pip install pyedflib")

        f         = pyedflib.EdfReader(edf_path)
        ch_names  = list(f.getSignalLabels())
        ch_fs     = [f.getSampleFrequency(i) for i in range(f.signals_in_file)]
        n_samples = list(f.getNSamples())
        f._close()
        del f
        return ch_names, ch_fs, n_samples

    # ------------------------------------------------------------------
    # Seizure-interval parsing  (intervals, not a dense mask)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_tsv_intervals(tsv_path: str, n_target: int) -> List[Tuple[int, int]]:
        """Parse ``*_events.tsv`` → list of ``(start_sample, end_sample)`` pairs.

        Interval endpoints are clipped to ``[0, n_target]`` and expressed in
        target-FS (250 Hz) samples.  Storing intervals instead of a dense
        binary mask avoids allocating arrays of hundreds of MB for long
        recordings.
        """
        intervals: List[Tuple[int, int]] = []
        try:
            df = pd.read_csv(tsv_path, sep='\t')
        except Exception:
            return intervals

        if 'onset' not in df.columns or 'duration' not in df.columns:
            return intervals

        if 'trial_type' in df.columns:
            def _is_seiz(val):
                if pd.isna(val):
                    return False
                v = str(val).lower()
                return any(kw in v for kw in _SEIZURE_KW)
            seiz_df = df[df['trial_type'].apply(_is_seiz)]
        else:
            seiz_df = df

        for _, row in seiz_df.iterrows():
            try:
                onset = float(row['onset'])
                dur   = float(row['duration']) if not pd.isna(row['duration']) else 0.0
            except (ValueError, TypeError):
                continue
            s = max(0, int(onset * SeizeIT2Dataset.FS))
            e = min(n_target, int((onset + dur) * SeizeIT2Dataset.FS))
            if e > s:
                intervals.append((s, e))

        return intervals

    # ------------------------------------------------------------------
    # Window label from interval list  (O(n_seizures), negligible)
    # ------------------------------------------------------------------

    def _compute_label(self, meta: _RecordingMeta, start: int) -> int:
        end = start + self.time_steps
        seiz_samples = sum(
            max(0, min(end, iv_e) - max(start, iv_s))
            for iv_s, iv_e in meta.seizure_ivs
        )
        return 1 if seiz_samples / self.time_steps >= self.seizure_boundary else 0

    # ------------------------------------------------------------------
    # Channel selection  (index-level, no data)
    # ------------------------------------------------------------------

    def _select_channel_indices(
        self,
        ch_names: List[str],
        ch_fs: List[float],
        ch_types: Optional[List[str]] = None,
    ) -> Tuple[List[int], List[float]]:
        """Return ``(sel_indices, sel_fs)`` in the desired modality order.

        Classification priority:
        1. BIDS ``*_channels.tsv`` ordered type list (positional, reliable).
        2. Keyword heuristic on the EDF channel label (fallback when no TSV).

        Supported modality strings: 'eeg', 'ecg', 'emg', 'imu', 'mov'.
        'imu' = acc + gyro combined.
        'mov' = motion channels declared as type 'mov' in channels.tsv
                (SeizeIT2 uses this for accelerometer + gyroscope).
        """
        buckets: Dict[str, List[Tuple[int, float]]] = {
            'eeg': [], 'ecg': [], 'emg': [], 'acc': [], 'gyro': [], 'mov': []
        }
        for i, (name, fs) in enumerate(zip(ch_names, ch_fs)):
            # Prefer positional type from channels.tsv; fall back to keyword
            if ch_types and i < len(ch_types):
                ctype = ch_types[i]
            else:
                ctype = _classify_channel(name)
            if ctype in buckets:
                buckets[ctype].append((i, fs))

        sel_idx: List[int]   = []
        sel_fs:  List[float] = []
        for mod in self.modalities:
            if mod == 'imu':
                for idx, fs in buckets['acc'] + buckets['gyro']:
                    sel_idx.append(idx)
                    sel_fs.append(fs)
            elif mod in buckets:
                for idx, fs in buckets[mod]:
                    sel_idx.append(idx)
                    sel_fs.append(fs)
        return sel_idx, sel_fs

    # ------------------------------------------------------------------
    # Index cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self) -> Optional[str]:
        """Return a deterministic cache file path, or None if cache_dir is unset."""
        if not self.cache_dir:
            return None
        import hashlib
        key = hashlib.md5(
            (
                self.data_root
                + str(self.subjects)
                + str(self.modalities)
                + str(self.time_steps)
                + str(self.step)
                + str(self.seizure_boundary)
            ).encode()
        ).hexdigest()[:16]
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, f"seizeit2_idx_{key}.pkl")

    def _load_cache(self, path: str) -> bool:
        """Try to load index from *path*. Returns True on success."""
        try:
            import pickle
            with open(path, "rb") as fh:
                data = pickle.load(fh)
            self._metas        = data["metas"]
            self._index        = data["index"]
            self._subj_wins    = data["subj_wins"]
            self.subject_to_windows = {sid: [] for sid in self._subj_wins}
            log_info(f"SeizeIT2 index loaded from cache: {path}")
            return True
        except Exception as exc:
            log_info(f"SeizeIT2 cache load failed ({exc}), rebuilding index.")
            return False

    def _save_cache(self, path: str):
        """Atomically write index to *path* (temp-file + rename, race-safe)."""
        import pickle, tempfile
        tmp = path + f".tmp{os.getpid()}"
        try:
            with open(tmp, "wb") as fh:
                pickle.dump(
                    {"metas": self._metas,
                     "index": self._index,
                     "subj_wins": self._subj_wins},
                    fh, protocol=4,
                )
            os.replace(tmp, path)   # atomic on POSIX
            log_info(f"SeizeIT2 index cached to: {path}")
        except Exception as exc:
            log_info(f"SeizeIT2 cache save failed (non-fatal): {exc}")
            try:
                os.remove(tmp)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Core index build  (fast — only header + TSV reads)
    # ------------------------------------------------------------------

    def _build_index(self):
        # ---- Try cache first ----
        cp = self._cache_path()
        if cp and self._load_cache(cp):
            return

        _logged_first = False   # print one-time channel diagnostic

        for subject_id in self.subjects:
            groups = list(self._find_recording_groups(subject_id))
            if not groups:
                log_info(f"Subject {subject_id}: no EDF files found, skipping")
                continue

            subj_wins: List[_WindowRecord] = []

            for grp in groups:
                eeg_edf    = grp['eeg_edf']
                tsv_path   = grp['events_tsv']
                modal_edfs = grp['modal_edfs']   # {mod: edf_path}

                # ---- EEG header (determines recording duration) ----
                try:
                    ch_names, ch_fs, n_samps = self._read_edf_header(eeg_edf)
                except Exception as exc:
                    log_info(f"  Header read failed for {eeg_edf}: {exc}")
                    continue

                ref_fs   = max(ch_fs)
                n_target = (int(max(n_samps) * self.FS / ref_fs)
                            if abs(ref_fs - self.FS) > 1 else int(max(n_samps)))

                # ---- Collect channels from each modality EDF ----
                all_indices: List[int]   = []
                all_fs:      List[float] = []
                all_edfs:    List[str]   = []

                for mod in self.modalities:
                    if mod == 'eeg':
                        # EEG: classify channels within the EEG EDF
                        sel_idx, sel_fs_m = self._select_channel_indices(
                            ch_names, ch_fs, ch_types=[])
                        src_edf = eeg_edf
                    elif mod in modal_edfs:
                        # Other modalities: dedicated EDF → take ALL channels
                        try:
                            m_names, m_fs, _ = self._read_edf_header(modal_edfs[mod])
                        except Exception as exc:
                            log_info(f"  Header read failed for {modal_edfs[mod]}: {exc}")
                            continue
                        sel_idx  = list(range(len(m_names)))
                        sel_fs_m = list(m_fs)
                        src_edf  = modal_edfs[mod]
                    else:
                        continue   # modality not present for this run

                    all_indices.extend(sel_idx)
                    all_fs.extend(sel_fs_m)
                    all_edfs.extend([src_edf] * len(sel_idx))

                if not all_indices:
                    log_info(f"  No channels selected for {os.path.basename(eeg_edf)}, skipping")
                    continue

                # One-time diagnostic
                if not _logged_first:
                    _logged_first = True
                    log_info(f"[CHANNEL DIAG] {os.path.basename(eeg_edf)}")
                    log_info(f"  modal_edfs found: { {k: os.path.basename(v) for k,v in modal_edfs.items()} }")
                    # Per-modality breakdown
                    counts = {}
                    for mod in self.modalities:
                        counts[mod] = sum(1 for e in all_edfs
                                         if (e == eeg_edf and mod == 'eeg')
                                         or (mod in modal_edfs and e == modal_edfs[mod]))
                    log_info(f"  Per-modality channel counts: {counts}")
                    log_info(f"  Total channels selected: {len(all_indices)} "
                             f"from {len(set(all_edfs))} EDF file(s)")

                # ---- Seizure intervals ----
                seizure_ivs: List[Tuple[int, int]] = []
                if tsv_path:
                    seizure_ivs = self._parse_tsv_intervals(tsv_path, n_target)
                else:
                    log_info(f"  No events TSV for {os.path.basename(eeg_edf)}, "
                             "treating as background-only recording")

                meta = _RecordingMeta(
                    subject_id     = subject_id,
                    edf_path       = eeg_edf,
                    sel_ch_indices = all_indices,
                    sel_ch_fs      = all_fs,
                    n_target       = n_target,
                    seizure_ivs    = seizure_ivs,
                    sel_ch_edfs    = all_edfs,
                )
                rec_idx = len(self._metas)
                self._metas.append(meta)

                # ---- Sliding-window records (index only) ----
                for start in range(0, n_target - self.time_steps + 1, self.step):
                    label  = self._compute_label(meta, start)
                    record = _WindowRecord(rec_idx=rec_idx, start_sample=start, label=label)
                    self._index.append(record)
                    subj_wins.append(record)

            if subj_wins:
                self._subj_wins[subject_id]       = subj_wins
                self.subject_to_windows[subject_id] = []   # empty — keys only
                n_seiz = sum(r.label for r in subj_wins)
                log_info(f"  {subject_id}: {len(subj_wins)} windows "
                         f"(seiz={n_seiz}, bg={len(subj_wins)-n_seiz})")

        if not self._index:
            raise ValueError(
                f"No windows found under '{self.data_root}'. "
                "Check data_root and subject list."
            )

        # ---- Save cache for next run ----
        cp = self._cache_path()
        if cp:
            self._save_cache(cp)

    # ------------------------------------------------------------------
    # Optional class balancing  (index-only, no data movement)
    # ------------------------------------------------------------------

    def _apply_balance(self):
        seiz_recs = [r for r in self._index if r.label == 1]
        bg_recs   = [r for r in self._index if r.label == 0]
        n_seiz    = len(seiz_recs)
        max_bg    = min(len(bg_recs), n_seiz * self.balance_ratio)
        log_info(f"Balancing: bg {len(bg_recs)} → {max_bg}, seizure={n_seiz}")
        np.random.seed(42)
        bg_keep   = [bg_recs[i] for i in np.random.choice(len(bg_recs), max_bg, replace=False)]
        self._index = seiz_recs + bg_keep
        # Invalidate per-subject sequences (ordering no longer meaningful)
        self._subj_wins        = {sid: [] for sid in self.subjects}
        self.subject_to_windows = {sid: [] for sid in self.subjects}
        log_info("Balancing applied; subject_to_windows invalidated")

    # ------------------------------------------------------------------
    # Signal reading helpers  (called lazily from __getitem__ / get_subject_sequence)
    # ------------------------------------------------------------------

    def _read_one_window(self, f, meta: _RecordingMeta, start_sample: int) -> np.ndarray:
        """Read one window from an *already-open* EdfReader ``f``.

        Uses ``readSignal(chn, start=, n=)`` — the standard pyedflib random-access
        API — to read exactly ``time_steps`` worth of samples per channel without
        loading the entire signal.  Resamples to ``time_steps`` when the channel's
        native rate differs from ``FS`` (e.g. IMU at 25 Hz).

        Returns:
            ``[C, T]`` float32 array, un-normalised.
        """
        channels = []
        for ch_idx, ch_fs in zip(meta.sel_ch_indices, meta.sel_ch_fs):
            native_start = int(start_sample * ch_fs / self.FS)
            native_len   = max(1, int(self.time_steps * ch_fs / self.FS))

            sig = f.readSignal(ch_idx, start=native_start, n=native_len).astype(np.float32)

            if len(sig) != self.time_steps:
                from scipy.signal import resample
                sig = resample(sig, self.time_steps)
            channels.append(sig)

        return np.stack(channels, axis=0)   # [C, T]

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Channel-wise z-score normalisation."""
        if self.global_stats is not None:
            mean = self.global_stats['mean'].reshape(-1, 1)
            std  = self.global_stats['std'].reshape(-1, 1) + 1e-8
        else:
            mean = x.mean(axis=1, keepdims=True)
            std  = x.std(axis=1, keepdims=True) + 1e-8
        return (x - mean) / std

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ImageFolder-style lazy read with per-worker file-handle caching.

        The EdfReader for each recording is opened once per worker process
        and reused across all subsequent calls that touch the same file.
        This eliminates the repeated header-parse overhead that stalls
        DataLoader workers when open/close is done per sample.
        """
        record = self._index[idx]
        meta   = self._metas[record.rec_idx]

        f = _get_edf_reader(meta.edf_path)
        x = self._read_one_window(f, meta, record.start_sample)

        x = self._normalize(x)
        return torch.from_numpy(x), torch.tensor(record.label, dtype=torch.long)

    def get_subject_sequence(self, subject_id: str) -> List[Tuple[torch.Tensor, int]]:
        """Load all windows for one subject into memory (for SequentialDataset).

        .. warning::
            SeizeIT2 contains ~330 K windows per subject on average.
            Calling this method will attempt to allocate several GB of RAM
            for a single subject.  For this dataset, prefer the plain
            ``DataLoader(SeizeIT2Dataset(...), ...)`` path (ImageFolder-style)
            rather than ``SequentialDataset``.

        Opens each EDF file once and reads windows sequentially.

        Raises:
            RuntimeError: when ``balance=True``.
        """
        if self.balance:
            raise RuntimeError(
                "get_subject_sequence() is unavailable when balance=True "
                "because global downsampling invalidates per-subject order."
            )
        records = self._subj_wins.get(subject_id, [])
        if not records:
            log_info(f"Subject {subject_id} has no indexed windows.")
            return []

        from collections import defaultdict
        by_rec: Dict[int, List[Tuple[int, _WindowRecord]]] = defaultdict(list)
        for pos, rec in enumerate(records):
            by_rec[rec.rec_idx].append((pos, rec))

        result: List[Optional[Tuple[torch.Tensor, int]]] = [None] * len(records)

        for rec_idx, pos_recs in by_rec.items():
            meta = self._metas[rec_idx]
            try:
                f = _get_edf_reader(meta.edf_path)
                for pos, rec in pos_recs:
                    x = self._read_one_window(f, meta, rec.start_sample)
                    x = self._normalize(x)
                    result[pos] = (torch.from_numpy(x).float(), int(rec.label))
            except Exception as exc:
                log_info(f"  Error streaming {meta.edf_path}: {exc}")

        return [item for item in result if item is not None]


# -----------------------------------------------------------------------
# IterableDataset variant — sequential per-recording reads (NFS-friendly)
# -----------------------------------------------------------------------

class SeizeIT2IterableDataset(torch.utils.data.IterableDataset):
    """Sequential-read wrapper around :class:`SeizeIT2Dataset` for NFS.

    **Why sequential reads?**
    Instead of random-seeking into EDF files (``readSignal(ch, start=, n=)``,
    one NFS round-trip per window), each recording is read **in full** once
    (``readSignal(ch)`` — a single sequential read), then all windows are
    sliced from the in-memory numpy array at zero I/O cost.

    **BPTT support**
    Each yielded item is a *sequence* of ``bptt_steps`` consecutive windows
    from the same recording: ``(x [bptt_steps, C, T], y [bptt_steps])``.
    The DataLoader collates B such items into ``[B, bptt_steps, C, T]``.
    ``AgentSequentialTrainer._wrap_plain_batch`` detects the 4-D tensor and
    constructs a proper ``SequentialBatch`` — the BPTT loop then runs exactly
    as it does for ``SequentialDataset``, including agent gating decisions.

    Crucially, windows within each BPTT chunk are **temporally consecutive**
    (drawn in order from the same recording), which is required for valid
    BPTT. The *chunks themselves* are shuffled each epoch.

    DDP / multi-worker
    ------------------
    * Rank-level partition: rank ``r`` processes every ``world_size``-th recording.
    * Worker-level partition: within a rank, each worker processes every
      ``num_workers``-th recording from the rank's slice.
    * Call ``set_epoch(e)`` at the start of each epoch to change the shuffle seed.
    """

    def __init__(
        self,
        base: "SeizeIT2Dataset",
        bptt_steps: int = 10,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
        expected_channels: Optional[int] = None,
    ):
        super().__init__()
        self._base              = base
        self._bptt              = bptt_steps
        self._rank              = rank
        self._world_size        = world_size
        self._shuffle           = shuffle
        self._epoch             = 0
        # If set, recordings whose channel count != expected_channels are
        # skipped to avoid empty-slice crashes when the model's sensor_ranges
        # extend beyond the data tensor's channel dimension.
        self._expected_channels = expected_channels

        # Pre-build rec_idx → [_WindowRecord, ...]  (windows in temporal order)
        from collections import defaultdict
        self._rec_wins: Dict[int, List[_WindowRecord]] = defaultdict(list)
        for rec in base._index:
            self._rec_wins[rec.rec_idx].append(rec)

        # Stable list of recording indices owned by this rank
        all_recs = sorted(self._rec_wins.keys())
        self._my_recs = [r for i, r in enumerate(all_recs) if i % world_size == rank]

    def set_epoch(self, epoch: int):
        """Call at the start of each epoch so the shuffle seed changes."""
        self._epoch = epoch

    # ------------------------------------------------------------------

    def __iter__(self):
        try:
            import pyedflib
        except ImportError:
            raise ImportError("pyedflib is required: pip install pyedflib")

        worker_info = torch.utils.data.get_worker_info()

        # Split recordings across workers within this rank
        rec_list = list(self._my_recs)
        if worker_info is not None:
            nw, wid = worker_info.num_workers, worker_info.id
            rec_list = [r for i, r in enumerate(rec_list) if i % nw == wid]

        # Epoch-level shuffle of recording order
        rng = np.random.default_rng(self._epoch * 997 + self._rank * 31
                                    + (worker_info.id if worker_info else 0))
        if self._shuffle:
            rng.shuffle(rec_list)

        base  = self._base
        bptt  = self._bptt
        n_recs = len(rec_list)

        for rec_pos, rec_idx in enumerate(rec_list):
            meta     = base._metas[rec_idx]
            edf_name = os.path.basename(meta.edf_path)
            records  = self._rec_wins[rec_idx]

            # Need at least bptt_steps consecutive windows
            n_chunks = len(records) // bptt
            if n_chunks == 0:
                continue

            # log_info(f"  [rec {rec_pos+1}/{n_recs}] Loading {edf_name} "
            #          f"({len(records)} windows → {n_chunks} BPTT chunks) …")
            try:
                # --- Sequential full-channel read (fast on NFS) ---
                # Group channels by source EDF so each file is opened only once.
                t0 = __import__("time").time()
                from collections import defaultdict as _dd
                edf_ch_groups = _dd(list)   # edf_path → [(out_i, ch_idx, ch_fs)]
                src_edfs = meta.sel_ch_edfs if meta.sel_ch_edfs else [meta.edf_path] * len(meta.sel_ch_indices)
                for out_i, (src, ch_idx, ch_fs) in enumerate(
                        zip(src_edfs, meta.sel_ch_indices, meta.sel_ch_fs)):
                    edf_ch_groups[src].append((out_i, ch_idx, ch_fs))

                channels_buf = [None] * len(meta.sel_ch_indices)
                for src_edf, ch_infos in edf_ch_groups.items():
                    f = pyedflib.EdfReader(src_edf)
                    for out_i, ch_idx, ch_fs in ch_infos:
                        sig = f.readSignal(ch_idx).astype(np.float32)
                        if abs(ch_fs - base.FS) > 1:
                            from scipy.signal import resample
                            sig = resample(sig, int(len(sig) * base.FS / ch_fs))
                        # Trim to EEG-derived n_target (other modalities may be longer)
                        channels_buf[out_i] = sig[:meta.n_target]
                    f._close()
                data = np.stack(channels_buf, axis=0)   # [C, N_total]
                # log_info(f"    → {data.nbytes/1e6:.0f} MB in "
                #          f"{__import__('time').time()-t0:.1f}s")
            except Exception as exc:
                log_info(f"  Skip {meta.edf_path}: {exc}")
                continue

            # Skip recordings that don't have the expected number of channels.
            # Some SeizeIT2 files are missing one or more modalities (e.g. no
            # ECG sensor), so their channel count is less than num_modal.
            # The model's sensor_ranges are fixed at construction time; if the
            # data tensor is too narrow, a later slice like x[:, 2:3, :] on a
            # 2-channel tensor would produce [B, 0, T] and crash the conv.
            if self._expected_channels is not None and data.shape[0] != self._expected_channels:
                log_info(f"  Skip {edf_name}: {data.shape[0]} channels "
                         f"(expected {self._expected_channels})")
                continue

            # --- Yield the full recording as one sequence (temporal order preserved) ---
            # train_step_bptt will split into BPTT chunks internally, so that
            # mem_running_context can be passed across consecutive chunks.
            # Do NOT shuffle windows within a recording — that would break temporal continuity.
            valid_records = [
                r for r in records
                if r.start_sample + base.time_steps <= data.shape[1]
            ]
            if not valid_records:
                continue

            x_seq = np.stack([
                base._normalize(
                    data[:, r.start_sample : r.start_sample + base.time_steps].copy()
                )
                for r in valid_records
            ], axis=0).astype(np.float32)   # [n_windows, C, T]

            y_seq = np.array([r.label for r in valid_records], dtype=np.int64)

            yield (
                torch.from_numpy(x_seq),   # [n_windows, C, T]
                torch.from_numpy(y_seq),   # [n_windows]
            )

    def __len__(self) -> int:
        """Number of recordings owned by this rank (one yield per recording)."""
        return len(self._my_recs)


# -----------------------------------------------------------------------
# Convenience dataloader factory
# -----------------------------------------------------------------------

def get_dataloaders(
    data_root: str,
    train_subjects: Optional[List[str]] = None,
    test_subjects: Optional[List[str]] = None,
    batch_size: int = 32,
    time_steps: int = 500,
    step: int = 250,
    modalities: Tuple[str, ...] = ('eeg', 'ecg', 'emg', 'imu'),
    balance: bool = False,
    apply_diff: bool = False,
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create train / test :class:`DataLoader` objects for SeizeIT2.

    **ImageFolder-style (default)**: each ``DataLoader`` iteration yields
    a plain ``(x [B, C, T], y [B])`` batch read lazily from EDF.  No data
    is pre-loaded into memory.  Compatible with
    :class:`~trainer.sigma_delta_joint_trainer.SigmaDeltaJointTrainer`
    which already handles plain batches via its ``else`` branch.

    :class:`SequentialDataset` is intentionally *not* used here because
    SeizeIT2 contains ~330 K windows per subject; loading one subject's
    sequence into RAM would require several GB.

    Default subject split: sub-001 … sub-100 training,
    sub-101 … sub-125 test.

    Returns:
        ``(train_loader, test_loader)``
    """
    if train_subjects is None:
        train_subjects = [f"sub-{i:03d}" for i in range(1, 101)]
    if test_subjects is None:
        test_subjects  = [f"sub-{i:03d}" for i in range(101, 126)]

    train_base = SeizeIT2Dataset(
        data_root, train_subjects,
        time_steps=time_steps, step=step,
        modalities=modalities, balance=balance,
        cache_dir=cache_dir,
    )
    test_base = SeizeIT2Dataset(
        data_root, test_subjects,
        time_steps=time_steps, step=step,
        modalities=modalities, balance=False,
        cache_dir=cache_dir,
    )

    if apply_diff:
        train_base = DeltaDataset(train_base, axis=-1)
        test_base  = DeltaDataset(test_base,  axis=-1)

    # Plain DataLoader — each worker reads windows independently via __getitem__.
    # * multiprocessing_context='forkserver': safer than 'fork' for C extensions
    #   (pyedflib).  Workers are spawned clean; the per-worker _worker_edf_cache
    #   is populated lazily on the first __getitem__ in each process.
    # * prefetch_factor=2: keep two batches in flight per worker so the GPU
    #   never waits on I/O.
    # pin_memory=False: the EDF data is on a network filesystem (/mnt/vstor).
    # pin_memory=True uses Unix-socket fd-passing between worker processes and
    # the pin_memory thread; on slow/NFS mounts this causes the thread to stall
    # and the DataLoader to deadlock.  CPU→GPU transfers are fast regardless.
    _pf = 2 if num_workers > 0 else None
    train_loader = DataLoader(
        train_base, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=_pf,
    )
    test_loader = DataLoader(
        test_base, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=_pf,
    )
    return train_loader, test_loader


# -----------------------------------------------------------------------
# Pre-extraction utility  (offline, run once)
# -----------------------------------------------------------------------

def preextract_to_dir(
    seizeit2_dataset: "SeizeIT2Dataset",
    output_dir: str,
    dtype: type = np.float16,
    skip_existing: bool = True,
) -> None:
    """Extract every window in *seizeit2_dataset* to numpy files in *output_dir*.

    Produces two files per recording run::

        {subject_id}_{rec_idx:04d}_data.npy    float16  [n_windows, C, T]
        {subject_id}_{rec_idx:04d}_labels.npy  int8     [n_windows]

    Once extracted, use :class:`SeizeIT2PreextractedDataset` for fast
    random access with large batch sizes — no EDF parsing at training time.

    Args:
        seizeit2_dataset: A fully initialised :class:`SeizeIT2Dataset`.
        output_dir:       Directory to write ``.npy`` files (created if absent).
        dtype:            Storage dtype for signal data (``float16`` halves disk).
        skip_existing:    Skip recordings whose output files already exist.
    """
    try:
        import pyedflib
    except ImportError:
        raise ImportError("pyedflib is required: pip install pyedflib")

    from collections import defaultdict as _dd

    os.makedirs(output_dir, exist_ok=True)
    base = seizeit2_dataset

    # Group windows by recording
    by_rec: Dict[int, List[_WindowRecord]] = _dd(list)
    for win in base._index:
        by_rec[win.rec_idx].append(win)

    n_recs = len(by_rec)
    log_info(f"[preextract] {n_recs} recordings → {output_dir}")

    for pos, (rec_idx, wins) in enumerate(sorted(by_rec.items())):
        meta  = base._metas[rec_idx]
        stem  = f"{meta.subject_id}_{rec_idx:04d}"
        dpath = os.path.join(output_dir, f"{stem}_data.npy")
        lpath = os.path.join(output_dir, f"{stem}_labels.npy")

        if skip_existing and os.path.exists(dpath) and os.path.exists(lpath):
            log_info(f"  [{pos+1}/{n_recs}] {stem}: skip (exists)")
            continue

        # --- Read full recording from EDF(s) ---
        try:
            edf_ch_groups: Dict[str, list] = _dd(list)
            src_edfs = (meta.sel_ch_edfs if meta.sel_ch_edfs
                        else [meta.edf_path] * len(meta.sel_ch_indices))
            for out_i, (src, ch_idx, ch_fs) in enumerate(zip(
                    src_edfs, meta.sel_ch_indices, meta.sel_ch_fs)):
                edf_ch_groups[src].append((out_i, ch_idx, ch_fs))

            channels_buf = [None] * len(meta.sel_ch_indices)
            for src_edf, ch_infos in edf_ch_groups.items():
                f = pyedflib.EdfReader(src_edf)
                for out_i, ch_idx, ch_fs in ch_infos:
                    sig = f.readSignal(ch_idx).astype(np.float32)
                    if abs(ch_fs - base.FS) > 1:
                        from scipy.signal import resample as _rs
                        sig = _rs(sig, int(len(sig) * base.FS / ch_fs))
                    channels_buf[out_i] = sig[:meta.n_target]
                f._close()
            raw = np.stack(channels_buf, axis=0)   # [C, N_total]
        except Exception as exc:
            log_info(f"  [{pos+1}/{n_recs}] {stem}: read failed ({exc}), skip")
            continue

        # --- Slice and normalise windows ---
        valid = [w for w in wins
                 if w.start_sample + base.time_steps <= raw.shape[1]]
        if not valid:
            continue

        x_all = np.stack([
            base._normalize(
                raw[:, w.start_sample : w.start_sample + base.time_steps].copy()
            ).astype(dtype)
            for w in valid
        ], axis=0)                                  # [n_windows, C, T]  float16
        y_all = np.array([w.label for w in valid], dtype=np.int8)

        np.save(dpath, x_all)
        np.save(lpath, y_all)
        log_info(f"  [{pos+1}/{n_recs}] {stem}: {len(valid)} windows "
                 f"({x_all.nbytes / 1e6:.0f} MB)")


# -----------------------------------------------------------------------
# Fast random-access dataset over pre-extracted numpy files
# -----------------------------------------------------------------------

import threading
_tls = threading.local()   # per-thread memmap cache (safe with DataLoader workers)


class SeizeIT2PreextractedDataset(Dataset):
    """Random-access dataset backed by pre-extracted ``.npy`` files.

    Each ``__getitem__`` reads a single ``[C, T]`` float32 window from a
    memory-mapped file — no EDF parsing, no NFS round-trips.  Enables
    ``batch_size=512`` with ``shuffle=True`` and many DataLoader workers.

    The in-memory index is compact: one ``(file_path, window_offset, label)``
    tuple per recording (not per window), so even 16 M windows occupy <1 MB.

    Usage::

        ds = SeizeIT2PreextractedDataset(
            preextracted_dir="/scratch/seizeit2_windows",
            subjects=train_subjects,
            expected_channels=15,
        )
        loader = DataLoader(ds, batch_size=512, shuffle=True, num_workers=8,
                            pin_memory=True)
    """

    def __init__(
        self,
        preextracted_dir: str,
        subjects: List[str],
        expected_channels: Optional[int] = None,
    ):
        self._dir = preextracted_dir
        # Per-recording metadata
        self._rec_paths:  List[str]        = []   # data .npy path
        self._rec_labels: List[np.ndarray] = []   # int8 label arrays (small)
        self._offsets:    List[int]        = [0]  # cumulative window counts

        subject_set = set(subjects)

        for fname in sorted(os.listdir(preextracted_dir)):
            if not fname.endswith("_data.npy"):
                continue
            stem    = fname[: -len("_data.npy")]          # sub-001_0000
            parts   = stem.split("_")
            # subject id = everything except the last numeric token (rec_idx)
            subj_id = "_".join(parts[:-1])                # sub-001
            if subj_id not in subject_set:
                continue

            dpath = os.path.join(preextracted_dir, fname)
            lpath = os.path.join(preextracted_dir, f"{stem}_labels.npy")
            if not os.path.exists(lpath):
                continue

            # Peek shape without loading signal data
            arr = np.load(dpath, mmap_mode="r")
            if expected_channels is not None and arr.shape[1] != expected_channels:
                continue

            labels = np.load(lpath).astype(np.int8)       # fully load labels (~KB)
            self._rec_paths.append(dpath)
            self._rec_labels.append(labels)
            self._offsets.append(self._offsets[-1] + len(labels))

        total = self._offsets[-1]
        log_info(f"SeizeIT2PreextractedDataset: {total:,} windows "
                 f"from {len(self._rec_paths)} recordings")

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._offsets[-1]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Binary search to find which recording this index belongs to
        import bisect
        rec_i   = bisect.bisect_right(self._offsets, idx) - 1
        win_i   = idx - self._offsets[rec_i]
        label   = int(self._rec_labels[rec_i][win_i])

        # Per-thread memmap cache: open each file at most once per worker
        cache = getattr(_tls, "mmap_cache", None)
        if cache is None:
            _tls.mmap_cache = {}
            cache = _tls.mmap_cache

        dpath = self._rec_paths[rec_i]
        if dpath not in cache:
            cache[dpath] = np.load(dpath, mmap_mode="r")  # [n_windows, C, T]

        x = cache[dpath][win_i].astype(np.float32)        # [C, T]
        return torch.from_numpy(x), torch.tensor(label, dtype=torch.long)


# -----------------------------------------------------------------------
# BPTT-capable IterableDataset over pre-extracted numpy files
# -----------------------------------------------------------------------

class SeizeIT2PreextractedIterableDataset(torch.utils.data.IterableDataset):
    """Sequential BPTT dataset backed by pre-extracted ``.npy`` files.

    Identical temporal semantics to :class:`SeizeIT2IterableDataset` but
    reads from local numpy memmap files instead of NFS EDF files — orders
    of magnitude faster, enabling larger batch sizes while preserving
    within-recording temporal order required for valid BPTT.

    Each yielded item is the **full recording** as
    ``(x [n_windows, C, T] float32, y [n_windows] int64)``.
    The caller (``_pad_collate`` + ``train_step_bptt``) handles BPTT
    chunking so that ``mem_running_context`` flows across chunks.

    DDP / multi-worker partitioning mirrors :class:`SeizeIT2IterableDataset`.
    Call ``set_epoch(e)`` at the start of each epoch.
    """

    def __init__(
        self,
        preextracted_dir: str,
        subjects: List[str],
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
        expected_channels: Optional[int] = None,
    ):
        super().__init__()
        self._dir              = preextracted_dir
        self._rank             = rank
        self._world_size       = world_size
        self._shuffle          = shuffle
        self._expected_ch      = expected_channels
        self._epoch            = 0

        # Collect all (data_path, label_path) pairs for the requested subjects
        subject_set = set(subjects)
        all_recs: List[Tuple[str, str]] = []
        for fname in sorted(os.listdir(preextracted_dir)):
            if not fname.endswith("_data.npy"):
                continue
            stem    = fname[: -len("_data.npy")]
            parts   = stem.split("_")
            subj_id = "_".join(parts[:-1])
            if subj_id not in subject_set:
                continue
            dpath = os.path.join(preextracted_dir, fname)
            lpath = os.path.join(preextracted_dir, f"{stem}_labels.npy")
            if not os.path.exists(lpath):
                continue
            if expected_channels is not None:
                arr = np.load(dpath, mmap_mode="r")
                if arr.shape[1] != expected_channels:
                    continue
            all_recs.append((dpath, lpath))

        # Rank-level partition (stable across epochs)
        self._my_recs = [r for i, r in enumerate(all_recs) if i % world_size == rank]
        log_info(f"SeizeIT2PreextractedIterableDataset: rank {rank} owns "
                 f"{len(self._my_recs)} / {len(all_recs)} recordings")

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self._my_recs)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        rec_list = list(self._my_recs)
        if worker_info is not None:
            nw, wid = worker_info.num_workers, worker_info.id
            rec_list = [r for i, r in enumerate(rec_list) if i % nw == wid]

        rng = np.random.default_rng(self._epoch * 997 + self._rank * 31
                                    + (worker_info.id if worker_info else 0))
        if self._shuffle:
            rng.shuffle(rec_list)

        cache = getattr(_tls, "mmap_cache", None)
        if cache is None:
            _tls.mmap_cache = {}
            cache = _tls.mmap_cache

        n_recs = len(rec_list)
        for pos, (dpath, lpath) in enumerate(rec_list):
            if dpath not in cache:
                cache[dpath] = np.load(dpath, mmap_mode="r")  # [n_windows, C, T]
            mmap = cache[dpath]

            labels = np.load(lpath).astype(np.int64)           # [n_windows]
            n = len(labels)
            if n == 0:
                continue

            log_info(f"  [rec {pos+1}/{n_recs}] {os.path.basename(dpath)} "
                     f"({n} windows)")

            # Copy to contiguous float32 array (cheap — local SSD)
            x = mmap[:n].astype(np.float32)                    # [n_windows, C, T]

            yield (
                torch.from_numpy(x),                           # [n_windows, C, T]
                torch.from_numpy(labels),                      # [n_windows]
            )


# -----------------------------------------------------------------------
# Quick smoke-test
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import time
    import traceback

    data_root = "/mnt/vstor/CSE_ECSE_GXD234/data/ds005873-1.1.0"   # ← update before running
    test_subs = ["sub-001"]

    print("=" * 70)
    print("SeizeIT2 lazy loader smoke-test")
    print("=" * 70)

    try:
        # 1. Init (should be near-instant — header scan only)
        print("\n1. Building index (lazy) …")
        t0   = time.time()
        base = SeizeIT2Dataset(
            data_root=data_root, subjects=test_subs,
            time_steps=500, step=250,
            modalities=('eeg', 'ecg', 'emg', 'imu'), balance=False,
        )
        print(f"   Index built in {time.time()-t0:.2f}s — {len(base)} windows")

        # 2. __getitem__ (lazy read)
        print("\n2. __getitem__ (lazy read) …")
        t0     = time.time()
        x0, y0 = base[0]
        print(f"   First window in {time.time()-t0:.3f}s  shape={x0.shape}  label={y0.item()}")

        # 3. SequentialDataset (one EDF open per recording)
        print("\n3. SequentialDataset …")
        t0  = time.time()
        seq = SequentialDataset(base, subject_ids=test_subs)
        print(f"   SequentialDataset built in {time.time()-t0:.2f}s")

        loader = DataLoader(seq, batch_size=1, shuffle=False,
                            collate_fn=collate_sequential_batch)
        batch  = next(iter(loader))
        print(f"   sequences shape : {batch.sequences.shape}")
        print(f"   labels shape    : {batch.labels.shape}")
        print(f"   seq_lengths     : {batch.seq_lengths.tolist()}")

        # 4. DeltaDataset + SequentialDataset
        print("\n4. DeltaDataset + SequentialDataset …")
        delta   = DeltaDataset(base, axis=-1)
        seq_d   = SequentialDataset(delta, subject_ids=test_subs)
        loader_d = DataLoader(seq_d, batch_size=1, shuffle=False,
                              collate_fn=collate_sequential_batch)
        batch_d = next(iter(loader_d))
        print(f"   delta sequences shape: {batch_d.sequences.shape}")

        # 5. Verify differencing is reversible
        print("\n5. Verifying cumsum(delta) ≈ original …")
        orig      = batch.sequences[0, 0, 0]
        delta_sig = batch_d.sequences[0, 0, 0]
        recovered = torch.cumsum(delta_sig, dim=0)
        err       = (recovered - orig).abs().max().item()
        print(f"   Max recovery error: {err:.2e}")
        assert err < 1e-5, "Differencing not reversible!"
        print("   OK")

        print("\n" + "=" * 70)
        print("All tests passed.")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"\nData not found – update `data_root` at the top of __main__.\n{e}")
    except Exception:
        traceback.print_exc()
