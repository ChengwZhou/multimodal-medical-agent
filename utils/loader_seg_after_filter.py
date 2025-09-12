# =====================================
# File: load_optimized.py
# ScientISST MOVE — EDF multi-device, multi-modal windowed loader (OPTIMIZED)
#   - 预处理所有信号数据，避免重复计算
#   - 缓存处理后的数据
#   - 优化内存使用
#   - 支持lazy loading和预加载模式
# =====================================

from __future__ import annotations
import os
import re
import math
import pickle
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.distributed import get_rank, is_initialized

try:
    import pyedflib
except Exception as e:
    raise ImportError("This loader requires 'pyEDFlib'. Install via: pip install pyEDFlib")

from utils.filters import apply_filter, resample_to, zscore

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_info(msg):
    if not is_initialized() or get_rank() == 0:
        logger.info(msg)


# -----------------------------
# 数据结构
# -----------------------------

@dataclass
class PreprocessedSignal:
    """预处理后的信号数据"""
    data: np.ndarray  # 已处理的信号数据
    fs: float  # 目标采样率
    original_fs: float  # 原始采样率


@dataclass
class PreprocessedDevice:
    """预处理后的设备数据"""
    path: str
    start_time_epoch: float
    duration_sec: float
    signals: Dict[str, PreprocessedSignal]
    annotations: List[Tuple[float, float, str]]


@dataclass
class WindowIndex:
    """窗口索引，只存储必要信息"""
    subject_id: str
    window_idx: int  # 在该subject中的窗口序号
    t0: float  # 绝对时间
    dur: float
    y: Optional[int]


# -----------------------------
# 通道选择规则
# -----------------------------

DEFAULT_CHANNEL_PATTERNS = {
    'ecg': [r'ECG'],
    'ppg': [r'PPG', r'BVP'],
    'eda': [r'EDA', r'GSR'],
    'emg': [r'EMG'],
    'temp': [r'TEMP', r'Temperature'],
    'c_acc': [r'CHEST.*ACC.*X', r'CHEST.*ACC.*Y', r'CHEST.*ACC.*Z'],
    'w_acc': [r'WRIST.*ACC.*X', r'WRIST.*ACC.*Y', r'WRIST.*ACC.*Z'],
}

DEVICE_CHANNEL_HINTS = {
    'empatica': {
        'ppg': [r'BVP', r'PPG'],
        'eda': [r'EDA'],
        'temp': [r'TEMP', r'Temperature'],
        'w_acc': [r'ACC.*X', r'ACC.*Y', r'ACC.*Z'],
    },
    'chest': {
        'ecg_gel': [r'ECG'],
        'ecg_textile': [r'ECG'],
        'c_acc': [r'ACC.*X', r'ACC.*Y', r'ACC.*Z'],
    },
    'forearm': {
        'emg': [r'EMG'],
        'eda': [r'EDA'],
        'ppg': [r'PPG'],
    }
}

FILTER_CONFIG = {
    'ppg': 'ppg',
    'eda': 'eda',
    'ecg_gel': 'ecg',
    'ecg_textile': 'ecg',
    'emg': None,  # EMG不滤波
    'temp': None,
    'ax': 'acc', 'ay': 'acc', 'az': 'acc',
    'cx': 'acc', 'cy': 'acc', 'cz': 'acc'
}

TARGET_FS = 100  # 统一目标采样率


def _match_first(label_list: List[str], patterns: List[str]) -> Optional[str]:
    """匹配第一个符合条件的通道"""
    for pat in patterns:
        regex = re.compile(pat, re.IGNORECASE)
        for lab in label_list:
            if regex.search(lab):
                return lab
    return None


# -----------------------------
# EDF读取和预处理
# -----------------------------

def _read_and_preprocess_edf(path: str, device_type: str) -> PreprocessedDevice:
    """读取并预处理EDF文件"""
    f = pyedflib.EdfReader(path)
    try:
        n_signals = f.signals_in_file
        labels = [f.getLabel(i).strip() for i in range(n_signals)]

        # 获取基本信息
        dt = f.getStartdatetime()
        start_epoch = dt.timestamp() if hasattr(dt, 'timestamp') else 0.0
        file_dur = float(f.getFileDuration())

        # 读取注释
        try:
            anns = f.readAnnotations()
            onsets = anns[0]
            durations = anns[1]
            ann_labels = [a.decode('utf-8') if isinstance(a, (bytes, bytearray)) else str(a) for a in anns[2]]
            annotations = [(float(o), float(d), s) for o, d, s in zip(onsets, durations, ann_labels)]
        except Exception:
            annotations = []

        # 预处理信号
        processed_signals = {}
        channel_hints = DEVICE_CHANNEL_HINTS.get(device_type, {})

        for signal_key, patterns in channel_hints.items():
            matched_label = _match_first(labels, patterns)
            if matched_label is None:
                continue

            # 读取原始数据
            signal_idx = labels.index(matched_label)
            raw_data = f.readSignal(signal_idx).astype(np.float32)
            original_fs = float(f.getSampleFrequency(signal_idx))

            # 应用滤波
            filter_type = FILTER_CONFIG.get(signal_key)
            # if filter_type:
            #     processed_data = apply_filter(raw_data, original_fs, filter_type)
            # else:
            processed_data = raw_data.copy()

            # 重采样到目标频率
            if abs(original_fs - TARGET_FS) > 1e-6:
                processed_data = resample_to(processed_data, original_fs, TARGET_FS)

            # Z-score标准化
            processed_data = zscore(processed_data)

            processed_signals[signal_key] = PreprocessedSignal(
                data=processed_data,
                fs=TARGET_FS,
                original_fs=original_fs
            )

    finally:
        f.close()

    return PreprocessedDevice(
        path=path,
        start_time_epoch=start_epoch,
        duration_sec=file_dur,
        signals=processed_signals,
        annotations=annotations
    )


def validate_tensor_dict(tensor_dict: Dict[str, torch.Tensor], dtype=torch.float32) -> Dict[str, torch.Tensor]:
    """Ensure all values are torch Tensors with requested dtype (cpu)"""
    out = {}
    for k, v in tensor_dict.items():
        if isinstance(v, np.ndarray):
            t = torch.from_numpy(v.astype(np.float32))
        elif isinstance(v, torch.Tensor):
            t = v
        else:
            t = torch.tensor(v, dtype=torch.float32)
        # ensure dtype
        if t.dtype != dtype:
            t = t.to(dtype=dtype)
        out[k] = t
    return out

# -----------------------------
# 缓存管理
# -----------------------------

class DataCache:
    """数据缓存管理器"""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache = {}
        self._lock = threading.RLock()

    def get_cache_path(self, subject_id: str, device_type: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{subject_id}_{device_type}.pkl"

    def load_from_cache(self, subject_id: str, device_type: str) -> Optional[PreprocessedDevice]:
        """从缓存加载数据"""
        cache_key = f"{subject_id}_{device_type}"

        # 先检查内存缓存
        with self._lock:
            if cache_key in self._memory_cache:
                return self._memory_cache[cache_key]

        # 检查磁盘缓存
        cache_path = self.get_cache_path(subject_id, device_type)
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                with self._lock:
                    self._memory_cache[cache_key] = data
                return data
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_path}: {e}")

        return None

    def save_to_cache(self, subject_id: str, device_type: str, data: PreprocessedDevice):
        """保存数据到缓存"""
        cache_key = f"{subject_id}_{device_type}"

        # 保存到内存
        with self._lock:
            self._memory_cache[cache_key] = data

        # 保存到磁盘
        cache_path = self.get_cache_path(subject_id, device_type)
        if cache_path:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
            except Exception as e:
                logger.warning(f"Failed to save cache {cache_path}: {e}")


# -----------------------------
# 优化后的数据集
# -----------------------------

class MoveEDFWindowDataset(Dataset):
    """优化后的MOVE EDF数据集"""

    def __init__(
            self,
            root: str,
            window_sec: float = 1.0,
            stride_sec: Optional[float] = 0.5,
            label_map: Optional[Dict[str, int]] = None,
            min_coverage: float = 0.95,
            none_policy: str = "extra_class",
            ignore_index: int = -100,
            cache_dir: Optional[str] = None,
            preload: bool = True,
            num_workers: int = 1
    ):
        super().__init__()
        self.root = root
        self.window_sec = float(window_sec)
        self.stride_sec = float(stride_sec) if stride_sec is not None else float(window_sec)
        self.min_coverage = float(min_coverage)
        self.none_policy = none_policy
        self.ignore_index = ignore_index
        self.preload = preload
        self.num_workers = num_workers

        # 初始化缓存
        self.cache = DataCache(cache_dir)

        # 扫描subjects
        self.subjects = self._scan_subjects()
        log_info(f"Found {len(self.subjects)} subjects")

        # 预处理所有设备数据
        self.preprocessed_devices: Dict[str, Dict[str, PreprocessedDevice]] = {}
        self._preprocess_all_devices()

        # 构建标签映射
        self._build_label_map(label_map)

        # 构建窗口索引
        self._build_window_indices()

        # 预计算所有窗口数据
        self.precomputed_windows: Dict[Tuple[str, int], Dict[str, torch.Tensor]] = {}
        if preload:
            self._precompute_windows()

    def _scan_subjects(self) -> List[str]:
        """扫描有效的subjects"""
        subjects = []
        for sid in sorted(os.listdir(self.root)):
            subj_dir = os.path.join(self.root, sid)
            if not os.path.isdir(subj_dir):
                continue
            required_files = ['empatica.edf', 'scientisst_chest.edf', 'scientisst_forearm.edf']
            if all(os.path.exists(os.path.join(subj_dir, f)) for f in required_files):
                subjects.append(sid)
        return subjects

    def _preprocess_single_subject(self, subject_id: str) -> Dict[str, PreprocessedDevice]:
        """预处理单个subject的所有设备数据"""
        subject_dir = os.path.join(self.root, subject_id)
        devices = {}

        device_files = {
            'wrist': 'empatica.edf',
            'chest': 'scientisst_chest.edf',
            'forearm': 'scientisst_forearm.edf'
        }

        for device_type, filename in device_files.items():
            # 尝试从缓存加载
            cached_data = self.cache.load_from_cache(subject_id, device_type)
            if cached_data is not None:
                devices[device_type] = cached_data
                continue

            # 处理新数据
            file_path = os.path.join(subject_dir, filename)
            device_data = _read_and_preprocess_edf(file_path, device_type)

            # 保存到缓存
            self.cache.save_to_cache(subject_id, device_type, device_data)
            devices[device_type] = device_data

        return devices

    def _preprocess_all_devices(self):
        """并行预处理所有设备数据"""
        log_info("Preprocessing device data...")

        if self.num_workers == 1:
            # 单线程处理
            for subject_id in self.subjects:
                self.preprocessed_devices[subject_id] = self._preprocess_single_subject(subject_id)
        else:
            # 多线程处理
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_subject = {
                    executor.submit(self._preprocess_single_subject, sid): sid
                    for sid in self.subjects
                }

                for future in as_completed(future_to_subject):
                    subject_id = future_to_subject[future]
                    try:
                        self.preprocessed_devices[subject_id] = future.result()
                        log_info(f"Processed subject {subject_id}")
                    except Exception as e:
                        logger.error(f"Failed to process subject {subject_id}: {e}")

        log_info(f"Preprocessing completed for {len(self.preprocessed_devices)} subjects")

    def _build_label_map(self, label_map: Optional[Dict[str, int]]):
        """构建标签映射"""
        if label_map is None:
            all_labels = set()
            for subject_devices in self.preprocessed_devices.values():
                for device in subject_devices.values():
                    for _, _, label in device.annotations:
                        normalized = self._normalize_label(label)
                        all_labels.add(normalized)

            sorted_labels = sorted(all_labels)
            self.label_map = {label: i for i, label in enumerate(sorted_labels)}
        else:
            self.label_map = label_map.copy()

        # 处理None标签
        if self.none_policy == "extra_class" and "__none__" not in self.label_map:
            self.label_map["__none__"] = len(self.label_map)

    def _normalize_label(self, label: str) -> str:
        """标签标准化"""
        if label.startswith("lift"):
            return "lift"
        if label.startswith("walk_before"):
            return "walk_before"
        return label

    def _build_window_indices(self):
        """构建窗口索引"""
        self.window_indices: List[WindowIndex] = []
        self.subject_windows: Dict[str, List[WindowIndex]] = {}

        for subject_id, devices in self.preprocessed_devices.items():
            chest, forearm, wrist = devices['chest'], devices['forearm'], devices['wrist']

            # 计算有效时间范围
            start_max = max(chest.start_time_epoch, forearm.start_time_epoch, wrist.start_time_epoch)
            end_min = min(
                chest.start_time_epoch + chest.duration_sec,
                forearm.start_time_epoch + forearm.duration_sec,
                wrist.start_time_epoch + wrist.duration_sec
            )

            overlap_duration = max(0.0, end_min - start_max)
            if overlap_duration < self.window_sec * self.min_coverage:
                logger.warning(f"Subject {subject_id} has insufficient overlap duration: {overlap_duration:.2f}s")
                continue

            # 生成窗口
            num_windows = int(math.floor((overlap_duration - self.window_sec) / self.stride_sec) + 1)
            subject_window_list = []

            for i in range(num_windows):
                t0 = start_max + i * self.stride_sec
                label = self._get_majority_label(devices, t0, self.window_sec)

                # 处理None标签
                if label is None:
                    if self.none_policy == "ignore":
                        label = self.ignore_index
                    elif self.none_policy == "extra_class":
                        label = self.label_map["__none__"]

                window_idx = WindowIndex(
                    subject_id=subject_id,
                    window_idx=i,
                    t0=t0,
                    dur=self.window_sec,
                    y=label
                )

                self.window_indices.append(window_idx)
                subject_window_list.append(window_idx)

            if subject_window_list:
                self.subject_windows[subject_id] = subject_window_list

        log_info(f"Built {len(self.window_indices)} windows across {len(self.subject_windows)} subjects")

    def _get_majority_label(self, devices: Dict[str, PreprocessedDevice], t0: float, duration: float) -> Optional[int]:
        """获取窗口内的多数标签"""
        label_votes: Dict[str, float] = {}

        for device in devices.values():
            for onset, length, label in device.annotations:
                abs_onset = device.start_time_epoch + float(onset)
                abs_end = abs_onset + float(length if length > 0 else 0.0)

                # 计算重叠时间
                window_start, window_end = t0, t0 + duration
                overlap = max(0.0, min(abs_end, window_end) - max(abs_onset, window_start))

                if overlap > 0:
                    normalized_label = self._normalize_label(label)
                    label_votes[normalized_label] = label_votes.get(normalized_label, 0.0) + overlap

        if not label_votes:
            return None

        majority_label = max(label_votes.items(), key=lambda x: x[1])[0]
        return self.label_map.get(majority_label)

    def _extract_window_data(self, subject_id: str, t0: float, duration: float) -> Dict[str, torch.Tensor]:
        """从预处理数据中提取窗口数据"""
        devices = self.preprocessed_devices[subject_id]
        wrist, chest, forearm = devices['wrist'], devices['chest'], devices['forearm']

        def extract_segment(device: PreprocessedDevice, signal_key: str) -> np.ndarray:
            if signal_key not in device.signals:
                return np.zeros(int(TARGET_FS * duration), dtype=np.float32)

            signal = device.signals[signal_key]
            rel_start = max(0.0, t0 - device.start_time_epoch)

            start_idx = int(round(rel_start * signal.fs))
            end_idx = int(round((rel_start + duration) * signal.fs))

            segment = signal.data[start_idx:end_idx]

            # 确保长度正确
            target_length = int(round(duration * signal.fs))
            if len(segment) < target_length:
                segment = np.pad(segment, (0, target_length - len(segment)), mode='constant')
            elif len(segment) > target_length:
                segment = segment[:target_length]

            return segment.astype(np.float32)

        # 提取各种信号
        result = {}

        # ECG (chest)
        result['ecg_chest_gel'] = extract_segment(chest, 'ecg_gel')
        result['ecg_chest_textile'] = extract_segment(chest, 'ecg_textile')

        # EDA
        result['eda_forearm_scientisst'] = extract_segment(forearm, 'eda')
        result['eda_wrist_e4'] = extract_segment(wrist, 'eda')

        # PPG
        result['ppg_forearm_scientisst'] = extract_segment(forearm, 'ppg')
        result['ppg_wrist_e4'] = extract_segment(wrist, 'ppg')

        # EMG
        result['emg_forearm'] = extract_segment(forearm, 'emg')

        # Temperature
        result['temp_wrist'] = extract_segment(wrist, 'temp')

        # Accelerometer - chest
        chest_acc_x = extract_segment(chest, 'cx') if 'cx' in chest.signals else np.zeros(int(TARGET_FS * duration))
        chest_acc_y = extract_segment(chest, 'cy') if 'cy' in chest.signals else np.zeros(int(TARGET_FS * duration))
        chest_acc_z = extract_segment(chest, 'cz') if 'cz' in chest.signals else np.zeros(int(TARGET_FS * duration))
        result['c_acc'] = np.stack([chest_acc_x, chest_acc_y, chest_acc_z], axis=0)

        # Accelerometer - wrist
        wrist_acc_x = extract_segment(wrist, 'ax') if 'ax' in wrist.signals else np.zeros(int(TARGET_FS * duration))
        wrist_acc_y = extract_segment(wrist, 'ay') if 'ay' in wrist.signals else np.zeros(int(TARGET_FS * duration))
        wrist_acc_z = extract_segment(wrist, 'az') if 'az' in wrist.signals else np.zeros(int(TARGET_FS * duration))
        result['w_acc'] = np.stack([wrist_acc_x, wrist_acc_y, wrist_acc_z], axis=0)

        # 转换为torch tensor，添加channel dimension
        tensor_result = {}
        for key, data in result.items():
            if data.ndim == 1:  # 单通道信号
                tensor_result[key] = torch.from_numpy(data).unsqueeze(0)  # [1, T]
            else:  # 多通道信号 (如加速度计)
                tensor_result[key] = torch.from_numpy(data)  # [C, T]

        return tensor_result

    def _precompute_windows(self):
        """预计算所有窗口数据"""
        log_info("Precomputing window data...")

        for i, window_idx in enumerate(self.window_indices):
            if i % 20000 == 0:
                log_info(f"Precomputed {i}/{len(self.window_indices)} windows")

            window_data = self._extract_window_data(
                window_idx.subject_id,
                window_idx.t0,
                window_idx.dur
            )

            cache_key = (window_idx.subject_id, window_idx.window_idx)
            self.precomputed_windows[cache_key] = window_data

        log_info(f"Precomputation completed for {len(self.precomputed_windows)} windows")

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        window_idx = self.window_indices[idx]

        if self.preload:
            # 从预计算缓存获取
            cache_key = (window_idx.subject_id, window_idx.window_idx)
            x_dict = self.precomputed_windows[cache_key]
        else:
            # 实时计算
            x_dict = self._extract_window_data(
                window_idx.subject_id,
                window_idx.t0,
                window_idx.dur
            )
        x_dict = validate_tensor_dict(x_dict, dtype=torch.float32)

        return x_dict, window_idx.y

    def get_subject_sequence(self, subject_id: str) -> List[Tuple[Dict[str, torch.Tensor], int]]:
        """获取某个subject的完整序列"""
        if subject_id not in self.subject_windows:
            return []

        sequence = []
        for window_idx in self.subject_windows[subject_id]:
            if self.preload:
                cache_key = (subject_id, window_idx.window_idx)
                x_dict = self.precomputed_windows[cache_key]
            else:
                x_dict = self._extract_window_data(subject_id, window_idx.t0, window_idx.dur)

            # 验证并确保数据类型正确
            x_dict = validate_tensor_dict(x_dict, torch.float32)
            sequence.append((x_dict, window_idx.y))

        return sequence


# -----------------------------
# 工具函数
# -----------------------------

def split_by_subject(dataset: MoveEDFWindowDataset, val_ratio: float = 0.2):
    """按subject划分训练/验证集"""
    rng = np.random.default_rng(42)
    subject_ids = list(dataset.subject_windows.keys())
    n_val = max(1, int(round(len(subject_ids) * val_ratio)))
    val_subjects = set(rng.choice(subject_ids, size=n_val, replace=False))

    train_indices, val_indices = [], []
    for i, window_idx in enumerate(dataset.window_indices):
        if window_idx.subject_id in val_subjects:
            val_indices.append(i)
        else:
            train_indices.append(i)

    return train_indices, val_indices


def filter_labels(dataset: MoveEDFWindowDataset, remove_labels: List[str]) -> MoveEDFWindowDataset:
    """过滤指定标签"""
    remove_set = set(remove_labels)
    remove_ids = {dataset.label_map[label] for label in remove_labels if label in dataset.label_map}

    if not remove_ids:
        return dataset

    # 重建标签映射
    new_label_map = {}
    id_mapping = {}
    new_id = 0

    for label, old_id in dataset.label_map.items():
        if label in remove_set:
            continue
        new_label_map[label] = new_id
        id_mapping[old_id] = new_id
        new_id += 1

    # 过滤窗口索引
    filtered_indices = []
    filtered_subject_windows = {}
    filtered_precomputed = {}

    for window_idx in dataset.window_indices:
        if window_idx.y in remove_ids:
            continue

        # 更新标签ID
        new_window_idx = WindowIndex(
            subject_id=window_idx.subject_id,
            window_idx=window_idx.window_idx,
            t0=window_idx.t0,
            dur=window_idx.dur,
            y=id_mapping[window_idx.y]
        )

        filtered_indices.append(new_window_idx)

        # 更新subject windows
        if window_idx.subject_id not in filtered_subject_windows:
            filtered_subject_windows[window_idx.subject_id] = []
        filtered_subject_windows[window_idx.subject_id].append(new_window_idx)

        # 更新预计算数据
        if dataset.preload:
            cache_key = (window_idx.subject_id, window_idx.window_idx)
            if cache_key in dataset.precomputed_windows:
                filtered_precomputed[cache_key] = dataset.precomputed_windows[cache_key]

    # 更新数据集属性
    dataset.label_map = new_label_map
    dataset.window_indices = filtered_indices
    dataset.subject_windows = filtered_subject_windows
    if dataset.preload:
        dataset.precomputed_windows = filtered_precomputed

    log_info(f"Filtered dataset: {len(filtered_indices)} windows, {len(new_label_map)} labels")
    return dataset