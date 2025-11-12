import sys
import os
import torch.distributed as dist

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import Dataset, DataLoader
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_info(msg):
    from torch.distributed import is_initialized, get_rank
    if not is_initialized() or get_rank() == 0:
        logger.info(msg)


@dataclass
class SequentialBatch:
    """Sequential batch with padding information"""
    sequences: torch.Tensor  # [B, max_seq_len, C, T] - padded sequences
    labels: torch.Tensor  # [B, max_seq_len] - padded labels (-100 for padding)
    seq_lengths: torch.Tensor  # [B] - actual sequence lengths for each subject
    subject_ids: List[str]  # Subject IDs for this batch


class SequentialDataset(Dataset):
    """
    Dataset wrapper that groups data by subject and maintains temporal order
    """

    def __init__(self, scientisst_dataset, subject_ids: Optional[List[str]] = None):
        self.base_dataset = scientisst_dataset
        self.subject_ids = subject_ids or list(scientisst_dataset.subject_to_windows.keys())

        # Group sequences by subject
        self.subject_sequences = {}
        all_lengths = []
        for subject_id in self.subject_ids:
            sequence = scientisst_dataset.get_subject_sequence(subject_id)
            if sequence:  # Only include subjects with data
                self.subject_sequences[subject_id] = sequence
                all_lengths.append(len(sequence))
        self.max_length = min(all_lengths)

        # Update subject list to only include those with data
        self.subject_ids = list(self.subject_sequences.keys())
        log_info(f"Sequential dataset initialized with {len(self.subject_ids)} subjects")

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> Tuple[str, List[Tuple[torch.Tensor, int]]]:
        subject_id = self.subject_ids[idx]
        sequence = self.subject_sequences[subject_id]
        return subject_id, sequence


def collate_sequential_batch(batch: List[Tuple[str, List[Tuple[torch.Tensor, int]]]]) -> SequentialBatch:
    """
    Collate function for sequential batches with padding
    Uses -100 for padding labels (standard ignore_index)
    """
    subject_ids = []
    x_sequences = []
    y_sequences = []
    lengths = []

    for subject_id, sequence in batch:
        subject_ids.append(subject_id)
        lengths.append(len(sequence))

        # Extract x and y from sequence
        x_seq = torch.stack([x for x, y in sequence])  # [seq_len, C, T]
        y_seq = torch.tensor([y for x, y in sequence], dtype=torch.long)  # [seq_len]

        x_sequences.append(x_seq)
        y_sequences.append(y_seq)

    # Pad sequences to same length
    max_len = max(lengths)
    batch_size = len(batch)
    C, T = x_sequences[0].shape[1], x_sequences[0].shape[2]

    # Initialize padded tensors
    sequences = torch.zeros(batch_size, max_len, C, T, dtype=torch.float32)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)  # Use -100 for padding

    # Fill with actual data
    for i, (x_seq, y_seq, length) in enumerate(zip(x_sequences, y_sequences, lengths)):
        sequences[i, :length] = x_seq
        labels[i, :length] = y_seq

    return SequentialBatch(
        sequences=sequences,
        labels=labels,
        seq_lengths=torch.tensor(lengths, dtype=torch.long),
        subject_ids=subject_ids
    )


def collate_sequential_batch_sync(batch, global_max_len):
    if not batch:
        return None

    subject_ids = [s[0] for s in batch]
    sequences = [s[1] for s in batch]

    lengths = [len(seq) for seq in sequences]
    batch_size = len(batch)

    if batch_size == 0:
        return None

    # 取第一个非空序列的形状
    first_valid = next(s for s in sequences if len(s) > 0)
    C, T = first_valid[0][0].shape

    # 在 CPU 上创建 padded tensor
    padded_seqs = torch.zeros(batch_size, global_max_len, C, T, dtype=torch.float32)  # CPU
    padded_labels = torch.full((batch_size, global_max_len), -100, dtype=torch.long)  # CPU

    for i, seq in enumerate(sequences):
        if len(seq) == 0:
            continue
        x_seq = torch.stack([x for x, y in seq])
        y_seq = torch.tensor([y for x, y in seq], dtype=torch.long)

        L = len(seq)
        L = min(global_max_len, L)
        padded_seqs[i, :L] = x_seq[:L]
        padded_labels[i, :L] = y_seq[:L]

    return SequentialBatch(
        sequences=padded_seqs,      # CPU
        labels=padded_labels,       # CPU
        seq_lengths=torch.tensor(lengths, dtype=torch.long),
        subject_ids=subject_ids
    )