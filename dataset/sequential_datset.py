import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import Dataset, DataLoader

from typing import List, Tuple, Optional
from dataclasses import dataclass



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
        for subject_id in self.subject_ids:
            sequence = scientisst_dataset.get_subject_sequence(subject_id)
            if sequence:  # Only include subjects with data
                self.subject_sequences[subject_id] = sequence

        # Update subject list to only include those with data
        self.subject_ids = list(self.subject_sequences.keys())
        print(f"Sequential dataset initialized with {len(self.subject_ids)} subjects")

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
