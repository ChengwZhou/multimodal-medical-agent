import numpy as np
import torch
from torch.utils.data import Dataset


class DeltaDataset(Dataset):
    """Wrapper dataset that applies differencing to the input data of a base dataset."""

    def __init__(self, base_dataset: Dataset, axis: int = -1):
        """
        Args:
            base_dataset: The original dataset to wrap (e.g., ScientISSTMOVEDataset or MHealthDataset)
            axis: Axis along which to apply differencing (default: -1, last axis, typically time)
        """
        self.base_dataset = base_dataset
        self.axis = axis

    def apply_differencing(self, x: np.ndarray) -> np.ndarray:
        """
        Apply differencing to signal along specified axis.
        For 1D: [t0, t1-t0, t2-t1, t3-t2, ...]
        For multi-channel (e.g., [C, T]): applies independently to each channel.
        First value remains unchanged, subsequent values are differences.

        Args:
            x: Input signal array, can be 1D [T] or multi-dim e.g. [C, T]

        Returns:
            Differenced signal of same shape as input
        """
        if x.shape[self.axis] <= 1:
            return x.copy()

        # Create slices for diff
        slices_prev = [slice(None)] * x.ndim
        slices_prev[self.axis] = slice(None, -1)
        slices_curr = [slice(None)] * x.ndim
        slices_curr[self.axis] = slice(1, None)

        # Compute differences
        diff_x = np.zeros_like(x)
        diff_x[tuple(slices_prev)] = x[tuple(slices_prev)]  # Copy all previous values
        diff_x[tuple(slices_curr)] = x[tuple(slices_curr)] - x[tuple(slices_prev)]  # Compute differences

        # Correct: set first slice to original first values
        slices_first = [slice(None)] * x.ndim
        slices_first[self.axis] = slice(0, 1)
        diff_x[tuple(slices_first)] = x[tuple(slices_first)]

        return diff_x

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]

        # Convert to numpy for differencing
        x_np = x.numpy() if isinstance(x, torch.Tensor) else x

        # Apply differencing
        x_diff = self.apply_differencing(x_np)

        # Convert back to torch tensor
        x_diff = torch.from_numpy(x_diff).to(dtype=x.dtype)

        return x_diff, y

    def __getattr__(self, name):
        """Delegate attribute access to the base dataset."""
        return getattr(self.base_dataset, name)


