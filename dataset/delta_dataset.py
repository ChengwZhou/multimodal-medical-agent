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
        Apply differencing along the specified axis.
        The output has the same shape as input, with:
        - First element: original value x[0]
        - Subsequent elements: differences x[t] - x[t-1]

        This ensures cumsum(output) recovers the original signal.
        """
        if x.shape[self.axis] <= 1:
            return x.copy()

        # Normalize axis to positive index
        axis = self.axis if self.axis >= 0 else x.ndim + self.axis

        # Compute differences: x[t] - x[t-1] for t >= 1
        # np.diff returns shape [..., T-1, ...]
        diff = np.diff(x, axis=axis)

        # Get the first element along the axis
        # We'll keep this as the first element in output
        first = np.take(x, indices=[0], axis=axis)  # shape: [..., 1, ...]

        # Concatenate [first, diff] along the axis
        # This gives us: [x[0], x[1]-x[0], x[2]-x[1], ..., x[T-1]-x[T-2]]
        delta_x = np.concatenate([first, diff], axis=axis)

        # Verify the shape is preserved
        assert delta_x.shape == x.shape, \
            f"Shape mismatch: input {x.shape} vs output {delta_x.shape}"

        return delta_x

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

    def get_subject_sequence(self, subject_id: int):
        if not hasattr(self.base_dataset, 'get_subject_sequence'):
            raise AttributeError("base_dataset has no get_subject_sequence")

        raw_seq = self.base_dataset.get_subject_sequence(subject_id)
        diff_seq = []
        for x, y in raw_seq:
            # print(x[0][:10])
            x_np = x.numpy() if isinstance(x, torch.Tensor) else np.array(x)
            x_diff = self.apply_differencing(x_np)
            x_diff = torch.from_numpy(x_diff).float()
            diff_seq.append((x_diff, int(y)))
            # print(x_diff[0][:10])
        return diff_seq