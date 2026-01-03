import numpy as np
import torch
from torch.utils.data import Dataset


def windows_from_array(arr, window=128, stride=64):
    """Create sliding windows from array of shape (N, C).

    Returns np.array shape (n_windows, window, C)
    """
    N, C = arr.shape
    if N < window:
        raise ValueError("Array length must be >= window")
    windows = []
    for start in range(0, N - window + 1, stride):
        windows.append(arr[start : start + window])
    return np.stack(windows)


class SlidingWindowDataset(Dataset):
    def __init__(self, windows):
        # windows: numpy array (n_windows, window, n_channels)
        self.windows = windows.astype('float32')

    def __len__(self):
        return self.windows.shape[0]

    def __getitem__(self, idx):
        arr = self.windows[idx]  # (window, channels)
        # convert to (channels, window) for Conv1D
        return torch.from_numpy(arr.T)
