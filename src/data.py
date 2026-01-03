import numpy as np
import pandas as pd
import random

from typing import Tuple


def load_safe_csv(path: str) -> np.ndarray:
    """Load accelerometer CSV (AccX,AccY,AccZ) into numpy array shape (N, 3)."""
    df = pd.read_csv(path)
    return df[['AccX','AccY','AccZ']].to_numpy(dtype=np.float32)


def window_data(data: np.ndarray, window_size: int, step: int) -> np.ndarray:
    """Slice data into overlapping windows: returns (num_windows, window_size, channels)"""
    N, C = data.shape
    windows = []
    for start in range(0, N - window_size + 1, step):
        windows.append(data[start:start+window_size])
    return np.stack(windows, axis=0)


def generate_anomalies(windows: np.ndarray, seed: int = 42, anomaly_rate: float=1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Generate anomalous windows by perturbing safe windows.

    Returns (X, y) where y==1 indicates safe (Class 1), y==0 indicates anomalous (Class 2).
    anomaly_rate controls how many anomalies to generate per safe window (e.g., 1.0 -> equal number).
    """
    rng = np.random.RandomState(seed)
    num_safe = len(windows)
    num_anom = int(num_safe * anomaly_rate)

    anoms = []
    for i in range(num_anom):
        w = windows[rng.randint(0, num_safe)].copy()
        anom_type = rng.choice(['spike','drift','noise','scale','drop'])
        if anom_type == 'spike':
            # add large spike at random time and channel
            t = rng.randint(0, w.shape[0])
            c = rng.randint(0, w.shape[1])
            w[t, c] += rng.normal(loc=0, scale=15.0)
        elif anom_type == 'drift':
            # linear drift across window
            factor = rng.uniform(-0.05, 0.05)
            drift = np.linspace(0, factor*w.shape[0], w.shape[0])[:, None]
            w += drift
        elif anom_type == 'noise':
            w += rng.normal(scale=2.0, size=w.shape)
        elif anom_type == 'scale':
            s = rng.uniform(1.5, 3.0)
            w *= s
        elif anom_type == 'drop':
            # zero out a short segment
            t = rng.randint(0, max(1, w.shape[0]//4))
            start = rng.randint(0, w.shape[0]-t)
            w[start:start+t] = 0.0
        anoms.append(w)

    X = np.concatenate([windows, np.stack(anoms, axis=0)], axis=0)
    y = np.concatenate([np.ones(len(windows), dtype=np.int64), np.zeros(len(anoms), dtype=np.int64)])

    # shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


# A small PyTorch Dataset wrapper (import lazily to avoid hard dependency in non-training contexts)
class AccelerometerDataset:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # return (channels, seq_len) for Conv1d -> channels first
        x = self.X[idx].T.copy()
        return x, self.y[idx]
