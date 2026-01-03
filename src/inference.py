import os
import numpy as np
import torch
from src.model import ConvAutoencoder
from src.utils import load_scaler, scale_windows
from src.dataset import windows_from_array


def load_model_and_meta(checkpoint_dir='checkpoints', device='cpu'):
    model = ConvAutoencoder(in_channels=3)
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_model.pth'), map_location=device))
    model.to(device)
    model.eval()
    scaler = load_scaler(os.path.join(checkpoint_dir, 'scaler.joblib'))
    thresh = float(np.load(os.path.join(checkpoint_dir, 'threshold.npy')))
    return model, scaler, thresh


def classify_series(arr, model, scaler, thresh, window=128, stride=64, device='cpu'):
    """arr: numpy array shape (N,3). Returns per-window anomaly flags and per-sample flags by max pooling"""
    windows = windows_from_array(arr, window=window, stride=stride)
    windows_scaled = scale_windows(windows, scaler)
    X = torch.from_numpy(windows_scaled.astype('float32')).permute(0, 2, 1)  # (n_windows, C, L)
    with torch.no_grad():
        X = X.to(device)
        recon = model(X)
        per_window_err = torch.mean((recon - X) ** 2, dim=(1, 2)).cpu().numpy()
    is_window_anom = per_window_err > thresh
    # map windows back to samples by marking any sample that belongs to an anomalous window
    n = arr.shape[0]
    sample_flags = np.zeros(n, dtype=bool)
    idx = 0
    for i, flag in enumerate(is_window_anom):
        start = i * stride
        end = start + window
        sample_flags[start:end] = sample_flags[start:end] | flag
    return is_window_anom, per_window_err, sample_flags
