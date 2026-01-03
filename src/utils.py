import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler


def load_csv(path):
    df = pd.read_csv(path)
    return df


def compute_magnitude(df):
    return np.sqrt(df['AccX'] ** 2 + df['AccY'] ** 2 + df['AccZ'] ** 2)


def save_scaler(scaler, path):
    joblib.dump(scaler, path)


def load_scaler(path):
    return joblib.load(path)


def fit_channel_scaler(windows):
    """Fit StandardScaler per-channel. windows shape: (n_windows, window_len, n_channels)"""
    # reshape to (n_windows * window_len, n_channels)
    n_w, wlen, n_ch = windows.shape
    arr = windows.reshape(-1, n_ch)
    scaler = StandardScaler().fit(arr)
    return scaler


def scale_windows(windows, scaler):
    n_w, wlen, n_ch = windows.shape
    arr = windows.reshape(-1, n_ch)
    arr_scaled = scaler.transform(arr)
    return arr_scaled.reshape(n_w, wlen, n_ch)
