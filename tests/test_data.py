import numpy as np
from src.data import load_safe_csv, window_data, generate_anomalies


def test_windowing_and_anomalies():
    data = load_safe_csv('safe_data.csv')
    windows = window_data(data, window_size=32, step=8)
    assert windows.ndim == 3
    assert windows.shape[1] == 32

    X, y = generate_anomalies(windows, seed=123, anomaly_rate=0.5)
    assert X.shape[0] == len(windows) + int(len(windows)*0.5)
    assert X.shape[1] == windows.shape[1]
    assert set(y) <= {0,1}
