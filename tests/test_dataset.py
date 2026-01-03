import numpy as np
from src.dataset import windows_from_array


def test_windows_basic():
    arr = np.arange(300).reshape(100, 3)
    w = windows_from_array(arr, window=20, stride=10)
    assert w.shape[1] == 20
    assert w.shape[2] == 3
    # count windows
    assert w.shape[0] == (100 - 20) // 10 + 1
