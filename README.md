# Railway CNN Autoencoder Anomaly Detection 🛤️

This project trains a 1D-CNN autoencoder on *safe* accelerometer readings (`safe_data.csv`) so that deviations are flagged as anomalies (unsafe).

Quick start

1. Create a virtual env and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Train a model (defaults: window=128, stride=64):

```bash
python -m src.train --data safe_data.csv --outdir checkpoints
```

3. Run inference from Python:

```python
from src.inference import load_model_and_meta, classify_series
import pandas as pd

model, scaler, thresh = load_model_and_meta('checkpoints')
df = pd.read_csv('safe_data.csv')
arr = df[['AccX','AccY','AccZ']].values
is_win, errs, sample_flags = classify_series(arr, model, scaler, thresh)
```

Notebook

There is a notebook at `notebooks/cnn_autoencoder_eda_train.ipynb` that demonstrates EDA, training, and synthetic anomaly evaluation.

Notes

- The training script saves `best_model.pth`, `scaler.joblib`, `val_errors.npy`, and `threshold.npy` in the output directory.
- Threshold selection uses the 99th percentile by default; tune `--threshold_pct` as needed.
