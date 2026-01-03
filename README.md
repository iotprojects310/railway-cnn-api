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

Supabase classification script

The `scripts/classify_supabase.py` script can fetch rows from a Supabase table and classify each row as **Class 1 (safe)** or **Class 2 (unsafe)**. It writes results to a CSV (default `supabase_classification1.csv`).

Usage examples:

```bash
# Single run and write results to supabase_classification1.csv
export SUPABASE_URL="https://<your-project>.supabase.co"
export SUPABASE_KEY="<service-key>"
export PYTHONPATH=.
python scripts/classify_supabase.py --table sensor_data --checkpoint checkpoints --out_csv supabase_classification1.csv

# Run in watch mode: poll the table and rewrite CSV when changes are detected
python scripts/classify_supabase.py --table sensor_data --checkpoint checkpoints --watch --interval 5
```

The script also supports `--update` which will write the `safety_class` back into the table per-row (requires a unique integer primary key column such as `id`).

New: direct website delivery

You can make the repository *push* classification results directly to your website (instead of writing a CSV) by:

1. Exposing a small receiver endpoint on your website (example: `web/receive_classifications.php` in this repo). The endpoint expects a POST JSON payload and will save the latest payload to a local cache file.

2. Setting GitHub Actions secrets in the repository: `SUPABASE_URL`, `SUPABASE_KEY`, `WEBSITE_ENDPOINT`, and `WEBSITE_TOKEN` (optional, used for `Authorization: Bearer` when posting).

3. The workflow `.github/workflows/publish-classification.yml` runs every minute and will execute `scripts/classify_supabase.py` which now can `--post_url` to your website. The workflow uses secrets and will POST classification JSON directly to your site so the website receives data directly (no CSV required).

Note about real-time frequency

- GitHub Actions' minimum cron granularity is 1 minute, so you cannot run an Action every 5s. For true sub-minute, immediate updates, use Supabase Realtime (direct DB change to client) or host a small always-on server (or serverless function) that subscribes to Supabase changes and POSTs immediately to your website.
