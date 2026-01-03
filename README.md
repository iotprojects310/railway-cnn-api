# Railway Accelerometer CNN Classifier

This project implements a simple 1D Convolutional Neural Network to classify accelerometer windows as:
- Class 1: same behavior as provided `safe_data.csv` (safe)
- Class 2: different behavior (anomalous)

Quickstart

1. Install Python dependencies:

   pip install -r requirements.txt

2. Train a quick model (small example):

   python -m src.train --epochs 5 --window_size 64 --batch_size 32

Files added
- `src/data.py`: data loading, windowing and anomaly generation utilities
- `src/model.py`: simple Conv1D classifier
- `src/train.py`: training loop and evaluation
- `requirements.txt`: Python package requirements

Notes
- Anomalies are synthetically generated from the safe windows; replace or extend the generator with real anomalous data if available.
- The scripts are kept minimal for clarity and easy adaptation.
