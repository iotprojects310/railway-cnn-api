import argparse
import numpy as np
import torch
from .data import load_safe_csv, window_data, generate_anomalies, AccelerometerDataset
from .model import Conv1DClassifier


def predict_on_windows(model, X_windows, device='cpu'):
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X_windows).float().to(device)  # (N, seq_len, channels)
        xb = xb.permute(0, 2, 1)  # (N, channels, seq_len)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
    return preds, probs


def main(args):
    data = load_safe_csv(args.csv)
    windows = window_data(data, args.window_size, args.step)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Conv1DClassifier(in_channels=3, num_classes=2).to(device)
    ckpt = torch.load(args.model, map_location=device)
    model.load_state_dict(ckpt['model_state'])

    # run prediction on a handful of windows
    sample = windows[:min(10, len(windows))]
    preds, probs = predict_on_windows(model, sample, device=device)

    print('Index  Pred  Prob(Class0)  Prob(Class1)')
    for i,(p,pr) in enumerate(zip(preds, probs)):
        print(f'{i:5d}  {p:4d}  {pr[0]:.4f}        {pr[1]:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='safe_data.csv')
    parser.add_argument('--model', default='models/best_model_small.pt')
    parser.add_argument('--window_size', type=int, default=64)
    parser.add_argument('--step', type=int, default=16)
    args = parser.parse_args()
    main(args)
