import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from tqdm import tqdm

from src.dataset import windows_from_array, SlidingWindowDataset
from src.model import ConvAutoencoder
from src.utils import load_csv, fit_channel_scaler, scale_windows, save_scaler


def split_windows(windows, val_frac=0.2, seed=0):
    n = windows.shape[0]
    n_val = int(n * val_frac)
    n_train = n - n_val
    return windows[:n_train], windows[n_train:]


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    df = load_csv(args.data)
    arr = df[['AccX', 'AccY', 'AccZ']].values

    windows = windows_from_array(arr, window=args.window, stride=args.stride)
    np.random.seed(args.seed)
    np.random.shuffle(windows)

    train_w, val_w = split_windows(windows, val_frac=args.val_frac)

    scaler = fit_channel_scaler(train_w)
    save_scaler(scaler, os.path.join(args.outdir, 'scaler.joblib'))
    train_w = scale_windows(train_w, scaler)
    val_w = scale_windows(val_w, scaler)

    train_ds = SlidingWindowDataset(train_w)
    val_ds = SlidingWindowDataset(val_w)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = ConvAutoencoder(in_channels=3, latent_channels=args.latent).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    best_val = float('inf')
    os.makedirs(args.outdir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            batch = batch.to(device)
            recon = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)
        train_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        errors = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                loss = criterion(recon, batch)
                val_loss += loss.item() * batch.size(0)
                # compute per-window errors
                per_sample = torch.mean((recon - batch) ** 2, dim=(1, 2)).cpu().numpy()
                errors.append(per_sample)
        val_loss /= len(val_loader.dataset)
        errors = np.concatenate(errors, axis=0)

        print(f"Epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        # checkpoint best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(args.outdir, 'best_model.pth'))
            np.save(os.path.join(args.outdir, 'val_errors.npy'), errors)

    # choose threshold as 99th percentile of val errors
    val_errors = np.load(os.path.join(args.outdir, 'val_errors.npy'))
    thresh = np.percentile(val_errors, args.threshold_pct)
    np.save(os.path.join(args.outdir, 'threshold.npy'), np.array([thresh]))
    print(f"Saved model and threshold={thresh:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='safe_data.csv')
    parser.add_argument('--outdir', default='checkpoints')
    parser.add_argument('--window', type=int, default=128)
    parser.add_argument('--stride', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--val_frac', type=float, default=0.2)
    parser.add_argument('--threshold_pct', type=float, default=99.0)
    parser.add_argument('--latent', type=int, default=32)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    train(args)
